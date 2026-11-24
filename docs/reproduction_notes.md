# FastGaMer 复现说明:逐条对照论文

本文件把论文每一处可实现的细节映射到代码,并明确声明"论文未给"而由本复现推断的部分,以及"无法完全重现"的部分。

## 1. 总体框架(Sec. 3, Fig. 3)

| 论文 | 代码 |
|---|---|
| `I_HR` → 256×256 缩略图 `I_thumb` | `FastGaMer.make_thumb`(bilinear);数据集也可提供整图缩略图 |
| Encoder → 全局特征 `F_thumb` | `models/encoder.py::GlobalEncoder`(strided conv + 全局池化 → 向量) |
| 由 `F_thumb` 并行生成 4 个算子 | `FastGaMer.forward` 中并行调用四个子模块 |
| 高分辨率操作 network-free | 网格切片 / LUT 变换 / 仿射调制均为逐点/查表算子(`F.grid_sample` + 1×1) |

## 2. 公式(Eq. 1–3, Sec. 3)

- (1) `GM_pred^log = Q_max · GM_norm^log`,`GM_norm ∈ [-1,1]` → `fastgamer.py`(`q_max * gm_norm`),`gainmap.normalize_gainmap`。
- (2) `I_HDR^log = log2(I_HR^γ + offset) + GM_pred^log` → `gainmap.reconstruct_hdr_linear`。
- (3) `I_HDR^lin = 2^(I_HDR^log) − offset` → 同上。
- γ=2.2,offset=1/64 → `gainmap.GAMMA/OFFSET`,配置 `model.gamma/offset`。
- GT 反推 `GM_gt^log = log2(HDR+offset) − log2(SDR^γ+offset)`(逐通道)→ `gainmap.compute_gt_gainmap_np/compute_gt_gainmap`。
- **数值验证**:`reconstruct(SDR, compute_gt(SDR,HDR)) == HDR`,最大误差 ~1e-6;torch 与 numpy 实现一致(~1e-6)。

## 3. Scale Estimation(Sec. 3.1)

| 论文 | 代码 |
|---|---|
| 两层 MLP:`F_thumb → Q_max` | `models/scale_head.py`(Linear→ReLU→Linear→softplus,保正) |
| 仅从整图缩略图估计,保持全局一致 | 训练时数据集用整图缩略图(非 patch)计算 `I_thumb` 与 `Q_max` GT |

`Q_max` 为**单个正标量**(stops),量纲与 `GM_gt^log` 一致(未除以 `log2(peak)`,区别于姊妹库 GMNet 的归一化)。

## 4. 双边网格(Sec. 3.2)

| 论文 | 代码 |
|---|---|
| 轻量 MLP 生成 K 个 `Nb×Nb×Nb` 网格,K=3, Nb=8 | `models/bilateral_grid.py`,MLP → `(B,K,Nb,Nb,Nb)` |
| 复用输入 RGB 作为 range guidance | 网格 k 的切片 guidance = 输入通道 k(要求 K==in_nc) |
| 切片后与输入 RGB 经 1×1 投影融合成 `I_grid` | 单次批量 3D `grid_sample`(B·K 批)+ `nn.Conv2d(K+3→3, 1×1)` |

初始化:MLP 末层置零(网格≈0)、1×1 融合初始化为直通 RGB,使 `I_grid≈I_HR`,训练从恒等起步。

## 5. 3D LUT(Sec. 3.3)

| 论文 | 代码 |
|---|---|
| 两层 MLP 预测 `T`,维度 `3×Nt×Nt×Nt`,Nt=17 | `models/lut3d.py`,MLP → 残差,叠加到 identity LUT |
| 对 `I_grid` 三线性采样 → `GM_LUT^log` | 5D `F.grid_sample`(padding=border, align_corners) |

**轴序验证**:对 identity LUT,`transform(identity, img) == img`,最大误差 ~1e-7(确认 (x,y,z)=(B,G,R) 轴序正确)。残差零初始化 → LUT 从恒等映射起步(平滑、单调,正则起始≈0)。

## 6. 神经调制器(Sec. 3.4)

| 论文 | 代码 |
|---|---|
| 小 MLP 预测逐通道 (α,β) | `models/modulator.py`,Linear→ReLU→Linear→(2·C) |
| Eq.4:`GM_norm = tanh(GM_LUT ⊙ (1+α) + β)` | 广播到全分辨率后逐通道仿射 + tanh |

零初始化 → 起始为恒等仿射(α=0,β=0)。

## 7. 损失(Eq. 5, Sec. 3)

`L = ‖GM_norm − GM_gt-norm‖₁ + λ1‖GM_pred − GM_gt‖₁ + λ2(Ls + Lm)`,λ1=3, λ2=0.1 → `losses.py::FastGaMerLoss`。

- 数据项一:归一化 log-GM 的 L1(相对动态范围)。
- 数据项二:最终尺度 log-GM 对原始未归一化 GT 的 L1(全局尺度)。
- `Ls`(平滑,TV,含 Zeng 等的边界 bin ×2 加权)/`Lm`(单调,ReLU 前向差)= Image-Adaptive 3D LUT(Zeng 等)标准正则 → `lut_smoothness_monotonicity`。
- **佐证**:姊妹库 GMNet 的 base 损失恰为 `L1(GM_norm)×1 + L1(GM_pred)×3`,`q_pix_weight=3` 与本论文 λ1=3 一致。

## 8. 训练(Sec. 4, Implementation Details)

| 论文 | 代码/配置 |
|---|---|
| Adam,β1=0.9, β2=0.99 | `train.py`,`config.train.betas=[0.9,0.99]` |
| lr=2e-4 | `config.train.lr` |
| MultiStep 在 200k/400k/600k/800k ×0.5,无 warmup | `MultiStepLR(milestones=[200000,400000,600000,800000], gamma=0.5)` |
| 256×256 随机裁剪 + 随机水平翻转 + 旋转 | `data/dataset.py`(`random_crop`+`augment`),`config.data` |
| batch=16,V100 | `config.train.batch_size=16`(本机用 4090) |
| **验证**:60 步冒烟训练 loss 4.30→(200 步)1.16,各项下降、无 NaN | ✓ |

## 9. 评测(Sec. 4, Tab. 1/2)

| 域 | 指标 | 代码 |
|---|---|---|
| 线性(按 GT 峰值归一化到 [0,1]) | PSNR / SSIM / SRSIM | `metrics.calculate_normalized_psnr`,`calculate_ssim`,`calculate_srsim` |
| PQ(`pq_oetf(linear×203nits)`) | PSNR / SSIM / SRSIM | 同上,先 `to_pq` |
| HDR | ΔE_ITP(BT.2124) | `metrics.delta_e_itp`:`colour.RGB_to_ICtCp`(Dolby2016,L_p=10000),`720·√(ΔI²+0.25ΔCt²+ΔCp²)`,sRGB→BT.2020 后转**绝对 nits**(白点 203,由 colour 内部按 L_p 归一) |
| HDR | HDR-VDP-3 | `metrics.hdr_vdp3` 外部钩子(见下) |

- **PSNR**:线性域采用 `-10·log10(mean(((a-b)/peak)²))`,`peak=max(GT)`(与姊妹库一致);PQ 域用标准 `20·log10(1/√mse)`。
- **SSIM**:BasicSR 高斯窗(11×11,σ=1.5,C1=(0.01·255)²,C2=(0.03·255)²),逐通道平均,输入 [0,255]。
- **SRSIM**:据 Zhang & Li(ICIP 2012)自实现——谱残差视觉显著性(scale 0.25 **双三次**重采样,3×3 均值残差,**固定 10×10 σ=3.8 高斯核**,对齐参考 `spectralResidueSaliency.m`)+ Scharr 梯度幅值相似;C1=0.4, C2=225, α=0.5;整图下采样因子 `F=round(min(H,W)/256)`。

## 10. 效率(Sec. 4, Tab. 3)

见 `benchmark.py`。本机实测:参数 0.6336M(论文 0.64M)、4K 显存 ~908MB(论文 962MB)、2K 运行 2.24ms(论文 3.07ms)、4K 运行 fp32 13.5ms / **fp16 7.46ms**(论文 6.2ms@V100)。

---

## 无法完全重现 / 推断实现(诚实声明)

1. **原始数据集**:论文用 RAISE RAW 经 Adobe Camera Raw 局部 TMO 人工生成(8150 合成对)+ Adobe Indigo/iPhone 12 Pro Max 实拍(82 对)。该管线依赖 Adobe 软件与人工调参,**不可代码重现**。本复现以 `synth_degrade.py`(忠实的两阶段前向:全局辐射压缩 + 空间可变局部自适应)+ `GainMapDataset` 替代,可端到端训练/评测;绝对指标数值因数据不同而不可与论文表格逐一对齐。

2. **HDR-VDP-3**:Mantiuk 等的 MATLAB 工具箱,未内置。以 `metrics.register_hdr_vdp3(fn)` 提供外部钩子(可包 MATLAB engine 或 Python 端口);未注册时返回 NaN 并告警。论文作者亦使用外部工具。

3. **论文未给的实现细节(按最合理且轻量方式推断,已使 params/显存对齐 Tab.3)**:
   - Encoder 具体通道/深度(`enc_widths=[16,32,48,64,64]`,feat_dim=64)。
   - 各生成器 MLP 隐藏维(均 32)。
   - 1×1 融合的具体形式(`Conv2d(K+3→3)`)。
   - LUT 以 identity 为基底、预测残差(标准做法,利于稳定与正则)。
   - `Q_max` 用 softplus 保正;各模块零初始化以恒等起步。
   这些不改变方法结构,仅为可运行实现;如作者公布官方代码,可据此微调对齐。

4. **4K 运行时**:`F.grid_sample`(fp32)在 4K 下受坐标张量(~600MB)带宽限制,慢于论文 V100 上的融合/部署内核;fp16 下 7.46ms 已接近 6.2ms。FLOPs 计数为透明的 hook 估计(Conv/Linear + 高分辨率 grid_sample 8-tap 解析项),与论文的计数约定可能不同(1.2G vs 0.48G),不影响架构忠实度。

5. **双边网格的训练/测试空间尺度**(论文设计的固有属性,非代码偏差):网格由整图缩略图特征 `F_thumb` 生成,切片时空间坐标用 `linspace(-1,1)` 铺满所传入的 `I_HR`。训练时 `I_HR` 是 256 随机裁剪、测试时是整图,故网格空间单元→场景位置的对应关系在两阶段不同(仅影响网格的空间维;range/guidance 维基于强度,完全一致;LUT 与调制器空间不变)。这忠实复现了论文"单一整图 `F_thumb`"的描述——论文的训练/测试一致性保证(Sec.3.1)仅针对标量 `Q_max`,未声明网格空间单元固定对应场景位置。如需消除,可让网格输入与裁剪空间对齐(需论文/作者确认),本复现保持与论文一致的实现。

## 对抗式审查结论(workflow,11 个 agent,含对抗验证)

分 5 维度(gainmap 数学 / 模型架构 / 损失 / 指标 / 数据与训练)对代码逐条比对论文,每条发现再由独立 agent 对抗式验证。结论:

- **gainmap 数学 / 训练超参**:无发现(完全对齐 Eq.1-3 与 Sec.4)。
- **[已修·高危] ΔE_ITP 双重归一化**:`colour.RGB_to_ICtCp(L_p=10000)` 内部已按 L_p 归一(需绝对 nits 输入),原代码额外乘 `203/10000` 造成白点被压入近黑区、ΔE_ITP 系统性偏低约 6-7 倍。已改为乘绝对 nits(白点 203)。验证:白点 I 分量 0.03→**0.5807**;测试集 dEITP 落入论文合理区间。
- **[已修·低危] Ls 边界加权**:补上 Zeng 等 TV_3D 的边界 bin ×2 加权,与被引参考 [34] 精确一致。
- **[已修·低危] SR-SIM 重采样/高斯核**:双线性→双三次、自动核→固定 10×10(σ=3.8),对齐参考 `SR_SIM.m`。
- **[已修·低危] 合成退化局部色彩失真**:局部阶段增加空间可变的逐通道增益场,使 "Color Distortion"(Fig.1b)出现在局部退化中,GT 彩色 GM 的局部分量非平凡。
- **[驳回] 1×1 融合是全分辨率可学习 conv**:验证判定不成立——1×1 逐点算子属标准做法,论文"network-free"指无重型空间网络。
- **[存疑→低危] 双边网格训练/测试空间尺度**:见上第 5 条,属论文设计固有属性,保持一致。
