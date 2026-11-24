# FastGaMer 复现(CVPR 2026)

> **FastGaMer: Efficient GainMap Learning for Practical Inverse Tone Mapping**
> Yuanshen Guan, Ruikang Xu, Chang Chen, Yinuo Liao, Dehua Song, Fenglong Song, Zhiwei Xiong.
> USTC & Huawei Noah's Ark Lab.

本仓库是对该论文的**精确复现**:一个极轻量、分辨率无关的逆色调映射(ITM)框架。它不直接回归 HDR,而是重建**三通道彩色增益图(color gain map)**,并显式对齐 local-TMO 的退化结构——用**动态双边网格**逆局部退化、**可学习 3D LUT + 轻量神经调制器**恢复全局辐射映射、**MLP** 估计绝对照度尺度 `Q_max`。所有高分辨率操作 network-free,重神经计算只在 256×256 缩略图上进行。

复现严格遵循论文 Eq.1–5 与全部超参。已在本机 `conda gmnet`(torch 2.5.1+cu121,RTX 4090)端到端跑通:数据合成 → 训练 → 评测 → 推理 → 基准。

## 目录结构

```
FastGaMer/
├── configs/fastgamer.yaml     # 全部超参锚定论文
├── fastgamer/
│   ├── gainmap.py             # Eq.1-3 正/反变换、GT-GM、PQ、Q_max 归一化
│   ├── models/                # encoder / scale_head / bilateral_grid / lut3d / modulator / fastgamer
│   ├── losses.py              # Eq.5:L1(norm)+λ1·L1(pred)+λ2(Ls+Lm)
│   ├── metrics.py             # 线性/PQ 域 PSNR·SSIM·SRSIM、ΔE_ITP、HDR-VDP3 钩子
│   └── data/                  # dataset(SDR-GM/SDR-HDR)+ synth_degrade(合成 local-TMO)
├── scripts/                   # make_dataset / train / test / infer / benchmark
└── docs/reproduction_notes.md # 逐条对照论文 + 已知偏差声明
```

## 方法流水线(Sec.3 / Fig.3)

```
I_HR ──resize──► I_thumb(256²) ──Encoder──► F_thumb(全局向量)
                                              │
        ┌──────────────┬──────────────┬──────┴───────┐
      Q_max        双边网格         3D LUT        调制器(α,β)
   (MLP,Sec3.1)  (K=3,Nb=8,Sec3.2) (Nt=17,Sec3.3)  (Sec3.4)
        │              │               │               │
I_HR ───┼──网格切片───► I_grid ──LUT变换──► GM_LUT ──tanh仿射──► GM_norm∈[-1,1]
        │                                                        │
        └──────────► GM_pred = Q_max · GM_norm ◄─────────────────┘   (Eq.1)
I_HDR^lin = 2^( log2(I_HR^2.2 + 1/64) + GM_pred ) − 1/64              (Eq.2,3)
```

## 快速上手

所有命令用已验证的环境:`PY=/home/hp/miniforge3/envs/gmnet/bin/python`(或任意含 torch/cv2/colour/scipy 的环境)。

```bash
cd /mnt/data/wcm/FastGaMer

# 1) 合成一个可训练的 SDR-彩色GM 数据集(镜像论文的前向 local-TMO 退化)
$PY scripts/make_dataset.py --out datasets/fastgamer --split train \
    --image_dir <一批普通图像目录> --limit 200 --max_size 2048
$PY scripts/make_dataset.py --out datasets/fastgamer --split test \
    --image_dir <另一批图像目录> --limit 20 --max_size 2048
#   若已有相对线性 HDR(.npy/.exr/.hdr),改用 --hdr_dir <目录>,直接作为 HDR 使用。

# 2) 训练(默认锚定论文:Adam 0.9/0.99, lr 2e-4, MultiStepLR@200/400/600/800k ×0.5, batch16)
$PY scripts/train.py --config configs/fastgamer.yaml
#   冒烟测试:$PY scripts/train.py --config configs/fastgamer.yaml --max_iter 60 --batch 4

# 3) 评测(复现 Tab.1/Tab.2 的三域指标)
$PY scripts/test.py --config configs/fastgamer.yaml --ckpt experiments/fastgamer/model_latest.pth --split test

# 4) 单图推理 SDR→HDR(输出 PQ 预览 + 相对线性 .npy + GM 可视化)
$PY scripts/infer.py --ckpt experiments/fastgamer/model_latest.pth --input <SDR.png> --out out/demo

# 5) 效率基准(复现 Tab.3:Params/Runtime/FLOPs/Memory @4K/2K)
$PY scripts/benchmark.py --config configs/fastgamer.yaml            # fp32
$PY scripts/benchmark.py --config configs/fastgamer.yaml --half     # fp16(部署)
```

## 与论文的量化对齐(本机 RTX 4090 实测)

| 指标 | 论文 Tab.3 @4K | 本复现 |
|---|---|---|
| 参数量 | 0.64 M | **0.6336 M** |
| 4K 显存 | 962 MB | **~908 MB** |
| 4K 运行时 | 6.20 ms (V100) | 13.5 ms (fp32) / **7.46 ms (fp16)** |
| 2K 运行时 | 3.07 ms | **2.24 ms (fp32) / 1.44 ms (fp16)** |

参数量与显存高度吻合;2K 运行时优于论文;4K fp16 运行时接近论文(V100 上的差异主要来自其融合/部署内核与精度)。详见 `docs/reproduction_notes.md`。

## 关键实现约定

- **增益图数学**:`GM_gt^log = log2(HDR+1/64) − log2(SDR^2.2+1/64)`(逐通道);`Q_max = max(|GM_gt^log|)`;`GM_norm = GM_gt^log/Q_max ∈ [-1,1]`。重建见 Eq.2,3。γ=2.2, offset=1/64(Adobe/Google 约定)。
- **损失**:`L = ‖GM_norm−GT_norm‖₁ + 3‖GM_pred−GT‖₁ + 0.1(Ls+Lm)`。Ls/Lm 为 Image-Adaptive 3D LUT(Zeng 等)的平滑/单调正则。
- **指标域**:线性域按 GT 峰值归一化到 [0,1];PQ 域 `pq_oetf(linear×203nits)`;ΔE_ITP 用 `colour.RGB_to_ICtCp`(Dolby2016,L_p=10000),`720·√(ΔI²+0.25ΔCt²+ΔCp²)`。

## 数据说明

论文原始数据来自 RAISE RAW 经 Adobe Camera Raw 局部色调映射人工生成 + Adobe Indigo/iPhone 实拍,**无法用代码完全重现**。本仓库提供:
1. `synth_degrade.py`:忠实的前向 local-TMO 模型(全局辐射压缩 + 空间可变局部自适应),从相对线性 HDR 造自洽 SDR-彩色GM 对;
2. `GainMapDataset`:`sdr_hdr`(实时按 Eq. 算彩色 GM,免 8bit 量化)与 `sdr_gm`(预存 GM + metadata)双模式。

## 依赖

见 `requirements.txt`。核心:torch / numpy / opencv-python / colour-science / PyYAML / scipy。
`tqdm`/`matplotlib`/`thop` 可选(缺失时自动降级)。HDR-VDP-3 为外部 MATLAB 工具箱,用
`fastgamer.metrics.register_hdr_vdp3(fn)` 注册后启用。
