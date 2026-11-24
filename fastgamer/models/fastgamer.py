"""FastGaMer main model (Sec. 3, Fig. 3).

Pipeline
--------
1. Downsample the high-resolution SDR ``I_HR`` to a 256x256 thumbnail
   ``I_thumb`` (or use a separately-provided whole-image thumbnail so the scale
   stays globally consistent -- Sec. 3.1).
2. Encode the thumbnail into a global feature ``F_thumb``.
3. From ``F_thumb`` predict, in parallel, the absolute scale ``Q_max`` and three
   image-adaptive operators: a bilateral grid, a 3D LUT, and a neural modulator.
4. Apply the operators *sequentially* to the full-resolution ``I_HR``
   (all network-free): grid slicing -> ``I_grid`` -> LUT transform ->
   ``GM_LUT^log`` -> affine+tanh modulation -> ``GM_norm^log``.
5. Scale: ``GM_pred^log = Q_max * GM_norm^log`` (Eq. 1), then reconstruct the
   linear HDR image via Eq. 2 & 3.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import gainmap
from .encoder import GlobalEncoder
from .scale_head import ScaleHead
from .bilateral_grid import BilateralGridGenerator
from .lut3d import LUT3DGenerator
from .modulator import NeuralModulator


class FastGaMer(nn.Module):
    def __init__(self,
                 in_nc: int = 3,
                 out_nc: int = 3,
                 feat_dim: int = 64,
                 thumb_size: int = 256,
                 num_grids: int = 3,
                 grid_size: int = 8,
                 lut_dim: int = 17,
                 enc_widths=(16, 32, 48, 64, 64),
                 hidden_scale: int = 32,
                 hidden_grid: int = 32,
                 hidden_lut: int = 32,
                 hidden_mod: int = 32,
                 gamma: float = gainmap.GAMMA,
                 offset: float = gainmap.OFFSET):
        super().__init__()
        self.thumb_size = thumb_size
        self.gamma = gamma
        self.offset = offset

        self.encoder = GlobalEncoder(in_nc, tuple(enc_widths), feat_dim)
        self.scale_head = ScaleHead(feat_dim, hidden_scale)
        self.grid = BilateralGridGenerator(feat_dim, hidden_grid, num_grids, grid_size, in_nc)
        self.lut = LUT3DGenerator(feat_dim, hidden_lut, lut_dim, out_nc)
        self.modulator = NeuralModulator(feat_dim, hidden_mod, out_nc)

    def make_thumb(self, i_hr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(i_hr, size=(self.thumb_size, self.thumb_size),
                             mode="bilinear", align_corners=False)

    def forward(self, i_hr: torch.Tensor, i_thumb: Optional[torch.Tensor] = None) -> dict:
        if i_thumb is None:
            i_thumb = self.make_thumb(i_hr)

        feat = self.encoder(i_thumb)                        # (B, feat_dim)
        q_max = self.scale_head(feat)                       # (B,1,1,1)

        i_grid, grids = self.grid(feat, i_hr)              # (B,3,H,W)
        gm_lut, lut = self.lut(feat, i_grid)              # (B,3,H,W), (B,3,Nt,Nt,Nt)
        gm_norm = self.modulator(feat, gm_lut)            # (B,3,H,W) in [-1,1]

        gm_pred = q_max * gm_norm                          # Eq. 1
        hdr_lin = gainmap.reconstruct_hdr_linear(i_hr, gm_pred, self.gamma, self.offset)

        return {
            "gm_norm": gm_norm,     # normalized log-GM in [-1, 1]
            "gm_pred": gm_pred,     # absolute log-GM (stops)
            "q_max": q_max,         # (B,1,1,1) absolute scale
            "lut": lut,             # (B,3,Nt,Nt,Nt) for smoothness/monotonicity reg
            "grids": grids,         # (B,K,Nb,Nb,Nb)
            "i_grid": i_grid,       # spatially-modulated base
            "gm_lut": gm_lut,       # pre-modulation log-GM
            "hdr_lin": hdr_lin,     # reconstructed relative linear HDR
        }


def build_model(cfg: dict) -> FastGaMer:
    """Instantiate FastGaMer from a config dict (see configs/fastgamer.yaml)."""
    m = cfg.get("model", {})
    return FastGaMer(
        in_nc=m.get("in_nc", 3),
        out_nc=m.get("out_nc", 3),
        feat_dim=m.get("feat_dim", 64),
        thumb_size=m.get("thumb_size", 256),
        num_grids=m.get("num_grids", 3),
        grid_size=m.get("grid_size", 8),
        lut_dim=m.get("lut_dim", 17),
        enc_widths=tuple(m.get("enc_widths", (16, 32, 48, 64, 64))),
        hidden_scale=m.get("hidden_scale", 32),
        hidden_grid=m.get("hidden_grid", 32),
        hidden_lut=m.get("hidden_lut", 32),
        hidden_mod=m.get("hidden_mod", 32),
        gamma=m.get("gamma", gainmap.GAMMA),
        offset=m.get("offset", gainmap.OFFSET),
    )
