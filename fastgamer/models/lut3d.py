"""Learnable 3D LUT for global SDR-to-GM translation (Sec. 3.3).

A lightweight two-layer MLP predicts the parameters ``T`` of a 3D LUT of
dimension ``3 x Nt x Nt x Nt`` (Nt=17 in the paper) from the global thumbnail
feature ``F_thumb``. The LUT directly processes the intermediate high-resolution
``I_grid`` through tri-linear sampling, producing a scene-aware intensity/color
remapping from the SDR domain to the log-GM domain (``GM_LUT^log``). Being a
table lookup, the high-resolution transform is network-free.

The MLP predicts a *residual* on an identity LUT so training starts from a
well-behaved, smooth, monotonic mapping.

Axis convention (verified against an identity LUT): the volume is indexed
``T[b, c, i_r, i_g, i_b]`` with (axis0, axis1, axis2) = (R, G, B). For 3D
``grid_sample`` the coordinate tuple is ordered ``(x=W, y=H, z=D) = (B, G, R)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_identity_lut(nt: int) -> torch.Tensor:
    """Identity 3D LUT of shape (3, Nt, Nt, Nt): output(r,g,b) = (r,g,b)."""
    axis = torch.linspace(0.0, 1.0, nt)
    r = axis.view(nt, 1, 1).expand(nt, nt, nt)   # varies along axis0 (R)
    g = axis.view(1, nt, 1).expand(nt, nt, nt)   # varies along axis1 (G)
    b = axis.view(1, 1, nt).expand(nt, nt, nt)   # varies along axis2 (B)
    return torch.stack([r, g, b], dim=0).contiguous()  # (3, Nt, Nt, Nt)


class LUT3DGenerator(nn.Module):
    def __init__(self, feat_dim: int = 64, hidden: int = 32, lut_dim: int = 17,
                 out_nc: int = 3):
        super().__init__()
        self.Nt = lut_dim
        self.out_nc = out_nc
        out_size = out_nc * lut_dim ** 3
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_size),
        )
        # Zero-init residual -> LUT starts exactly at identity.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("identity", build_identity_lut(lut_dim))  # (3,Nt,Nt,Nt)

    def generate(self, feat: torch.Tensor) -> torch.Tensor:
        """F_thumb -> LUT of shape (B, out_nc, Nt, Nt, Nt) = identity + residual."""
        b = feat.shape[0]
        residual = self.mlp(feat).view(b, self.out_nc, self.Nt, self.Nt, self.Nt)
        return self.identity.unsqueeze(0) + residual

    def transform(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Tri-linear LUT lookup of ``img`` (B,3,H,W in [0,1]) -> (B,out_nc,H,W)."""
        b, _, h, w = img.shape
        rgb = img.clamp(0.0, 1.0)
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        bl = rgb[:, 2:3]
        # grid coords ordered (x=W=B, y=H=G, z=D=R), scaled to [-1, 1].
        coords = torch.stack([bl, g, r], dim=-1) * 2.0 - 1.0        # (B,1,H,W,3)
        coords = coords.view(b, 1, h, w, 3)
        out = F.grid_sample(lut, coords, mode="bilinear",
                            padding_mode="border", align_corners=True)  # (B,out,1,H,W)
        return out[:, :, 0]

    def forward(self, feat: torch.Tensor, img: torch.Tensor):
        lut = self.generate(feat)
        gm_lut = self.transform(lut, img)
        return gm_lut, lut
