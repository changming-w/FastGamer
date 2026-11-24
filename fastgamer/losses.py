"""FastGaMer training objective (Eq. 5).

    L = || GM_norm - GM_gt_norm ||_1
      + lambda1 * || GM_pred - GM_gt ||_1
      + lambda2 * ( Ls + Lm )

with ``lambda1 = 3`` (emphasize both absolute scale and relative dynamic range)
and ``lambda2 = 0.1`` (LUT smoothness + monotonicity). The two data terms are the
per-pixel L1 losses on (a) the normalized log-GM against its normalized
reference and (b) the final scaled log-GM against the original unnormalized
reference. ``Ls``/``Lm`` are the standard Image-Adaptive 3D LUT regularizers
(Zeng et al.): total variation across the three LUT axes and a ReLU monotonicity
penalty.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tv_boundary_weighted(dif: torch.Tensor, axis: int) -> torch.Tensor:
    """Mean of ``dif^2`` with the two boundary bins along ``axis`` weighted x2,
    matching the official Zeng et al. TV_3D smoothness regularizer."""
    w = torch.ones_like(dif)
    idx = [slice(None)] * dif.dim()
    idx[axis] = 0
    w[tuple(idx)] = 2.0
    idx[axis] = dif.shape[axis] - 1
    w[tuple(idx)] = 2.0
    return (w * dif.pow(2)).mean()


def lut_smoothness_monotonicity(lut: torch.Tensor):
    """Return (Ls, Lm) for a LUT of shape (B, C, Nt, Nt, Nt).

    Axes (2, 3, 4) correspond to the (R, G, B) input dimensions. Ls is the
    boundary-weighted TV (squared adjacent differences, x2 at the two boundary
    bins per axis -- Zeng et al.); Lm is the ReLU of the forward differences
    (penalizing non-monotonic, decreasing entries).
    """
    dif_r = lut[:, :, :-1, :, :] - lut[:, :, 1:, :, :]
    dif_g = lut[:, :, :, :-1, :] - lut[:, :, :, 1:, :]
    dif_b = lut[:, :, :, :, :-1] - lut[:, :, :, :, 1:]

    ls = (_tv_boundary_weighted(dif_r, 2)
          + _tv_boundary_weighted(dif_g, 3)
          + _tv_boundary_weighted(dif_b, 4))
    lm = (F.relu(dif_r).mean() + F.relu(dif_g).mean() + F.relu(dif_b).mean())
    return ls, lm


class FastGaMerLoss(nn.Module):
    def __init__(self, lambda1: float = 3.0, lambda2: float = 0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.l1 = nn.L1Loss()

    def forward(self, pred: dict, target: dict) -> dict:
        """
        pred:   output dict of FastGaMer (needs 'gm_norm', 'gm_pred', 'lut').
        target: dict with 'gm_norm' (normalized GT) and 'gm_pred' (absolute GT).
        """
        l_norm = self.l1(pred["gm_norm"], target["gm_norm"])
        l_scale = self.l1(pred["gm_pred"], target["gm_pred"])
        ls, lm = lut_smoothness_monotonicity(pred["lut"])

        total = l_norm + self.lambda1 * l_scale + self.lambda2 * (ls + lm)
        return {
            "total": total,
            "l_norm": l_norm.detach(),
            "l_scale": l_scale.detach(),
            "l_smooth": ls.detach(),
            "l_mono": lm.detach(),
        }
