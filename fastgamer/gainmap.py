"""Gain-map mathematics for FastGaMer (CVPR 2026), reproduced exactly per paper.

This module is the single source of truth for the forward/inverse gain-map
formulation described in Sec. 3 of the paper:

    (1)  GM_pred^log = Q_max * GM_norm^log          ,  GM_norm^log in [-1, 1]
    (2)  I_HDR^log   = log2( I_HR^gamma + offset ) + GM_pred^log
    (3)  I_HDR^lin   = 2^( I_HDR^log ) - offset

with ``gamma = 2.2`` and ``offset = 1/64`` (industry-standard Adobe/Google gain
map convention, Eq. 2 in the paper). The ground-truth (color) gain map used for
supervision is the per-channel log-ratio between the linear HDR image and the
gamma-linearized SDR image:

    GM_gt^log   = log2( I_HDR^lin + offset ) - log2( I_HR^gamma + offset )
    Q_max^gt    = max( | GM_gt^log | )   (single scalar per image, Sec. 3.1)
    GM_gt-norm  = GM_gt^log / Q_max^gt   (in [-1, 1])

Both NumPy (for offline data preparation) and Torch (for the in-graph training /
reconstruction path) implementations are provided and kept numerically
identical.

PQ (SMPTE ST 2084) constants and the 203 nit reference white match the sister
GMNet codebase so that PQ-domain metrics are comparable.
"""
from __future__ import annotations

import numpy as np

try:  # torch is optional for pure data-prep usage
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Constants (paper Eq. 2 + industry convention)
# --------------------------------------------------------------------------- #
GAMMA: float = 2.2            # SDR -> linear power
OFFSET: float = 1.0 / 64.0    # 0.015625, additive offset in the log-gain map
REF_WHITE_NITS: float = 203.0 # SDR diffuse white in nits (for PQ-domain metrics)

# SMPTE ST 2084 / BT.2100 PQ constants (identical to sister codebase pq_oetf).
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875


# --------------------------------------------------------------------------- #
# NumPy path (offline dataset preparation)
# --------------------------------------------------------------------------- #
def sdr_to_linear_np(sdr: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Gamma-decode an SDR image in [0, 1] to relative linear light: I_HR^gamma."""
    return np.power(np.clip(sdr, 0.0, None), gamma)


def compute_gt_gainmap_np(sdr: np.ndarray, hdr_lin: np.ndarray,
                          gamma: float = GAMMA, offset: float = OFFSET) -> np.ndarray:
    """Per-channel ground-truth log-gain map (Sec. 3 / inverse of Eq. 2).

    Args:
        sdr:      gamma-encoded SDR in [0, 1], shape (H, W, C).
        hdr_lin:  relative linear HDR (1.0 == SDR diffuse white), shape (H, W, C).
    Returns:
        log-domain gain map, same shape, in stops (log2).
    """
    sdr_lin_off = np.power(np.clip(sdr, 0.0, None), gamma) + offset
    hdr_off = np.clip(hdr_lin, 0.0, None) + offset
    return np.log2(hdr_off) - np.log2(sdr_lin_off)


def normalize_gainmap_np(gm_log: np.ndarray, eps: float = 1e-6):
    """Normalize a log-gain map to [-1, 1] by its per-image max absolute value.

    Returns ``(gm_norm, q_max)`` where ``q_max`` is a single scalar (Sec. 3.1).
    """
    q_max = float(np.max(np.abs(gm_log)))
    q_max = max(q_max, eps)
    gm_norm = np.clip(gm_log / q_max, -1.0, 1.0)
    return gm_norm.astype(np.float32), np.float32(q_max)


def reconstruct_hdr_linear_np(sdr: np.ndarray, gm_pred_log: np.ndarray,
                              gamma: float = GAMMA, offset: float = OFFSET) -> np.ndarray:
    """Reconstruct relative linear HDR from SDR + log-gain map (Eq. 2 & 3)."""
    sdr_lin_off = np.power(np.clip(sdr, 0.0, None), gamma) + offset
    hdr_log = np.log2(sdr_lin_off) + gm_pred_log
    hdr_lin = np.power(2.0, hdr_log) - offset
    return np.clip(hdr_lin, 0.0, None)


def pq_oetf_np(x_nits: np.ndarray) -> np.ndarray:
    """Linear light in nits -> PQ code value in [0, 1] (ST 2084 OETF)."""
    ym = np.power(np.clip(x_nits, 0.0, None) / 10000.0, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * ym) / (1.0 + _PQ_C3 * ym), _PQ_M2)


def pq_eotf_np(x: np.ndarray) -> np.ndarray:
    """PQ code value in [0, 1] -> linear light in nits (ST 2084 EOTF)."""
    x = np.clip(x, 0.0, None)
    p = np.power(x, 1.0 / _PQ_M2)
    num = np.clip(p - _PQ_C1, 0.0, None)
    den = _PQ_C2 - _PQ_C3 * p
    return 10000.0 * np.power(num / np.clip(den, 1e-12, None), 1.0 / _PQ_M1)


def linear_to_pq_np(hdr_lin_relative: np.ndarray,
                    ref_white_nits: float = REF_WHITE_NITS) -> np.ndarray:
    """Relative linear HDR (1.0 == diffuse white) -> PQ code values in [0, 1].

    Matches the sister codebase metric recipe: ``pq_oetf(linear * 203)``.
    """
    return pq_oetf_np(hdr_lin_relative * ref_white_nits)


# --------------------------------------------------------------------------- #
# Torch path (in-graph training / reconstruction / eval)
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    def sdr_to_linear(sdr: "torch.Tensor", gamma: float = GAMMA) -> "torch.Tensor":
        return torch.clamp(sdr, min=0.0) ** gamma

    def reconstruct_hdr_linear(sdr: "torch.Tensor", gm_pred_log: "torch.Tensor",
                               gamma: float = GAMMA, offset: float = OFFSET) -> "torch.Tensor":
        """Eq. 2 & 3 in Torch (differentiable). ``sdr`` in [0, 1], gm in stops."""
        sdr_lin_off = torch.clamp(sdr, min=0.0) ** gamma + offset
        hdr_log = torch.log2(sdr_lin_off) + gm_pred_log
        hdr_lin = torch.pow(2.0, hdr_log) - offset
        return torch.clamp(hdr_lin, min=0.0)

    def compute_gt_gainmap(sdr: "torch.Tensor", hdr_lin: "torch.Tensor",
                           gamma: float = GAMMA, offset: float = OFFSET) -> "torch.Tensor":
        sdr_lin_off = torch.clamp(sdr, min=0.0) ** gamma + offset
        hdr_off = torch.clamp(hdr_lin, min=0.0) + offset
        return torch.log2(hdr_off) - torch.log2(sdr_lin_off)

    def normalize_gainmap(gm_log: "torch.Tensor", eps: float = 1e-6):
        """Per-sample [-1, 1] normalization. gm_log: (B, C, H, W).

        Returns ``(gm_norm, q_max)`` with ``q_max`` of shape (B, 1, 1, 1).
        """
        b = gm_log.shape[0]
        q_max = gm_log.abs().reshape(b, -1).amax(dim=1).clamp(min=eps)
        q_max = q_max.view(b, 1, 1, 1)
        gm_norm = torch.clamp(gm_log / q_max, -1.0, 1.0)
        return gm_norm, q_max

    def pq_oetf(x_nits: "torch.Tensor") -> "torch.Tensor":
        ym = torch.clamp(x_nits, min=0.0).div(10000.0).pow(_PQ_M1)
        return ((_PQ_C1 + _PQ_C2 * ym) / (1.0 + _PQ_C3 * ym)).pow(_PQ_M2)

    def linear_to_pq(hdr_lin_relative: "torch.Tensor",
                     ref_white_nits: float = REF_WHITE_NITS) -> "torch.Tensor":
        return pq_oetf(hdr_lin_relative * ref_white_nits)
