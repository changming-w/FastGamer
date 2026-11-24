"""Synthetic local-TMO degradation (forward model of Fig. 1b).

The paper builds its dataset by running RAW imagery through Adobe Camera Raw's
*local* tone mapper, which the authors characterize as a **global radiometric
compression** (dynamic-range / gamut / global-contrast shaping) composed with a
**spatially-varying local adaptation** (exposure, contrast remapping, color
distortion). That exact pipeline needs Adobe software and manual grading and
cannot be reproduced in code. This module provides a faithful *forward model* of
that degradation so SDR--color-GM training pairs can be generated from linear HDR
imagery without Adobe, mirroring the same two-stage structure FastGaMer is
designed to invert.

Given a relative-linear HDR image ``hdr_lin`` (1.0 == diffuse white, highlights
> 1), it produces a gamma-encoded 8-bit-style SDR in [0, 1]:

    1. Global compression  : a global highlight-rolloff tone curve (Reinhard-like)
                             plus a mild per-channel gamut compression toward a
                             desaturated point.
    2. Local adaptation    : a smooth, spatially-varying exposure/contrast field
                             (low-frequency random field) that locally lifts
                             shadows / compresses highlights -- the hard part ITM
                             must undo.
    3. OETF                : encode to gamma space (SDR = linear ** (1/gamma)).

The corresponding color gain map GT is then computed by ``gainmap`` from the SDR
and the original ``hdr_lin`` (never from the degraded intermediate), so the pair
is self-consistent with Eq. 2/3.
"""
from __future__ import annotations

import numpy as np
import cv2

from .. import gainmap


def _smooth_random_field(h: int, w: int, ctrl: int, lo: float, hi: float,
                         rng: np.random.Generator) -> np.ndarray:
    """A smooth low-frequency field in [lo, hi], generated from a coarse grid."""
    coarse = rng.uniform(lo, hi, size=(ctrl, ctrl)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, lo, hi)[..., None]


def global_compression(hdr_lin: np.ndarray, key: float = 0.5,
                       gamut_strength: float = 0.15) -> np.ndarray:
    """Global dynamic-range + gamut compression on relative-linear HDR."""
    lum = (0.2126 * hdr_lin[..., 0] + 0.7152 * hdr_lin[..., 1]
           + 0.0722 * hdr_lin[..., 2])[..., None]
    # Extended Reinhard highlight rolloff, keyed to the scene mean.
    l_white = max(float(np.percentile(lum, 99.5)), 1e-3)
    scale = key / (float(np.mean(lum)) + 1e-6)
    l_scaled = lum * scale
    l_out = l_scaled * (1.0 + l_scaled / (l_white * scale) ** 2) / (1.0 + l_scaled)
    ratio = l_out / (lum + 1e-6)
    comp = hdr_lin * ratio
    # Mild per-channel gamut compression toward the luminance (desaturate highlights).
    comp = comp * (1.0 - gamut_strength) + l_out * gamut_strength
    return np.clip(comp, 0.0, None)


def local_adaptation(lin: np.ndarray, rng: np.random.Generator,
                     ctrl: int = 6, exp_range=(0.75, 1.4),
                     contrast_range=(0.85, 1.2), color_range=(0.9, 1.1)) -> np.ndarray:
    """Spatially-varying exposure + local-contrast + per-channel color remapping.

    Mirrors the paper's local degradation stage, which lists *exposure
    adjustment, contrast remapping, and color distortion* (Fig. 1b) -- i.e.
    spatially-varying, channel-wise color-ratio distortions. Beyond a luminance
    exposure/contrast field, this applies three independent smooth per-channel
    gain fields so the resulting color gain map has a genuine local, per-channel
    component for the bilateral grid to invert.
    """
    h, w = lin.shape[:2]
    exposure = _smooth_random_field(h, w, ctrl, *exp_range, rng=rng)
    contrast = _smooth_random_field(h, w, ctrl, *contrast_range, rng=rng)
    # Independent spatially-varying gain per RGB channel -> channel-wise distortion.
    color = np.concatenate([_smooth_random_field(h, w, ctrl, *color_range, rng=rng)
                            for _ in range(3)], axis=-1)
    out = lin * exposure
    local_mean = cv2.GaussianBlur(out, (0, 0), sigmaX=max(h, w) / 64.0)
    out = local_mean + (out - local_mean) * contrast
    out = out * color
    return np.clip(out, 0.0, None)


def degrade_hdr_to_sdr(hdr_lin: np.ndarray, rng: np.random.Generator,
                       gamma: float = gainmap.GAMMA,
                       local: bool = True) -> np.ndarray:
    """Full forward local-TMO: relative-linear HDR -> gamma-encoded SDR in [0,1]."""
    comp = global_compression(hdr_lin)
    if local:
        comp = local_adaptation(comp, rng)
    # Normalize so the diffuse-white region maps near 1.0, then clip to display.
    norm = max(float(np.percentile(comp, 99.0)), 1e-3)
    sdr_lin = np.clip(comp / norm, 0.0, 1.0)
    sdr = np.power(sdr_lin, 1.0 / gamma)                 # OETF (inverse of Eq.2 gamma)
    return np.clip(sdr, 0.0, 1.0).astype(np.float32)


def make_pair(hdr_lin: np.ndarray, seed: int = 0, local: bool = True):
    """Produce a self-consistent (sdr, hdr_lin, gm_norm, gm_log, q_max) tuple.

    ``hdr_lin`` is re-derived from the produced SDR via the SAME formula the model
    inverts, so the pair exactly satisfies Eq. 2/3 (no train/eval mismatch).
    """
    rng = np.random.default_rng(seed)
    sdr = degrade_hdr_to_sdr(hdr_lin, rng, local=local)
    # Recover the relative-linear HDR consistent with this SDR (rescale HDR so its
    # diffuse white matches the SDR's, keeping the SDR fixed as ground reference).
    gm_log = gainmap.compute_gt_gainmap_np(sdr, _match_white(sdr, hdr_lin))
    gm_norm, q_max = gainmap.normalize_gainmap_np(gm_log)
    hdr_recon = gainmap.reconstruct_hdr_linear_np(sdr, gm_log)
    return sdr, hdr_recon, gm_norm, gm_log, q_max


def _match_white(sdr: np.ndarray, hdr_lin: np.ndarray,
                 gamma: float = gainmap.GAMMA) -> np.ndarray:
    """Rescale HDR so its diffuse white aligns with the SDR's linear white.

    Ensures the gain map is dominated by genuine highlight/gamut headroom rather
    than a global exposure offset between SDR and HDR.
    """
    sdr_lin = np.power(np.clip(sdr, 0, 1), gamma)
    s_white = max(float(np.percentile(sdr_lin, 99.0)), 1e-4)
    h_white = max(float(np.percentile(hdr_lin, 99.0)), 1e-4)
    return hdr_lin * (s_white / h_white)
