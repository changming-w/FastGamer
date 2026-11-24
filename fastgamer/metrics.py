"""Evaluation metrics for FastGaMer (Sec. 4, Tab. 1 & 2).

Reported in three domains, matching the paper:

* Linear domain -- prediction and reference are each normalized by the
  ground-truth peak to [0, 1]; then PSNR / SSIM / SRSIM.
* PQ domain -- prediction and reference are PQ-encoded (``pq_oetf(linear*203)``);
  then PSNR / SSIM / SRSIM. PQ aligns better with perceived contrast.
* HDR domain -- the color-difference metric Delta E_ITP (ITU-R BT.2124) and the
  perceptual index HDR-VDP-3.

All inputs are NumPy ``float`` arrays of shape ``(H, W, 3)`` in *relative linear*
HDR (``1.0 == SDR diffuse white == 203 nits``), RGB channel order.

PSNR / SSIM formulas and PQ constants match the sister GMNet codebase; SR-SIM is
implemented from Zhang & Li, "SR-SIM: A fast and high performance IQA index
based on spectral residual" (ICIP 2012); Delta E_ITP uses colour-science's
``RGB_to_ICtCp``. HDR-VDP-3 is exposed as an optional external hook.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import cv2

from . import gainmap

try:
    import colour
    _HAS_COLOUR = True
except Exception:  # pragma: no cover
    _HAS_COLOUR = False


# --------------------------------------------------------------------------- #
# Domain conversions
# --------------------------------------------------------------------------- #
def to_pq(linear_relative: np.ndarray) -> np.ndarray:
    """Relative linear HDR -> PQ code values in [0, 1]."""
    return np.clip(gainmap.linear_to_pq_np(linear_relative), 0.0, 1.0)


def normalize_by_peak(pred: np.ndarray, gt: np.ndarray):
    """Normalize both arrays by the GT peak to (approx) [0, 1]."""
    peak = float(np.max(gt))
    peak = max(peak, 1e-8)
    return np.clip(pred / peak, 0.0, 1.0), np.clip(gt / peak, 0.0, 1.0), peak


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #
def calculate_psnr(img1: np.ndarray, img2: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def calculate_normalized_psnr(img1: np.ndarray, img2: np.ndarray, norm: float) -> float:
    """-10*log10( mean( ((img1-img2)/norm)^2 ) ) -- the sister-repo variant."""
    a = img1.astype(np.float64) / norm
    b = img2.astype(np.float64) / norm
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


# --------------------------------------------------------------------------- #
# SSIM (BasicSR / OpenCV Gaussian-window, inputs in [0, 255])
# --------------------------------------------------------------------------- #
def _ssim_single(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM for HWC images in [0, 255], averaged over channels."""
    if img1.ndim == 2:
        return _ssim_single(img1, img2)
    return float(np.mean([_ssim_single(img1[..., c], img2[..., c])
                          for c in range(img1.shape[2])]))


# --------------------------------------------------------------------------- #
# SR-SIM (Zhang & Li, ICIP 2012), inputs in [0, 255]
# --------------------------------------------------------------------------- #
def _rgb2gray_601(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """MATLAB ``fspecial('gaussian', size, sigma)`` equivalent (normalized)."""
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


_SR_GAUSS = _fspecial_gaussian(10, 3.8)   # reference spectralResidueSaliency post-filter


def _spectral_residual_saliency(y: np.ndarray) -> np.ndarray:
    """Spectral-residual visual saliency map (Hou & Zhang 2007) at scale 0.25.

    Matches the reference ``spectralResidueSaliency.m`` used by SR-SIM: bicubic
    resample, 3x3 mean-filter spectral residual, and a fixed 10x10 Gaussian
    (sigma=3.8) post-filter.
    """
    scale = 0.25
    h, w = y.shape
    small = cv2.resize(y, (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_CUBIC)
    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-12)
    phase = np.angle(fft)
    avg = cv2.blur(log_amp, (3, 3), borderType=cv2.BORDER_REPLICATE)
    residual = log_amp - avg
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.filter2D(sal, -1, _SR_GAUSS, borderType=cv2.BORDER_REPLICATE)
    mn, mx = float(sal.min()), float(sal.max())
    sal = (sal - mn) / (mx - mn + 1e-12)                       # mat2gray
    return cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)


def _scharr_gradient(y: np.ndarray) -> np.ndarray:
    dx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], np.float64) / 16.0
    dy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], np.float64) / 16.0
    ix = cv2.filter2D(y, -1, dx, borderType=cv2.BORDER_REPLICATE)
    iy = cv2.filter2D(y, -1, dy, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(ix ** 2 + iy ** 2)


def calculate_srsim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SR-SIM for images in [0, 255] (HWC or HW)."""
    y1 = _rgb2gray_601(img1.astype(np.float64))
    y2 = _rgb2gray_601(img2.astype(np.float64))
    rows, cols = y1.shape
    f = max(1, round(min(rows, cols) / 256.0))
    if f > 1:
        ker = (f, f)
        y1 = cv2.blur(y1, ker)[::f, ::f]
        y2 = cv2.blur(y2, ker)[::f, ::f]

    sal1 = _spectral_residual_saliency(y1)
    sal2 = _spectral_residual_saliency(y2)
    grad1 = _scharr_gradient(y1)
    grad2 = _scharr_gradient(y2)

    c1, c2, alpha = 0.40, 225.0, 0.50
    s_vs = (2 * sal1 * sal2 + c1) / (sal1 ** 2 + sal2 ** 2 + c1)
    s_gm = (2 * grad1 * grad2 + c2) / (grad1 ** 2 + grad2 ** 2 + c2)
    weight = np.maximum(sal1, sal2)
    sim = s_vs * (s_gm ** alpha) * weight
    return float(sim.sum() / (weight.sum() + 1e-12))


# --------------------------------------------------------------------------- #
# Delta E_ITP (ITU-R BT.2124)
# --------------------------------------------------------------------------- #
def delta_e_itp(pred_lin: np.ndarray, gt_lin: np.ndarray,
                ref_white_nits: float = gainmap.REF_WHITE_NITS,
                convert_primaries: bool = True) -> float:
    """Delta E_ITP = 720 * sqrt( dI^2 + 0.25*dCt^2 + dCp^2 ), mean over pixels.

    Inputs are relative-linear sRGB-primary HDR. They are (optionally) converted
    to BT.2020 linear and expressed in ABSOLUTE nits (diffuse white == 203 nits)
    before ICtCp -- ``colour.RGB_to_ICtCp`` applies the L_p=10000 PQ
    normalization internally (its ST 2084 inverse EOTF divides by L_p), so the
    input must be absolute nits, not pre-divided by 10000.
    """
    if not _HAS_COLOUR:
        return float("nan")

    def prep(x):
        x = np.clip(x, 0.0, None).astype(np.float64)
        if convert_primaries:
            x = colour.RGB_to_RGB(x, colour.models.RGB_COLOURSPACE_sRGB,
                                  colour.models.RGB_COLOURSPACE_BT2020,
                                  apply_cctf_decoding=False, apply_cctf_encoding=False)
        x = np.clip(x, 0.0, None) * ref_white_nits               # absolute nits
        return colour.RGB_to_ICtCp(x)                            # Dolby 2016, L_p=10000

    a = prep(pred_lin)
    b = prep(gt_lin)
    d = np.sqrt((a[..., 0] - b[..., 0]) ** 2
                + 0.25 * (a[..., 1] - b[..., 1]) ** 2
                + (a[..., 2] - b[..., 2]) ** 2)
    return float(720.0 * np.mean(d))


# --------------------------------------------------------------------------- #
# HDR-VDP-3 (optional external hook)
# --------------------------------------------------------------------------- #
def hdr_vdp3(pred_lin: np.ndarray, gt_lin: np.ndarray, **kwargs):
    """Optional HDR-VDP-3 hook.

    HDR-VDP-3 is a MATLAB toolbox (Mantiuk et al.) not bundled here. If a Python
    callable is registered via ``register_hdr_vdp3`` (e.g. wrapping a MATLAB
    engine or a compiled port), it is used; otherwise returns NaN with a warning.
    """
    if _HDR_VDP3_IMPL is not None:
        return _HDR_VDP3_IMPL(pred_lin, gt_lin, **kwargs)
    warnings.warn("HDR-VDP-3 not available; returning NaN. Register an "
                  "implementation with metrics.register_hdr_vdp3(fn).")
    return float("nan")


_HDR_VDP3_IMPL = None


def register_hdr_vdp3(fn):
    """Register a callable ``fn(pred_lin, gt_lin, **kwargs) -> float``."""
    global _HDR_VDP3_IMPL
    _HDR_VDP3_IMPL = fn


# --------------------------------------------------------------------------- #
# High-level evaluation over both domains
# --------------------------------------------------------------------------- #
def evaluate(pred_lin: np.ndarray, gt_lin: np.ndarray, with_hdrvdp: bool = False) -> dict:
    """Compute all paper metrics for a single relative-linear HDR pair."""
    pred_lin = np.asarray(pred_lin, np.float64)
    gt_lin = np.asarray(gt_lin, np.float64)

    # Linear domain (peak-normalized to [0, 1]).
    pn, gn, peak = normalize_by_peak(pred_lin, gt_lin)
    lin = {
        "psnr": calculate_normalized_psnr(pred_lin, gt_lin, np.max(gt_lin) + 1e-8),
        "ssim": calculate_ssim(pn * 255.0, gn * 255.0),
        "srsim": calculate_srsim(pn * 255.0, gn * 255.0),
    }

    # PQ domain.
    pp, gp = to_pq(pred_lin), to_pq(gt_lin)
    pq = {
        "psnr": calculate_psnr(pp, gp, peak=1.0),
        "ssim": calculate_ssim(pp * 255.0, gp * 255.0),
        "srsim": calculate_srsim(pp * 255.0, gp * 255.0),
    }

    hdr = {"deltaEITP": delta_e_itp(pred_lin, gt_lin)}
    if with_hdrvdp:
        hdr["hdrvdp3"] = hdr_vdp3(pred_lin, gt_lin)

    return {"linear": lin, "pq": pq, "hdr": hdr}
