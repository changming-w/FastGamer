"""Small utilities: config, seeding, image IO, augmentation, checkpoints."""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------- #
# Config / seeding
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Image IO
# --------------------------------------------------------------------------- #
def imread_srgb(path: str) -> np.ndarray:
    """Read an 8/16-bit image as float RGB in [0, 1] (HWC)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = img[..., None].repeat(3, axis=2)
    if img.shape[2] == 4:
        img = img[..., :3]
    maxv = 65535.0 if img.dtype == np.uint16 else 255.0
    img = img[..., ::-1].astype(np.float32) / maxv        # BGR->RGB
    return np.ascontiguousarray(img)


def imwrite_srgb(path: str, img: np.ndarray) -> None:
    """Write a float RGB image in [0, 1] as 8-bit PNG/JPG."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = np.clip(img[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def read_hdr_linear(path: str) -> np.ndarray:
    """Read a relative-linear HDR image (.npy float32 HWC preferred)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., ::-1]
    return np.ascontiguousarray(img.astype(np.float32))


# --------------------------------------------------------------------------- #
# Augmentation (BasicSR-style flip + 90-degree rotation)
# --------------------------------------------------------------------------- #
def augment(imgs, hflip=True, rot=True, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    do_h = hflip and rng.random() < 0.5
    do_v = rot and rng.random() < 0.5
    do_t = rot and rng.random() < 0.5

    def _f(img):
        if do_h:
            img = img[:, ::-1, :]
        if do_v:
            img = img[::-1, :, :]
        if do_t:
            img = img.transpose(1, 0, 2)
        return np.ascontiguousarray(img)

    return [_f(im) for im in imgs]


def random_crop(imgs, size: int, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng()
    h, w = imgs[0].shape[:2]
    if h < size or w < size:
        imgs = [_pad_to(im, size) for im in imgs]
        h, w = imgs[0].shape[:2]
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return [im[top:top + size, left:left + size] for im in imgs]


def _pad_to(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")


def to_chw_tensor(img: np.ndarray):
    """HWC float ndarray -> CHW torch float tensor."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer=None, scheduler=None, step: int = 0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("step", 0)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
