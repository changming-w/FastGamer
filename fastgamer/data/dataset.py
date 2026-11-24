"""SDR -- color Gain Map dataset (Sec. 4, "Proposed Dataset").

The paper trains on >8,000 4K SDR--color-GM pairs. Public paired datasets with
three-channel color gain maps do not exist, so this loader supports the on-disk
layout produced by ``scripts/make_dataset.py`` (synthetic local-TMO pairs) and
computes the color gain map on the fly from the SDR and the relative-linear HDR,
exactly per ``gainmap`` (Eq. 2/3) -- avoiding any 8-bit GM quantization.

On-disk layout (root/<split>/):
    sdr/<name>.png     8-bit gamma-encoded SDR in [0, 1]
    hdr/<name>.npy     float32 HWC relative-linear HDR (1.0 == diffuse white)
    thumb/<name>.png   (optional) 256x256 whole-image SDR thumbnail
    list.txt           one <name> per line

Two modes:
    'sdr_hdr' (default) -- load SDR + HDR, compute color GM & Q_max online.
    'sdr_gm'            -- load SDR + precomputed GM (.npy in [-1,1]) + meta txt
                          (Q_max); used for LQGT-compatible / precomputed data.

A whole-image thumbnail is always produced (from ``thumb/`` if present, else by
resizing the full SDR) so that Q_max stays globally consistent (Sec. 3.1), even
when training on 256x256 crops.
"""
from __future__ import annotations

import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .. import gainmap
from ..utils import imread_srgb, read_hdr_linear, augment, random_crop, to_chw_tensor


class GainMapDataset(Dataset):
    def __init__(self, root: str, split: str = "train", patch_size: int = 256,
                 thumb_size: int = 256, mode: str = "sdr_hdr",
                 augment_data: bool = True, is_train: bool = True, seed: int = 0):
        self.dir = os.path.join(root, split)
        self.patch_size = patch_size
        self.thumb_size = thumb_size
        self.mode = mode
        self.augment_data = augment_data
        self.is_train = is_train
        list_path = os.path.join(self.dir, "list.txt")
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"missing manifest: {list_path}")
        with open(list_path) as f:
            self.names = [ln.strip() for ln in f if ln.strip()]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.names)

    # ---- loading ---------------------------------------------------------- #
    def _load_sample(self, name: str):
        sdr = imread_srgb(os.path.join(self.dir, "sdr", f"{name}.png"))
        thumb_path = os.path.join(self.dir, "thumb", f"{name}.png")
        if os.path.isfile(thumb_path):
            thumb = imread_srgb(thumb_path)
        else:
            thumb = sdr
        thumb = cv2.resize(thumb, (self.thumb_size, self.thumb_size),
                           interpolation=cv2.INTER_AREA)

        if self.mode == "sdr_hdr":
            hdr = read_hdr_linear(os.path.join(self.dir, "hdr", f"{name}.npy"))
            gm_log = gainmap.compute_gt_gainmap_np(sdr, hdr)
            gm_norm, q_max = gainmap.normalize_gainmap_np(gm_log)
        elif self.mode == "sdr_gm":
            gm_norm = np.load(os.path.join(self.dir, "gm", f"{name}.npy")).astype(np.float32)
            q_max = np.float32(np.loadtxt(os.path.join(self.dir, "meta", f"{name}.txt")).reshape(-1)[0])
            gm_log = gm_norm * q_max
            hdr = gainmap.reconstruct_hdr_linear_np(sdr, gm_log)
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return sdr, hdr, gm_norm, gm_log, np.float32(q_max), thumb

    # ---- item ------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        name = self.names[idx]
        sdr, hdr, gm_norm, gm_log, q_max, thumb = self._load_sample(name)

        if self.is_train:
            sdr, hdr, gm_norm, gm_log = random_crop([sdr, hdr, gm_norm, gm_log],
                                                    self.patch_size, self.rng)
            if self.augment_data:
                sdr, hdr, gm_norm, gm_log = augment([sdr, hdr, gm_norm, gm_log],
                                                    hflip=True, rot=True, rng=self.rng)

        return {
            "i_hr": to_chw_tensor(sdr),
            "i_thumb": to_chw_tensor(thumb),
            "gm_norm": to_chw_tensor(gm_norm),
            "gm_pred": to_chw_tensor(gm_log),          # absolute log-GM GT
            "hdr_lin": to_chw_tensor(hdr),
            "q_max": torch.tensor([q_max], dtype=torch.float32).view(1, 1, 1),
            "name": name,
        }
