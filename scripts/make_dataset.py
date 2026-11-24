#!/usr/bin/env python
"""Build a synthetic SDR--color-GM dataset (mirrors the paper's forward local-TMO).

The paper derives its data from RAISE RAW processed by Adobe Camera Raw (Sec. 4);
that manual pipeline is not reproducible in code. This script instead produces
self-consistent SDR/HDR/GM pairs using ``fastgamer.data.synth_degrade`` so the
whole training/eval pipeline is runnable end-to-end.

Two input sources:
  --hdr_dir    directory of relative-linear HDR files (.npy/.exr/.hdr), used as-is.
  --image_dir  directory of ordinary images; a plausible HDR with highlight/gamut
               headroom is synthesized from each (for demos / smoke tests).

Output layout (root/<split>/): sdr/<name>.png, hdr/<name>.npy,
thumb/<name>.png, list.txt  -- consumed by ``fastgamer.data.GainMapDataset``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.data import synth_degrade
from fastgamer import gainmap
from fastgamer.utils import imread_srgb, imwrite_srgb, read_hdr_linear


def synthesize_hdr_from_image(img_srgb: np.ndarray, rng: np.random.Generator,
                              highlight_gain: float = 6.0,
                              gamut_gain: float = 0.4) -> np.ndarray:
    """Fabricate a relative-linear HDR (highlights > 1) from an sRGB image.

    Linearizes the image, then multiplicatively boosts bright regions to create
    genuine highlight headroom and slightly saturates them to create WCG headroom
    -- so the resulting color gain map is non-trivial and per-channel.
    """
    lin = np.power(np.clip(img_srgb, 0, 1), gainmap.GAMMA)
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])[..., None]
    t0 = float(np.percentile(lum, 85.0))
    s = np.clip((lum - t0) / (1.0 - t0 + 1e-6), 0.0, 1.0)      # highlight mask
    s = s * s * (3 - 2 * s)                                     # smoothstep
    boost = 1.0 + highlight_gain * s
    hdr = lin * boost
    # Per-channel gamut headroom: push saturated highlights further from gray.
    gray = lum
    hdr = hdr + gamut_gain * s * (hdr - gray)
    return np.clip(hdr, 0.0, None).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dataset root")
    ap.add_argument("--split", default="train")
    ap.add_argument("--hdr_dir", default=None, help="dir of relative-linear HDR files")
    ap.add_argument("--image_dir", default=None, help="dir of ordinary images")
    ap.add_argument("--limit", type=int, default=0, help="max samples (0 = all)")
    ap.add_argument("--max_size", type=int, default=0, help="downscale long side to this (0=off)")
    ap.add_argument("--no_local", action="store_true", help="disable local adaptation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.hdr_dir or args.image_dir):
        ap.error("provide --hdr_dir or --image_dir")

    out_dir = os.path.join(args.out, args.split)
    for sub in ("sdr", "hdr", "thumb"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    if args.hdr_dir:
        paths = sorted(sum([glob.glob(os.path.join(args.hdr_dir, e))
                            for e in ("*.npy", "*.exr", "*.hdr")], []))
        src_mode = "hdr"
    else:
        paths = sorted(sum([glob.glob(os.path.join(args.image_dir, e))
                            for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")], []))
        src_mode = "image"
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        ap.error(f"no inputs found in {args.hdr_dir or args.image_dir}")

    names = []
    for i, p in enumerate(paths):
        name = f"{i:05d}_{os.path.splitext(os.path.basename(p))[0]}"
        rng = np.random.default_rng(args.seed + i)
        if src_mode == "hdr":
            hdr = read_hdr_linear(p)
        else:
            img = imread_srgb(p)
            hdr = synthesize_hdr_from_image(img, rng)
        if args.max_size > 0:
            h, w = hdr.shape[:2]
            if max(h, w) > args.max_size:
                sc = args.max_size / max(h, w)
                hdr = cv2.resize(hdr, (round(w * sc), round(h * sc)),
                                 interpolation=cv2.INTER_AREA)

        sdr, hdr_recon, gm_norm, gm_log, q_max = synth_degrade.make_pair(
            hdr, seed=args.seed + i, local=not args.no_local)

        imwrite_srgb(os.path.join(out_dir, "sdr", f"{name}.png"), sdr)
        np.save(os.path.join(out_dir, "hdr", f"{name}.npy"), hdr_recon.astype(np.float32))
        thumb = cv2.resize(sdr, (256, 256), interpolation=cv2.INTER_AREA)
        imwrite_srgb(os.path.join(out_dir, "thumb", f"{name}.png"), thumb)
        names.append(name)
        print(f"[{i+1}/{len(paths)}] {name}  q_max={q_max:.3f}  "
              f"hdr_peak={hdr_recon.max():.2f}  shape={sdr.shape}")

    with open(os.path.join(out_dir, "list.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"\nWrote {len(names)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
