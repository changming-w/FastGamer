#!/usr/bin/env python
"""Efficiency benchmark (Tab. 3): Params, Runtime, FLOPs, Memory at 4K and 2K.

FLOPs are counted with a lightweight hook-based counter (Conv2d, Linear) plus an
analytic term for the high-resolution ``grid_sample`` operations (bilateral-grid
slicing + 3D-LUT lookup), which dominate the network-free high-res path.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.models import build_model
from fastgamer import utils

RES = {"4K": (2160, 3840), "2K": (1080, 1920), "1K": (540, 960)}


class FlopCounter:
    """Count MACs for Conv2d / Linear via forward hooks (1 MAC = 1 mul + 1 add)."""

    def __init__(self, model):
        self.model = model
        self.macs = 0
        self.handles = []

    def _conv_hook(self, module, inp, out):
        oc = out.shape[1]
        out_hw = out.shape[2] * out.shape[3]
        kh, kw = module.kernel_size
        ic = module.in_channels // module.groups
        self.macs += oc * out_hw * ic * kh * kw

    def _linear_hook(self, module, inp, out):
        n = out.numel() // out.shape[-1]
        self.macs += n * module.in_features * module.out_features

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def grid_sample_macs(model, h, w):
    """Analytic MACs for the high-res grid_sample ops (trilinear = 8 taps)."""
    k = model.grid.K                       # K slices, one 3D grid_sample each
    lut_c = model.lut.out_nc               # 3D-LUT lookup outputs 3 channels
    taps = 8                               # trilinear interpolation taps
    # each output element: taps multiply-adds; grids: K single-channel samples.
    return (k + lut_c) * h * w * taps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastgamer.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--half", action="store_true", help="benchmark in fp16 (deployment)")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    dtype = torch.half if (args.half and device == "cuda") else torch.float
    if dtype == torch.half:
        model = model.half()
        print("Benchmarking in fp16.\n")

    params = utils.count_parameters(model)
    print(f"Parameters: {params/1e6:.4f} M\n")
    print(f"{'Res':>4s} | {'Runtime(ms)':>12s} | {'FLOPs(G)':>10s} | {'Memory(MB)':>11s}")
    print("-" * 52)

    for tag, (h, w) in RES.items():
        x = torch.rand(1, 3, h, w, device=device, dtype=dtype)
        with torch.no_grad():
            with FlopCounter(model) as fc:
                model(x)
            net_macs = fc.macs
            gs_macs = grid_sample_macs(model, h, w)
            gflops = 2.0 * (net_macs + gs_macs) / 1e9      # FLOPs = 2 * MACs

            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.warmup):
                    model(x)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.runs):
                    model(x)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / args.runs
                mem = torch.cuda.max_memory_allocated() / 1e6
            else:
                import time
                t0 = time.time()
                for _ in range(args.runs):
                    model(x)
                ms = (time.time() - t0) / args.runs * 1000
                mem = float("nan")

        print(f"{tag:>4s} | {ms:12.2f} | {gflops:10.3f} | {mem:11.1f}")

    print("\n(Paper Tab.3 target @4K: ~0.64M params, 6.20 ms, 0.48 GFLOPs, ~962 MB.)")


if __name__ == "__main__":
    main()
