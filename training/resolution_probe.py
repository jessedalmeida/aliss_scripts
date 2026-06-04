#!/usr/bin/env python3
"""resolution_probe.py - Decide model input resolution for the needle model.

Two independent measurements that together set the Step-2 gate:

  (A) Needle survival: downsample real needle masks to each candidate size,
      upsample back, and measure IoU vs the original. A thin needle can vanish
      under aggressive downscaling; this finds the smallest size that keeps it.
      Also reports the thinnest needle width per mask, since that is what
      actually limits how far you can downscale.

  (B) Throughput: time a forward pass of a representative ResNet-UNet-ish
      encoder/decoder at each size, on whatever device is present, in the same
      precision you will deploy (AMP/half on CUDA). Reports FPS.

Run on the WORKSTATION (RTX 4070 Mobile) so the FPS numbers are real.

Usage:
    # survival only (no torch needed):
    python resolution_probe.py --masks-glob 'auto-annotating/annotations/*/masks/*_needle_mask.png'
    # both:
    python resolution_probe.py --masks-glob '...*.png' --time-model
"""

from __future__ import annotations

import argparse
import glob
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

CANDIDATE_SIZES = (384, 512, 640, 768, 896)
NATIVE = 1080


# --------------------------------------------------------------------------- #
# (A) needle survival
# --------------------------------------------------------------------------- #
def thinnest_width(mask: np.ndarray) -> float:
    """Approximate the needle's minimum width via distance transform.

    The max of the distance transform on the mask is the radius of the largest
    inscribed disk; for a thin elongated needle that radius ~ half the width.
    Returns full width in pixels (0 if mask empty).
    """
    if cv2 is None or not mask.any():
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    # median of the ridge (skeleton-ish) widths is more stable than the single max
    ridge = dt[dt > 0.5 * dt.max()]
    return float(2.0 * np.median(ridge)) if ridge.size else 0.0


def survival_iou(mask: np.ndarray, size: int) -> float:
    """IoU of (downsample to size -> upsample back) vs original binary mask."""
    h, w = mask.shape
    small = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA)
    small = (small > 0.5).astype(np.uint8)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    back = (back > 0.5).astype(np.uint8)
    inter = np.logical_and(mask, back).sum()
    union = np.logical_or(mask, back).sum()
    return float(inter / union) if union else 1.0


def run_survival(mask_paths: list[str], sizes) -> None:
    if cv2 is None:
        raise SystemExit("opencv required for survival test: pip install opencv-python-headless")
    print(f"\n=== (A) needle survival over {len(mask_paths)} masks ===")
    widths: list[float] = []
    empties = 0
    iou_by_size: dict[int, list[float]] = {s: [] for s in sizes}
    survive_by_size: dict[int, int] = {s: 0 for s in sizes}  # masks keeping IoU>=0.5

    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8)
        if not m.any():
            empties += 1
            continue
        widths.append(thinnest_width(m))
        for s in sizes:
            iou = survival_iou(m, s)
            iou_by_size[s].append(iou)
            if iou >= 0.5:
                survive_by_size[s] += 1

    n = len(widths)
    if n == 0:
        raise SystemExit("no non-empty masks read; check --masks-glob")
    print(f"non-empty masks: {n}  (empty skipped: {empties})")
    print(f"needle width px: median={statistics.median(widths):.1f}  "
          f"p10={np.percentile(widths, 10):.1f}  min={min(widths):.1f}")
    print(f"{'size':>6} {'meanIoU':>8} {'p10IoU':>8} {'minIoU':>8} {'IoU>=0.5':>9}")
    for s in sizes:
        v = iou_by_size[s]
        print(f"{s:>6} {statistics.mean(v):>8.3f} {np.percentile(v,10):>8.3f} "
              f"{min(v):>8.3f} {survive_by_size[s]/n:>8.1%}")
    print("Pick the smallest size where p10 IoU stays high (needle preserved on "
          "the hardest frames), then confirm it clears the FPS bar below.")


# --------------------------------------------------------------------------- #
# (B) throughput
# --------------------------------------------------------------------------- #
def build_dummy_model():
    """A representative ResNet34-encoder / light-decoder seg+kp net for timing.

    Architecture only needs to be the right *shape and cost*; weights are random.
    Three heads: 1-ch mask, K heatmaps, K visibility logits.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet34

    K = 2  # needle tip + tail for v1

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            enc = resnet34(weights=None)
            self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
            self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
            def up(ci, co):
                return nn.Sequential(
                    nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.d4, self.d3, self.d2, self.d1 = up(512, 256), up(256, 128), up(128, 64), up(64, 64)
            self.mask_head = nn.Conv2d(64, 1, 1)
            self.hm_head = nn.Conv2d(64, K, 1)
            self.vis_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, K))

        def forward(self, x):
            x = self.stem(x)
            x = self.l1(x); x = self.l2(x); x = self.l3(x); f = self.l4(x)
            d = self.d4(f); d = self.d3(d); d = self.d2(d); d = self.d1(d)
            return self.mask_head(d), self.hm_head(d), self.vis_head(f)

    return Net()


def run_timing(sizes, iters: int) -> None:
    import torch
    print(f"\n=== (B) forward-pass throughput ({iters} iters/size) ===")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("device: CPU  (FPS here is NOT representative of the workstation GPU)")

    model = build_dummy_model().to(device).eval()
    use_amp = device == "cuda"
    print(f"precision: {'AMP/half' if use_amp else 'fp32'}")
    print(f"{'size':>6} {'ms/frame':>9} {'FPS':>7} {'VRAM MB':>9}")

    for s in sizes:
        x = torch.randn(1, 3, s, s, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else _nullctx()
            # warmup
            for _ in range(5):
                with ctx:
                    model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                with ctx:
                    model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
        vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
        print(f"{s:>6} {dt*1e3:>9.2f} {1/dt:>7.1f} {vram:>9.0f}")
    print("Gate: smallest size that is >=15 FPS with headroom AND survived (A).")


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolution + throughput probe")
    ap.add_argument("--masks-glob", default=None, help="glob for real *_needle_mask.png files")
    ap.add_argument("--max-masks", type=int, default=400, help="cap masks sampled for survival")
    ap.add_argument("--time-model", action="store_true", help="run the GPU timing sweep")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--sizes", default=None, help="comma list to override candidate sizes")
    args = ap.parse_args()

    sizes = tuple(int(x) for x in args.sizes.split(",")) if args.sizes else CANDIDATE_SIZES

    if args.masks_glob:
        paths = sorted(glob.glob(args.masks_glob))
        if len(paths) > args.max_masks:
            # even stride sample so we span many bags, not just the first
            step = len(paths) / args.max_masks
            paths = [paths[int(i * step)] for i in range(args.max_masks)]
        run_survival(paths, sizes)
    else:
        print("[skip] no --masks-glob given; skipping survival test")

    if args.time_model:
        run_timing(sizes, args.iters)
    else:
        print("\n[skip] --time-model not set; skipping throughput sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
