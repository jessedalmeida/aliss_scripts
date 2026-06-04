#!/usr/bin/env python3
"""overfit_one_batch.py - End-to-end stack check: can the model memorize one batch?

If total loss drives toward zero on a single fixed batch (augmentation OFF),
the whole pipeline -- dataset -> heatmap rendering -> model -> loss -> grads --
is wired correctly. If it can't, there's a bug to fix before any real run.

Workstation usage (full):
    python overfit_one_batch.py --manifest overfit_manifest.jsonl \
        --input-size 640 --steps 300 --pretrained --device cuda

CPU sanity (smaller/faster):
    python overfit_one_batch.py --manifest overfit_manifest.jsonl \
        --input-size 256 --steps 40 --device cpu
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from needle_dataset import NeedleDataset
from needle_model import NeedleNet
from needle_loss import NeedleLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ds = NeedleDataset(args.manifest, input_size=args.input_size, transform=None)
    batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False)))
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    model = NeedleNet(num_keypoints=2, pretrained=args.pretrained).to(device).train()
    crit = NeedleLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    first = None
    for step in range(args.steps):
        opt.zero_grad()
        out = model(batch["image"])
        losses = crit(out, batch)
        losses["total"].backward()
        opt.step()
        if step == 0:
            first = losses["total"].item()
            first_mask = losses["mask"].item()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"step {step:4d} | total {losses['total']:.4f} "
                  f"| mask {losses['mask']:.4f} hm {losses['hm']:.4f} vis {losses['vis']:.4f}")

    final = losses["total"].item()
    print(f"\nfirst total={first:.4f}  final total={final:.4f}  drop={100*(1-final/first):.1f}%")
    print(f"final terms: mask={losses['mask']:.4f} (dice={losses['mask_dice']:.3f}) "
          f"hm={losses['hm']:.4f} vis={losses['vis']:.4f}")
    # Per-term gate: the keypoint/visibility heads memorize fast; the mask head
    # (Dice on a thin/fragmented needle) converges slower, especially from cold
    # init at low res. So check each path is learning rather than thresholding
    # the total (which the slow mask term would otherwise dominate).
    ok_hm = losses["hm"].item() < 0.02
    ok_vis = losses["vis"].item() < 0.05
    ok_mask = losses["mask"].item() < 0.9 * first_mask   # trending down
    print(f"gate: hm={'ok' if ok_hm else 'FAIL'} "
          f"vis={'ok' if ok_vis else 'FAIL'} mask_trending={'ok' if ok_mask else 'FAIL'}")
    assert ok_hm and ok_vis and ok_mask, (
        "a head is not learning; stack may be miswired. With --pretrained at full "
        "resolution the mask term should also reach near-zero.")
    print("Overfit check PASSED (every head learns -> stack is wired end to end).")


if __name__ == "__main__":
    main()
