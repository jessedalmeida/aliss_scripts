#!/usr/bin/env python3
"""make_splits.py - Build bag-level train/val/test splits from a manifest.

Splitting is BY BAG, never by frame: adjacent frames in a bag are near
-duplicates, so a frame-level split leaks. Validation and test are drawn only
from no-board bags (the deployment domain we judge on). Board bags are
train-only. Any no-board bags not needed for val/test fall into train too, so
no deployment-domain data is wasted.

Selection priority:
  1. Explicit --val-bags / --test-bags (comma-separated names) win.
  2. Otherwise pick --num-val / --num-test no-board bags deterministically
     (seeded shuffle of the sorted no-board bag names).

Usage:
    python make_splits.py --manifest dataset/manifest.jsonl --out dataset/splits.json
    python make_splits.py --manifest dataset/manifest.jsonl --out dataset/splits.json \
        --test-bags chicken_1 --val-bags chicken_2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_bag_stats(manifest_path: Path) -> dict[str, dict]:
    """Aggregate the manifest into per-bag stats."""
    bags: dict[str, dict] = {}
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bag = rec["bag"]
            b = bags.setdefault(
                bag, {"has_board": rec["has_board"], "frames": 0, "kp_frames": 0}
            )
            b["frames"] += 1
            # a frame counts toward keypoint coverage if the needle pair is present
            kp = rec["keypoints"]
            if kp["needle_tip"]["xy"] is not None and kp["needle_tail"]["xy"] is not None:
                b["kp_frames"] += 1
    return bags


def choose_splits(
    bags: dict[str, dict],
    val_bags: set[str],
    test_bags: set[str],
    num_val: int,
    num_test: int,
    seed: int,
) -> dict[str, list[str]]:
    noboard = sorted(b for b, s in bags.items() if not s["has_board"])
    board = sorted(b for b, s in bags.items() if s["has_board"])

    explicit = val_bags | test_bags
    if explicit:
        unknown = explicit - set(bags)
        if unknown:
            raise SystemExit(f"--val/--test-bags reference unknown bags: {sorted(unknown)}")
        board_in_eval = [b for b in explicit if bags[b]["has_board"]]
        if board_in_eval:
            print(f"[warn] board bags placed in val/test (not deployment domain): {board_in_eval}")
        chosen_val, chosen_test = sorted(val_bags), sorted(test_bags)
    else:
        pool = noboard[:]
        random.Random(seed).shuffle(pool)
        need = num_val + num_test
        if len(pool) < need:
            print(f"[warn] only {len(pool)} no-board bags but {need} requested for val+test; "
                  f"splits will be thin and metrics noisy.")
        chosen_test = sorted(pool[:num_test])
        chosen_val = sorted(pool[num_test:num_test + num_val])

    eval_set = set(chosen_val) | set(chosen_test)
    train = sorted(b for b in bags if b not in eval_set)  # all board + leftover no-board
    return {"train": train, "val": chosen_val, "test": chosen_test}


def split_frame_counts(bags: dict[str, dict], names: list[str]) -> tuple[int, int, int, int]:
    frames = sum(bags[b]["frames"] for b in names)
    kp = sum(bags[b]["kp_frames"] for b in names)
    nb = sum(1 for b in names if not bags[b]["has_board"])
    return len(names), frames, kp, nb


def verify(splits: dict[str, list[str]], bags: dict[str, dict]) -> None:
    all_named = splits["train"] + splits["val"] + splits["test"]
    # no bag in two splits
    assert len(all_named) == len(set(all_named)), "a bag appears in multiple splits"
    # full coverage
    assert set(all_named) == set(bags), "splits do not cover all bags exactly once"
    # val/test are deployment domain only (warn-level handled upstream; assert non-board here is too strict
    # because explicit override is allowed, so we only assert non-empty)
    for s in ("val", "test"):
        if not splits[s]:
            print(f"[warn] {s} split is EMPTY")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build bag-level train/val/test splits")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-bags", default=None, help="explicit comma-separated val bag names")
    ap.add_argument("--test-bags", default=None, help="explicit comma-separated test bag names")
    ap.add_argument("--num-val", type=int, default=2)
    ap.add_argument("--num-test", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bags = load_bag_stats(Path(args.manifest))
    if not bags:
        raise SystemExit("manifest is empty")

    def csv(v):
        return {x.strip() for x in v.split(",") if x.strip()} if v else set()

    splits = choose_splits(
        bags, csv(args.val_bags), csv(args.test_bags),
        args.num_val, args.num_test, args.seed,
    )
    verify(splits, bags)

    out = {
        "manifest": str(Path(args.manifest)),
        "seed": args.seed,
        "splits": splits,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote {args.out}")
    for s in ("train", "val", "test"):
        nbags, frames, kp, nb = split_frame_counts(bags, splits[s])
        print(f"  {s:5s}: {nbags:2d} bags | {frames:5d} frames | "
              f"{kp:5d} with keypoints | {nb} no-board bags")
        if s in ("val", "test") and splits[s]:
            for b in splits[s]:
                flag = "" if bags[b]["kp_frames"] else "  <-- NO keypoints (mask-only eval)"
                dom = "no-board" if not bags[b]["has_board"] else "BOARD"
                print(f"           {b} [{dom}]{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
