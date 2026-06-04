#!/usr/bin/env python3
"""needle_sampler.py - Domain-balanced, bag-equalized training sampler.

Counters the ~10:1 board:no-board imbalance so each epoch draws close to a
target no-board fraction, while preventing any single large bag from dominating
its domain (adjacent frames in a bag are near-duplicates).

Per-frame weight for a frame in bag b of domain D:

    w = domain_mass(D) * (1 / num_bags_in_D) * (1 / frames_in_b)

so that:
  - summing over a bag's frames gives domain_mass(D)/num_bags_in_D
    -> every bag in a domain contributes EQUAL probability mass (bag-equalized);
  - summing over a domain gives domain_mass(D)
    -> realized domain ratio matches the target.

target_noboard_frac: fraction of sampled frames that should be no-board.
    0.5 => 1:1 board:no-board; 0.667 => 1:2 board:no-board (no-board majority).
If a domain has no bags in the split, its mass goes to the other domain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_frame_weights(records: list[dict], target_noboard_frac: float = 0.5) -> np.ndarray:
    """Per-frame sampling weights (un-normalized is fine for WeightedRandomSampler)."""
    # frames per bag, and each bag's domain
    bag_frames: dict[str, int] = defaultdict(int)
    bag_board: dict[str, bool] = {}
    for r in records:
        bag_frames[r["bag"]] += 1
        bag_board[r["bag"]] = r["has_board"]

    board_bags = [b for b, isb in bag_board.items() if isb]
    noboard_bags = [b for b, isb in bag_board.items() if not isb]

    # resolve domain masses, handling an empty domain
    if not noboard_bags:
        mass = {True: 1.0, False: 0.0}
    elif not board_bags:
        mass = {True: 0.0, False: 1.0}
    else:
        mass = {True: 1.0 - target_noboard_frac, False: target_noboard_frac}

    n_bags = {True: len(board_bags), False: len(noboard_bags)}

    weights = np.empty(len(records), np.float64)
    for i, r in enumerate(records):
        b = r["bag"]
        dom = bag_board[b]
        if n_bags[dom] == 0 or mass[dom] == 0.0:
            weights[i] = 0.0
        else:
            weights[i] = mass[dom] / n_bags[dom] / bag_frames[b]
    return weights


def build_sampler(records: list[dict], target_noboard_frac: float = 0.5,
                  num_samples: int | None = None, seed: int | None = None) -> WeightedRandomSampler:
    import torch
    w = compute_frame_weights(records, target_noboard_frac)
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=num_samples if num_samples is not None else len(records),
        replacement=True,
        generator=g,
    )


def load_train_records(manifest_path: str | Path, splits_path: str | Path | None = None) -> list[dict]:
    records = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
    if splits_path is not None:
        train = set(json.loads(Path(splits_path).read_text())["splits"]["train"])
        records = [r for r in records if r["bag"] in train]
    return records
