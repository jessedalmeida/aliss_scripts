#!/usr/bin/env python3
"""needle_dataset.py - Dataset for needle mask + tip/tail keypoint training.

Reads the JSONL manifest (and optionally splits.json) and yields, per frame:

    image      (3, S, S)  float32, ImageNet-normalized        [S = input_size]
    mask       (1, S, S)  float32 in {0,1}
    heatmaps   (K, Hh, Wh) float32, Gaussian peak=1 at each visible keypoint
    hm_mask    (K,)        float32, 1 where the heatmap is supervised
    vis_target (K,)        float32, 1 if visible, 0 if occluded
    vis_mask   (K,)        float32, 1 where visibility is supervised (labeled)
    meta       dict        bag, frame, has_board

Per-keypoint label logic (the three cases):
    visible  (xy set, not occluded): render heatmap; hm_mask=1; vis_target=1; vis_mask=1
    occluded (xy set, occluded):     no heatmap;     hm_mask=0; vis_target=0; vis_mask=1
    absent   (xy null):              no heatmap;     hm_mask=0; vis_target=0; vis_mask=0

Augmentation (Step 4) and the weighted sampler (Step 5) live elsewhere; this
file is the plain mapping from manifest -> tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_KEYPOINTS = ("needle_tip", "needle_tail")
NATIVE = 1080  # annotation pixel coords are in native frame resolution


def render_gaussian(hm: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    """Add a unit-peak 2D Gaussian centered at (cx, cy) onto hm in place."""
    H, W = hm.shape
    radius = int(3 * sigma)
    x0, x1 = max(0, int(cx) - radius), min(W, int(cx) + radius + 1)
    y0, y1 = max(0, int(cy) - radius), min(H, int(cy) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)


class NeedleDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        splits_path: str | Path | None = None,
        split: str | None = None,
        input_size: int = 640,
        heatmap_stride: int = 4,
        sigma: float = 2.0,
        keypoints: tuple[str, ...] = DEFAULT_KEYPOINTS,
        normalize: bool = True,
        transform=None,
    ):
        self.input_size = input_size
        self.heatmap_stride = heatmap_stride
        self.hm_size = input_size // heatmap_stride
        self.sigma = sigma
        self.keypoints = keypoints
        self.normalize = normalize
        self.transform = transform  # Step-4 hook; receives/returns a sample dict

        records = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        if split is not None:
            if splits_path is None:
                raise ValueError("split given but splits_path is None")
            bags = set(json.loads(Path(splits_path).read_text())["splits"][split])
            records = [r for r in records if r["bag"] in bags]
        if not records:
            raise RuntimeError(f"no records (split={split})")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"image not readable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"mask not readable: {path}")
        m = cv2.resize(m, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.float32)

    def _build_raw(self, rec: dict) -> dict:
        """Load image+mask at input_size and keypoints in input_size pixel space.

        Returns raw arrays the transform (Step 4) can operate on before heatmaps
        are rendered: image (S,S,3 uint8), mask (S,S float), kps (K,2 float with
        NaN for absent), vis (K,) and labeled (K,) flags.
        """
        img = self._load_image(rec["image"])          # (S,S,3) uint8
        mask = self._load_mask(rec["mask"])            # (S,S) float {0,1}
        K = len(self.keypoints)
        kps = np.full((K, 2), np.nan, np.float32)
        vis = np.zeros(K, np.float32)
        labeled = np.zeros(K, np.float32)
        s_in = self.input_size / NATIVE                # native(1080) -> input_size
        for k, name in enumerate(self.keypoints):
            kp = rec["keypoints"][name]
            xy = kp["xy"]
            if xy is None:                             # absent
                continue
            labeled[k] = 1.0                           # visibility is supervised
            kps[k] = [xy[0] * s_in, xy[1] * s_in]
            if kp["visible"]:
                vis[k] = 1.0                           # occluded -> stays 0
        return {"image": img, "mask": mask, "kps": kps, "vis": vis, "labeled": labeled}

    def _finalize(self, raw: dict, rec: dict) -> dict:
        """Render heatmaps from (possibly transformed) raw arrays -> tensors."""
        img, mask = raw["image"], raw["mask"]
        kps, vis, labeled = raw["kps"], raw["vis"], raw["labeled"]
        K = len(self.keypoints)

        heatmaps = np.zeros((K, self.hm_size, self.hm_size), np.float32)
        hm_mask = np.zeros(K, np.float32)
        s_hm = 1.0 / self.heatmap_stride               # input_size px -> heatmap grid
        for k in range(K):
            # supervise the heatmap only where the point is labeled AND visible
            if labeled[k] > 0 and vis[k] > 0 and np.isfinite(kps[k]).all():
                hm_mask[k] = 1.0
                render_gaussian(heatmaps[k], kps[k, 0] * s_hm, kps[k, 1] * s_hm, self.sigma)

        img_f = img.astype(np.float32) / 255.0
        if self.normalize:
            img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(img_f.transpose(2, 0, 1)).contiguous(),
            "mask": torch.from_numpy(mask[None].astype(np.float32)).contiguous(),
            "heatmaps": torch.from_numpy(heatmaps),
            "hm_mask": torch.from_numpy(hm_mask),
            "vis_target": torch.from_numpy(vis.astype(np.float32)),   # 1 visible, 0 occluded/out
            "vis_mask": torch.from_numpy(labeled.astype(np.float32)), # 1 where supervised
            "meta": {"bag": rec["bag"], "frame": rec["frame"], "has_board": rec["has_board"]},
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        raw = self._build_raw(rec)
        if self.transform is not None:
            raw = self.transform(raw)
        return self._finalize(raw, rec)


def heatmap_argmax(hm: torch.Tensor) -> tuple[int, int]:
    """Return (x, y) of the peak of a single 2D heatmap tensor."""
    flat = int(torch.argmax(hm))
    return flat % hm.shape[-1], flat // hm.shape[-1]
