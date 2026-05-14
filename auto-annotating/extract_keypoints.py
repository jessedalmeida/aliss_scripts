#!/usr/bin/env python3
"""
extract_keypoints.py - Derive needle tip/tail keypoints from propagated masks.

This is a prototype keypoint extractor for the current workflow:
- Use propagated `needle_mask` frames as the primary signal.
- Skeletonize the needle mask.
- Extract the two skeleton endpoints.
- Keep tip/tail identities stable across frames.
- Use one selected seed frame where you click tip and tail once.

Outputs one `keypoints.json` per bag.

Example
-------
python3 extract_keypoints.py --ann-dir ./annotations --bag ch_linearx --seed-ui
python3 extract_keypoints.py --ann-dir ./annotations --all --seed-ui

Notes
-----
- Tip/tail ordering is anchored by the clicked seed-frame tip/tail.
- This is intended as a fast annotation aid, not a final ML keypoint model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def collect_bags(ann_dir: Path, bag_name: str | None, all_bags: bool) -> list[Path]:
    if bag_name:
        return [ann_dir / bag_name]
    if all_bags:
        return sorted([p for p in ann_dir.iterdir() if p.is_dir() and (p / "masks").exists()])
    return sorted([p for p in ann_dir.iterdir() if p.is_dir() and (p / "masks").exists()])


def frame_index_from_filename(path: Path) -> int | None:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def parse_tip_hint(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("--tip-hint must be 'x,y'")
    return int(parts[0]), int(parts[1])


def load_mask(mask_path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Pure-numpy Zhang-Suen thinning for binary masks."""
    img = (binary > 0).astype(np.uint8).copy()
    if img.ndim != 2:
        raise ValueError("Expected a 2D binary image")

    changed = True
    rows, cols = img.shape
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[0:rows, 1:cols + 1]
            p3 = padded[0:rows, 2:cols + 2]
            p4 = padded[1:rows + 1, 2:cols + 2]
            p5 = padded[2:rows + 2, 2:cols + 2]
            p6 = padded[2:rows + 2, 1:cols + 1]
            p7 = padded[2:rows + 2, 0:cols]
            p8 = padded[1:rows + 1, 0:cols]
            p9 = padded[0:rows, 0:cols]
            p1 = img

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            cond1 = (p1 == 1) & (neighbors >= 2) & (neighbors <= 6)
            cond2 = transitions == 1
            if step == 0:
                cond3 = (p2 * p4 * p6 == 0)
                cond4 = (p4 * p6 * p8 == 0)
            else:
                cond3 = (p2 * p4 * p8 == 0)
                cond4 = (p2 * p6 * p8 == 0)

            remove = cond1 & cond2 & cond3 & cond4
            if np.any(remove):
                img[remove] = 0
                changed = True

    return img.astype(np.uint8)


def skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Return endpoints as (x, y) coordinates from a skeleton image."""
    skel = (skeleton > 0).astype(np.uint8)
    if not np.any(skel):
        return []

    padded = np.pad(skel, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 2:] +
        padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    endpoint_mask = (center == 1) & (neighbors == 1)
    ys, xs = np.where(endpoint_mask)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def farthest_pair(points: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(points) < 2:
        return None
    best = None
    best_dist = -1.0
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dist > best_dist:
                best_dist = dist
                best = (points[i], points[j])
    return best


def pca_axis_endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return []
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ axis
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    return [(int(pts[i_min, 0]), int(pts[i_min, 1])), (int(pts[i_max, 0]), int(pts[i_max, 1]))]


def choose_endpoints(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    skel = zhang_suen_thinning(mask)
    endpoints = skeleton_endpoints(skel)
    pair = farthest_pair(endpoints) if len(endpoints) >= 2 else None
    if pair is None:
        fallback = pca_axis_endpoints(mask)
        if len(fallback) < 2:
            return None
        pair = (fallback[0], fallback[1])
    return pair


def order_tip_tail(
    pair: tuple[tuple[int, int], tuple[int, int]],
    tip_seed: tuple[int, int] | None,
    tail_seed: tuple[int, int] | None,
    prev_tip: tuple[int, int] | None,
    prev_tail: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = pair

    if prev_tip is not None and prev_tail is not None:
        cost_ab = math.dist(a, prev_tip) + math.dist(b, prev_tail)
        cost_ba = math.dist(b, prev_tip) + math.dist(a, prev_tail)
        return (a, b) if cost_ab <= cost_ba else (b, a)

    if tip_seed is not None and tail_seed is not None:
        cost_ab = math.dist(a, tip_seed) + math.dist(b, tail_seed)
        cost_ba = math.dist(b, tip_seed) + math.dist(a, tail_seed)
        return (a, b) if cost_ab <= cost_ba else (b, a)

    if tip_seed is not None:
        return (a, b) if math.dist(a, tip_seed) <= math.dist(b, tip_seed) else (b, a)

    # Deterministic fallback if no hint exists yet.
    return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)


def load_frame_image(bag_dir: Path, frame_idx: int) -> np.ndarray | None:
    frame_candidates = [
        bag_dir / "frames" / f"frame_{frame_idx:06d}.jpg",
        bag_dir / "frames" / f"frame_{frame_idx:06d}.png",
        bag_dir / "frames_jpg" / f"{frame_idx:06d}.jpg",
        bag_dir / "frames" / f"{frame_idx:06d}.jpg",
        bag_dir / "frames" / f"{frame_idx:06d}.png",
    ]
    for candidate in frame_candidates:
        if candidate.exists():
            img = cv2.imread(str(candidate))
            if img is not None:
                return img
    return None


class SeedKeypointSession:
    """Click tip and tail on a single seed frame to anchor ordering."""

    def __init__(self, frame: np.ndarray, mask: np.ndarray, bag_stem: str, frame_idx: int):
        self.frame = frame
        self.mask = mask
        self.bag_stem = bag_stem
        self.frame_idx = frame_idx
        self.window_name = f"Keypoint Seed - {bag_stem}"
        self.points: dict[str, tuple[int, int] | None] = {"tip": None, "tail": None}
        self.current_label = "tip"
        self.done = False
        self.quit_all = False
        self.display = frame.copy()

    def _render(self):
        vis = self.frame.copy()
        if self.mask is not None and np.any(self.mask):
            ys, xs = np.where(self.mask > 0)
            vis[ys, xs] = (0.7 * vis[ys, xs] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)

        colors = {"tip": (0, 0, 255), "tail": (255, 0, 0)}
        for label, pt in self.points.items():
            if pt is None:
                continue
            cv2.circle(vis, pt, 2, colors[label], -1)
            cv2.circle(vis, pt, 4, (255, 255, 255), 1)
            cv2.putText(vis, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[label], 2)

        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        lines = [
            f"Bag: {self.bag_stem}  |  Seed frame: {self.frame_idx}",
            f"Click {self.current_label} with L-click; N toggles tip/tail; Z undo; C clear; ENTER accept",
            f"tip={self.points['tip']}  tail={self.points['tail']}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, h - 60 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        self.display = vis

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.points[self.current_label] = (int(x), int(y))
        if self.current_label == "tip":
            self.current_label = "tail"
        self._render()

    def run(self) -> dict | None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1080, 720)
        self._render()
        cv2.imshow(self.window_name, self.display)
        cv2.waitKey(1)
        try:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        except cv2.error as exc:
            raise RuntimeError("OpenCV could not attach the seed-point callback.") from exc

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("n"):
                self.current_label = "tail" if self.current_label == "tip" else "tip"
                self._render()
            elif key == ord("z"):
                if self.points["tail"] is not None and self.current_label == "tail":
                    self.points["tail"] = None
                    self.current_label = "tail"
                elif self.points["tip"] is not None:
                    self.points["tip"] = None
                    self.current_label = "tip"
                self._render()
            elif key == ord("c"):
                self.points = {"tip": None, "tail": None}
                self.current_label = "tip"
                self._render()
            elif key in (13, 32):
                if self.points["tip"] is None or self.points["tail"] is None:
                    print("  [!] Please click both tip and tail before accepting.")
                    continue
                break
            elif key in (ord("q"), 27):
                self.quit_all = True
                cv2.destroyWindow(self.window_name)
                return None

        cv2.destroyWindow(self.window_name)
        return {
            "frame_idx": self.frame_idx,
            "tip": [int(self.points["tip"][0]), int(self.points["tip"][1])],
            "tail": [int(self.points["tail"][0]), int(self.points["tail"][1])],
        }


def extract_bag_keypoints(
    bag_dir: Path,
    seed_ui: bool,
    seed_frame_idx: int | None,
    tip_hint: tuple[int, int] | None,
    save_preview: bool = False,
) -> dict:
    masks_dir = bag_dir / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks dir: {masks_dir}")

    mask_paths = sorted(masks_dir.glob("frame_*_needle_mask.png"))
    if not mask_paths:
        raise RuntimeError(f"No needle masks found in {masks_dir}")

    preview_dir = bag_dir / "keypoints_preview"
    if save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if seed_frame_idx is None:
        seed_frame_idx = frame_index_from_filename(mask_paths[0])

    seed_points = None
    if seed_ui:
        seed_mask_path = next((p for p in mask_paths if frame_index_from_filename(p) == seed_frame_idx), None)
        if seed_mask_path is None:
            raise RuntimeError(f"No needle mask found for seed frame {seed_frame_idx}")
        seed_frame = load_frame_image(bag_dir, seed_frame_idx)
        if seed_frame is None:
            raise RuntimeError(f"Could not load seed frame image for frame {seed_frame_idx}")
        seed_mask = load_mask(seed_mask_path)
        session = SeedKeypointSession(seed_frame, seed_mask, bag_dir.name, seed_frame_idx)
        seed_points = session.run()
        if seed_points is None:
            return {
                "bag_stem": bag_dir.name,
                "source": "mask_skeleton_endpoints_with_seed_ui",
                "seed_frame_idx": seed_frame_idx,
                "seed_points": None,
                "frames": {},
            }

    frame_keypoints: dict[str, dict] = {}
    prev_tip = None
    prev_tail = None

    for mask_path in mask_paths:
        frame_idx = frame_index_from_filename(mask_path)
        if frame_idx is None:
            continue

        mask = load_mask(mask_path)
        if mask is None or not np.any(mask):
            frame_keypoints[f"{frame_idx:06d}"] = {
                "needle_tip": None,
                "needle_tail": None,
                "status": "missing_mask",
            }
            continue

        pair = choose_endpoints(mask)
        if pair is None:
            frame_keypoints[f"{frame_idx:06d}"] = {
                "needle_tip": None,
                "needle_tail": None,
                "status": "no_endpoints",
            }
            continue

        if frame_idx == seed_frame_idx and seed_points is not None:
            tip_seed = tuple(seed_points["tip"])
            tail_seed = tuple(seed_points["tail"])
            tip, tail = order_tip_tail(pair, tip_seed, tail_seed, None, None)
        else:
            tip, tail = order_tip_tail(
                pair,
                tip_hint if prev_tip is None and seed_points is None else None,
                None,
                prev_tip,
                prev_tail,
            )
        prev_tip, prev_tail = tip, tail

        frame_keypoints[f"{frame_idx:06d}"] = {
            "needle_tip": [int(tip[0]), int(tip[1])],
            "needle_tail": [int(tail[0]), int(tail[1])],
            "status": "ok",
            "method": "mask_skeleton_endpoints",
        }

        if save_preview:
            frame_path_candidates = [
                bag_dir / "frames" / f"frame_{frame_idx:06d}.jpg",
                bag_dir / "frames" / f"frame_{frame_idx:06d}.png",
                bag_dir / "frames_jpg" / f"{frame_idx:06d}.jpg",
            ]
            frame_img = None
            for candidate in frame_path_candidates:
                if candidate.exists():
                    frame_img = cv2.imread(str(candidate))
                    if frame_img is not None:
                        break
            if frame_img is not None:
                vis = frame_img.copy()
                ys, xs = np.where(mask > 0)
                vis[ys, xs] = (0.7 * vis[ys, xs] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.circle(vis, tip, 5, (0, 0, 255), -1)
                cv2.circle(vis, tail, 5, (255, 0, 0), -1)
                cv2.putText(vis, "tip", (tip[0] + 6, tip[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(vis, "tail", (tail[0] + 6, tail[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(str(preview_dir / f"frame_{frame_idx:06d}.png"), vis)

    return {
        "bag_stem": bag_dir.name,
        "source": "mask_skeleton_endpoints_with_seed_ui" if seed_ui else "mask_skeleton_endpoints",
        "seed_frame_idx": seed_frame_idx,
        "seed_points": seed_points,
        "tip_hint": list(tip_hint) if tip_hint is not None else None,
        "frames": frame_keypoints,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract needle tip/tail keypoints from bag masks")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", default=None, help="Specific bag to process")
    parser.add_argument("--all", action="store_true", help="Process all bags with masks")
    parser.add_argument("--seed-ui", action="store_true", help="Open a UI to click tip/tail on a seed frame")
    parser.add_argument("--seed-frame", type=int, default=None, help="Seed frame index to label (default: first mask frame)")
    parser.add_argument("--tip-hint", default=None, help="Fallback tip hint as x,y if seed UI is disabled")
    parser.add_argument("--preview", action="store_true", help="Save preview overlays per frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        print(f"[ERROR] ann-dir does not exist: {ann_dir}")
        return 1

    bag_dirs = collect_bags(ann_dir, args.bag, args.all)
    if not bag_dirs:
        print(f"[ERROR] No bags found in {ann_dir}")
        return 1

    for bag_dir in bag_dirs:
        try:
            result = extract_bag_keypoints(
                bag_dir,
                seed_ui=args.seed_ui,
                seed_frame_idx=args.seed_frame,
                tip_hint=parse_tip_hint(args.tip_hint),
                save_preview=args.preview,
            )
        except Exception as exc:
            print(f"[ERROR] {bag_dir.name}: {exc}")
            continue

        out_path = bag_dir / "keypoints.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[OK] wrote {out_path}")
        print(f"     frames={len(result['frames'])} preview={'on' if args.preview else 'off'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
