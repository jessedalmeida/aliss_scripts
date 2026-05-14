#!/usr/bin/env python3
"""
annotate_keypoints.py - Annotate needle tip/tail using optical flow tracking.

User labels tip/tail on the first frame manually, then optical flow (Lucas-Kanade)
propagates the tracking forward and backward. User can interactively correct
predictions when flow drifts.

Example:
    python3 annotate_keypoints.py --ann-dir ./annotations --bag ch_suture1
    python3 annotate_keypoints.py --ann-dir ./annotations --bag ch_suture1 --seed-frame 10

Controls (Seed UI):
  Left-click: place tip, then tail
  Z: undo
  ENTER: confirm
  Q: quit

Controls (Tracking Verification):
  LEFT/RIGHT arrows or A/D: prev/next frame
  Left-click: correct tip or tail (next click is opposite)
    T: toggle which point the next click edits
    R: re-anchor flow from the current corrected frame forward
        O: toggle occluded mode for the selected point in the current frame
  ENTER: accept all predictions
  Q: quit without saving
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def collect_bags(ann_dir: Path, bag_name: str | None) -> list[Path]:
    if bag_name:
        return [ann_dir / bag_name]
    return sorted([p for p in ann_dir.iterdir() if p.is_dir() and (p / "frames").exists()])


def find_frame_paths(frames_dir: Path) -> list[Path]:
    """Return sorted list of frame image paths."""
    exts = ["*.png", "*.jpg", "*.jpeg"]
    frames = []
    for ext in exts:
        frames.extend(sorted(frames_dir.glob(ext)))
    return sorted(set(frames))  # deduplicate and sort


def frame_index_from_path(path: Path) -> int | None:
    """Extract frame index from filename (e.g., 'frame_000042.jpg' -> 42 or '000042.jpg' -> 42)."""
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


class SeedKeypointUI:
    """Interactive UI to click tip/tail on a seed frame."""

    def __init__(self, frame: np.ndarray, frame_idx: int, bag_stem: str):
        self.frame = frame
        self.frame_idx = frame_idx
        self.bag_stem = bag_stem
        self.window_name = f"Seed Keypoints - {bag_stem}"
        self.points = {"tip": None, "tail": None}
        self.current_label = "tip"
        self.display = frame.copy()

    def _render(self):
        vis = self.frame.copy()
        colors = {"tip": (0, 0, 255), "tail": (255, 0, 0)}
        for label, pt in self.points.items():
            if pt is None:
                continue
            cv2.circle(vis, pt, 1, colors[label], -1)
            cv2.circle(vis, pt, 4, (255, 255, 255), 1)
            cv2.putText(vis, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[label], 2)

        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        lines = [
            f"Bag: {self.bag_stem}  |  Seed frame: {self.frame_idx}",
            f"Click {self.current_label} (red=tail, blue=tip). Z=undo, ENTER=done, Q=quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, h - 50 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        self.display = vis

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.points[self.current_label] = (int(x), int(y))
        self.current_label = "tail" if self.current_label == "tip" else "tip"
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

            if key == ord("z"):
                if self.points["tail"] is not None:
                    self.points["tail"] = None
                    self.current_label = "tail"
                elif self.points["tip"] is not None:
                    self.points["tip"] = None
                    self.current_label = "tip"
                self._render()
            elif key in (13, 32):  # ENTER or SPACE
                if self.points["tip"] is None or self.points["tail"] is None:
                    print("  [!] Please label both tip and tail.")
                    continue
                break
            elif key in (ord("q"), 27):  # Q or ESC
                cv2.destroyWindow(self.window_name)
                return None

        cv2.destroyWindow(self.window_name)
        return {
            "frame_idx": self.frame_idx,
            "tip": list(self.points["tip"]),
            "tail": list(self.points["tail"]),
        }


class OpticalFlowTracker:
    """Track tip/tail keypoints using Lucas-Kanade optical flow."""

    def __init__(self, frames: list[np.ndarray], seed_frame_idx: int, seed_tip: tuple, seed_tail: tuple):
        self.frames = frames
        self.seed_idx = seed_frame_idx
        self.n_frames = len(frames)

        # Convert frames to grayscale for flow computation
        self.gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]

        # Initialize tracking at seed frame
        self.tracked = {seed_frame_idx: {"tip": seed_tip, "tail": seed_tail}}

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def _track_frame_to_frame(self, gray1: np.ndarray, gray2: np.ndarray, points_dict: dict) -> dict:
        """Track points from gray1 to gray2 using Lucas-Kanade."""
        points = np.array([points_dict["tip"], points_dict["tail"]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            new_points, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None, **self.lk_params)
            if new_points is None:
                return None
            new_points = new_points.reshape(-1, 2)
            return {"tip": tuple(new_points[0]), "tail": tuple(new_points[1])}
        except Exception:
            return None

    def track_forward(self):
        """Track from seed frame forward to end."""
        for i in range(self.seed_idx, self.n_frames - 1):
            if i not in self.tracked:
                continue
            prev_points = self.tracked[i]
            next_points = self._track_frame_to_frame(self.gray_frames[i], self.gray_frames[i + 1], prev_points)
            if next_points is not None:
                self.tracked[i + 1] = next_points

    def track_backward(self):
        """Track from seed frame backward to start."""
        for i in range(self.seed_idx, 0, -1):
            if i not in self.tracked:
                continue
            prev_points = self.tracked[i]
            next_points = self._track_frame_to_frame(self.gray_frames[i], self.gray_frames[i - 1], prev_points)
            if next_points is not None:
                self.tracked[i - 1] = next_points

    def reseed_from(self, frame_idx: int, tip: tuple[float, float], tail: tuple[float, float]):
        """Treat a corrected frame as a new anchor and repropagate forward only."""
        anchor = {"tip": (float(tip[0]), float(tip[1])), "tail": (float(tail[0]), float(tail[1]))}
        self.tracked[frame_idx] = anchor

        # Re-track forward from the corrected frame only.
        # Earlier frames are left untouched so a local correction does not
        # overwrite already-good keypoints behind the anchor.
        current = anchor
        for i in range(frame_idx, self.n_frames - 1):
            self.tracked[i] = current
            next_points = self._track_frame_to_frame(self.gray_frames[i], self.gray_frames[i + 1], current)
            if next_points is None:
                break
            current = next_points
        self.tracked[self.n_frames - 1 if self.n_frames else frame_idx] = current

    def get_predictions(self) -> dict[int, dict]:
        """Return keypoints for all tracked frames."""
        return self.tracked


class VerificationUI:
    """Interactive UI to verify and correct optical flow predictions."""

    def __init__(self, frames: list[np.ndarray], predictions: dict, bag_stem: str, frame_indices: list, reseed_callback=None):
        self.frames = frames
        self.predictions = predictions  # {frame_idx: {"tip": (x,y), "tail": (x,y)}}
        self.bag_stem = bag_stem
        self.frame_indices = frame_indices  # map from UI frame num to actual frame idx
        self.n_frames = len(frames)
        self.current_frame_num = 0
        self.corrections = {}  # {frame_idx: {"tip": (x,y), "tail": (x,y)}}
        self.current_label = "tip"
        self.occluded = {}  # {frame_idx: {"tip": bool, "tail": bool}}
        self.window_name = f"Verify Keypoints - {bag_stem}"
        self.display = None
        self.reseed_callback = reseed_callback

    def _get_frame_idx(self, frame_num: int) -> int:
        return self.frame_indices[frame_num] if frame_num < len(self.frame_indices) else frame_num

    def _frame_occlusion_state(self, frame_idx: int) -> dict[str, bool]:
        return self.occluded.setdefault(frame_idx, {"tip": False, "tail": False})

    def _is_occluded(self, frame_idx: int, label: str) -> bool:
        return self.occluded.get(frame_idx, {}).get(label, False)

    def _render(self):
        frame = self.frames[self.current_frame_num].copy()
        frame_idx = self._get_frame_idx(self.current_frame_num)

        # Show prediction or correction
        keypoints = self.corrections.get(frame_idx)
        if keypoints is None:
            keypoints = self.predictions.get(frame_idx)
            status = "predicted"
        else:
            status = "corrected"

        if keypoints is not None:
            tip, tail = keypoints.get("tip"), keypoints.get("tail")
            occluded = self._frame_occlusion_state(frame_idx)
            if not occluded["tip"] and tip:
                cv2.circle(frame, tuple(map(int, tip)), 1, (0, 0, 255), -1)
                cv2.circle(frame, tuple(map(int, tip)), 4, (255, 255, 255), 1)
                cv2.putText(frame, "tip", (int(tip[0]) + 6, int(tip[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if not occluded["tail"] and tail:
                cv2.circle(frame, tuple(map(int, tail)), 1, (255, 0, 0), -1)
                cv2.circle(frame, tuple(map(int, tail)), 4, (255, 255, 255), 1)
                cv2.putText(frame, "tail", (int(tail[0]) + 6, int(tail[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        lines = [
            f"Frame {self.current_frame_num}/{self.n_frames - 1}  [{status}]",
            "A/D or arrows=navigate  Left-click=correct selected point  ENTER=accept all  Q=quit",
            f"Next correction: {self.current_label.upper()}  (T to toggle)  R=reanchor  O=toggle {self.current_label} occluded:{'on' if self._is_occluded(frame_idx, self.current_label) else 'off'}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - 50 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        self.display = frame

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame_idx = self._get_frame_idx(self.current_frame_num)
        if frame_idx not in self.corrections:
            # Get current prediction or initialize
            pred = self.predictions.get(frame_idx, {"tip": None, "tail": None})
            self.corrections[frame_idx] = {"tip": pred.get("tip"), "tail": pred.get("tail")}

        # Update the selected label (current_label)
        self.corrections[frame_idx][self.current_label] = (float(x), float(y))
        self._render()

    def _current_points(self) -> dict | None:
        """Return the prediction/correction currently shown for the active frame."""
        frame_idx = self._get_frame_idx(self.current_frame_num)
        keypoints = self.corrections.get(frame_idx)
        if keypoints is None:
            keypoints = self.predictions.get(frame_idx)
        return keypoints

    def _reanchor_current_frame(self):
        """Promote the current frame to a new seed and rerun flow from there."""
        frame_idx = self._get_frame_idx(self.current_frame_num)
        keypoints = self._current_points()
        if keypoints is None or keypoints.get("tip") is None or keypoints.get("tail") is None:
            print("  [!] Need both tip and tail on this frame before re-anchoring.")
            return

        if self.reseed_callback is None:
            print("  [!] Re-seeding is not available.")
            return

        self.reseed_callback(frame_idx, keypoints["tip"], keypoints["tail"])

        # Corrections before the anchor are still valid; downstream frames will be refreshed.
        for idx in list(self.corrections.keys()):
            if idx >= frame_idx:
                del self.corrections[idx]

        self.current_frame_num = min(self.current_frame_num, self.n_frames - 1)
        self._render()

    def run(self) -> dict:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1080, 720)
        self._render()
        cv2.imshow(self.window_name, self.display)
        cv2.waitKey(1)

        try:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        except cv2.error as exc:
            raise RuntimeError("OpenCV could not attach the verification callback.") from exc

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue

            if key in (ord("a"), 81, 2424832):  # A or left arrow
                self.current_frame_num = max(0, self.current_frame_num - 1)
                self._render()
            elif key in (ord("d"), 83, 2555904):  # D or right arrow
                self.current_frame_num = min(self.n_frames - 1, self.current_frame_num + 1)
                self._render()
            elif key == ord("t"):  # T to toggle tip/tail
                self.current_label = "tail" if self.current_label == "tip" else "tip"
                self._render()
            elif key == ord("o"):  # O to toggle occlusion mode
                frame_idx = self._get_frame_idx(self.current_frame_num)
                occluded = self._frame_occlusion_state(frame_idx)
                occluded[self.current_label] = not occluded[self.current_label]
                self._render()
            elif key == ord("r"):  # R to re-anchor the flow on this frame
                self._reanchor_current_frame()
            elif key in (13, 32):  # ENTER or SPACE
                break
            elif key in (ord("q"), 27):  # Q or ESC
                cv2.destroyWindow(self.window_name)
                return None

        cv2.destroyWindow(self.window_name)

        # Merge predictions and corrections
        final = {}
        all_frame_indices = sorted(set(self.predictions.keys()) | set(self.corrections.keys()))
        for frame_idx in all_frame_indices:
            keypoints = self.corrections.get(frame_idx, self.predictions.get(frame_idx))
            if keypoints is None:
                continue

            occluded = self._frame_occlusion_state(frame_idx)
            final[frame_idx] = {
                "tip": None if occluded["tip"] else keypoints.get("tip"),
                "tail": None if occluded["tail"] else keypoints.get("tail"),
                "occluded": {"tip": occluded["tip"], "tail": occluded["tail"]},
            }
        return final


def annotate_bag_keypoints(bag_dir: Path, seed_frame_idx: int | None = None) -> dict:
    """Interactively annotate and track tip/tail for a bag using optical flow."""
    frames_dir = bag_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    frame_paths = find_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Load all frames
    frames = []
    frame_indices = []
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(img)
            fidx = frame_index_from_path(path)
            frame_indices.append(fidx if fidx is not None else len(frame_indices))

    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError("Could not load any frames")

    # Determine seed frame
    if seed_frame_idx is None:
        seed_frame_idx = 0
    seed_frame_idx = max(0, min(seed_frame_idx, n_frames - 1))

    print(f"  Loaded {n_frames} frames.")
    print(f"  Seed frame index: {seed_frame_idx}")

    # Seed UI: user labels tip/tail
    print("\n  Opening seed frame for tip/tail labeling…")
    seed_ui = SeedKeypointUI(frames[seed_frame_idx], frame_indices[seed_frame_idx], bag_dir.name)
    seed_result = seed_ui.run()
    if seed_result is None:
        print("  User quit seed labeling.")
        return None

    seed_tip = tuple(seed_result["tip"])
    seed_tail = tuple(seed_result["tail"])
    print(f"  Seed labeled: tip={seed_tip}, tail={seed_tail}")

    # Optical flow tracking
    print("\n  Tracking with optical flow…")
    tracker = OpticalFlowTracker(frames, seed_frame_idx, seed_tip, seed_tail)
    tracker.track_forward()
    tracker.track_backward()
    predictions = tracker.get_predictions()
    print(f"  Tracked {len(predictions)} frames")

    # Verification UI: user can correct predictions
    print("\n  Opening verification UI…")
    verifier = VerificationUI(frames, predictions, bag_dir.name, frame_indices, reseed_callback=tracker.reseed_from)
    final = verifier.run()
    if final is None:
        print("  User quit without saving.")
        return None

    # If the verifier re-anchored the flow, use the updated tracker output.
    predictions = tracker.get_predictions()

    # Format output
    result = {
        "bag_stem": bag_dir.name,
        "source": "optical_flow_tracking",
        "seed_frame_idx": frame_indices[seed_frame_idx],
        "seed_points": {"frame_idx": frame_indices[seed_frame_idx], "tip": _point_to_list(seed_tip), "tail": _point_to_list(seed_tail)},
        "frames": {},
    }

    for frame_num, actual_idx in enumerate(frame_indices):
        if frame_num in range(len(frames)):
            keypoints = final.get(actual_idx)
            if keypoints is not None:
                result["frames"][f"{actual_idx:06d}"] = {
                    "needle_tip": _point_to_list(keypoints["tip"]),
                    "needle_tail": _point_to_list(keypoints["tail"]),
                    "occluded": keypoints.get("occluded", {"tip": False, "tail": False}),
                    "status": "ok",
                }

    return result


def _point_to_list(point) -> list[float] | None:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def main():
    parser = argparse.ArgumentParser(description="Annotate needle tip/tail with optical flow tracking")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", required=True, help="Bag stem to annotate")
    parser.add_argument("--seed-frame", type=int, default=0, help="Seed frame index (default: 0)")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        print(f"[ERROR] ann-dir does not exist: {ann_dir}")
        return 1

    bag_dirs = collect_bags(ann_dir, args.bag)
    if not bag_dirs:
        print(f"[ERROR] No bag found: {args.bag}")
        return 1

    for bag_dir in bag_dirs:
        print(f"\n{'='*60}")
        print(f"BAG: {bag_dir.name}")
        print(f"{'='*60}")

        try:
            result = annotate_bag_keypoints(bag_dir, seed_frame_idx=args.seed_frame)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return 1

        if result is None:
            print(f"[SKIP] {bag_dir.name}")
            continue

        out_path = bag_dir / "keypoints.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n✓ Saved keypoints → {out_path}")
        print(f"  Frames: {len(result['frames'])}  Source: {result['source']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
