#!/usr/bin/env python3
"""
visualize_masks.py - Quick playback of frames with overlaid segmentation masks
and optional keypoints.

Usage
-----
python3 visualize_masks.py --ann-dir ./annotations --bag ch_linearx
python3 visualize_masks.py --ann-dir ./annotations --bag ch_linearx --edit-keypoints

Controls
--------
SPACE   pause/resume
LEFT    previous frame
RIGHT   next frame
ESC     quit
Click the Play/Pause button in the top-right to toggle playback.
Click left/right thirds of the image to step backward/forward.
If `keypoints.json` exists, tip/tail points are also overlaid.
In `--edit-keypoints` mode, drag tip/tail points to update `keypoints.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (0, 255, 100),
    (255, 100, 0),
    (100, 0, 255),
    (255, 255, 0),
]


def collect_frames(frames_dir: Path) -> list[Path]:
    frame_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(sorted(frames_dir.glob(pattern)))
    return sorted(frame_paths)


def load_keypoints(keypoints_path: Path) -> dict | None:
    if not keypoints_path.exists():
        return None
    try:
        with open(keypoints_path) as f:
            return json.load(f)
    except Exception:
        return None


def save_keypoints(keypoints_path: Path, keypoints_data: dict) -> None:
    with open(keypoints_path, "w") as f:
        json.dump(keypoints_data, f, indent=2)
        f.write("\n")


def frame_index_from_name(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem


def load_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 128).astype(np.uint8)


def overlay_masks(frame: np.ndarray, mask_paths: list[Path]) -> np.ndarray:
    display = frame.copy()
    h, w = display.shape[:2]

    for idx, mask_path in enumerate(mask_paths):
        mask = load_mask(mask_path, (h, w))
        if mask is None:
            continue

        color = COLORS[idx % len(COLORS)]
        color_layer = np.zeros_like(display)
        color_layer[mask == 1] = color
        blended = cv2.addWeighted(display, 0.72, color_layer, 0.28, 0)
        display[mask == 1] = blended[mask == 1]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 2)

        label = mask_path.stem.split("_", 2)[-1] if "_" in mask_path.stem else mask_path.stem
        cv2.putText(display, label, (10, 28 + 24 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return display


def overlay_keypoints(frame: np.ndarray, keypoints_data: dict | None, frame_key: str) -> np.ndarray:
    if not keypoints_data:
        return frame

    frame_entries = keypoints_data.get("frames", {})
    entry = frame_entries.get(frame_key)
    if not entry:
        return frame

    display = frame.copy()
    points = [
        ("needle_tip", (0, 0, 255), "tip"),
        ("needle_tail", (255, 0, 0), "tail"),
    ]
    for name, color, label in points:
        pt = entry.get(name)
        if not pt or len(pt) != 2:
            continue
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(display, (x, y), 4, color, -1)
        cv2.circle(display, (x, y), 7, (255, 255, 255), 1)
        cv2.putText(display, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return display


def keypoint_positions_for_frame(keypoints_data: dict | None, frame_key: str) -> dict[str, tuple[int, int]]:
    if not keypoints_data:
        return {}
    entry = keypoints_data.get("frames", {}).get(frame_key)
    if not entry:
        return {}
    result = {}
    for name in ("needle_tip", "needle_tail"):
        pt = entry.get(name)
        if pt and len(pt) == 2:
            result[name] = (int(pt[0]), int(pt[1]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize frames with overlaid masks")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", required=True, help="Bag name to visualize")
    parser.add_argument("--edit-keypoints", action="store_true", help="Allow dragging tip/tail and save keypoints.json")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    bag_dir = ann_dir / args.bag
    frames_dir = bag_dir / "frames"
    masks_dir = bag_dir / "masks"
    keypoints_path = bag_dir / "keypoints.json"

    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return 1
    if not masks_dir.exists():
        print(f"[ERROR] Masks directory not found: {masks_dir}")
        return 1

    frame_paths = collect_frames(frames_dir)
    if not frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}")
        return 1

    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    keypoints_data = load_keypoints(keypoints_path)
    if keypoints_data:
        print(f"Found keypoints in {keypoints_path}")
    if args.edit_keypoints and not keypoints_data:
        print(f"[WARN] --edit-keypoints was requested but no keypoints.json found at {keypoints_path}")

    window_name = f"Masks - {args.bag}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused = False
    callback_ready = False
    fps = 30
    keypoint_edit_enabled = bool(args.edit_keypoints and keypoints_data)
    drag_state = {"active": False, "name": None}
    modified = False

    def on_mouse(event, x, y, flags, param):
        nonlocal paused, frame_idx, modified
        if keypoint_edit_enabled and drag_state["active"]:
            if event == cv2.EVENT_MOUSEMOVE:
                idx_key = frame_index_from_name(frame_paths[frame_idx])
                if idx_key.isdigit():
                    entry = keypoints_data.get("frames", {}).get(idx_key)
                    if entry and drag_state["name"] in entry and len(entry[drag_state["name"]]) == 2:
                        entry[drag_state["name"]] = [int(x), int(y)]
                        modified = True
            elif event == cv2.EVENT_LBUTTONUP:
                drag_state["active"] = False
                drag_state["name"] = None
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        cur_frame = cv2.imread(str(frame_paths[frame_idx]))
        if cur_frame is None:
            return
        h, w = cur_frame.shape[:2]

        btn_w, btn_h = 110, 34
        bx1, by1 = w - btn_w - 10, 10
        bx2, by2 = w - 10, 10 + btn_h
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            paused = not paused
            return

        if keypoint_edit_enabled:
            idx_key = frame_index_from_name(frame_paths[frame_idx])
            kp_positions = keypoint_positions_for_frame(keypoints_data, idx_key)
            if kp_positions:
                for name, pt in kp_positions.items():
                    if (x - pt[0]) ** 2 + (y - pt[1]) ** 2 <= 10 ** 2:
                        drag_state["active"] = True
                        drag_state["name"] = name
                        modified = True
                        return

        if x < w // 3:
            frame_idx = (frame_idx - 1) % len(frame_paths)
        elif x > (2 * w) // 3:
            frame_idx = (frame_idx + 1) % len(frame_paths)

    while True:
        frame_path = frame_paths[frame_idx]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Could not load frame: {frame_path}")
            frame_idx = (frame_idx + 1) % len(frame_paths)
            continue

        h, w = frame.shape[:2]
        cv2.resizeWindow(window_name, w, h)

        idx_str = frame_index_from_name(frame_path)
        mask_candidates = sorted(masks_dir.glob(f"frame_{int(idx_str):06d}_*.png")) if idx_str.isdigit() else []
        if not mask_candidates:
            mask_candidates = sorted(masks_dir.glob(f"*{idx_str}*.png"))

        display = overlay_masks(frame, mask_candidates)
        display = overlay_keypoints(display, keypoints_data, idx_str)

        if keypoint_edit_enabled:
            cv2.putText(display, "EDIT KEYPOINTS: drag tip/tail, press S to save", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw play/pause button in the top-right corner.
        btn_w, btn_h = 110, 34
        bx1, by1 = w - btn_w - 10, 10
        bx2, by2 = w - 10, 10 + btn_h
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        btn_label = "Play" if paused else "Pause"
        cv2.putText(display, btn_label, (bx1 + 16, by1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(
            display,
            f"Frame {frame_idx + 1}/{len(frame_paths)}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow(window_name, display)

        if not callback_ready:
            cv2.waitKey(1)
            try:
                cv2.setMouseCallback(window_name, on_mouse)
                callback_ready = True
            except cv2.error:
                pass

        delay = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:
            break
        if key == 32:
            paused = not paused
        elif key in (ord("s"), ord("S")) and keypoint_edit_enabled and keypoints_data:
            save_keypoints(keypoints_path, keypoints_data)
            modified = False
            print(f"Saved edited keypoints -> {keypoints_path}")
        elif key in (81, 2424832):
            frame_idx = (frame_idx - 1) % len(frame_paths)
        elif key in (83, 2555904):
            frame_idx = (frame_idx + 1) % len(frame_paths)

        if not paused:
            frame_idx = (frame_idx + 1) % len(frame_paths)

    cv2.destroyAllWindows()

    if keypoint_edit_enabled and keypoints_data and modified:
        save_keypoints(keypoints_path, keypoints_data)
        print(f"Saved edited keypoints -> {keypoints_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
