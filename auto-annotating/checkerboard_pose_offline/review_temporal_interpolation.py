#!/usr/bin/env python3
"""
Review temporal interpolations for failed poses before applying them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visualize_poses import (
    choose_camera_yaml,
    collect_frames,
    draw_pose_overlay,
    load_camera_yaml,
    load_json as load_pose_json,
)
from temporal_interpolation import compute_temporal_fill, save_interpolated_preview


def overlay_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    display = image.copy()
    h, w = display.shape[:2]
    panel_h = min(150, max(100, 24 * len(lines) + 20))
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (10, h - panel_h + 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return display


def select_review_frame(bag_dir: Path, frame_key: str) -> tuple[np.ndarray | None, str]:
    """Load frame image for review."""
    frames_dir = bag_dir / "frames"
    frame_paths = collect_frames(frames_dir)
    
    frame_path = None
    for candidate in frame_paths:
        if candidate.stem.endswith(frame_key) or frame_key in candidate.stem:
            frame_path = candidate
            break
    
    if frame_path is None:
        try:
            frame_idx = int(frame_key)
        except ValueError:
            return None, "invalid frame key"
        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        if not frame_path.exists():
            frame_path = frames_dir / f"{frame_idx:06d}.png"
    
    if not frame_path.exists():
        return None, f"not found: {frame_path.name}"
    
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, f"unreadable: {frame_path.name}"
    
    return image, frame_path.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Review temporal interpolations for failed poses")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", required=True, help="Bag stem")
    parser.add_argument("--poses-json", default=None, help="Path to original poses.json")
    parser.add_argument("--camera-yaml", default=None, help="Path to ves_camera.yaml")
    args = parser.parse_args()

    bag_dir = Path(args.ann_dir) / args.bag
    poses_json = Path(args.poses_json) if args.poses_json else bag_dir / "poses.json"
    
    if not poses_json.exists():
        print(f"[ERROR] poses.json not found: {poses_json}")
        return 1

    print("[INFO] Computing temporal interpolations...")
    temporal_results = compute_temporal_fill(poses_json)
    
    successful = [k for k, v in temporal_results.items() if v.get("status") == "success"]
    failed = [k for k, v in temporal_results.items() if v.get("status") != "success"]
    
    print(f"[INFO] Successfully computed {len(successful)} temporal interpolations")
    if failed:
        print(f"[WARN] {len(failed)} frames could not be interpolated:")
        for frame_key in failed[:5]:
            reason = temporal_results[frame_key].get("reason", "unknown")
            print(f"       {frame_key}: {reason}")
        if len(failed) > 5:
            print(f"       ... and {len(failed) - 5} more")

    if not successful:
        print("[ERROR] No frames could be interpolated")
        return 1

    print("\n[INFO] Saving preview to poses_temporal_preview.json")
    preview_path = bag_dir / "poses_temporal_preview.json"
    save_interpolated_preview(poses_json, temporal_results, preview_path)
    print(f"[INFO] Preview saved to {preview_path}")

    print("\n[INFO] Loading interpolated poses preview for review...")
    preview_payload = load_pose_json(preview_path)
    frames = preview_payload.get("frames", {})
    
    camera_yaml_path = choose_camera_yaml(Path(args.ann_dir), bag_dir, args.camera_yaml)
    camera_data = load_camera_yaml(camera_yaml_path)
    camera_matrix = np.asarray(camera_data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)[:, :3]
    distortion = np.asarray(camera_data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
    
    params = preview_payload.get("parameters", {})
    
    accepted = set()
    rejected = set()
    current_idx = 0
    
    while current_idx < len(successful):
        frame_key = successful[current_idx]
        frame_data = frames.get(frame_key, {})
        temporal_info = frame_data.get("temporal_source", {})
        
        frame, source_desc = select_review_frame(bag_dir, frame_key)
        if frame is None:
            print(f"[WARN] Could not load image for frame {frame_key}")
            current_idx += 1
            continue
        
        display = draw_pose_overlay(
            frame,
            camera_matrix,
            distortion,
            frame_data,
            squares_x=int(params.get("squares_x", 4)),
            squares_y=int(params.get("squares_y", 5)),
            square_size=float(params.get("square_size", 0.002)),
            draw_corners=True,
        )
        
        before_idx = temporal_info.get("before_idx")
        after_idx = temporal_info.get("after_idx")
        before_dist = temporal_info.get("before_distance")
        after_dist = temporal_info.get("after_distance")
        t_factor = temporal_info.get("interpolation_factor", 0.5)
        
        display = overlay_text(
            display,
            [
                f"Frame {frame_key} | Temporal Interpolation",
                f"Between {before_idx} ({before_dist} steps before) and {after_idx} ({after_dist} steps after)",
                f"Interpolation factor t={t_factor:.3f}",
                "Y accept | N reject | Left/Right navigate | S save | Q quit",
            ],
        )
        
        cv2.imshow(f"Temporal Interpolation Review - {args.bag}", display)
        key_code = cv2.waitKeyEx(0)
        
        if key_code in (ord("q"), 27):
            break
        if key_code in (ord("s"),):
            break
        if key_code in (ord("y"), ord("a")):
            accepted.add(frame_key)
            print(f"{frame_key}: accepted")
            current_idx += 1
            continue
        if key_code in (ord("n"), ord("r")):
            rejected.add(frame_key)
            print(f"{frame_key}: rejected")
            current_idx += 1
            continue
        if key_code in (81, 2424832):  # Left arrow
            current_idx = max(0, current_idx - 1)
            continue
        if key_code in (83, 2555904):  # Right arrow
            current_idx = min(len(successful) - 1, current_idx + 1)
            continue
        
        current_idx += 1

    cv2.destroyAllWindows()

    print(f"\n[SUMMARY] Accepted: {len(accepted)}, Rejected: {len(rejected)}, Total reviewed: {len(successful)}")
    
    if accepted:
        print(f"\n[INFO] Applying {len(accepted)} accepted interpolations...")
        for frame_key in accepted:
            frame_data = frames[frame_key]
            frame_data["status"] = "interpolated_accepted"
        
        with open(poses_json, "w") as fh:
            json.dump(preview_payload, fh, indent=2)
            fh.write("\n")
        
        print(f"[SUCCESS] Updated {poses_json} with accepted interpolations")
    else:
        print("[INFO] No interpolations accepted. Original poses.json unchanged.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
