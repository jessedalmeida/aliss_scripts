#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from correct_poses import open_corner_picker
from checkerboard_pose_offline import (
    CameraModel,
    CheckerboardInfo,
    generate_checkerboard_info,
    load_camera_model,
)
from flow_seed import try_optical_flow_seed
import copy


@dataclass
class ReviewDecision:
    status: str
    note: str = ""


def load_poses_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_review_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    reviews = payload.get("reviews", {})
    return {str(key): value for key, value in reviews.items()}


def save_review_json(path: Path, bag_stem: str, review_map: dict[str, dict[str, Any]]) -> None:
    payload = {
        "bag_stem": bag_stem,
        "source": "failed_frame_review",
        "reviews": review_map,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def select_review_frame(bag_dir: Path, frame_key: str, frame_data: dict, show_diagnostics: bool, diagnostics_grid: bool = False) -> tuple[np.ndarray | None, str]:
    # If diagnostics_grid is explicitly requested and a diagnostics image exists, show it.
    diag_path = frame_data.get("diagnostics_image")
    if diagnostics_grid and show_diagnostics and diag_path:
        img = cv2.imread(str(diag_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img, f"diagnostics: {diag_path}"
    frames_dir = bag_dir / "frames"
    frame_paths = collect_frames(frames_dir)

    # Prefer matching any existing frame file (jpg/png) by stem containing the key
    for candidate in frame_paths:
        # candidate.stem may be '000007' or 'frame_000007'
        if candidate.stem.endswith(frame_key) or frame_key in candidate.stem:
            img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if img is not None:
                return img, f"frame: {candidate.name}"

    # Fallback: try common filename patterns
    try:
        idx = int(frame_key)
    except (ValueError, TypeError):
        return None, "no image available"

    candidates = [
        frames_dir / f"frame_{idx:06d}.jpg",
        frames_dir / f"{idx:06d}.jpg",
        frames_dir / f"frame_{idx:06d}.jpeg",
        frames_dir / f"{idx:06d}.jpeg",
        frames_dir / f"frame_{idx:06d}.png",
        frames_dir / f"{idx:06d}.png",
    ]
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                return img, f"frame: {p.name}"

    return None, "no image available"


def overlay_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    display = image.copy()
    h, w = display.shape[:2]
    panel_h = min(120, max(80, 24 * len(lines) + 20))
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (10, h - panel_h + 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return display


def show_pose_overlay_for_frame(
    bag_dir: Path,
    ann_dir: Path,
    bag_name: str,
    frame_key: str,
    poses_payload: dict,
    camera_yaml_path: Path,
) -> None:
    frames_dir = bag_dir / "frames"
    frame_paths = collect_frames(frames_dir)
    if not frame_paths:
        return

    frame_path = None
    for candidate in frame_paths:
        if candidate.stem.endswith(frame_key) or frame_key in candidate.stem:
            frame_path = candidate
            break
    if frame_path is None:
        try:
            frame_idx = int(frame_key)
        except ValueError:
            return
        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        if not frame_path.exists():
            frame_path = frames_dir / f"{frame_idx:06d}.png"
        if not frame_path.exists():
            return

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return

    camera_data = load_camera_yaml(camera_yaml_path)
    camera_matrix = np.asarray(camera_data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)[:, :3]
    distortion = np.zeros((5, 1), dtype=np.float64)
    frame_entries = poses_payload.get("frames", {})
    pose_data = frame_entries.get(frame_key, {})
    params = poses_payload.get("parameters", {})
    display = draw_pose_overlay(
        image,
        camera_matrix,
        distortion,
        pose_data,
        squares_x=int(params.get("squares_x", 4)),
        squares_y=int(params.get("squares_y", 5)),
        square_size=float(params.get("square_size", 0.002)),
        draw_corners=True,
    )
    display = overlay_text(
        display,
        [
            f"Pose overlay for {bag_name} frame {frame_key}",
            "Press any key to return to failed-frame review",
        ],
    )
    window_name = f"Pose Overlay - {bag_name} - {frame_key}"
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# Temporal interpolation removed — optical-flow seeding is the default preview path.


def handle_manual_correction(
    bag_dir: Path,
    frame_key: str,
    poses_json_path: Path,
    poses_payload: dict,
    camera_yaml_path: Path,
) -> bool:
    """
    Open corner picker for manual pose correction.
    Returns True if pose was corrected and saved, False otherwise.
    """
    frames_dir = bag_dir / "frames"
    frame_paths = collect_frames(frames_dir)
    if not frame_paths:
        print(f"[WARN] No frames found in {frames_dir}")
        return False

    frame_path = None
    for candidate in frame_paths:
        if candidate.stem.endswith(frame_key) or frame_key in candidate.stem:
            frame_path = candidate
            break
    if frame_path is None:
        try:
            frame_idx = int(frame_key)
        except ValueError:
            return False
        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        if not frame_path.exists():
            frame_path = frames_dir / f"{frame_idx:06d}.png"
    
    if not frame_path.exists():
        print(f"[WARN] Frame image not found: {frame_path}")
        return False

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] Could not read image: {frame_path}")
        return False

    try:
        camera = load_camera_model(camera_yaml_path, rectified_input=True)
    except Exception as e:
        print(f"[ERROR] Failed to load camera model: {e}")
        return False

    params = poses_payload.get("parameters", {})
    squares_x = int(params.get("squares_x", 4))
    squares_y = int(params.get("squares_y", 5))
    square_size = float(params.get("square_size", 0.002))
    
    try:
        checkerboard = generate_checkerboard_info(squares_x, squares_y, square_size)
    except Exception as e:
        print(f"[ERROR] Failed to generate checkerboard info: {e}")
        return False

    print("\n[INFO] Opening corner picker. Click checkerboard corners on the image.")
    result = open_corner_picker(image, camera, checkerboard, pixel_noise_sigma=1.0)
    
    if result is None:
        print("[INFO] Manual correction cancelled")
        return False

    try:
        frame_entries = poses_payload.get("frames", {})
        frame_data = frame_entries.get(frame_key, {})
        frame_data["status"] = "ok"
        frame_data["pose"] = result.get("pose")
        frame_data["detector"] = "manual_correction"
        frame_data["rms_reprojection_error"] = result.get("rms_reprojection_error", 0.0)
        frame_data["failure_reason"] = ""
        
        with open(poses_json_path, "w", encoding="utf-8") as fh:
            json.dump(poses_payload, fh, indent=2)
            fh.write("\n")
        
        print(f"[SUCCESS] Corrected pose saved to poses.json for frame {frame_key}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save corrected pose: {e}")
        return False



def main() -> int:
    parser = argparse.ArgumentParser(description="Review failed checkerboard pose frames")
    parser.add_argument("--ann-dir", required=True)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--poses-json", default=None)
    parser.add_argument("--review-json", default=None, help="Path to save review decisions (default: <bag>/pose_reviews.json)")
    parser.add_argument("--show-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-show-diagnostics", dest="show_diagnostics", action="store_false")
    parser.add_argument("--diagnostics-grid", action="store_true", default=False, help="Show 2x2 diagnostics grid when available")
    parser.add_argument("--read-only", action="store_true", help="Do not save any review decisions")
    args = parser.parse_args()

    bag_dir = Path(args.ann_dir) / args.bag
    poses_json = Path(args.poses_json) if args.poses_json else bag_dir / "poses.json"
    if not poses_json.exists():
        print(f"[ERROR] poses.json not found: {poses_json}")
        return 1

    payload = load_pose_json(poses_json)

    failed = [(key, value) for key, value in payload.get("frames", {}).items() if value.get("status") != "ok"]
    if not failed:
        print("No failed frames found")
        return 0

    failed = sorted(failed, key=lambda item: int(item[0]))
    review_path = Path(args.review_json) if args.review_json else bag_dir / "pose_reviews.json"
    review_map = load_review_json(review_path)
    camera_yaml_path = choose_camera_yaml(Path(args.ann_dir), bag_dir, None)

    print(f"Found {len(failed)} failed frame(s)")
    print("Keys: Y=accept  N=reject  M=manual edit  Left/Right=prev/next  S=save  Q=quit")

    current_idx = 0
    while 0 <= current_idx < len(failed):
        key, value = failed[current_idx]
        frame, source_desc = select_review_frame(bag_dir, key, value, args.show_diagnostics, diagnostics_grid=args.diagnostics_grid)
        if frame is None:
            print(f"[WARN] Could not load image for frame {key}")
            current_idx += 1
            continue

        # load camera for overlay drawing
        camera_data = load_camera_yaml(camera_yaml_path)
        camera_matrix = np.asarray(camera_data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)[:, :3]
        distortion = np.zeros((5, 1), dtype=np.float64)

        # Attempt optical-flow seeding automatically (preview only)
        seeded_preview = None
        seeded_info = None
        if not args.read_only:
            try:
                seeded_info = try_optical_flow_seed(bag_dir, key, payload, camera_yaml_path)
            except Exception:
                seeded_info = None

        if seeded_info:
            preview_frame_data = copy.deepcopy(value)
            preview_frame_data["pose"] = seeded_info.get("pose")
            preview_frame_data["detector"] = seeded_info.get("detector", "optical_flow_seed_preview")
            preview_frame_data["rms_reprojection_error"] = seeded_info.get("rms_reprojection_error", 0.0)
            preview_frame_data["status"] = "preview"
            display_img = draw_pose_overlay(
                frame,
                camera_matrix,
                distortion,
                preview_frame_data,
                squares_x=int(payload.get("parameters", {}).get("squares_x", 4)),
                squares_y=int(payload.get("parameters", {}).get("squares_y", 5)),
                square_size=float(payload.get("parameters", {}).get("square_size", 0.002)),
                draw_corners=True,
            )
            status_line = f"Frame {key} | auto-seeded preview (optical-flow) | rms={preview_frame_data.get('rms_reprojection_error'):.2f}"
            review_line = f"Review: {review_map.get(key, {}).get('status','unreviewed')} | source: {source_desc}"
            display = overlay_text(display_img, [status_line, review_line, "Y accept seed | N reject | F re-run seed | M manual edit | S save | Q quit"])
        else:
            status_line = f"Frame {key} | detector={value.get('detector','unknown')} | failure={value.get('failure_reason','unknown')}"
            review_line = f"Review: {review_map.get(key, {}).get('status','unreviewed')} | source: {source_desc}"
            display = overlay_text(frame, [status_line, review_line, "F try flow | M manual edit | S save | Q quit"]) 

        cv2.imshow(f"Failed Frame Review - {args.bag}", display)
        key_code = cv2.waitKeyEx(0)

        if key_code in (ord("q"), 27):
            break
        if key_code in (ord("s"),):
            if not args.read_only:
                save_review_json(review_path, args.bag, review_map)
                print(f"Saved review decisions -> {review_path}")
            continue
        if key_code in (ord("y"), ord("a")):
            # If an auto-seed preview exists, accept and save it; otherwise mark accepted
            if seeded_info and not args.read_only:
                frame_entries = payload.get("frames", {})
                frame_data = frame_entries.get(key, {})
                frame_data["status"] = "ok"
                frame_data["pose"] = seeded_info.get("pose")
                # save tracked corners if available so overlays and downstream tools can use them
                if seeded_info.get("corners"):
                    frame_data["corners"] = seeded_info.get("corners")
                if seeded_info.get("corner_indices"):
                    frame_data["corner_indices"] = seeded_info.get("corner_indices")
                frame_data["detector"] = seeded_info.get("detector", "optical_flow_seed")
                frame_data["rms_reprojection_error"] = seeded_info.get("rms_reprojection_error", 0.0)
                frame_data["failure_reason"] = ""
                with open(poses_json, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                    fh.write("\n")
                review_map[key] = {"status": "optical_flow_applied", "note": f"seeded from neighbor {seeded_info.get('used_neighbor')} ({seeded_info.get('num_tracked')} pts)"}
                print(f"{key}: optical-flow seed saved")
            else:
                review_map[key] = {"status": "accepted", "note": ""}
                print(f"{key}: accepted (no seed to save)")
            current_idx += 1
            continue
        if key_code in (ord("n"), ord("r")):
            review_map[key] = {"status": "rejected", "note": ""}
            print(f"{key}: rejected")
            current_idx += 1
            continue
        if key_code in (ord("f"),):
            print(f"{key}: trying optical-flow seeding...")
            if not args.read_only:
                seeded = try_optical_flow_seed(bag_dir, key, payload, camera_yaml_path)
                if seeded:
                    frame_entries = payload.get("frames", {})
                    frame_data = frame_entries.get(key, {})
                    frame_data["status"] = "ok"
                    frame_data["pose"] = seeded.get("pose")
                    if seeded.get("corners"):
                        frame_data["corners"] = seeded.get("corners")
                    if seeded.get("corner_indices"):
                        frame_data["corner_indices"] = seeded.get("corner_indices")
                    frame_data["detector"] = seeded.get("detector", "optical_flow_seed")
                    frame_data["rms_reprojection_error"] = seeded.get("rms_reprojection_error", 0.0)
                    with open(poses_json, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2)
                        fh.write("\n")
                    review_map[key] = {"status": "optical_flow_applied", "note": f"seeded from neighbor {seeded.get('used_neighbor')} ({seeded.get('num_tracked')} pts)"}
                    print(f"{key}: optical-flow seed saved")
                    # do not advance — redisplay this frame so you can inspect the applied seed
                    continue
                else:
                    print(f"{key}: optical-flow seeding failed")
            continue
        if key_code in (ord("m"),):
            review_map[key] = {"status": "manual_edit", "note": "needs manual pose correction"}
            print(f"{key}: marked for manual_edit")
            show_pose_overlay_for_frame(bag_dir, Path(args.ann_dir), args.bag, key, payload, camera_yaml_path)
            if not args.read_only:
                corrected = handle_manual_correction(bag_dir, key, poses_json, payload, camera_yaml_path)
                if corrected:
                    review_map[key] = {"status": "manual_correction_applied", "note": "manually corrected via corner picking"}
                    print(f"{key}: manual correction applied")
            current_idx += 1
            continue
        if key_code in (81, 2424832):
            current_idx = max(0, current_idx - 1)
            continue
        if key_code in (83, 2555904):
            current_idx = min(len(failed) - 1, current_idx + 1)
            continue

        current_idx += 1

    cv2.destroyAllWindows()
    if not args.read_only:
        save_review_json(review_path, args.bag, review_map)
        print(f"Saved review decisions -> {review_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
