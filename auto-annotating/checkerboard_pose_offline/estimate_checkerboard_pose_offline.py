#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from checkerboard_pose_offline import (
    OfflineCheckerboardEstimator,
    discover_frame_paths,
    frame_index_from_path,
    generate_checkerboard_info,
    load_camera_model,
)


def collect_bags(ann_dir: Path, bag_name: str | None, all_bags: bool) -> list[Path]:
    if bag_name:
        return [ann_dir / bag_name]
    if all_bags:
        return sorted([p for p in ann_dir.iterdir() if p.is_dir() and (p / "frames").exists()])
    return sorted([p for p in ann_dir.iterdir() if p.is_dir() and (p / "frames").exists()])


def parse_steps(text: str) -> list[str]:
    steps = [step.strip() for step in text.split(",") if step.strip()]
    return steps or ["clahe"]


def choose_camera_yaml(ann_dir: Path, bag_dir: Path, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        ann_dir / bag_dir.name / "ves_camera.yaml",
        bag_dir / "ves_camera.yaml",
        ann_dir / "ves_camera.yaml",
        ann_dir.parent / "ves_camera.yaml",
        ann_dir.parent.parent / "ves_camera.yaml",
        Path.cwd() / "ves_camera.yaml",
        Path.cwd().parent / "ves_camera.yaml",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find ves_camera.yaml. Pass --camera-yaml or place it beside the bag annotations.")


def process_bag(args: argparse.Namespace, bag_dir: Path, output_json: Path) -> bool:
    frames_dir = bag_dir / "frames"
    if not frames_dir.exists():
        print(f"[WARN] Frames dir not found: {frames_dir}")
        return False

    frame_paths = discover_frame_paths(frames_dir)
    if not frame_paths:
        print(f"[WARN] No frames found in {frames_dir}")
        return False

    camera_yaml = choose_camera_yaml(Path(args.ann_dir), bag_dir, args.camera_yaml)
    camera = load_camera_model(camera_yaml, rectified_input=args.rectified_input)
    checkerboard = generate_checkerboard_info(args.squares_x, args.squares_y, args.square_size)

    estimator = OfflineCheckerboardEstimator(
        camera=camera,
        checkerboard=checkerboard,
        pixel_noise_sigma=args.pixel_noise_sigma,
        rectified_input=args.rectified_input,
        detector_mode=args.detector_mode,
        preprocess_steps=parse_steps(args.preprocess_steps),
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        blur_kernel_size=args.blur_kernel_size,
        gamma_value=args.gamma_value,
        use_temporal_roi=args.use_temporal_roi,
        temporal_roi_padding_factor=args.temporal_roi_padding_factor,
        detection_timeout_ms=args.detection_timeout_ms,
        save_failed_diagnostics=args.save_failed_diagnostics,
        failed_diagnostics_dir=Path(args.failed_diagnostics_dir) / bag_dir.name,
        detection_cache_file=Path(args.detection_cache_file) if args.detection_cache_file else None,
    )

    print(f"\n{'=' * 60}")
    print(f"Bag: {bag_dir.name}")
    print(f"Frames: {len(frame_paths)}")
    print(f"Camera YAML: {camera_yaml}")
    print(f"Board: {args.squares_x}x{args.squares_y} squares, size={args.square_size}")
    print(f"Detector: {args.detector_mode}, preprocess={parse_steps(args.preprocess_steps)}")
    print(f"{'=' * 60}")

    result = {
        "bag_stem": bag_dir.name,
        "source": "offline_checkerboard_pose_estimation",
        "board_type": "checkerboard",
        "parameters": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_size": args.square_size,
            "rectified_input": args.rectified_input,
            "detector_mode": args.detector_mode,
            "preprocess_steps": parse_steps(args.preprocess_steps),
            "clahe_clip_limit": args.clahe_clip_limit,
            "clahe_tile_grid_size": args.clahe_tile_grid_size,
            "blur_kernel_size": args.blur_kernel_size,
            "gamma_value": args.gamma_value,
            "use_orientation_markers": args.use_orientation_markers,
            "use_temporal_roi": args.use_temporal_roi,
            "temporal_roi_padding_factor": args.temporal_roi_padding_factor,
            "detection_timeout_ms": args.detection_timeout_ms,
            "save_failed_diagnostics": args.save_failed_diagnostics,
            "pixel_noise_sigma": args.pixel_noise_sigma,
        },
        "camera_yaml": str(camera_yaml),
        "frames": {},
    }

    if args.overwrite and output_json.exists():
        output_json.unlink()

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            continue

        frame_idx = frame_index_from_path(frame_path)
        if frame_idx is None:
            frame_idx = len(result["frames"])
        frame_key = f"{frame_idx:06d}"
        frame_result = estimator.process_frame(frame, frame_key)
        result["frames"][frame_key] = frame_result

        if frame_result["status"] == "ok":
            pose = frame_result["pose"]
            print(f"  [OK] {frame_key} detector={frame_result['detector']} rms={frame_result['rms_reprojection_error']:.2f}px")
        else:
            print(f"  [FAIL] {frame_key} {frame_result['failure_reason']}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    print(f"Saved: {output_json}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline checkerboard pose estimation for extracted frames")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", nargs="+", help="Bag stem(s) to process")
    parser.add_argument("--all", action="store_true", help="Process every bag directory under --ann-dir")
    parser.add_argument("--output-json", default=None, help="Override output JSON path (single bag only)")
    parser.add_argument("--camera-yaml", default=None, help="Path to ves_camera.yaml")
    parser.add_argument("--rectified-input", action="store_true", default=True, help="Treat images as rectified/undistorted (default: true)")
    parser.add_argument("--raw-input", dest="rectified_input", action="store_false", help="Treat images as raw distorted frames")
    parser.add_argument("--squares-x", type=int, default=4)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-size", type=float, default=0.002)
    parser.add_argument("--pixel-noise-sigma", type=float, default=1.0)
    parser.add_argument("--detector-mode", choices=["auto", "sb", "legacy", "fast"], default="sb")
    parser.add_argument("--preprocess-steps", default="clahe", help="Comma-separated preprocessing steps: clahe, normalize, blur, denoise, gamma")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--blur-kernel-size", type=int, default=3)
    parser.add_argument("--gamma-value", type=float, default=1.0)
    parser.add_argument("--use-orientation-markers", action="store_true", default=True)
    parser.add_argument("--no-orientation-markers", dest="use_orientation_markers", action="store_false")
    parser.add_argument("--use-temporal-roi", action="store_true", default=False)
    parser.add_argument("--temporal-roi-padding-factor", type=float, default=0.75)
    parser.add_argument("--detection-timeout-ms", type=int, default=10000)
    parser.add_argument("--save-failed-diagnostics", action="store_true", default=False)
    parser.add_argument("--failed-diagnostics-dir", default="/tmp/checkerboard_pose_offline_diagnostics")
    parser.add_argument("--detection-cache-file", default=None, help="Optional JSON cache of successful detections")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        print(f"[ERROR] ann-dir does not exist: {ann_dir}")
        return 1

    bags = collect_bags(ann_dir, args.bag[0] if args.bag and len(args.bag) == 1 else None, args.all)
    if args.bag and len(args.bag) > 1:
        bags = [ann_dir / bag for bag in args.bag]
    if not bags:
        print("[ERROR] No bags found")
        return 1

    if args.output_json and len(bags) != 1:
        print("[ERROR] --output-json is only supported when processing one bag")
        return 1

    ok = True
    for bag_dir in bags:
        output_json = Path(args.output_json) if args.output_json else bag_dir / "poses.json"
        ok = process_bag(args, bag_dir, output_json) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
