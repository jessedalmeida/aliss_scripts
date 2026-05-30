#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def collect_frames(frames_dir: Path) -> list[Path]:
    frame_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(sorted(frames_dir.glob(pattern)))
    return sorted(frame_paths)


def frame_index_from_path(path: Path) -> str:
    stem = path.stem
    for part in stem.split("_"):
        if part.isdigit():
            return part
    if stem.isdigit():
        return stem
    return stem


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_camera_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
    raise FileNotFoundError("Could not find ves_camera.yaml")


def pose_to_rvec_tvec(pose_data: dict) -> tuple[np.ndarray, np.ndarray]:
    position = pose_data.get("position", [0.0, 0.0, 0.0])
    quaternion = pose_data.get("quaternion", [0.0, 0.0, 0.0, 1.0])
    qx, qy, qz, qw = [float(v) for v in quaternion]

    # Quaternion to rotation matrix
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    rot = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.asarray(position, dtype=np.float64).reshape(3, 1)
    return rvec, tvec


def rotation_matrix_to_euler_deg(rot: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def print_pose_summary(frame_key: str, pose_data: dict) -> None:
    pose = pose_data.get("pose") if isinstance(pose_data, dict) else None
    if not pose:
        print(f"frame {frame_key}: pose unavailable")
        return

    position = [float(v) for v in pose.get("position", [0.0, 0.0, 0.0])]
    rvec, _ = pose_to_rvec_tvec(pose)
    rot, _ = cv2.Rodrigues(rvec)
    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_euler_deg(rot)

    cov_list = pose.get("covariance", [])
    std_pos_mm = [float("nan")] * 3
    std_rot_deg = [float("nan")] * 3
    if isinstance(cov_list, list) and len(cov_list) == 36:
        cov = np.asarray(cov_list, dtype=np.float64).reshape(6, 6)
        diag = np.diag(cov)
        std_pos_m = np.sqrt(np.maximum(diag[:3], 0.0))
        std_rot_rad = np.sqrt(np.maximum(diag[3:], 0.0))
        std_pos_mm = [float(v * 1e3) for v in std_pos_m]
        std_rot_deg = [float(math.degrees(v)) for v in std_rot_rad]

    print(
        f"frame {frame_key}: "
        f"mean pos [mm]=({position[0] * 1e3:.3f}, {position[1] * 1e3:.3f}, {position[2] * 1e3:.3f}) "
        f"mean rpy [deg]=({roll_deg:.3f}, {pitch_deg:.3f}, {yaw_deg:.3f}) "
        f"std pos [mm]=({std_pos_mm[0]:.3f}, {std_pos_mm[1]:.3f}, {std_pos_mm[2]:.3f}) "
        f"std rot [deg]=({std_rot_deg[0]:.3f}, {std_rot_deg[1]:.3f}, {std_rot_deg[2]:.3f})"
    )


def board_points_from_params(squares_x: int, squares_y: int, square_size: float) -> np.ndarray:
    pattern_x = squares_x - 1
    pattern_y = squares_y - 1
    center_x = 0.5 * (pattern_x - 1) * square_size
    center_y = 0.5 * (pattern_y - 1) * square_size
    points = []
    for y in range(pattern_y):
        for x in range(pattern_x):
            points.append((x * square_size - center_x, y * square_size - center_y, 0.0))
    return np.asarray(points, dtype=np.float64)


def draw_pose_overlay(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    pose_data: dict,
    squares_x: int,
    squares_y: int,
    square_size: float,
    draw_corners: bool = True,
    draw_detected_corners: bool = True,
) -> np.ndarray:
    display = image.copy()
    if not pose_data or not pose_data.get("pose"):
        cv2.putText(display, "POSE: unavailable", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        reason = pose_data.get("failure_reason", "") if pose_data else ""
        if reason:
            cv2.putText(display, reason, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return display

    pose = pose_data["pose"]
    rvec, tvec = pose_to_rvec_tvec(pose)
    axis_length = 0.5 * min(squares_x - 1, squares_y - 1) * square_size
    cv2.drawFrameAxes(display, camera_matrix, distortion, rvec, tvec, axis_length)

    if draw_corners:
        object_points = board_points_from_params(squares_x, squares_y, square_size)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        projected = projected.reshape(-1, 2)
        for idx, (x, y) in enumerate(projected):
            x_int = int(round(x))
            y_int = int(round(y))
            cv2.circle(display, (x_int, y_int), 2, (0, 255, 255), -1)
            cv2.putText(display, str(idx + 1), (x_int + 5, y_int - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if draw_detected_corners:
        detected_corners = pose_data.get("corners", [])
        if detected_corners:
            for idx, corner in enumerate(detected_corners):
                if not corner or len(corner) < 2:
                    continue
                x_int = int(round(float(corner[0])))
                y_int = int(round(float(corner[1])))
                cv2.circle(display, (x_int, y_int), 3, (0, 128, 255), 1)
                cv2.putText(display, str(idx + 1), (x_int + 5, y_int + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)

    rms = pose_data.get("rms_reprojection_error")
    if rms is not None:
        cv2.putText(display, f"RMS: {float(rms):.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if pose_data.get("status") and pose_data.get("status") != "ok":
        cv2.putText(display, f"STATUS: {pose_data.get('status')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return display


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize saved offline checkerboard poses")
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", required=True, help="Bag stem to visualize")
    parser.add_argument("--poses-json", default=None, help="Path to poses.json")
    parser.add_argument("--camera-yaml", default=None, help="Path to ves_camera.yaml")
    parser.add_argument("--step", type=int, default=1, help="Frame step while browsing")
    parser.add_argument("--no-corners", action="store_true", help="Do not draw projected checkerboard corners")
    parser.add_argument("--no-detected-corners", action="store_true", help="Do not draw the detected corner locations from poses.json")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    bag_dir = ann_dir / args.bag
    frames_dir = bag_dir / "frames"
    poses_json = Path(args.poses_json) if args.poses_json else bag_dir / "poses.json"
    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return 1
    if not poses_json.exists():
        print(f"[ERROR] poses.json not found: {poses_json}")
        return 1

    camera_yaml_path = choose_camera_yaml(ann_dir, bag_dir, args.camera_yaml)
    camera_data = load_camera_yaml(camera_yaml_path)
    camera_matrix = np.asarray(camera_data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)[:, :3]
    # Match the estimator: frames are treated as rectified, so distortion is zeroed.
    distortion = np.zeros((5, 1), dtype=np.float64)

    payload = load_json(poses_json)
    frame_entries = payload.get("frames", {})
    frame_paths = collect_frames(frames_dir)
    if not frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}")
        return 1

    idx = 0
    while True:
        frame_path = frame_paths[idx]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            idx = (idx + max(1, args.step)) % len(frame_paths)
            continue

        frame_key = frame_index_from_path(frame_path)
        pose_data = frame_entries.get(frame_key, {})
        print_pose_summary(frame_key, pose_data)
        display = draw_pose_overlay(
            frame,
            camera_matrix,
            distortion,
            pose_data,
            squares_x=int(payload.get("parameters", {}).get("squares_x", 4)),
            squares_y=int(payload.get("parameters", {}).get("squares_y", 5)),
            square_size=float(payload.get("parameters", {}).get("square_size", 0.002)),
            draw_corners=not args.no_corners,
            draw_detected_corners=not args.no_detected_corners,
        )

        status = pose_data.get("status", "missing")
        cv2.putText(display, f"{args.bag}  frame {frame_key}  status={status}", (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(f"Pose Viewer - {args.bag}", display)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("d"), 83, 2555904, ord(" ")):
            idx = min(len(frame_paths) - 1, idx + max(1, args.step))
        elif key in (ord("a"), 81, 2424832):
            idx = max(0, idx - max(1, args.step))
        else:
            idx = min(len(frame_paths) - 1, idx + max(1, args.step))

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
