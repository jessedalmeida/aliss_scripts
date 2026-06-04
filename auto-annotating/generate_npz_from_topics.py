#!/usr/bin/env python3
"""Generate NPZ files from annotated topic MCAPs compatible with `npz_dataloader`.

Usage:
  python generate_npz_from_topics.py --mcap /path/to/annotated.mcap --out /path/to/out.npz

This reads the split-topic annotated MCAPs produced by `repack_bag.py` and emits
an NPZ containing keys expected by `npz_dataloader.NPZTrackingDataParser`:

  f{idx:06d}_raw_mask      (H, W) uint8
  f{idx:06d}_mask_points   (H, W) uint8 (visible keypoints drawn)
  f{idx:06d}_kp_needle_tip    (2,) float32
  f{idx:06d}_kp_needle_tail   (2,) float32
  f{idx:06d}_kp_left_arm_tip  (2,) float32
  f{idx:06d}_kp_right_arm_tip (2,) float32
  f{idx:06d}_kp_<name>_v      () uint8  -- visibility: 1 visible/usable, 0 not visible
  f{idx:06d}_pose          (4,4) float32  -- only frames with pose are written
  f{idx:06d}_arm_cp        (7,) float32  -- optional, if PoseStamped present

Only frames that have both a mask and a checkerboard pose are written, with a
fallback to mask+keypoints when no pose frames exist. All four keypoints are
exported with a binary visibility flag; when a keypoint is not visible (v=0) its
coordinates are unreliable and should be ignored/masked in the training loss.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2 as _cv2


def _pose_from_pose_with_cov(msg) -> np.ndarray:
    # msg.pose.pose.position, msg.pose.pose.orientation
    T = np.eye(4, dtype=np.float32)
    px = float(msg.pose.pose.position.x)
    py = float(msg.pose.pose.position.y)
    pz = float(msg.pose.pose.position.z)
    qx = float(msg.pose.pose.orientation.x)
    qy = float(msg.pose.pose.orientation.y)
    qz = float(msg.pose.pose.orientation.z)
    qw = float(msg.pose.pose.orientation.w)

    # Build rotation matrix from quaternion
    # quaternion -> rotation matrix (x, y, z, w)
    x, y, z, w = qx, qy, qz, qw
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

    T[:3, :3] = R
    T[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return T


def _pose_from_pose_stamped(msg) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    px = float(msg.pose.position.x)
    py = float(msg.pose.position.y)
    pz = float(msg.pose.position.z)
    qx = float(msg.pose.orientation.x)
    qy = float(msg.pose.orientation.y)
    qz = float(msg.pose.orientation.z)
    qw = float(msg.pose.orientation.w)

    x, y, z, w = qx, qy, qz, qw
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return T


def generate_npz_from_mcap(mcap_path: Path, out_path: Path) -> None:
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import get_typestore
        from rosbags.typesys import Stores
    except Exception as e:
        raise RuntimeError("rosbags (rosbags) is required to read MCAP files. Install with `pip install rosbags`.") from e

    try:
        from repack_bag import (
            MASK_TOPIC,
            KEYPOINTS_TOPIC,
            CHECKERBOARD_POSE_TOPIC,
            RIGHT_TIP_POSE_TOPIC,
            LEFT_TIP_POSE_TOPIC,
            register_custom_types,
        )
    except Exception:
        # Fallbacks in case module isn't importable; these are the defaults used by repack_bag
        MASK_TOPIC = "/needle_tracking/mask"
        KEYPOINTS_TOPIC = "/needle_tracking/keypoints"
        CHECKERBOARD_POSE_TOPIC = "/checkerboard/pose"
        RIGHT_TIP_POSE_TOPIC = "/needle_tracking/right_tip_pose"
        LEFT_TIP_POSE_TOPIC = "/needle_tracking/left_tip_pose"

        def register_custom_types(typestore):
            return

    frames = defaultdict(dict)

    with Reader(str(mcap_path)) as reader:
        connections = list(reader.connections)
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        register_custom_types(typestore)

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            try:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            except Exception:
                # skip messages we cannot deserialize
                continue

            # group by timestamp (ns)
            ts = int(timestamp)

            if topic == MASK_TOPIC:
                # sensor_msgs/Image: height, width, data (bytes)
                try:
                    h = int(msg.height)
                    w = int(msg.width)
                    data = bytes(msg.data)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    if arr.size == h * w:
                        mask = arr.reshape((h, w)).copy()
                    else:
                        # attempt reshape with step
                        mask = arr[: h * w].reshape((h, w)).copy()
                    frames[ts]["mask"] = mask
                except Exception:
                    continue

            elif topic == KEYPOINTS_TOPIC:
                # custom NeedleTrackingKeypoints: 4 keypoints + binary visibility
                # (1 = visible/usable, 0 = not visible/coords unreliable)
                try:
                    def _xy(pt):
                        return np.array([float(pt.x), float(pt.y)], dtype=np.float32)
                    def _vis(field):
                        return int(getattr(msg, field, 0) or 0)
                    frames[ts]["kp"] = {
                        "needle_tip":    _xy(msg.needle_tip),
                        "needle_tail":   _xy(msg.needle_tail),
                        "left_arm_tip":  _xy(msg.left_arm_tip),
                        "right_arm_tip": _xy(msg.right_arm_tip),
                    }
                    frames[ts]["vis"] = {
                        "needle_tip":    _vis("needle_tip_visibility"),
                        "needle_tail":   _vis("needle_tail_visibility"),
                        "left_arm_tip":  _vis("left_arm_tip_visibility"),
                        "right_arm_tip": _vis("right_arm_tip_visibility"),
                    }
                except Exception:
                    continue

            elif topic == CHECKERBOARD_POSE_TOPIC:
                try:
                    T = _pose_from_pose_with_cov(msg)
                    frames[ts]["pose"] = T
                except Exception:
                    continue

            elif topic == RIGHT_TIP_POSE_TOPIC or topic == LEFT_TIP_POSE_TOPIC:
                try:
                    # geometry_msgs/PoseWithCovarianceStamped (use same extractor as checkerboard)
                    # extract pose.pose.pose
                    T = _pose_from_pose_with_cov(msg)
                    px = float(msg.pose.pose.position.x)
                    py = float(msg.pose.pose.position.y)
                    pz = float(msg.pose.pose.position.z)
                    qx = float(msg.pose.pose.orientation.x)
                    qy = float(msg.pose.pose.orientation.y)
                    qz = float(msg.pose.pose.orientation.z)
                    qw = float(msg.pose.pose.orientation.w)
                    frames[ts].setdefault("tip_poses", []).append(np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float32))
                except Exception:
                    continue

    # Now assemble NPZ entries in timestamp order where both mask and pose are present
    valid_ts = [ts for ts, d in sorted(frames.items()) if "mask" in d and "pose" in d and "kp" in d]
    if not valid_ts:
        print(f"Warning: no frames with all valid annotations, trying without checkerboard pose...")
        valid_ts = [ts for ts, d in sorted(frames.items()) if "mask" in d and "kp" in d]

        if not valid_ts:
            raise RuntimeError(f"No valid annotated frames (mask+pose+keypoints) found in {mcap_path}")

    data_dict = {}
    for idx, ts in enumerate(valid_ts):
        prefix = f"f{idx:06d}_"
        d = frames[ts]

        mask = d.get("mask")
        kp = d.get("kp")
        pose = d.get("pose")

        # raw mask
        data_dict[f"{prefix}raw_mask"] = np.array(mask, dtype=np.uint8)

        # mask_points: draw tiny circles only at VISIBLE keypoints (vis==1)
        mask_points = np.zeros_like(mask, dtype=np.uint8)
        vis = d.get("vis") or {}
        try:
            if kp is not None:
                for name in ("needle_tip", "needle_tail", "left_arm_tip", "right_arm_tip"):
                    pt = kp.get(name)
                    if pt is not None and vis.get(name, 0) == 1 and np.isfinite(pt).all():
                        _cv2.circle(mask_points, (int(pt[0]), int(pt[1])), 3, 255, -1)
        except Exception:
            pass
        data_dict[f"{prefix}mask_points"] = mask_points

        # keypoints: all 4, each (x, y) plus a binary visibility flag
        # (1 = visible/usable, 0 = not visible -> coords unreliable, ignore in loss)
        if kp is not None:
            for name in ("needle_tip", "needle_tail", "left_arm_tip", "right_arm_tip"):
                pt = kp.get(name)
                if pt is not None:
                    data_dict[f"{prefix}kp_{name}"] = np.array(pt, dtype=np.float32)
                data_dict[f"{prefix}kp_{name}_v"] = np.array(vis.get(name, 0), dtype=np.uint8)

        # pose
        if pose is not None:
            data_dict[f"{prefix}pose"] = np.array(pose, dtype=np.float32)

        # arm control point: prefer right/left tip pose if present -> store as arm_cp
        tip_poses = d.get("tip_poses")
        if tip_poses:
            data_dict[f"{prefix}arm_cp"] = np.array(tip_poses[0], dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **data_dict)
    print(f"Saved NPZ: {out_path} with {len(valid_ts)} frames")


def main():
    parser = argparse.ArgumentParser(description="Generate .npz from annotated MCAP topics or a directory of MCAPs")
    parser.add_argument("--input", required=True, help="Path to an .mcap file or a directory containing .mcap files (recursively searched)")
    parser.add_argument("--out-dir", required=False, help="Output directory to write .npz files. If omitted, writes next to each .mcap")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if input_path.is_file():
        # Single file
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (input_path.stem + ".npz")
        else:
            out_path = input_path.with_suffix(".npz")
        generate_npz_from_mcap(input_path, out_path)
    elif input_path.is_dir():
        # Recursive search for .mcap files
        mcap_files = list(input_path.rglob("*.mcap"))
        if not mcap_files:
            print(f"No .mcap files found under {input_path}")
            return

        for mcap in sorted(mcap_files):
            if out_dir:
                out_path = out_dir / (mcap.stem + ".npz")
            else:
                out_path = mcap.with_suffix(".npz")
            try:
                print(f"Processing: {mcap} -> {out_path}")
                generate_npz_from_mcap(mcap, out_path)
            except Exception as e:
                print(f"Failed to process {mcap}: {e}")
    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
