#!/usr/bin/env python3
"""
03_repack_bag.py  —  Run on either machine (no GPU needed)
===========================================================
Takes the original .mcap bag and the generated mask PNGs, then writes a new
.mcap bag that contains the original topics PLUS two new topics:

  /needle_tracking/needle_mask   — sensor_msgs/msg/Image  (mono8, binary)

The new bag can be replayed alongside your NeedleTrackerNode. The node
already subscribes to a needle_mask topic, so no code changes are needed.

USAGE
-----
python 03_repack_bag.py \
    --bag     /path/to/original/suture1 \
    --ann-dir ./annotations \
    --out-dir ./annotated_bags

DEPENDENCIES
------------
  pip install rosbags opencv-python numpy tqdm

NOTE ON FRAME ALIGNMENT
-----------------------
Masks are matched to image messages by index (frame_N → Nth image message).
If you used --every-n > 1 during extraction, frames between saved ones will
receive a copy of the nearest available mask (nearest-neighbour fill).
This is fine for the tracker since the masks are used as soft guidance.

The original bag's image, robot, and TF topics are passed through unchanged.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from rosbags.rosbag2 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.stores.ros2_humble import (
        builtin_interfaces__msg__Time as Time,
        std_msgs__msg__Header as Header,
        sensor_msgs__msg__Image as RosImage,
    )
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


IMAGE_TOPICS = [
    "/ves_camera/image_rect",
    "/ves_camera/image",
]

MASK_TOPIC_MAP = {
    "needle_mask": "/needle_tracking/needle_mask",
}


def ns_to_time(ns: int) -> "Time":
    return Time(sec=ns // 10**9, nanosec=ns % 10**9)


def build_image_msg(mask_np: np.ndarray, timestamp_ns: int, frame_id: str = "camera") -> "RosImage":
    """Wrap a numpy uint8 mask as a sensor_msgs/Image (mono8)."""
    h, w = mask_np.shape[:2]
    header = Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id)
    data = mask_np.flatten().tolist()
    return RosImage(
        header=header,
        height=h,
        width=w,
        encoding="mono8",
        is_bigendian=False,
        step=w,
        data=data,
    )


def load_masks_for_bag(ann_dir: Path, bag_stem: str) -> dict[str, dict[int, np.ndarray]]:
    """
    Load mask PNGs from the masks/ subdirectory.
    Returns {label: {frame_idx: mask_np}}.
    """
    masks_dir = ann_dir / bag_stem / "masks"
    result: dict[str, dict[int, np.ndarray]] = {}

    if not masks_dir.exists():
        return result

    for label in ["needle_mask"]:
        files = sorted(masks_dir.glob(f"frame_*_{label}.png"))
        if not files:
            continue
        frames = {}
        for f in files:
            idx = int(f.stem.split("_")[1])
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames[idx] = img
        if frames:
            result[label] = frames

    return result


def nearest_mask(masks_by_frame: dict[int, np.ndarray], query_idx: int) -> np.ndarray | None:
    """Return the mask for the closest available frame index."""
    if not masks_by_frame:
        return None
    keys = np.array(sorted(masks_by_frame.keys()))
    nearest_key = keys[np.argmin(np.abs(keys - query_idx))]
    return masks_by_frame[nearest_key]


def repack_bag(bag_path: Path, ann_dir: Path, out_dir: Path):
    if not ROSBAGS_AVAILABLE:
        print("[ERROR] rosbags not installed. Run:  pip install rosbags")
        sys.exit(1)

    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent
    bag_stem = bag_dir.name
    out_bag_dir = out_dir / f"{bag_stem}_annotated"

    print(f"\n{'='*60}")
    print(f"Repacking: {bag_stem}")
    print(f"Output:    {out_bag_dir}")
    print(f"{'='*60}")

    # ── Load masks ────────────────────────────────────────────────────────
    masks = load_masks_for_bag(ann_dir, bag_stem)
    if not masks:
        print(f"  [WARN] No masks found for {bag_stem} — bag will be written without mask topics")
    else:
        for label, frames in masks.items():
            print(f"  Loaded {len(frames)} masks for '{label}'")

    # ── Load seeds for every_n info ────────────────────────────────────────
    seeds_path = ann_dir / bag_stem / "seeds.json"
    every_n = 3  # default assumption
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds = json.load(f)
        # Infer every_n from frame count vs message count if possible
        # (We just use nearest-neighbour so exact every_n doesn't matter)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    out_dir.mkdir(parents=True, exist_ok=True)

    if out_bag_dir.exists():
        print(f"  [WARN] Output bag already exists: {out_bag_dir}")
        print(f"         Delete it to repack again.")
        return False

    # ── Open reader and writer ────────────────────────────────────────────
    with Reader(bag_dir) as reader:
        available_topics = {c.topic: c for c in reader.connections}

        # Determine which image topic exists
        img_topic = next((t for t in IMAGE_TOPICS if t in available_topics), None)

        with Writer(out_bag_dir) as writer:
            # Register all original connections
            conn_map = {}  # original id → new connection
            for orig_conn in reader.connections:
                new_conn = writer.add_connection(
                    topic=orig_conn.topic,
                    msgtype=orig_conn.msgtype,
                    serialization_format=orig_conn.serialization_format,
                    offered_qos_profiles=orig_conn.offered_qos_profiles,
                )
                conn_map[orig_conn.id] = new_conn

            # Register new mask topics
            mask_conns = {}
            for label in masks:
                ros_topic = MASK_TOPIC_MAP.get(label)
                if ros_topic:
                    mask_conns[label] = writer.add_connection(
                        topic=ros_topic,
                        msgtype="sensor_msgs/msg/Image",
                    )

            # ── Stream messages ────────────────────────────────────────────
            img_frame_counter = 0
            total = reader.message_count

            iter_ = reader.messages()
            if TQDM_AVAILABLE:
                iter_ = tqdm(iter_, total=total, desc="  Messages", unit="msg")

            for connection, timestamp, rawdata in iter_:
                # Pass through original message
                new_conn = conn_map[connection.id]
                writer.write(new_conn, timestamp, rawdata)

                # When we hit an image message, also write the corresponding masks
                if connection.topic == img_topic and mask_conns:
                    for label, label_masks in masks.items():
                        ros_topic = MASK_TOPIC_MAP.get(label)
                        if ros_topic not in mask_conns:
                            continue

                        # Map image counter → nearest saved frame index
                        # (saved frames are every every_n images)
                        mask_np = nearest_mask(label_masks, img_frame_counter)
                        if mask_np is None:
                            img_frame_counter += 1
                            continue

                        # Resize mask to match if necessary
                        # (we don't know image size here, but masks should match)

                        msg = build_image_msg(mask_np, timestamp)
                        raw = typestore.serialize_cdr(msg, "sensor_msgs/msg/Image")
                        writer.write(mask_conns[label], timestamp, raw)

                    img_frame_counter += 1

    print(f"  ✓ Written to {out_bag_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Repack bags with annotation mask topics")
    parser.add_argument("--bag", nargs="+", required=True,
                        help="Path(s) to original bag directories")
    parser.add_argument("--ann-dir", required=True,
                        help="Annotation directory (output from 01_seed_annotator + 02_propagate)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write annotated bags into")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bag_str in args.bag:
        bag_path = Path(bag_str)
        if not bag_path.exists():
            print(f"[WARN] Bag not found: {bag_path}")
            continue
        repack_bag(bag_path, ann_dir, out_dir)

    print("\nDone. Replay with:")
    print("  ros2 bag play <annotated_bag> --topics /needle_tracking/needle_mask \\")
    print("                                          /ves_camera/image_rect \\")
    print("                                          /ves/right/joint/measured_jp ...")


if __name__ == "__main__":
    main()
