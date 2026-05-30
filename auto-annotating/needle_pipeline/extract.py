#!/usr/bin/env python3
"""
needle_pipeline.extract - headless frame extraction from a ROS2 .mcap bag.

This decouples extraction from seeding (the old seed_annotator.py fused them,
and its ros_image_to_cv2 had a displaced `return img` that made the success
path return None). Run it standalone or via the orchestrator:

    python -m needle_pipeline.extract --bag /path/to/bag --out ./annotations
    python -m needle_pipeline.extract --bag /path/to/bag --out ./annotations --every-n 5

Writes <out>/<bag_stem>/frames/000000.jpg, 000001.jpg, ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_TOPICS = ["/ves_camera/image_rect", "/ves_camera/image"]


def ros_image_to_cv2(msg) -> "np.ndarray | None":
    """Convert a sensor_msgs/Image to a BGR uint8 array. Returns None on failure."""
    encoding = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    try:
        if encoding in ("bgr8", "rgb8", "bgra8", "rgba8"):
            channels = 4 if "a" in encoding else 3
            img = data.reshape((h, w, channels))
            if encoding.startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif encoding == "bgra8":
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif encoding == "mono8":
            img = cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
        elif encoding == "16uc1":
            img16 = data.view(np.uint16).reshape((h, w))
            img = cv2.cvtColor((img16 / 256).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img = data.reshape((h, msg.step))[:, : w * 3].reshape((h, w, 3))
        return img            # <-- the fix: actually return on the success path
    except Exception:
        return None


def extract_frames(bag_path: Path, out_dir: Path, every_n: int = 3,
                   progress=None) -> list[Path]:
    """Save every Nth image message as a JPG. `progress` is an optional
    callback (saved_count, total_seen) for the GUI's progress bar."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print("[ERROR] rosbags not installed. Run: pip install rosbags", file=sys.stderr)
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent
    saved: list[Path] = []
    seen = 0
    kept = 0

    with AnyReader([bag_dir]) as reader:
        available = set(reader.topics)
        topic = next((t for t in IMAGE_TOPICS if t in available), None)
        if topic is None:
            print(f"[WARN] no image topic in {bag_dir}; available: {sorted(available)}")
            return []
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _ts, raw in reader.messages(connections=conns):
            if seen % every_n == 0:
                msg = reader.deserialize(raw, conn.msgtype)
                img = ros_image_to_cv2(msg)
                if img is not None:
                    fname = out_dir / f"{kept:06d}.jpg"
                    cv2.imwrite(str(fname), img)
                    saved.append(fname)
                    kept += 1
                    if progress and kept % 25 == 0:
                        progress(kept, seen)
            seen += 1

    print(f"  extracted {len(saved)} frames (every {every_n} of {seen}) -> {out_dir}")
    return sorted(saved)


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless frame extraction from a ROS2 bag")
    ap.add_argument("--bag", required=True, help="Original .mcap bag directory")
    ap.add_argument("--out", required=True, help="Annotation root directory")
    ap.add_argument("--every-n", type=int, default=3, help="Keep every Nth frame")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"[ERROR] bag not found: {bag_path}", file=sys.stderr)
        return 1
    stem = bag_path.stem if bag_path.is_file() else bag_path.name
    frames = extract_frames(bag_path, Path(args.out) / stem / "frames", args.every_n)
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
