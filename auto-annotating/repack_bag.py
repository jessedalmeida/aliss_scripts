#!/usr/bin/env python3
"""
03_repack_bag.py  -  Run on either machine (no GPU needed)
===========================================================
Takes the original .mcap bag and the generated annotation files, then writes a
new bag that keeps the camera stream needed for playback plus a new
`/needle_tracking/snapshot` topic.

Use `--output-mode topics` to keep the camera stream and publish each
annotation component on its own topic for visualization tools like Foxglove.

If a `poses.json` file is present alongside the other annotations, the script
also publishes `/checkerboard/pose` using the saved checkerboard pose and
covariance for each annotated frame.

Each snapshot is written at the original image timestamp for the extracted
frame, and bundles:
  - the camera image
  - the needle mask
  - the nearest-earlier arm joint/pose messages
  - the annotation keypoints

This is intended so a single replayed bag can drive the needle tracker or
suturing package without requiring the annotation files at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np

try:
    from rosbags.rosbag2 import Reader, StoragePlugin, Writer
    from rosbags.interfaces import ConnectionExtRosbag2
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    from rosbags.typesys.stores.ros2_humble import (
        builtin_interfaces__msg__Time as Time,
        geometry_msgs__msg__Point as Point,
        geometry_msgs__msg__Point32 as Point32,
        geometry_msgs__msg__Pose as Pose,
        geometry_msgs__msg__Pose2D as Pose2D,
        geometry_msgs__msg__PoseStamped as PoseStamped,
        geometry_msgs__msg__PoseWithCovariance as PoseWithCovariance,
        geometry_msgs__msg__PoseWithCovarianceStamped as PoseWithCovarianceStamped,
        geometry_msgs__msg__Quaternion as Quaternion,
        sensor_msgs__msg__Image as RosImage,
        sensor_msgs__msg__CameraInfo as CameraInfo,
        sensor_msgs__msg__RegionOfInterest as RegionOfInterest,
        sensor_msgs__msg__JointState as JointState,
        std_msgs__msg__Header as Header,
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

CAMERA_INFO_TYPES = {"sensor_msgs/msg/CameraInfo"}
SNAPSHOT_TOPIC = "/needle_tracking/snapshot"
KEYPOINTS_TOPIC = "/needle_tracking/keypoints"
MASK_TOPIC = "/needle_tracking/mask"
RIGHT_ARM_JP_TOPIC = "/needle_tracking/right_arm_jp"
LEFT_ARM_JP_TOPIC = "/needle_tracking/left_arm_jp"
RIGHT_ARM_CP_TOPIC = "/needle_tracking/right_arm_cp"
LEFT_ARM_CP_TOPIC = "/needle_tracking/left_arm_cp"
RIGHT_TIP_POSE_TOPIC = "/needle_tracking/right_tip_pose"
LEFT_TIP_POSE_TOPIC = "/needle_tracking/left_tip_pose"
CHECKERBOARD_POSE_TOPIC = "/checkerboard/pose"
PACKAGE_NAME = "aliss_ros_msg"
KEYPOINTS_TYPE = f"{PACKAGE_NAME}/msg/NeedleTrackingKeypoints"
SNAPSHOT_TYPE = f"{PACKAGE_NAME}/msg/NeedleTrackingSnapshot"
POSE_TYPE = "geometry_msgs/msg/PoseWithCovarianceStamped"

KEYPOINTS_MSG_DEF = """
geometry_msgs/Point32 needle_tip
geometry_msgs/Point32 needle_tail
geometry_msgs/Point32 left_arm_tip
geometry_msgs/Point32 right_arm_tip
geometry_msgs/Pose2D bounding_box_center
float64 bounding_box_size_x
float64 bounding_box_size_y
"""

SNAPSHOT_MSG_DEF = """
sensor_msgs/Image img
sensor_msgs/Image mask
sensor_msgs/JointState right_arm_jp
sensor_msgs/JointState left_arm_jp
geometry_msgs/PoseStamped right_arm_cp
geometry_msgs/PoseStamped left_arm_cp
geometry_msgs/PoseWithCovarianceStamped right_tip_pose
geometry_msgs/PoseWithCovarianceStamped left_tip_pose
NeedleTrackingKeypoints keypoints
builtin_interfaces/Time stamp
"""


def ns_to_time(ns: int) -> Time:
    return Time(sec=ns // 10**9, nanosec=ns % 10**9)


def normalize_topic(topic: str) -> str:
    return topic.lower()


def register_custom_types(typestore) -> None:
    """Register the custom message definitions under the needle_tracking package."""

    typestore.register(get_types_from_msg(KEYPOINTS_MSG_DEF, KEYPOINTS_TYPE))
    typestore.register(get_types_from_msg(SNAPSHOT_MSG_DEF, SNAPSHOT_TYPE))


def build_image_msg(image_np: np.ndarray, timestamp_ns: int, frame_id: str = "camera") -> RosImage:
    h, w = image_np.shape[:2]
    header = Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id)
    data = np.ascontiguousarray(image_np, dtype=np.uint8).reshape(-1)
    return RosImage(
        header=header,
        height=h,
        width=w,
        encoding="mono8",
        is_bigendian=False,
        step=w,
        data=data,
    )


def build_blank_joint_state(timestamp_ns: int, frame_id: str = "") -> JointState:
    return JointState(
        header=Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id),
        name=[],
        position=np.array([], dtype=np.float64),
        velocity=np.array([], dtype=np.float64),
        effort=np.array([], dtype=np.float64),
    )


def build_blank_pose_stamped(timestamp_ns: int, frame_id: str = "") -> PoseStamped:
    return PoseStamped(
        header=Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id),
        pose=Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def build_blank_pose_with_cov_stamped(timestamp_ns: int, frame_id: str = "") -> PoseWithCovarianceStamped:
    return PoseWithCovarianceStamped(
        header=Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id),
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
    )


def point32_from_xy(x: float, y: float, z: float = 0.0) -> Point32:
    return Point32(x=float(x), y=float(y), z=float(z))


def load_masks_for_bag(ann_dir: Path, bag_stem: str) -> dict[int, np.ndarray]:
    masks_dir = ann_dir / bag_stem / "masks"
    if not masks_dir.exists():
        return {}

    frames: dict[int, np.ndarray] = {}
    for mask_path in sorted(masks_dir.glob("frame_*_needle_mask.png")):
        parts = mask_path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            frame_idx = int(parts[1])
        except ValueError:
            continue
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames[frame_idx] = img
    return frames


def load_keypoints_for_bag(ann_dir: Path, bag_stem: str) -> dict[int, dict]:
    keypoints_path = ann_dir / bag_stem / "keypoints.json"
    if not keypoints_path.exists():
        return {}

    with open(keypoints_path) as f:
        payload = json.load(f)

    frames = payload.get("frames", {})
    result: dict[int, dict] = {}
    for frame_key, frame_data in frames.items():
        try:
            frame_idx = int(frame_key)
        except ValueError:
            continue
        result[frame_idx] = frame_data
    return result


def load_poses_for_bag(ann_dir: Path, bag_stem: str) -> dict[int, dict]:
    poses_path = ann_dir / bag_stem / "poses_smooth.json"
    if not poses_path.exists():
        poses_path = ann_dir / bag_stem / "poses.json"
    if not poses_path.exists():
        return {}

    with open(poses_path, encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames", {})
    result: dict[int, dict] = {}
    for frame_key, frame_data in frames.items():
        try:
            frame_idx = int(frame_key)
        except ValueError:
            continue
        result[frame_idx] = frame_data
    return result


def select_topic(connections, msgtype: str | None = None, keywords: tuple[str, ...] = ()) -> str | None:
    candidates = [conn.topic for conn in connections if msgtype is None or conn.msgtype == msgtype]
    if not candidates:
        return None

    if keywords:
        filtered = [topic for topic in candidates if all(keyword in normalize_topic(topic) for keyword in keywords)]
        if filtered:
            return sorted(filtered)[0]

        filtered = [topic for topic in candidates if any(keyword in normalize_topic(topic) for keyword in keywords)]
        if filtered:
            return sorted(filtered)[0]

    return sorted(candidates)[0]


def select_side_topic(connections, msgtype: str, side: str, fallback_index: int) -> str | None:
    candidates = [conn.topic for conn in connections if conn.msgtype == msgtype]
    if not candidates:
        return None

    filtered = [topic for topic in candidates if side in normalize_topic(topic)]
    if filtered:
        return sorted(filtered)[0]

    ordered = sorted(candidates)
    if fallback_index < len(ordered):
        return ordered[fallback_index]
    return ordered[0]


def count_messages_for_topic(bag_dir: Path, topic: str) -> int:
    with Reader(bag_dir) as reader:
        connection = next((conn for conn in reader.connections if conn.topic == topic), None)
        if connection is None:
            return 0
        return sum(1 for _ in reader.messages(connections=[connection]))


def infer_every_n(bag_dir: Path, image_topic: str, annotated_frame_count: int) -> int:
    if annotated_frame_count <= 0:
        return 1

    total_images = count_messages_for_topic(bag_dir, image_topic)
    if total_images <= 0:
        return 1

    inferred = max(1, round(total_images / annotated_frame_count))
    if abs((annotated_frame_count * inferred) - total_images) > inferred:
        print(
            f"  [WARN] Inferred frame stride {inferred} from {total_images} image messages and "
            f"{annotated_frame_count} annotated frames."
        )
    return inferred


def mask_bbox(mask_np: np.ndarray, fallback_points: list[tuple[float, float]] | None = None) -> tuple[Pose2D, float, float]:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        fallback_points = fallback_points or []
        if not fallback_points:
            return Pose2D(x=0.0, y=0.0, theta=0.0), 0.0, 0.0
        xs = np.array([pt[0] for pt in fallback_points], dtype=np.float32)
        ys = np.array([pt[1] for pt in fallback_points], dtype=np.float32)

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())

    center = Pose2D(x=(x_min + x_max) / 2.0, y=(y_min + y_max) / 2.0, theta=0.0)
    return center, x_max - x_min + 1.0, y_max - y_min + 1.0


def extract_xy_or_default(point_like, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if point_like is None:
        return default

    try:
        if len(point_like) < 2:
            return default
        return float(point_like[0]), float(point_like[1])
    except Exception:
        return default


def build_keypoints_msg(typestore, frame_data: dict, mask_np: np.ndarray,
                        right_pose: PoseWithCovarianceStamped | None,
                        left_pose: PoseWithCovarianceStamped | None,
                        right_tooltip=None, left_tooltip=None):
    keypoint_cls = typestore.types[KEYPOINTS_TYPE]

    needle_tip_xy = extract_xy_or_default(frame_data.get("needle_tip"))
    needle_tail_xy = extract_xy_or_default(frame_data.get("needle_tail"))

    def _tip_point(tooltip, pose):
        # Prefer the tool-tip PIXEL topic (x/y are pixels at .pose.pose.position);
        # fall back to the 3D tip-pose position when no tool-tip message is present.
        if tooltip is not None:
            pos = tooltip.pose.pose.position
            return point32_from_xy(float(pos.x), float(pos.y), 0.0)
        if pose is not None:
            return point32_from_xy(float(pose.pose.pose.position.x),
                                   float(pose.pose.pose.position.y),
                                   float(pose.pose.pose.position.z))
        return point32_from_xy(0.0, 0.0, 0.0)

    right_tip = _tip_point(right_tooltip, right_pose)
    left_tip = _tip_point(left_tooltip, left_pose)

    fallback_points = [pt for pt in (needle_tip_xy, needle_tail_xy) if any(abs(v) > 0.0 for v in pt)]
    bbox_center, bbox_size_x, bbox_size_y = mask_bbox(mask_np, fallback_points=fallback_points)

    return keypoint_cls(
        needle_tip=point32_from_xy(*needle_tip_xy),
        needle_tail=point32_from_xy(*needle_tail_xy),
        left_arm_tip=left_tip,
        right_arm_tip=right_tip,
        bounding_box_center=bbox_center,
        bounding_box_size_x=float(bbox_size_x),
        bounding_box_size_y=float(bbox_size_y),
    )


def build_pose_msg(timestamp_ns: int, frame_data: dict, default_frame_id: str = "endoscope_optical") -> PoseWithCovarianceStamped | None:
    pose_data = frame_data.get("pose") if isinstance(frame_data, dict) else None
    if not pose_data:
        return None

    position = pose_data.get("position", [0.0, 0.0, 0.0])
    quaternion = pose_data.get("quaternion", [0.0, 0.0, 0.0, 1.0])
    covariance = pose_data.get("covariance", [0.0] * 36)
    if len(covariance) != 36:
        covariance = list(covariance[:36]) + [0.0] * max(0, 36 - len(covariance))

    pose_msg = PoseWithCovarianceStamped(
        header=Header(
            stamp=ns_to_time(timestamp_ns),
            frame_id=str(pose_data.get("frame_id", default_frame_id) or default_frame_id),
        ),
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=Quaternion(
                    x=float(quaternion[0]),
                    y=float(quaternion[1]),
                    z=float(quaternion[2]),
                    w=float(quaternion[3]),
                ),
            ),
            covariance=np.asarray(covariance, dtype=np.float64),
        ),
    )
    return pose_msg


def build_snapshot_msg(
    typestore,
    image_msg: RosImage,
    mask_msg: RosImage,
    right_joint: JointState,
    left_joint: JointState,
    right_cp: PoseStamped,
    left_cp: PoseStamped,
    right_pose: PoseWithCovarianceStamped,
    left_pose: PoseWithCovarianceStamped,
    keypoints_msg,
    timestamp_ns: int,
):
    snapshot_cls = typestore.types[SNAPSHOT_TYPE]
    return snapshot_cls(
        img=image_msg,
        mask=mask_msg,
        right_arm_jp=right_joint,
        left_arm_jp=left_joint,
        right_arm_cp=right_cp,
        left_arm_cp=left_cp,
        right_tip_pose=right_pose,
        left_tip_pose=left_pose,
        keypoints=keypoints_msg,
        stamp=ns_to_time(timestamp_ns),
    )


def build_split_topic_messages(
    mask_np: np.ndarray,
    right_joint: JointState,
    left_joint: JointState,
    right_cp: PoseStamped,
    left_cp: PoseStamped,
    right_pose: PoseWithCovarianceStamped,
    left_pose: PoseWithCovarianceStamped,
    keypoints_msg,
    timestamp_ns: int,
    frame_id: str,
) -> dict[str, tuple[str, object]]:
    mask_msg = build_image_msg(mask_np, timestamp_ns, frame_id=frame_id)

    return {
        MASK_TOPIC: ("sensor_msgs/msg/Image", mask_msg),
        KEYPOINTS_TOPIC: (KEYPOINTS_TYPE, keypoints_msg),
        RIGHT_ARM_JP_TOPIC: ("sensor_msgs/msg/JointState", right_joint),
        LEFT_ARM_JP_TOPIC: ("sensor_msgs/msg/JointState", left_joint),
        RIGHT_ARM_CP_TOPIC: ("geometry_msgs/msg/PoseStamped", right_cp),
        LEFT_ARM_CP_TOPIC: ("geometry_msgs/msg/PoseStamped", left_cp),
        RIGHT_TIP_POSE_TOPIC: ("geometry_msgs/msg/PoseWithCovarianceStamped", right_pose),
        LEFT_TIP_POSE_TOPIC: ("geometry_msgs/msg/PoseWithCovarianceStamped", left_pose),
    }


def serialize_and_write(writer, connection, typestore, message, msgtype: str, timestamp: int) -> None:
    raw = typestore.serialize_cdr(message, msgtype)
    writer.write(connection, timestamp, raw)


def repack_bag(
    bag_path: Path,
    ann_dir: Path,
    out_dir: Path,
    every_n: int | None = None,
    output_mode: str = "snapshot",
):
    if not ROSBAGS_AVAILABLE:
        print("[ERROR] rosbags not installed. Run: pip install rosbags")
        sys.exit(1)

    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent
    bag_stem = bag_dir.name
    out_bag_dir = out_dir / f"{bag_stem}_annotated_{output_mode}"

    print(f"\n{'=' * 60}")
    print(f"Repacking: {bag_stem}")
    print(f"Output:    {out_bag_dir}")
    print(f"Annotation directory: {ann_dir / bag_stem}")
    print(f"Output mode: {output_mode}")
    print(f"{'=' * 60}")

    split_topics = output_mode == "topics"

    masks = load_masks_for_bag(ann_dir, bag_stem)
    keypoints = load_keypoints_for_bag(ann_dir, bag_stem)
    poses = load_poses_for_bag(ann_dir, bag_stem)
    annotated_indices = sorted(set(masks) & set(keypoints))
    pose_indices = sorted(idx for idx, frame_data in poses.items() if frame_data.get("pose") and frame_data.get("status") != "failed")

    if not annotated_indices and not pose_indices:
        print(f"  [WARN] No overlapping mask/keypoint frames found for {bag_stem}")
    else:
        print(f"  Loaded {len(annotated_indices)} needle-annotated frames and {len(pose_indices)} pose frames")

    # Try to find ves_camera.yaml in common locations (annotation dir, bag dir, ann root)
    camera_yaml = None
    yaml_paths = [
        ann_dir / bag_stem / "ves_camera.yaml",
        bag_dir / "ves_camera.yaml",
        ann_dir / "ves_camera.yaml",
    ]
    try:
        import yaml

        for p in yaml_paths:
            if p.exists():
                try:
                    with open(p, "r") as fh:
                        camera_yaml = yaml.safe_load(fh)
                    print(f"  Found camera YAML: {p}")
                    break
                except Exception:
                    camera_yaml = None
    except Exception:
        camera_yaml = None

    def build_camera_info_from_yaml(timestamp_ns: int, frame_id: str = "camera"):
        # Default empty CameraInfo
        if camera_yaml is None:
            return CameraInfo(header=Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id))

        img_w = int(camera_yaml.get("image_width") or camera_yaml.get("image_width", 0) or 0)
        img_h = int(camera_yaml.get("image_height") or camera_yaml.get("image_height", 0) or 0)
        model = camera_yaml.get("distortion_model", "plumb_bob")
        D = camera_yaml.get("distortion_coefficients", {}).get("data") if isinstance(camera_yaml.get("distortion_coefficients"), dict) else camera_yaml.get("distortion_coefficients")
        if D is None:
            D = camera_yaml.get("D") or camera_yaml.get("distortion") or []
        K = camera_yaml.get("camera_matrix", {}).get("data") if isinstance(camera_yaml.get("camera_matrix"), dict) else camera_yaml.get("camera_matrix")
        if K is None:
            K = camera_yaml.get("K")
        R = camera_yaml.get("rectification_matrix", {}).get("data") if isinstance(camera_yaml.get("rectification_matrix"), dict) else camera_yaml.get("rectification_matrix")
        if R is None:
            R = camera_yaml.get("R")
        P = camera_yaml.get("projection_matrix", {}).get("data") if isinstance(camera_yaml.get("projection_matrix"), dict) else camera_yaml.get("projection_matrix")
        if P is None:
            P = camera_yaml.get("P")

        # Ensure lists of correct lengths
        D = list(D) if D else []
        K = list(K) if K and len(K) == 9 else ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] if K else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        R = list(R) if R and len(R) == 9 else ([1.0] * 9 if R else [1.0] * 9)
        P = list(P) if P and len(P) == 12 else ([1.0] * 12 if P else [1.0] * 12)

        roi = RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)

        return CameraInfo(
            header=Header(stamp=ns_to_time(timestamp_ns), frame_id=frame_id),
            height=img_h,
            width=img_w,
            distortion_model=model,
            d=D,
            k=K,
            r=R,
            p=P,
            binning_x=int(camera_yaml.get("binning_x", 0) or 0),
            binning_y=int(camera_yaml.get("binning_y", 0) or 0),
            roi=roi,
        )

    with Reader(bag_dir) as reader:
        connections = list(reader.connections)
        available_topics = {conn.topic: conn for conn in connections}

        # Prefer the rectified image topic when available; fall back to any image topic.
        if "/ves_camera/image_rect" in available_topics:
            image_topic = "/ves_camera/image_rect"
        else:
            image_topic = next((topic for topic in IMAGE_TOPICS if topic in available_topics), None)
            if image_topic is None:
                image_topic = select_topic(connections, msgtype="sensor_msgs/msg/Image", keywords=("image",))
        if image_topic is None:
            print(f"  [ERROR] No camera image topic found in {bag_dir}")
            return False
        if image_topic != "/ves_camera/image_rect":
            print(f"  [WARN] '/ves_camera/image_rect' not present; falling back to {image_topic}")

        camera_info_topics = [
            conn.topic
            for conn in connections
            if conn.msgtype in CAMERA_INFO_TYPES or normalize_topic(conn.topic).endswith("camera_info")
        ]

        right_joint_topic = select_side_topic(connections, "sensor_msgs/msg/JointState", "right", fallback_index=0)
        left_joint_topic = select_side_topic(connections, "sensor_msgs/msg/JointState", "left", fallback_index=1)
        right_cp_topic = select_side_topic(connections, "geometry_msgs/msg/PoseStamped", "right", fallback_index=0)
        left_cp_topic = select_side_topic(connections, "geometry_msgs/msg/PoseStamped", "left", fallback_index=1)

        # tool-tip pixel topics: PoseWithCovarianceStamped whose position x/y are pixels.
        # Discover them first and exclude from the 3D tip-pose selection so the two
        # don't collide (both are PoseWithCovarianceStamped).
        def _find_tooltip(side: str) -> str | None:
            cands = [c.topic for c in connections
                     if c.msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped"
                     and "tool_tip_pixel" in normalize_topic(c.topic)
                     and side in normalize_topic(c.topic)]
            return cands[0] if cands else None
        right_tooltip_topic = _find_tooltip("right")
        left_tooltip_topic = _find_tooltip("left")
        _tooltip_topics = {t for t in (right_tooltip_topic, left_tooltip_topic) if t}
        if _tooltip_topics:
            print(f"  Tool-tip pixel topics: left={left_tooltip_topic}  right={right_tooltip_topic}")

        def _select_pose_side(side: str, fallback_index: int) -> str | None:
            cands = [c.topic for c in connections
                     if c.msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped"
                     and c.topic not in _tooltip_topics]
            sided = sorted(t for t in cands if side in normalize_topic(t))
            if sided:
                return sided[0]
            cands = sorted(cands)
            return cands[fallback_index] if fallback_index < len(cands) else (cands[0] if cands else None)
        right_pose_topic = _select_pose_side("right", 0)
        left_pose_topic = _select_pose_side("left", 1)

        image_connection = available_topics.get(image_topic)
        if image_connection is None:
            print(f"  [ERROR] Could not resolve image topic connection: {image_topic}")
            return False

        image_connection_ext = cast(ConnectionExtRosbag2, image_connection.ext)

        reference_annotated_count = len(annotated_indices) if annotated_indices else len(pose_indices)
        inferred_every_n = every_n if every_n is not None else infer_every_n(bag_dir, image_topic, reference_annotated_count)
        inferred_every_n = max(1, int(inferred_every_n))
        print(f"  Image topic: {image_topic}")
        print(f"  Frame stride: {inferred_every_n}")

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        register_custom_types(typestore)

        if out_bag_dir.exists():
            print(f"  [WARN] Output bag already exists: {out_bag_dir}")
            print("         Delete it to repack again.")
            return False

        out_dir.mkdir(parents=True, exist_ok=True)

        with Writer(
            out_bag_dir,
            version=Writer.VERSION_LATEST,
            storage_plugin=StoragePlugin.MCAP,
        ) as writer:
            # Add a camera_info connection that we will populate from ves_camera.yaml
            camera_info_conn = writer.add_connection(topic="/ves_camera/camera_info", msgtype="sensor_msgs/msg/CameraInfo", typestore=typestore)
            pass_through_conns = {}
            for topic in camera_info_topics:
                conn = available_topics.get(topic)
                if conn is None:
                    continue
                conn_ext = cast(ConnectionExtRosbag2, conn.ext)
                pass_through_conns[conn.id] = writer.add_connection(
                    topic=conn.topic,
                    msgtype=conn.msgtype,
                    typestore=typestore,
                    serialization_format=conn_ext.serialization_format,
                    offered_qos_profiles=conn_ext.offered_qos_profiles,
                )

            image_out_conn = None
            annotation_conns = {}
            snapshot_conn = None
            checkerboard_pose_conn = None
            if split_topics:
                image_out_conn = writer.add_connection(
                    topic=image_topic,
                    msgtype=image_connection.msgtype,
                    typestore=typestore,
                    serialization_format=image_connection_ext.serialization_format,
                    offered_qos_profiles=image_connection_ext.offered_qos_profiles,
                )
                annotation_topics = {
                    MASK_TOPIC: "sensor_msgs/msg/Image",
                    KEYPOINTS_TOPIC: KEYPOINTS_TYPE,
                    RIGHT_ARM_JP_TOPIC: "sensor_msgs/msg/JointState",
                    LEFT_ARM_JP_TOPIC: "sensor_msgs/msg/JointState",
                    RIGHT_ARM_CP_TOPIC: "geometry_msgs/msg/PoseStamped",
                    LEFT_ARM_CP_TOPIC: "geometry_msgs/msg/PoseStamped",
                    RIGHT_TIP_POSE_TOPIC: "geometry_msgs/msg/PoseWithCovarianceStamped",
                    LEFT_TIP_POSE_TOPIC: "geometry_msgs/msg/PoseWithCovarianceStamped",
                }
                if pose_indices:
                    annotation_topics[CHECKERBOARD_POSE_TOPIC] = POSE_TYPE
                for topic, msgtype in annotation_topics.items():
                    annotation_conns[topic] = writer.add_connection(topic=topic, msgtype=msgtype, typestore=typestore)
            else:
                snapshot_conn = writer.add_connection(topic=SNAPSHOT_TOPIC, msgtype=SNAPSHOT_TYPE, typestore=typestore)
                if pose_indices:
                    checkerboard_pose_conn = writer.add_connection(topic=CHECKERBOARD_POSE_TOPIC, msgtype=POSE_TYPE, typestore=typestore)

            latest_joint_right: JointState | None = None
            latest_joint_left: JointState | None = None
            latest_cp_right: PoseStamped | None = None
            latest_cp_left: PoseStamped | None = None
            latest_pose_right: PoseStamped | None = None
            latest_pose_left: PoseStamped | None = None
            latest_tooltip_right = None
            latest_tooltip_left = None

            img_frame_counter = 0
            saved_frame_counter = 0
            last_selected_mask_idx = None
            total = reader.message_count
            iter_ = reader.messages()
            if TQDM_AVAILABLE:
                iter_ = tqdm(iter_, total=total, desc="  Messages", unit="msg")

            for connection, timestamp, rawdata in iter_:
                if connection.topic in camera_info_topics:
                    writer.write(pass_through_conns[connection.id], timestamp, rawdata)
                    continue

                if split_topics and connection.id == image_connection.id:
                    assert image_out_conn is not None
                    writer.write(image_out_conn, timestamp, rawdata)

                if connection.topic == right_joint_topic and connection.msgtype == "sensor_msgs/msg/JointState":
                    latest_joint_right = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.topic == left_joint_topic and connection.msgtype == "sensor_msgs/msg/JointState":
                    latest_joint_left = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.topic == right_cp_topic and connection.msgtype == "geometry_msgs/msg/PoseStamped":
                    latest_cp_right = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.topic == left_cp_topic and connection.msgtype == "geometry_msgs/msg/PoseStamped":
                    latest_cp_left = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.topic == right_pose_topic and connection.msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped":
                    latest_pose_right = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.topic == left_pose_topic and connection.msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped":
                    latest_pose_left = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if right_tooltip_topic and connection.topic == right_tooltip_topic:
                    latest_tooltip_right = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if left_tooltip_topic and connection.topic == left_tooltip_topic:
                    latest_tooltip_left = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    continue

                if connection.id != image_connection.id:
                    continue

                image_msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                image_frame_idx = img_frame_counter

                if image_frame_idx % inferred_every_n == 0:
                    annotated_idx = saved_frame_counter
                    if annotated_idx in masks and annotated_idx in keypoints:
                        mask_np = masks[annotated_idx]
                        frame_data = keypoints[annotated_idx]
                        frame_id = getattr(image_msg.header, "frame_id", "camera") or "camera"
                        mask_msg = build_image_msg(mask_np, timestamp, frame_id=frame_id)

                        right_joint_msg = latest_joint_right or build_blank_joint_state(timestamp, frame_id=frame_id)
                        left_joint_msg = latest_joint_left or build_blank_joint_state(timestamp, frame_id=frame_id)
                        right_cp_msg = latest_cp_right or build_blank_pose_stamped(timestamp, frame_id=frame_id)
                        left_cp_msg = latest_cp_left or build_blank_pose_stamped(timestamp, frame_id=frame_id)
                        right_pose_msg = latest_pose_right or build_blank_pose_with_cov_stamped(timestamp, frame_id=frame_id)
                        left_pose_msg = latest_pose_left or build_blank_pose_with_cov_stamped(timestamp, frame_id=frame_id)

                        keypoints_msg = build_keypoints_msg(
                            typestore,
                            frame_data,
                            mask_np,
                            right_pose_msg,
                            left_pose_msg,
                            right_tooltip=latest_tooltip_right,
                            left_tooltip=latest_tooltip_left,
                        )

                        if split_topics:
                            assert image_out_conn is not None
                            for topic, (msgtype, message) in build_split_topic_messages(
                                mask_np,
                                right_joint_msg,
                                left_joint_msg,
                                right_cp_msg,
                                left_cp_msg,
                                right_pose_msg,
                                left_pose_msg,
                                keypoints_msg,
                                timestamp,
                                frame_id,
                            ).items():
                                serialize_and_write(
                                    writer,
                                    annotation_conns[topic],
                                    typestore,
                                    message,
                                    msgtype,
                                    timestamp,
                                )
                        else:
                            assert snapshot_conn is not None
                            snapshot_msg = build_snapshot_msg(
                                typestore,
                                image_msg,
                                mask_msg,
                                right_joint_msg,
                                left_joint_msg,
                                right_cp_msg,
                                left_cp_msg,
                                right_pose_msg,
                                left_pose_msg,
                                keypoints_msg,
                                timestamp,
                            )
                            serialize_and_write(writer, snapshot_conn, typestore, snapshot_msg, SNAPSHOT_TYPE, timestamp)
                        try:
                            camera_info_msg = build_camera_info_from_yaml(timestamp, frame_id=frame_id)
                            serialize_and_write(writer, camera_info_conn, typestore, camera_info_msg, "sensor_msgs/msg/CameraInfo", timestamp)
                        except Exception:
                            pass

                    if annotated_idx in poses:
                        pose_frame_data = poses[annotated_idx]
                        pose_msg = build_pose_msg(timestamp, pose_frame_data)
                        if pose_msg is not None:
                            if split_topics:
                                serialize_and_write(
                                    writer,
                                    annotation_conns[CHECKERBOARD_POSE_TOPIC],
                                    typestore,
                                    pose_msg,
                                    POSE_TYPE,
                                    timestamp,
                                )
                            else:
                                if checkerboard_pose_conn is not None:
                                    serialize_and_write(writer, checkerboard_pose_conn, typestore, pose_msg, POSE_TYPE, timestamp)
                        last_selected_mask_idx = annotated_idx

                    saved_frame_counter += 1

                img_frame_counter += 1

    if last_selected_mask_idx is None:
        print("  [WARN] No snapshot messages were written")
    else:
        print(f"  Last snapshot frame index: {last_selected_mask_idx}")

    print(f"  ✓ Written to {out_bag_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Repack bags for needle-tracker playback")
    parser.add_argument("--bag", nargs="+", required=True, help="Path(s) to original bag directories")
    parser.add_argument("--ann-dir", required=True, help="Annotation directory")
    parser.add_argument("--out-dir", required=True, help="Directory to write annotated bags into")
    parser.add_argument(
        "--output-mode",
        choices=["snapshot", "topics"],
        default="snapshot",
        help="Write a bundled NeedleTrackingSnapshot or separate Foxglove-friendly topics.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=None,
        help="Original frame stride used during extraction. If omitted, infer it from the bag and annotations.",
    )
    parser.add_argument(
        "--pack-each",
        action="store_true",
        help="If set and a provided bag path is a directory containing multiple bag folders, repack each child folder into its own annotated bag.",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bag_str in args.bag:
        bag_path = Path(bag_str)
        if not bag_path.exists():
            print(f"[WARN] Bag not found: {bag_path}")
            continue

        # If --pack-each is set and the path is a parent directory, iterate its child folders
        if args.pack_each and bag_path.is_dir():
            for child in sorted(bag_path.iterdir()):
                if not child.is_dir():
                    continue
                print(f"Packing child bag: {child.name}")
                repack_bag(child, ann_dir, out_dir, every_n=args.every_n, output_mode=args.output_mode)
            continue

        repack_bag(bag_path, ann_dir, out_dir, every_n=args.every_n, output_mode=args.output_mode)

    print("\nDone. Replay with:")
    print("  ros2 bag play <annotated_bag>")


if __name__ == "__main__":
    main()