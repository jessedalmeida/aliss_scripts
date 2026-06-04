#!/usr/bin/env python3
"""export_dataset.py - Build a training manifest from annotation bags.

Reads the offline-pipeline outputs for each bag (poses.json + keypoints.json +
needle masks + frame images) and emits a flat JSONL manifest plus a summary,
without the MCAP topics round-trip. This is the dataset interface for the
real-time needle segmentation + keypoint model.

Per-frame record (one JSON object per line in manifest.jsonl):
    bag            bag stem
    frame          int frame index
    image          path to the frame image
    mask           path to the needle mask PNG
    has_board      bool, BAG-LEVEL domain flag (board physically present)
    board_detected bool, THIS frame's checkerboard detection succeeded
    pose_status    raw poses.json status for the frame
    keypoints      {name: {"xy": [x, y] | null, "visible": bool}} for all 4
                   keypoints (needle_tip, needle_tail, left_arm_tip,
                   right_arm_tip). xy is null when unannotated; visible is
                   False when null or flagged occluded.

A frame is emitted only if both an image and a mask exist (mask is the v1
supervision signal). All four keypoints are always emitted for forward-compat;
v1 training consumes only the needle pair.

Usage:
    python export_dataset.py --ann-dir ./annotations --out ./dataset
    python export_dataset.py --ann-dir ./annotations --out ./dataset \
        --no-board-bags chicken_1,chicken_2     # explicit domain labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYPOINT_NAMES = ("needle_tip", "needle_tail", "left_arm_tip", "right_arm_tip")

# poses.json statuses that mean the board was actually detected this frame.
BOARD_DETECTED_STATUSES = {"ok", "high_rms"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def frame_index_from_name(name: str) -> int | None:
    """Pull the first all-digit token out of a filename stem or frame key."""
    for token in Path(name).stem.split("_"):
        if token.isdigit():
            return int(token)
    if name.isdigit():
        return int(name)
    return None


def resolve_image(bag_dir: Path, idx: int) -> Path | None:
    """Find the frame image for a given index across known layouts."""
    candidates = [
        bag_dir / "frames" / f"frame_{idx:06d}.jpg",
        bag_dir / "frames" / f"frame_{idx:06d}.png",
        bag_dir / "frames_jpg" / f"{idx:06d}.jpg",
        bag_dir / "frames" / f"{idx:06d}.jpg",
        bag_dir / "frames" / f"{idx:06d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def index_masks(bag_dir: Path) -> dict[int, Path]:
    """Map frame index -> needle mask path."""
    masks_dir = bag_dir / "masks"
    out: dict[int, Path] = {}
    if not masks_dir.exists():
        return out
    for p in sorted(masks_dir.glob("frame_*_needle_mask.png")):
        idx = frame_index_from_name(p.name)
        if idx is not None:
            out[idx] = p
    return out


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------------------------------- #
# normalization core (format-agnostic; reused by any serializer)
# --------------------------------------------------------------------------- #
def normalize_keypoints(kp_frame: dict) -> dict:
    """Normalize one keypoints.json frame entry into all-4-keypoint form.

    Handles both the rich target schema (with an `occluded` dict and arm tips)
    and the prototype schema (needle tip/tail only, no occlusion).
    """
    occluded = kp_frame.get("occluded") or {}
    out: dict[str, dict] = {}
    for name in KEYPOINT_NAMES:
        xy = kp_frame.get(name)
        if xy is None or len(xy) != 2:
            out[name] = {"xy": None, "visible": False}
            continue
        # visible = annotated AND not explicitly flagged occluded.
        is_occluded = bool(occluded.get(name, False))
        out[name] = {
            "xy": [float(xy[0]), float(xy[1])],
            "visible": not is_occluded,
        }
    return out


def empty_keypoints() -> dict:
    return {name: {"xy": None, "visible": False} for name in KEYPOINT_NAMES}


def bag_has_board(pose_frames: dict, override: bool | None) -> bool:
    """Bag-level domain flag.

    If the caller supplied an explicit label, trust it. Otherwise auto-detect:
    a bag is a board bag if the checkerboard was ever successfully detected.
    (A true no-board bag never detects a board; a board bag may still have
    frames where detection failed, which is why per-frame status alone is not
    a reliable domain label.)
    """
    if override is not None:
        return override
    return any(
        f.get("status") in BOARD_DETECTED_STATUSES for f in pose_frames.values()
    )


# --------------------------------------------------------------------------- #
# per-bag export
# --------------------------------------------------------------------------- #
def export_bag(bag_dir: Path, has_board_override: bool | None) -> tuple[list[dict], dict]:
    """Return (records, per-bag stats) for one bag directory."""
    poses = load_json(bag_dir / "poses.json").get("frames", {})
    kps = load_json(bag_dir / "keypoints.json").get("frames", {})
    masks = index_masks(bag_dir)

    has_board = bag_has_board(poses, has_board_override)

    # Re-key pose / keypoint dicts by int frame index for robust joining.
    pose_by_idx = {
        frame_index_from_name(k): v for k, v in poses.items()
        if frame_index_from_name(k) is not None
    }
    kp_by_idx = {
        frame_index_from_name(k): v for k, v in kps.items()
        if frame_index_from_name(k) is not None
    }

    records: list[dict] = []
    skipped_no_image = 0
    kp_coverage = {name: 0 for name in KEYPOINT_NAMES}

    # Masks are the v1 supervision anchor: iterate the frames that have one.
    for idx in sorted(masks):
        image = resolve_image(bag_dir, idx)
        if image is None:
            skipped_no_image += 1
            continue

        pose_entry = pose_by_idx.get(idx, {})
        pose_status = pose_entry.get("status")
        board_detected = pose_status in BOARD_DETECTED_STATUSES

        kp_entry = kp_by_idx.get(idx)
        keypoints = normalize_keypoints(kp_entry) if kp_entry else empty_keypoints()
        for name in KEYPOINT_NAMES:
            if keypoints[name]["xy"] is not None:
                kp_coverage[name] += 1

        records.append({
            "bag": bag_dir.name,
            "frame": idx,
            "image": str(image),
            "mask": str(masks[idx]),
            "has_board": has_board,
            "board_detected": board_detected,
            "pose_status": pose_status,
            "keypoints": keypoints,
        })

    stats = {
        "bag": bag_dir.name,
        "has_board": has_board,
        "frames_emitted": len(records),
        "masks_found": len(masks),
        "skipped_no_image": skipped_no_image,
        "keypoint_coverage": kp_coverage,
    }
    return records, stats


def collect_bag_dirs(ann_dir: Path) -> list[Path]:
    """Bag dirs are immediate subdirectories that contain a masks/ folder."""
    return sorted(p for p in ann_dir.iterdir() if p.is_dir() and (p / "masks").exists())


def export_dataset(
    ann_dir: Path,
    out_dir: Path,
    no_board_bags: set[str],
    board_bags: set[str],
) -> dict:
    bag_dirs = collect_bag_dirs(ann_dir)
    if not bag_dirs:
        raise RuntimeError(f"No bag directories (with masks/) found under {ann_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    all_stats: list[dict] = []
    total = 0
    with manifest_path.open("w") as fh:
        for bag_dir in bag_dirs:
            override: bool | None = None
            if bag_dir.name in no_board_bags:
                override = False
            elif bag_dir.name in board_bags:
                override = True
            records, stats = export_bag(bag_dir, override)
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
            total += len(records)
            all_stats.append(stats)

    board_bag_count = sum(1 for s in all_stats if s["has_board"])
    board_frames = sum(s["frames_emitted"] for s in all_stats if s["has_board"])
    noboard_frames = total - board_frames
    summary = {
        "ann_dir": str(ann_dir),
        "bags": len(all_stats),
        "bags_with_board": board_bag_count,
        "bags_without_board": len(all_stats) - board_bag_count,
        "frames_total": total,
        "frames_with_board": board_frames,
        "frames_without_board": noboard_frames,
        "per_bag": all_stats,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export training manifest from annotation bags")
    ap.add_argument("--ann-dir", required=True, help="Annotation root containing bag subdirs")
    ap.add_argument("--out", required=True, help="Output directory for manifest + summary")
    ap.add_argument("--no-board-bags", default=None,
                    help="Comma-separated bag names to force has_board=False")
    ap.add_argument("--board-bags", default=None,
                    help="Comma-separated bag names to force has_board=True")
    args = ap.parse_args()

    summary = export_dataset(
        Path(args.ann_dir),
        Path(args.out),
        _split_csv(args.no_board_bags),
        _split_csv(args.board_bags),
    )
    print(f"[OK] {summary['frames_total']} frames from {summary['bags']} bags "
          f"({summary['bags_with_board']} board / {summary['bags_without_board']} no-board)")
    print(f"     board frames={summary['frames_with_board']}  "
          f"no-board frames={summary['frames_without_board']}")
    print(f"     manifest -> {Path(args.out) / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
