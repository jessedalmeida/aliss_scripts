#!/usr/bin/env python3
"""Self-test for export_dataset.py using a synthetic annotation tree."""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import export_dataset as ed


def make_bag(root: Path, name: str, poses_frames: dict, kp_frames: dict, mask_indices):
    bag = root / name
    (bag / "masks").mkdir(parents=True)
    (bag / "frames").mkdir(parents=True)
    for idx in mask_indices:
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(
            bag / "masks" / f"frame_{idx:06d}_needle_mask.png")
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
            bag / "frames" / f"frame_{idx:06d}.jpg")
    (bag / "poses.json").write_text(json.dumps({"frames": poses_frames}))
    (bag / "keypoints.json").write_text(json.dumps({"frames": kp_frames}))
    return bag


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Board bag: frame 0 detected (ok), frame 1 detection failed.
        # Rich keypoint schema with an occluded tail on frame 1.
        make_bag(
            root, "board_bag",
            poses_frames={
                "000000": {"status": "ok"},
                "000001": {"status": "failed"},
            },
            kp_frames={
                "000000": {"needle_tip": [3, 3], "needle_tail": [5, 5],
                           "occluded": {"needle_tip": False, "needle_tail": False}},
                "000001": {"needle_tip": [4, 2], "needle_tail": [6, 6],
                           "occluded": {"needle_tip": False, "needle_tail": True}},
            },
            mask_indices=[0, 1],
        )

        # No-board bag: all failed, prototype schema (no `occluded`),
        # frame 1 has no keypoint entry at all.
        make_bag(
            root, "noboard_bag",
            poses_frames={"000000": {"status": "failed"}, "000001": {"status": "no_board"}},
            kp_frames={"000000": {"needle_tip": [2, 2], "needle_tail": [4, 4]}},
            mask_indices=[0, 1],
        )

        out = root / "dataset"
        summary = ed.export_dataset(root, out, no_board_bags=set(), board_bags=set())

        records = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        by = {(r["bag"], r["frame"]): r for r in records}

        # --- domain flags (auto-detected) ---
        assert summary["bags_with_board"] == 1, summary
        assert summary["bags_without_board"] == 1, summary
        assert by[("board_bag", 0)]["has_board"] is True
        assert by[("noboard_bag", 0)]["has_board"] is False

        # --- per-frame board detection vs bag-level domain ---
        assert by[("board_bag", 0)]["board_detected"] is True   # status ok
        assert by[("board_bag", 1)]["board_detected"] is False  # status failed
        assert by[("board_bag", 1)]["has_board"] is True        # still a board bag

        # --- occlusion -> visibility ---
        assert by[("board_bag", 0)]["keypoints"]["needle_tail"]["visible"] is True
        assert by[("board_bag", 1)]["keypoints"]["needle_tail"]["visible"] is False
        assert by[("board_bag", 1)]["keypoints"]["needle_tail"]["xy"] == [6.0, 6.0]

        # --- prototype schema (no `occluded`) -> visible when annotated ---
        assert by[("noboard_bag", 0)]["keypoints"]["needle_tip"]["visible"] is True

        # --- missing keypoint entry -> null xy, not visible ---
        assert by[("noboard_bag", 1)]["keypoints"]["needle_tip"]["xy"] is None
        assert by[("noboard_bag", 1)]["keypoints"]["needle_tip"]["visible"] is False

        # --- arm tips always present, null for v1 data ---
        assert by[("board_bag", 0)]["keypoints"]["left_arm_tip"]["xy"] is None

        # --- counts ---
        assert summary["frames_total"] == 4, summary
        assert summary["frames_with_board"] == 2 and summary["frames_without_board"] == 2

        # --- explicit override beats auto-detect ---
        out2 = root / "dataset2"
        s2 = ed.export_dataset(root, out2, no_board_bags={"board_bag"}, board_bags=set())
        assert s2["bags_with_board"] == 0, s2

        print("All self-tests passed.")


if __name__ == "__main__":
    main()
