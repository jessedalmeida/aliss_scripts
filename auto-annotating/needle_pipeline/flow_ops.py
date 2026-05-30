"""
needle_pipeline.flow_ops - keypoint tracking for the GUI.

Reuses OpticalFlowTracker from the existing annotate_keypoints.py (Lucas-Kanade)
so the browser can: place tip/tail on one frame, track across the sequence, then
re-anchor from any corrected frame. cv2/numpy only - no GPU - so it runs directly
in the web process for instant feedback. The legacy scripts must be importable
(the server puts --scripts-dir on sys.path at startup).
"""

from __future__ import annotations

from pathlib import Path

import cv2


def _load_frames(frame_paths: list[Path]):
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"could not read frame {p}")
        frames.append(img)
    return frames


def track_keypoints(frame_paths: list[Path], keys: list[str], seed_key: str,
                    tip, tail, direction: str = "both",
                    anchor: bool = False, existing: dict | None = None) -> dict:
    """Run LK tracking from a seeded frame and return a keypoints `frames` dict.

    seed_key:  frame key where tip/tail are given
    direction: "forward" | "backward" | "both"
    anchor:    if True, only re-track forward from seed (reseed_from) and keep
               frames before the seed as-is from `existing`.
    existing:  prior keypoints `frames` dict to preserve outside the retrack range.
    """
    from annotate_keypoints import OpticalFlowTracker

    seed_pos = keys.index(seed_key)
    frames = _load_frames(frame_paths)
    seed_tip = (float(tip[0]), float(tip[1]))
    seed_tail = (float(tail[0]), float(tail[1]))

    tracker = OpticalFlowTracker(frames, seed_pos, seed_tip, seed_tail)
    if anchor:
        tracker.reseed_from(seed_pos, seed_tip, seed_tail)
    else:
        if direction in ("forward", "both"):
            tracker.track_forward()
        if direction in ("backward", "both"):
            tracker.track_backward()

    preds = tracker.get_predictions()  # {pos: {"tip": (x,y), "tail": (x,y)}}
    out = dict(existing or {})
    for pos, kp in preds.items():
        key = keys[pos]
        out[key] = {
            "needle_tip": [int(round(kp["tip"][0])), int(round(kp["tip"][1]))],
            "needle_tail": [int(round(kp["tail"][0])), int(round(kp["tail"][1]))],
            "occluded": (existing or {}).get(key, {}).get("occluded", {"tip": False, "tail": False}),
            "status": "ok",
            "method": "optical_flow_gui",
        }
    return out
