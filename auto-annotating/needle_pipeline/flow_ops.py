"""
needle_pipeline.flow_ops - keypoint tracking for the GUI.

Reuses OpticalFlowTracker from the existing annotate_keypoints.py (Lucas-Kanade)
so the browser can: place keypoints on one frame, track across the sequence, then
re-anchor from any corrected frame. cv2/numpy only - no GPU - so it runs directly
in the web process for instant feedback. The legacy scripts must be importable
(the server puts --scripts-dir on sys.path at startup).

The underlying OpticalFlowTracker tracks exactly two points (tip+tail) per run, so
to support an arbitrary set of named keypoints we group them into pairs and run the
tracker once per pair, loading the frames only once.
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


def _track_pair(frames, keys, seed_pos, p0, p1, direction, anchor):
    """Track a single pair of points with OpticalFlowTracker.
    Returns {pos: {"tip": (x,y), "tail": (x,y)}} (tracker's native naming)."""
    from annotate_keypoints import OpticalFlowTracker
    a = (float(p0[0]), float(p0[1]))
    b = (float(p1[0]), float(p1[1]))
    tracker = OpticalFlowTracker(frames, seed_pos, a, b)
    if anchor:
        # Avoid reseed_from unless you know it is forward-only.
        tracker.track_forward()
    else:
        if direction in ("forward", "both"):
            tracker.track_forward()
        if direction in ("backward", "both"):
            tracker.track_backward()

    preds = tracker.get_predictions()

    if anchor:
        preds = {pos: kp for pos, kp in preds.items() if pos >= seed_pos}

    return preds


def track_keypoints(frame_paths: list[Path], keys: list[str], seed_key: str,
                    points: dict, direction: str = "both",
                    anchor: bool = False, existing: dict | None = None) -> dict:
    """Run LK tracking from a seeded frame for an arbitrary set of named keypoints.

    points:    {name: [x, y]} placed on the seed frame. Tracked in pairs (the
               underlying tracker is two-point); an odd point out is paired with
               itself harmlessly and its second slot ignored.
    seed_key:  frame key where the points are given.
    direction: "forward" | "backward" | "both".
    anchor:    if True, only re-track forward from the seed, preserving frames
               before it from `existing`.
    existing:  prior keypoints `frames` dict to preserve/merge.

    Returns a keypoints `frames` dict: {key: {<name>:[x,y], ..., occluded, status}}.
    """
    seed_pos = keys.index(seed_key)
    frames = _load_frames(frame_paths)  # load once, reuse across pairs

    # names = [n for n in points if points[n] is not None]
    preferred = ["needle_tip", "needle_tail", "left_arm_tip", "right_arm_tip"]
    names = [n for n in preferred if points.get(n) is not None]

    # Optional: include any future/custom keypoints after the known ones
    names += [n for n in points if n not in names and points[n] is not None]
    if not names:
        return dict(existing or {})

    # group into consecutive pairs; the tracker always needs two slots
    pairs = []
    i = 0
    while i < len(names):
        if i + 1 < len(names):
            pairs.append((names[i], names[i + 1]))
            i += 2
        else:
            pairs.append((names[i], names[i]))  # lone point: track against itself
            i += 1

    # per-frame accumulation of every tracked point
    merged: dict[int, dict] = {}
    for a, b in pairs:
        preds = _track_pair(frames, keys, seed_pos, points[a], points[b], direction, anchor)
        for pos, kp in preds.items():
            if anchor and pos < seed_pos:
                continue
            slot = merged.setdefault(pos, {})
            slot[a] = kp["tip"]
            if b != a:
                slot[b] = kp["tail"]

    out = dict(existing or {})
    for pos, kp in merged.items():
        key = keys[pos]
        prev = (existing or {}).get(key, {})
        entry = dict(prev)  # preserve any fields we aren't overwriting
        occ = dict(prev.get("occluded", {}))
        for name, xy in kp.items():
            entry[name] = [int(round(xy[0])), int(round(xy[1]))]
            occ.setdefault(name, False)
        entry["occluded"] = occ
        entry["status"] = "ok"
        entry["method"] = "optical_flow_gui"
        out[key] = entry
    return out
