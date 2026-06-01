#!/usr/bin/env python3
"""
needle_pipeline.server - FastAPI backend for the annotation GUI.

Launch (from the repo root, where the legacy stage scripts live):

    python -m needle_pipeline.server --ann-dir ./annotations --scripts-dir .

Then open http://localhost:8000  (over SSH: ssh -L 8000:localhost:8000 host)

Heavy stages (propagate, pose, smooth, ...) run as background subprocess jobs,
so torch/SAM2/ros are never imported into the web process. Interactive edits
(mask paint, keypoint drag, pose corner-fix) use cv2/numpy directly.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import stages as S
from .config import Manifest

app = FastAPI(title="Needle Annotation")

# Populated at startup (see create_app / __main__).
STATE: dict = {"ann_dir": None, "scripts_dir": None, "jobs": {}}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def manifest() -> Manifest:
    return Manifest(STATE["ann_dir"])


def ctx(m: Optional[Manifest] = None):
    m = m or manifest()
    return m.to_ctx(STATE["scripts_dir"])


def bag_dir(bag: str) -> Path:
    return STATE["ann_dir"] / bag


def frame_files(bag: str) -> list[Path]:
    d = bag_dir(bag) / "frames"
    if not d.exists():
        return []
    out: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        out.extend(d.glob(pat))
    return sorted(out)


def frame_key(path: Path) -> str:
    m = re.search(r"(\d+)", path.stem)
    return f"{int(m.group(1)):06d}" if m else path.stem


def frame_by_key(bag: str, key: str) -> Optional[Path]:
    for f in frame_files(bag):
        if frame_key(f) == key:
            return f
    return None


def mask_path(bag: str, key: str) -> Path:
    return bag_dir(bag) / "masks" / f"frame_{key}_needle_mask.png"


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------------
# worklist + status
# --------------------------------------------------------------------------

@app.get("/api/bags")
def list_bags():
    m = manifest()
    c = ctx(m)
    bags = S.discover_bags(c, m)
    out = []
    for bag in bags:
        tracks = m.tracks(bag)
        status = S.compute_status(c, bag, tracks)
        out.append({
            "name": bag,
            "tracks": tracks,
            "frames": len(frame_files(bag)),
            "repack_mode": m.bag(bag).get("repack_mode") or c.repack_mode,
            "poses_source": m.bag(bag).get("poses_source") or c.poses_source,
            "has_smoothed": (bag_dir(bag) / "poses_smooth.json").exists(),
            "has_reference": (bag_dir(bag) / "reference_seed.json").exists(),
            "stages": [
                {"name": s.name, "track": s.track, "kind": s.kind,
                 "status": status[s.name][0], "msg": status[s.name][1]}
                for s in S.STAGES
            ],
        })
    m.save()
    return {"bags": out, "config": m.data["config"]}


@app.post("/api/bags/{bag}/track")
async def set_track(bag: str, req: Request):
    body = await req.json()
    m = manifest()
    for t in ("needle", "checkerboard"):
        if t in body:
            m.set_track(bag, t, bool(body[t]))
    m.save()
    return {"ok": True, "tracks": m.tracks(bag)}


@app.post("/api/bags/{bag}/repack_mode")
async def set_repack_mode(bag: str, req: Request):
    body = await req.json()
    mode = body.get("mode")
    if mode not in ("snapshot", "topics"):
        raise HTTPException(400, "mode must be 'snapshot' or 'topics'")
    m = manifest()
    m.set_repack_mode(bag, mode)
    m.save()
    return {"ok": True, "repack_mode": mode}


@app.post("/api/bags/{bag}/poses_source")
async def set_poses_source(bag: str, req: Request):
    body = await req.json()
    src = body.get("source")
    if src not in ("auto", "smooth", "raw"):
        raise HTTPException(400, "source must be 'auto', 'smooth', or 'raw'")
    m = manifest()
    m.set_poses_source(bag, src)
    m.save()
    return {"ok": True, "poses_source": src}


@app.get("/api/bags/{bag}/pose_trajectory")
def pose_trajectory(bag: str, preview: bool = False):
    """Per-frame position trajectories for raw and smoothed poses, plus the
    per-frame deviation between them — for the GUI's smoothing plots.
    If preview=true and poses_recomputed.json exists, it's shown as the 'raw'
    series so you can see a recompute proposal before committing."""
    raw_name = "poses.json"
    if preview and (bag_dir(bag) / "poses_recomputed.json").exists():
        raw_name = "poses_recomputed.json"

    def _traj(fname: str):
        data = read_json(bag_dir(bag) / fname).get("frames", {})
        keys = sorted((k for k in data if k.isdigit()), key=int)
        idx, xs, ys, zs = [], [], [], []
        for k in keys:
            fr = data[k]
            pose = fr.get("pose") if isinstance(fr, dict) else None
            pos = (pose or {}).get("position") if pose else None
            if not pos:
                continue
            idx.append(int(k)); xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
        return {"idx": idx, "x": xs, "y": ys, "z": zs}

    raw = _traj(raw_name)
    smooth = _traj("poses_smooth.json")
    # per-frame deviation (m) on the common frames
    dev = {"idx": [], "dist": []}
    if smooth["idx"]:
        rmap = {i: (raw["x"][n], raw["y"][n], raw["z"][n]) for n, i in enumerate(raw["idx"])}
        for n, i in enumerate(smooth["idx"]):
            if i in rmap:
                rx, ry, rz = rmap[i]
                d = ((smooth["x"][n]-rx)**2 + (smooth["y"][n]-ry)**2 + (smooth["z"][n]-rz)**2) ** 0.5
                dev["idx"].append(i); dev["dist"].append(d)
    stats = {}
    if dev["dist"]:
        ds = dev["dist"]
        stats = {"mean_mm": 1000*sum(ds)/len(ds), "max_mm": 1000*max(ds), "frames": len(ds)}
    return {"raw": raw, "smooth": smooth, "deviation": dev, "stats": stats,
            "has_smoothed": bool(smooth["idx"])}


# --------------------------------------------------------------------------
# frames / masks (binary)
# --------------------------------------------------------------------------

@app.get("/api/bags/{bag}/frames")
def list_frames(bag: str):
    files = frame_files(bag)
    return {"keys": [frame_key(f) for f in files], "count": len(files)}


@app.get("/api/bags/{bag}/frame/{key}")
def get_frame(bag: str, key: str):
    f = frame_by_key(bag, key)
    if not f:
        raise HTTPException(404, "frame not found")
    media = "image/png" if f.suffix == ".png" else "image/jpeg"
    return Response(f.read_bytes(), media_type=media)


@app.get("/api/bags/{bag}/mask/{key}")
def get_mask(bag: str, key: str):
    p = mask_path(bag, key)
    if not p.exists():
        raise HTTPException(404, "no mask")
    return Response(p.read_bytes(), media_type="image/png")


@app.post("/api/bags/{bag}/mask/{key}")
async def save_mask(bag: str, key: str, req: Request):
    """Body is raw PNG bytes of the binary mask (any non-zero pixel = needle)."""
    import cv2
    raw = await req.body()
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise HTTPException(400, "could not decode PNG")
    out = (arr > 127).astype(np.uint8) * 255
    p = mask_path(bag, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), out)
    return {"ok": True, "path": str(p)}


# --------------------------------------------------------------------------
# SAM2 seeds (browser places prompts; the propagate job runs SAM2)
# --------------------------------------------------------------------------

@app.get("/api/bags/{bag}/seeds")
def get_seeds(bag: str):
    return read_json(bag_dir(bag) / "seeds.json")


@app.post("/api/bags/{bag}/seeds")
async def save_seeds(bag: str, req: Request):
    """Body: {"seeds": [{"frame_idx": int, "points": [[x,y]...], "labels": [1|0...]}]}.
    Written in the seed_frames schema propagate.py consumes (1=foreground/needle,
    0=background). Frames are extracted sequentially, so frame_idx == frame key int."""
    body = await req.json()
    incoming = [s for s in body.get("seeds", [])
                if s.get("points") and any(l == 1 for l in s.get("labels", []))]
    if not incoming:
        raise HTTPException(400, "need at least one positive point")
    seed_frames = [{
        "seed_frame_idx": int(s["frame_idx"]),
        "objects": {"needle_mask": {
            "points": [[int(x), int(y)] for x, y in s["points"]],
            "labels": [int(l) for l in s["labels"]],
        }},
    } for s in incoming]
    seeds = {
        "bag_stem": bag,
        "frame_count": len(frame_files(bag)),
        "seed_frames": seed_frames,
        # legacy single-seed keys (propagate falls back to these if needed)
        "seed_frame_idx": seed_frames[0]["seed_frame_idx"],
        "objects": seed_frames[0]["objects"],
    }
    p = bag_dir(bag) / "seeds.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(seeds, indent=2) + "\n")
    return {"ok": True, "seed_frames": len(seed_frames)}


# --------------------------------------------------------------------------
# cross-bag reference seeding (teach from another bag's annotated frame)
# --------------------------------------------------------------------------

@app.get("/api/seed_sources")
def seed_sources():
    """List bags that have usable seeds (can act as a teacher), with their
    seeded frames + clicks so the UI can preview the teaching example."""
    m = manifest()
    out = []
    for bag in S.discover_bags(m.to_ctx(STATE["scripts_dir"]), m):
        sd = read_json(bag_dir(bag) / "seeds.json")
        entries = sd.get("seed_frames")
        if not entries and sd.get("objects"):
            entries = [{"seed_frame_idx": sd.get("seed_frame_idx", 0), "objects": sd["objects"]}]
        if not entries:
            continue
        frames = []
        for e in entries:
            o = (e.get("objects") or {}).get("needle_mask")
            if o:
                frames.append({"frame_key": f"{int(e.get('seed_frame_idx', 0)):06d}",
                               "points": o.get("points", []), "labels": o.get("labels", [])})
        if frames:
            out.append({"bag": bag, "frames": frames})
    return {"sources": out}


@app.post("/api/bags/{bag}/reference")
async def set_reference(bag: str, req: Request):
    """Borrow a source bag's seed as this bag's teacher (and optionally others).
    Body: {source_bag, source_frame_key, points, labels, also_apply_to:[bag,...]}.
    Writes reference_seed.json to each target; propagate then uses crossprop."""
    body = await req.json()
    src, key = body.get("source_bag"), body.get("source_frame_key")
    points, labels = body.get("points"), body.get("labels")
    if not (src and key and points and labels):
        raise HTTPException(400, "need source_bag, source_frame_key, points, labels")
    if not any(l == 1 for l in labels):
        raise HTTPException(400, "reference needs at least one positive point")
    targets = [bag] + [t for t in (body.get("also_apply_to") or []) if t != bag]
    ref = {"source_bag": src, "source_frame_key": key,
           "points": [[int(x), int(y)] for x, y in points], "labels": [int(l) for l in labels]}
    written = []
    for t in targets:
        if t == src:
            continue  # a bag can't teach itself
        d = bag_dir(t)
        d.mkdir(parents=True, exist_ok=True)
        (d / "reference_seed.json").write_text(json.dumps(ref, indent=2) + "\n")
        written.append(t)
    return {"ok": True, "applied_to": written}


@app.delete("/api/bags/{bag}/reference")
def clear_reference(bag: str):
    p = bag_dir(bag) / "reference_seed.json"
    existed = p.exists()
    if existed:
        p.unlink()
    return {"ok": True, "removed": existed}


@app.get("/api/bags/{bag}/reference")
def get_reference(bag: str):
    return read_json(bag_dir(bag) / "reference_seed.json")


# --------------------------------------------------------------------------
# keypoints (canonical schema)
# --------------------------------------------------------------------------

@app.get("/api/bags/{bag}/keypoints")
def get_keypoints(bag: str):
    return read_json(bag_dir(bag) / "keypoints.json")


@app.post("/api/bags/{bag}/keypoints/track")
async def track_keypoints_ep(bag: str, req: Request):
    """LK-track tip/tail across frames from one seeded frame.
    Body: {seed_key, needle_tip:[x,y], needle_tail:[x,y],
           direction:"forward"|"backward"|"both", anchor:bool}.
    anchor=True re-tracks forward only, preserving frames before the seed."""
    from .flow_ops import track_keypoints
    body = await req.json()
    tip, tail = body.get("needle_tip"), body.get("needle_tail")
    seed_key = body.get("seed_key")
    if not (tip and tail and seed_key):
        raise HTTPException(400, "need seed_key, needle_tip, needle_tail")
    files = frame_files(bag)
    keys = [frame_key(f) for f in files]
    if seed_key not in keys:
        raise HTTPException(404, "seed frame not found")
    p = bag_dir(bag) / "keypoints.json"
    data = read_json(p)
    existing = data.get("frames", {}) if body.get("anchor") else None
    try:
        frames = track_keypoints(
            files, keys, seed_key, tip, tail,
            direction=body.get("direction", "both"),
            anchor=bool(body.get("anchor")), existing=existing)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        raise HTTPException(400, f"tracking failed: {exc}")
    out = {"bag_stem": bag, "source": "optical_flow_gui", "frames": frames}
    p.write_text(json.dumps(out, indent=2) + "\n")
    return {"ok": True, "tracked": len(frames)}


@app.post("/api/bags/{bag}/keypoints/{key}/suggest")
async def suggest_keypoint(bag: str, key: str, req: Request):
    """Suggest tip/tail for one frame from its mask (skeleton endpoints).
    Optional body hints disambiguate tip vs tail: {tip_hint, tail_hint}.
    The previous frame's saved tip/tail are used automatically for continuity."""
    from .keypoint_ops import suggest_from_mask
    body = await req.json() if await req.body() else {}
    mp = mask_path(bag, key)
    if not mp.exists():
        raise HTTPException(404, "no mask for this frame - run propagate or paint one first")
    # previous frame's saved keypoints (temporal hint)
    files = frame_files(bag)
    keys = [frame_key(f) for f in files]
    prev_tip = prev_tail = None
    if key in keys:
        i = keys.index(key)
        if i > 0:
            pv = read_json(bag_dir(bag) / "keypoints.json").get("frames", {}).get(keys[i - 1])
            if pv:
                prev_tip, prev_tail = pv.get("needle_tip"), pv.get("needle_tail")
    sug = suggest_from_mask(mp, body.get("tip_hint"), body.get("tail_hint"),
                            prev_tip, prev_tail)
    if sug is None:
        raise HTTPException(422, "could not find endpoints in mask")
    return {"ok": True, **sug}


@app.post("/api/bags/{bag}/keypoints/{key}")
async def save_keypoint(bag: str, key: str, req: Request):
    """Merge a single frame's tip/tail into keypoints.json (canonical schema)."""
    body = await req.json()
    p = bag_dir(bag) / "keypoints.json"
    data = read_json(p)
    data.setdefault("bag_stem", bag)
    data.setdefault("source", "gui_edit")
    frames = data.setdefault("frames", {})
    entry = frames.setdefault(key, {})
    for field in ("needle_tip", "needle_tail"):
        if field in body:
            entry[field] = body[field]  # [x, y] or None
    entry.setdefault("occluded", {"tip": False, "tail": False})
    if "occluded" in body:
        entry["occluded"].update(body["occluded"])
    entry["status"] = "ok"
    p.write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True}


# --------------------------------------------------------------------------
# poses (read + recompute from manually-fixed corners)
# --------------------------------------------------------------------------

@app.get("/api/bags/{bag}/poses")
def get_poses(bag: str):
    return read_json(bag_dir(bag) / "poses.json")


@app.get("/api/bags/{bag}/axes/{key}")
def get_axes(bag: str, key: str):
    """Project the checkerboard coordinate frame to pixels for the given frame.
    Returns {origin,x,y,z: [px,py]} or {axes:null} if the frame has no pose."""
    frame = read_json(bag_dir(bag) / "poses.json").get("frames", {}).get(key)
    if not frame or frame.get("status") != "ok" or not frame.get("pose"):
        return {"axes": None}
    try:
        from .pose_ops import project_axes
        axes = project_axes(ctx(), bag, key, frame)
    except Exception as exc:  # noqa: BLE001 - axes are decorative; never 500 the UI
        return {"axes": None, "error": str(exc)}
    return {"axes": axes}


@app.post("/api/bags/{bag}/pose/{key}/reflow")
async def reflow_pose(bag: str, key: str, req: Request):
    """Re-seed a bad frame's checkerboard corners by optical-flow from the nearest
    GOOD neighbor (status==ok with corners), then solve. Returns the new frame dict
    without saving, so the UI can show it for review before the user re-solves/saves.
    Body (optional): {"neighbor_radius": int, "save": bool}."""
    body = await req.json() if await req.body() else {}
    c = ctx()
    # flow_seed lives beside the legacy scripts
    try:
        from flow_seed import try_optical_flow_seed
    except ImportError as exc:
        raise HTTPException(500, f"flow_seed not importable (check --scripts-dir): {exc}")

    poses_path = bag_dir(bag) / "poses.json"
    payload = read_json(poses_path)
    if not payload.get("frames"):
        raise HTTPException(404, "no poses.json yet - run the pose stage first")
    # try_optical_flow_seed reads board dims from payload["parameters"]; backfill from config
    payload.setdefault("parameters", {}).update({
        "squares_x": c.board["squares_x"], "squares_y": c.board["squares_y"],
        "square_size": c.board["square_size"]})

    from .pose_ops import _camera_yaml
    try:
        result = try_optical_flow_seed(
            c.bag_dir(bag), key, payload, _camera_yaml(c, bag),
            neighbor_radius=int(body.get("neighbor_radius", 3)))
    except Exception as exc:  # noqa: BLE001 - surface to UI
        raise HTTPException(400, f"reflow failed: {exc}")
    if result is None:
        raise HTTPException(422, "no good neighbor within range / flow rejected - "
                                 "fix a nearby frame first, then reflow from it")
    result.setdefault("frame_key", key)
    if body.get("save"):
        payload["frames"][key] = result
        # don't persist the backfilled parameters block if it wasn't there originally
        poses_path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"ok": True, "frame": result}


@app.post("/api/bags/{bag}/pose/{key}")
async def recompute_pose(bag: str, key: str, req: Request):
    """Recompute a frame's pose from manually-placed checkerboard corners.
    Body: {"corners": [[x,y], ...]} ordered to match the board's object points."""
    from .pose_ops import recompute_from_corners
    body = await req.json()
    corners = body.get("corners")
    if not corners:
        raise HTTPException(400, "no corners provided")
    try:
        
        # frame_result = recompute_from_corners(ctx(), bag, key, corners)
        p = bag_dir(bag) / "poses.json"
        data = read_json(p)
        old_frame = data.get("frames", {}).get(key, {})
        old_pose = None
        old_rms = old_frame.get("rms_reprojection_error")

        if old_frame.get("pose") and old_rms is not None and old_rms < 3.0:
            old_pose = old_frame.get("pose")

        frame_result = recompute_from_corners(
            ctx(),
            bag,
            key,
            corners,
            initial_pose=old_pose,
        )
        frame_result["roi"] = old_frame.get("roi", [0, 0, 0, 0])
        frame_result["failure_stage"] = ""
        frame_result["diagnostics_image"] = old_frame.get("diagnostics_image")
    except Exception as exc:  # noqa: BLE001 - surface the reason to the UI
        raise HTTPException(400, f"pose solve failed: {exc}")
    p = bag_dir(bag) / "poses.json"
    data = read_json(p)
    data.setdefault("frames", {})[key] = frame_result
    p.write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True, "frame": frame_result}


@app.post("/api/bags/{bag}/pose/{key}/no_board")
def mark_no_board(bag: str, key: str):
    """Mark a frame as intentionally having no checkerboard, so it isn't counted
    as a failure and doesn't block review/repack. Toggles back to needing
    detection if already no_board."""
    p = bag_dir(bag) / "poses.json"
    data = read_json(p)
    frames = data.setdefault("frames", {})
    cur = frames.get(key, {})
    if cur.get("status") == "no_board":
        # un-mark: drop the entry so the pose stage will try it again next run
        frames.pop(key, None)
        state = "cleared"
    else:
        frames[key] = {"frame_key": key, "status": "no_board", "detector": "manual_no_board",
                       "failure_reason": "", "corners": None, "pose": None,
                       "rms_reprojection_error": 0.0}
        state = "no_board"
    p.write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True, "state": state}


@app.post("/api/bags/{bag}/poses/recompute")
async def recompute_all_poses(bag: str, req: Request):
    """Re-solve EVERY frame's pose from its stored corners using the current
    camera + board settings (fixes wrong camera_info or square_size without
    re-detecting). Writes to poses_recomputed.json for review — does NOT touch
    poses.json until committed. Body (optional): {square_size, squares_x, squares_y}."""
    from .pose_ops import recompute_from_corners
    body = await req.json() if await req.body() else {}
    c = ctx()
    # allow a one-off board override (e.g. correcting square_size) without persisting
    if body.get("square_size"): c.board["square_size"] = float(body["square_size"])
    if body.get("squares_x"): c.board["squares_x"] = int(body["squares_x"])
    if body.get("squares_y"): c.board["squares_y"] = int(body["squares_y"])
    src = read_json(bag_dir(bag) / "poses.json")
    frames = src.get("frames", {})
    if not frames:
        raise HTTPException(404, "no poses.json to recompute from")
    out = dict(src); out_frames = {}
    recomputed = skipped = failed = 0
    for key, fr in frames.items():
        if fr.get("status") == "no_board":
            out_frames[key] = fr; skipped += 1; continue
        corners = fr.get("corners")
        if not corners:
            out_frames[key] = fr; skipped += 1; continue
        try:
            new_fr = recompute_from_corners(
                c,
                bag,
                key,
                corners,
                initial_pose=(fr.get("pose") if isinstance(fr, dict) else None),
            )
            new_fr["roi"] = fr.get("roi", [0, 0, 0, 0])
            new_fr["failure_stage"] = ""
            new_fr["diagnostics_image"] = fr.get("diagnostics_image")
            new_fr["detector"] = fr.get("detector", new_fr.get("detector", "manual_corner"))
            out_frames[key] = new_fr
            recomputed += 1
        except Exception:
            out_frames[key] = fr; failed += 1
    out["frames"] = out_frames
    out["parameters"] = {**out.get("parameters", {}),
                         "square_size": c.board["square_size"],
                         "squares_x": c.board["squares_x"], "squares_y": c.board["squares_y"]}
    (bag_dir(bag) / "poses_recomputed.json").write_text(json.dumps(out, indent=2) + "\n")
    return {"ok": True, "recomputed": recomputed, "skipped": skipped, "failed": failed,
            "preview": "poses_recomputed.json"}


@app.post("/api/bags/{bag}/poses/commit")
async def commit_poses(bag: str, req: Request):
    """Promote a preview file to poses.json (backing up the current one first).
    Body: {"preview": "poses_recomputed.json"}."""
    body = await req.json()
    preview = body.get("preview", "poses_recomputed.json")
    src = bag_dir(bag) / preview
    if not src.exists():
        raise HTTPException(404, f"{preview} not found - run recompute first")
    dst = bag_dir(bag) / "poses.json"
    if dst.exists():
        backup = bag_dir(bag) / "poses_prev.json"
        backup.write_text(dst.read_text())
    dst.write_text(src.read_text())
    src.unlink()  # consume the preview
    # smoothed file is now stale relative to new raw poses
    return {"ok": True, "committed": preview, "note": "re-run smoothing to refresh poses_smooth.json"}


@app.post("/api/bags/{bag}/poses/resmooth")
async def resmooth_poses(bag: str, req: Request):
    """Re-run SE(3) smoothing on the current poses.json. Body (optional):
    {z_downweight: float (>1 trusts Z less), to_preview: bool}. Writes
    poses_smooth.json (or poses_smooth_preview.json if to_preview)."""
    body = await req.json() if await req.body() else {}
    from .pose_ops import resmooth
    try:
        info = resmooth(ctx(), bag,
                        z_downweight=float(body.get("z_downweight", 1.0)),
                        to_preview=bool(body.get("to_preview")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"re-smooth failed: {exc}")
    return {"ok": True, **info}

async def _run_job(job_id: str, cmd: list[str], cwd: str):
    job = STATE["jobs"][job_id]
    job["status"] = "running"
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    job["pid"] = proc.pid
    assert proc.stdout is not None
    async for line in proc.stdout:
        job["log"].append(line.decode(errors="replace").rstrip())
        job["log"] = job["log"][-500:]
    rc = await proc.wait()
    job["status"] = "done" if rc == 0 else "failed"
    job["rc"] = rc
    job["ended"] = time.time()


@app.post("/api/run")
async def run_stage(req: Request):
    """Body: {"bag": "...", "stage": "..."} - launch one auto stage as a job."""
    body = await req.json()
    bag, stage_name = body.get("bag"), body.get("stage")
    if stage_name not in S.STAGE_BY_NAME:
        raise HTTPException(400, "unknown stage")
    stage = S.STAGE_BY_NAME[stage_name]
    cmd = stage.build_cmd(ctx(), bag)
    if cmd is None:
        raise HTTPException(400, "stage is blocked (missing config/paths)")
    job_id = uuid.uuid4().hex[:8]
    STATE["jobs"][job_id] = {
        "id": job_id, "bag": bag, "stage": stage_name,
        "status": "queued", "log": [], "rc": None, "started": time.time(),
    }
    asyncio.create_task(_run_job(job_id, cmd, str(STATE["scripts_dir"])))
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": list(STATE["jobs"].values())}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = STATE["jobs"].get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job


# --------------------------------------------------------------------------
# static frontend
# --------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text())
    return HTMLResponse("<h1>index.html missing</h1>", status_code=500)


def create_app(ann_dir: Path, scripts_dir: Path) -> FastAPI:
    STATE["ann_dir"] = Path(ann_dir).resolve()
    STATE["scripts_dir"] = Path(scripts_dir).resolve()
    STATE["ann_dir"].mkdir(parents=True, exist_ok=True)
    # let interactive ops import the legacy algorithm modules
    sp = str(STATE["scripts_dir"])
    if sp not in sys.path:
        sys.path.insert(0, sp)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Needle annotation web server")
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--scripts-dir", default=".",
                    help="Where the legacy stage scripts live (default: cwd)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    create_app(Path(args.ann_dir), Path(args.scripts_dir))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
