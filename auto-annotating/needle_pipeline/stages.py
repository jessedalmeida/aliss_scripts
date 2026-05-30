"""
needle_pipeline.stages - the stage DAG, status computation, and auto runner.

Two optional tracks per bag run independently after extraction:

    extract ─┬─ seed ─ propagate ─ keypoints ─────────┐   (needle track)
             │                                          ├─ repack ─ npz
             └─ pose ─┬─ review ┐                       │   (checkerboard track)
                      └─────────┴─ smooth ──────────────┘

A stage whose track is disabled for a bag reports status "na" and is treated
as satisfied for downstream prerequisites.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import Ctx, Manifest

# status values
DONE, ATTENTION, READY, BLOCKED, WAITING, NA = (
    "done", "attention", "ready", "blocked", "waiting", "na",
)
SATISFYING = {DONE, ATTENTION, NA}   # counts as "prerequisite met"


@dataclass
class Stage:
    name: str
    track: str            # "needle" | "checkerboard" | "shared" | "output"
    kind: str             # "auto" | "manual"
    requires: list[str]
    build_cmd: Callable[[Ctx, str], Optional[list[str]]]
    is_done: Callable[[Ctx, str], bool]
    needs_attention: Callable[[Ctx, str], tuple[bool, str]] = lambda c, b: (False, "")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _frames(ctx: Ctx, bag: str) -> bool:
    d = ctx.bag_dir(bag) / "frames"
    return d.exists() and any(d.glob(p) for p in ("*.jpg", "*.jpeg", "*.png"))


def _extra(ctx: Ctx, stage: str) -> list[str]:
    return ctx.stage_args.get(stage, [])


def _board_args(ctx: Ctx) -> list[str]:
    b = ctx.board
    args = ["--squares-x", str(b["squares_x"]),
            "--squares-y", str(b["squares_y"]),
            "--square-size", str(b["square_size"])]
    if ctx.camera_yaml:
        args += ["--camera-yaml", str(ctx.camera_yaml)]
    return args


def _pose_failures(ctx: Ctx, bag: str) -> int:
    p = ctx.bag_dir(bag) / "poses.json"
    if not p.exists():
        return 0
    try:
        frames = json.loads(p.read_text()).get("frames", {})
    except Exception:
        return 0
    return sum(1 for f in frames.values() if f.get("status") != "ok")


# --------------------------------------------------------------------------
# stage command builders + done checks
# --------------------------------------------------------------------------

def _extract_cmd(ctx, bag):
    src = ctx.bag_sources.get(bag)
    if not src:
        return None
    return [sys.executable, "-m", "needle_pipeline.extract",
            "--bag", src, "--out", str(ctx.ann_dir),
            "--every-n", str(ctx.every_n), *_extra(ctx, "extract")]


def _has_ref_seed(ctx, bag) -> bool:
    return (ctx.bag_dir(bag) / "reference_seed.json").exists()


def _seed_cmd(ctx, bag):
    # frames already extracted -> seed only places SAM2 clicks
    src = ctx.bag_sources.get(bag)
    return [sys.executable, str(ctx.script("seed_annotator.py")),
            "--bag", src or str(ctx.bag_dir(bag)), "--out", str(ctx.ann_dir),
            "--skip-extract", *_extra(ctx, "seed")]


def _propagate_cmd(ctx, bag):
    if not ctx.sam2_repo:
        return None
    # if this bag borrows another bag's seed, use the cross-bag propagation path
    if _has_ref_seed(ctx, bag):
        return [sys.executable, "-m", "needle_pipeline.crossprop",
                "--ann-dir", str(ctx.ann_dir), "--bag", bag,
                "--sam2-repo", str(ctx.sam2_repo), "--model", ctx.model]
    return [sys.executable, str(ctx.script("propagate.py")),
            "--ann-dir", str(ctx.ann_dir), "--bag", bag,
            "--sam2-repo", str(ctx.sam2_repo), "--model", ctx.model,
            *_extra(ctx, "propagate")]


def _keypoints_cmd(ctx, bag):
    return [sys.executable, str(ctx.script("extract_keypoints.py")),
            "--ann-dir", str(ctx.ann_dir), "--bag", bag, *_extra(ctx, "keypoints")]


def _pose_cmd(ctx, bag):
    return [sys.executable, str(ctx.script("estimate_checkerboard_pose_offline.py")),
            "--ann-dir", str(ctx.ann_dir), "--bag", bag,
            *_board_args(ctx), *_extra(ctx, "pose")]


def _review_cmd(ctx, bag):
    return [sys.executable, str(ctx.script("review_failed_frames.py")),
            "--ann-dir", str(ctx.ann_dir), "--bag", bag, *_extra(ctx, "review")]


def _smooth_cmd(ctx, bag):
    return [sys.executable, str(ctx.script("smooth_poses_se3.py")),
            "--ann-dir", str(ctx.ann_dir), "--bag", bag, *_extra(ctx, "smooth")]


def _repack_mode(ctx, bag) -> str:
    # per-bag override wins over the global default; default "snapshot"
    per_bag = (ctx.bag_repack_modes or {}).get(bag)
    return per_bag or ctx.repack_mode or "snapshot"


def _poses_source(ctx, bag) -> str:
    per_bag = (ctx.bag_poses_sources or {}).get(bag)
    return per_bag or ctx.poses_source or "auto"


def _annotated_dir(ctx, bag):
    # repack_bag.py writes <out>/<bag>_annotated_<mode>
    return ctx.out_dir / f"{bag}_annotated_{_repack_mode(ctx, bag)}" if ctx.out_dir else None


def _repack_cmd(ctx, bag):
    src = ctx.bag_sources.get(bag)
    if not src or not ctx.out_dir:
        return None
    return [sys.executable, str(ctx.script("repack_bag.py")),
            "--bag", src, "--ann-dir", str(ctx.ann_dir),
            "--out-dir", str(ctx.out_dir),
            "--output-mode", _repack_mode(ctx, bag),
            "--poses", _poses_source(ctx, bag), *_extra(ctx, "repack")]


def _npz_cmd(ctx, bag):
    if not ctx.out_dir:
        return None
    annotated = _annotated_dir(ctx, bag)
    if not annotated or not annotated.exists():
        return None
    return [sys.executable, str(ctx.script("generate_npz_from_topics.py")),
            "--input", str(annotated), "--out-dir", str(ctx.out_dir),
            *_extra(ctx, "npz")]


STAGES: list[Stage] = [
    Stage("extract", "shared", "auto", [], _extract_cmd, lambda c, b: _frames(c, b)),
    Stage("seed", "needle", "manual", ["extract"], _seed_cmd,
          lambda c, b: (c.bag_dir(b) / "seeds.json").exists()
          or (c.bag_dir(b) / "reference_seed.json").exists()),
    Stage("propagate", "needle", "auto", ["seed"], _propagate_cmd,
          lambda c, b: (c.bag_dir(b) / "propagation_done").exists()
          and any((c.bag_dir(b) / "masks").glob("*.png")),
          lambda c, b: ((c.bag_dir(b) / "sanity_check.png").exists(),
                        f"QC {c.bag_dir(b) / 'sanity_check.png'}")),
    Stage("keypoints", "needle", "auto", ["propagate"], _keypoints_cmd,
          lambda c, b: (c.bag_dir(b) / "keypoints.json").exists()),
    Stage("pose", "checkerboard", "auto", ["extract"], _pose_cmd,
          lambda c, b: (c.bag_dir(b) / "poses.json").exists(),
          lambda c, b: (_pose_failures(c, b) > 0,
                        f"{_pose_failures(c, b)} detection failure(s) -> review")),
    Stage("review", "checkerboard", "manual", ["pose"], _review_cmd,
          lambda c, b: (c.bag_dir(b) / "poses.json").exists() and _pose_failures(c, b) == 0),
    Stage("smooth", "checkerboard", "auto", ["pose", "review"], _smooth_cmd,
          lambda c, b: (c.bag_dir(b) / "poses_smooth.json").exists()),
    Stage("repack", "output", "auto", ["keypoints", "smooth"], _repack_cmd,
          lambda c, b: bool(c.out_dir) and (_annotated_dir(c, b) is not None)
          and _annotated_dir(c, b).exists()),
    Stage("npz", "output", "auto", ["repack"], _npz_cmd,
          lambda c, b: bool(c.out_dir) and (c.out_dir / f"{b}.npz").exists()),
]
STAGE_BY_NAME = {s.name: s for s in STAGES}


# --------------------------------------------------------------------------
# applicability + status
# --------------------------------------------------------------------------

def applicable(tracks: dict, stage: Stage) -> bool:
    if stage.track in ("shared", "output"):
        return any(tracks.get(t, False) for t in ("needle", "checkerboard"))
    return bool(tracks.get(stage.track, False))


def compute_status(ctx: Ctx, bag: str, tracks: dict) -> dict[str, tuple[str, str]]:
    """Return {stage_name: (status, message)} for one bag."""
    out: dict[str, tuple[str, str]] = {}
    for stage in STAGES:
        if not applicable(tracks, stage):
            out[stage.name] = (NA, "")
            continue
        if stage.is_done(ctx, bag):
            flag, msg = stage.needs_attention(ctx, bag)
            out[stage.name] = (ATTENTION, msg) if flag else (DONE, "")
            continue
        # prerequisites
        unmet = [r for r in stage.requires
                 if applicable(tracks, STAGE_BY_NAME[r])
                 and out.get(r, (WAITING, ""))[0] not in SATISFYING]
        if unmet:
            out[stage.name] = (WAITING, f"needs {', '.join(unmet)}")
            continue
        if stage.build_cmd(ctx, bag) is None:
            out[stage.name] = (BLOCKED, "missing config (path not set)")
            continue
        out[stage.name] = (READY, "")
    return out


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def run_stage(ctx: Ctx, bag: str, stage: Stage, dry_run: bool = False) -> int:
    cmd = stage.build_cmd(ctx, bag)
    if cmd is None:
        return 1
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return 0
    rc = subprocess.run(cmd, cwd=str(ctx.scripts_dir)).returncode
    print(f"  [{'ok' if rc == 0 else f'FAIL rc={rc}'}] {bag}:{stage.name}")
    return rc


def run_bag(ctx: Ctx, bag: str, manifest: Manifest,
            dry_run: bool = False, only: Optional[set[str]] = None) -> dict:
    """Fixed-point: run every ready auto stage until no progress. Returns a
    summary dict with what ran, pending manual gates, blocks, and failures."""
    tracks = manifest.tracks(bag)
    ran: list[str] = []
    failed: Optional[tuple[str, int]] = None
    progressed = True
    while progressed and failed is None:
        progressed = False
        status = compute_status(ctx, bag, tracks)
        for stage in STAGES:
            if only and stage.name not in only:
                continue
            st, _ = status[stage.name]
            if st == READY and stage.kind == "auto":
                rc = run_stage(ctx, bag, stage, dry_run=dry_run)
                manifest.record_stage(bag, stage.name,
                                      status="ok" if rc == 0 else "fail",
                                      rc=rc, ts=time.time())
                manifest.save()
                ran.append(stage.name)
                if rc != 0:
                    failed = (stage.name, rc)
                    break
                if not dry_run:
                    progressed = True   # may unblock downstream
        if dry_run:
            break  # dry-run can't create artifacts, so one pass only

    status = compute_status(ctx, bag, tracks)
    manual = [(s, status[s.name][1]) for s in STAGES
              if status[s.name][0] == READY and s.kind == "manual"]
    blocked = [(s.name, status[s.name][1]) for s in STAGES
               if status[s.name][0] == BLOCKED]
    attention = [(s.name, status[s.name][1]) for s in STAGES
                 if status[s.name][0] == ATTENTION]
    return {
        "ran": ran,
        "failed": failed,
        "manual": [(s.name, ctx_cmd(ctx, bag, s)) for s, _ in manual],
        "blocked": blocked,
        "attention": attention,
        "complete": failed is None and not manual and not blocked
        and all(status[s.name][0] in (DONE, NA) for s in STAGES),
    }


def ctx_cmd(ctx: Ctx, bag: str, stage: Stage) -> str:
    cmd = stage.build_cmd(ctx, bag)
    return " ".join(cmd) if cmd else "(set required paths first)"


def discover_bags(ctx: Ctx, manifest: Manifest) -> list[str]:
    found = set(ctx.bag_sources) | set(manifest.data["bags"])
    if ctx.ann_dir.exists():
        markers = ("frames", "seeds.json", "masks", "poses.json")
        for d in ctx.ann_dir.iterdir():
            if d.is_dir() and any((d / m).exists() for m in markers):
                found.add(d.name)
    return sorted(found)
