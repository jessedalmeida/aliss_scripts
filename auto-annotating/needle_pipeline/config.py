"""
needle_pipeline.config - manifest + run context.

The manifest (annotations/manifest.json) is the single place paths and
per-bag settings live, so the CLI and the GUI share one source of truth.

Schema
------
{
  "config": {
    "out_dir": "...", "sam2_repo": "...", "model": "large", "every_n": 3,
    "camera_yaml": null,
    "board": {"squares_x": 4, "squares_y": 5, "square_size": 0.002},
    "bag_sources": {"<stem>": "/path/to/original/bag"},
    "stage_args": {"<stage>": ["--flag", "value"]}
  },
  "bags": {
    "<stem>": {
      "tracks": {"needle": true, "checkerboard": true},
      "stages": {"<stage>": {"status": "ok", "rc": 0, "ts": 1234567890.0}}
    }
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "manifest.json"

DEFAULT_BOARD = {"squares_x": 4, "squares_y": 5, "square_size": 0.002}
DEFAULT_MODEL = "large"
DEFAULT_EVERY_N = 3
TRACK_NAMES = ("needle", "checkerboard")


@dataclass
class Ctx:
    """Resolved run context: where everything lives + global settings."""
    ann_dir: Path
    scripts_dir: Path                 # where the legacy stage scripts live
    out_dir: Optional[Path] = None
    sam2_repo: Optional[Path] = None
    model: str = DEFAULT_MODEL
    every_n: int = DEFAULT_EVERY_N
    camera_yaml: Optional[Path] = None
    board: dict = field(default_factory=lambda: dict(DEFAULT_BOARD))
    bag_sources: dict[str, str] = field(default_factory=dict)
    stage_args: dict[str, list[str]] = field(default_factory=dict)
    repack_mode: str = "snapshot"                       # global default: snapshot | topics
    bag_repack_modes: dict[str, str] = field(default_factory=dict)  # per-bag override
    poses_source: str = "auto"                          # global default: auto | smooth | raw
    bag_poses_sources: dict[str, str] = field(default_factory=dict)  # per-bag override

    def script(self, name: str) -> Path:
        return self.scripts_dir / name

    def bag_dir(self, bag: str) -> Path:
        return self.ann_dir / bag


class Manifest:
    """Thin wrapper around manifest.json with load/save and per-bag access."""

    def __init__(self, ann_dir: Path):
        self.ann_dir = Path(ann_dir)
        self.path = self.ann_dir / MANIFEST_NAME
        self.data: dict = {"config": {}, "bags": {}}
        self.load()

    def load(self) -> "Manifest":
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                pass
        self.data.setdefault("config", {})
        self.data.setdefault("bags", {})
        return self

    def save(self) -> None:
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    # -- per-bag -----------------------------------------------------------
    def bag(self, stem: str) -> dict:
        b = self.data["bags"].setdefault(stem, {})
        b.setdefault("tracks", {"needle": True, "checkerboard": True})
        b.setdefault("stages", {})
        return b

    def tracks(self, stem: str) -> dict:
        return self.bag(stem)["tracks"]

    def set_track(self, stem: str, track: str, on: bool) -> None:
        if track not in TRACK_NAMES:
            raise ValueError(f"unknown track {track!r}")
        self.bag(stem)["tracks"][track] = bool(on)

    def set_repack_mode(self, stem: str, mode: str) -> None:
        if mode not in ("snapshot", "topics"):
            raise ValueError(f"unknown repack mode {mode!r}")
        self.bag(stem)["repack_mode"] = mode

    def set_poses_source(self, stem: str, source: str) -> None:
        if source not in ("auto", "smooth", "raw"):
            raise ValueError(f"unknown poses source {source!r}")
        self.bag(stem)["poses_source"] = source

    def record_stage(self, stem: str, stage: str, **fields) -> None:
        self.bag(stem)["stages"][stage] = fields

    # -- context -----------------------------------------------------------
    def to_ctx(self, scripts_dir: Path) -> Ctx:
        c = self.data["config"]
        per_bag_modes = {stem: b["repack_mode"] for stem, b in self.data["bags"].items()
                         if isinstance(b, dict) and b.get("repack_mode")}
        per_bag_poses = {stem: b["poses_source"] for stem, b in self.data["bags"].items()
                         if isinstance(b, dict) and b.get("poses_source")}
        return Ctx(
            ann_dir=self.ann_dir,
            scripts_dir=Path(scripts_dir),
            out_dir=Path(c["out_dir"]) if c.get("out_dir") else None,
            sam2_repo=Path(c["sam2_repo"]) if c.get("sam2_repo") else None,
            model=c.get("model", DEFAULT_MODEL),
            every_n=int(c.get("every_n", DEFAULT_EVERY_N)),
            camera_yaml=Path(c["camera_yaml"]) if c.get("camera_yaml") else None,
            board=dict(c.get("board", DEFAULT_BOARD)),
            bag_sources=dict(c.get("bag_sources", {})),
            stage_args=dict(c.get("stage_args", {})),
            repack_mode=c.get("repack_mode", "snapshot"),
            bag_repack_modes=per_bag_modes,
            poses_source=c.get("poses_source", "auto"),
            bag_poses_sources=per_bag_poses,
        )

    def update_config(self, ctx: Ctx) -> None:
        self.data["config"] = {
            "out_dir": str(ctx.out_dir) if ctx.out_dir else None,
            "sam2_repo": str(ctx.sam2_repo) if ctx.sam2_repo else None,
            "model": ctx.model,
            "every_n": ctx.every_n,
            "camera_yaml": str(ctx.camera_yaml) if ctx.camera_yaml else None,
            "board": ctx.board,
            "bag_sources": ctx.bag_sources,
            "stage_args": ctx.stage_args,
            "repack_mode": ctx.repack_mode,
            "poses_source": ctx.poses_source,
        }
