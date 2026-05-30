"""Needle-annotation pipeline: track-aware orchestration + GUI backend."""
from .config import Manifest, Ctx
from .stages import (
    STAGES, STAGE_BY_NAME, compute_status, run_bag, discover_bags, applicable,
    DONE, ATTENTION, READY, BLOCKED, WAITING, NA,
)

__all__ = [
    "Manifest", "Ctx", "STAGES", "STAGE_BY_NAME", "compute_status", "run_bag",
    "discover_bags", "applicable",
    "DONE", "ATTENTION", "READY", "BLOCKED", "WAITING", "NA",
]
