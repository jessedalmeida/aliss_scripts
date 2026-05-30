#!/usr/bin/env python3
"""
needle_pipeline CLI - status / run / track / config.

Run from the repo root (where the legacy stage scripts live):

    python -m needle_pipeline status --ann-dir ./annotations
    python -m needle_pipeline run    --ann-dir ./annotations
    python -m needle_pipeline run    --ann-dir ./annotations --bag suture1 --stage propagate
    python -m needle_pipeline track  --ann-dir ./annotations --bag suture1 --checkerboard off

First run, register paths (saved to the manifest):

    python -m needle_pipeline status --ann-dir ./annotations \
        --sam2-repo /path/to/sam2 --out-dir ./annotated_bags \
        --bag-source suture1=/data/bags/suture1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import stages as S
from .config import Manifest

COLOR = {
    S.DONE: "\033[32m", S.ATTENTION: "\033[33m", S.READY: "\033[36m",
    S.BLOCKED: "\033[31m", S.WAITING: "\033[90m", S.NA: "\033[90m",
}
GLYPH = {S.DONE: "OK", S.ATTENTION: "!!", S.READY: ">>",
         S.BLOCKED: "XX", S.WAITING: "..", S.NA: "--"}
RESET = "\033[0m"


def paint(status: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{COLOR.get(status, '')}{text}{RESET}"


def _overlay_config(args, m: Manifest) -> None:
    ctx = m.to_ctx(_scripts_dir(args))
    if getattr(args, "sam2_repo", None):
        ctx.sam2_repo = Path(args.sam2_repo)
    if getattr(args, "out_dir", None):
        ctx.out_dir = Path(args.out_dir)
    if getattr(args, "model", None):
        ctx.model = args.model
    if getattr(args, "every_n", None):
        ctx.every_n = args.every_n
    if getattr(args, "camera_yaml", None):
        ctx.camera_yaml = Path(args.camera_yaml)
    for bag_dir in getattr(args, "bag_source_dir", None) or []:
        root = Path(bag_dir).expanduser().resolve()
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix in {".bag", ".mcap"}:
                ctx.bag_sources[p.stem] = str(p)
            elif p.is_dir():
                # for rosbag dirs
                ctx.bag_sources[p.name] = str(p)
    for item in getattr(args, "bag_source", None) or []:
        if "=" in item:
            stem, path = item.split("=", 1)
            ctx.bag_sources[stem.strip()] = path.strip()
    for item in getattr(args, "stage_arg", None) or []:
        if "=" in item:
            stage, raw = item.split("=", 1)
            ctx.stage_args[stage.strip()] = raw.split()
    m.update_config(ctx)
    m.save()


def _scripts_dir(args) -> Path:
    if getattr(args, "scripts_dir", None):
        return Path(args.scripts_dir).resolve()
    # default: legacy scripts live one level up from this package
    return Path(__file__).resolve().parent.parent


def _load(args) -> tuple[Manifest, "S.Ctx"]:
    ann = Path(args.ann_dir).resolve()
    ann.mkdir(parents=True, exist_ok=True)
    m = Manifest(ann)
    _overlay_config(args, m)
    return m, m.to_ctx(_scripts_dir(args))


def cmd_status(args) -> int:
    m, ctx = _load(args)
    bags = [args.bag] if args.bag else S.discover_bags(ctx, m)
    if not bags:
        print("No bags yet. Register one: --bag-source STEM=/path/to/bag")
        return 0
    w = max(len(b) for b in bags)
    worklist: list[str] = []
    for bag in bags:
        tracks = m.tracks(bag)
        status = S.compute_status(ctx, bag, tracks)
        cells = [paint(status[s.name][0], f"{GLYPH[status[s.name][0]]} {s.name}")
                 for s in S.STAGES]
        tflag = "".join(t[0] for t in ("needle", "checkerboard") if tracks.get(t))
        print(f"{bag.ljust(w)} [{tflag or '-':2}]  " + "  ".join(cells))
        for s in S.STAGES:
            st, msg = status[s.name]
            if st in (S.READY, S.BLOCKED, S.ATTENTION) and not args.brief:
                line = f"  {bag}:{s.name} [{st}]" + (f" - {msg}" if msg else "")
                worklist.append(paint(st, line))
    m.save()
    if worklist and not args.brief:
        print("\nWorklist:")
        print("\n".join(worklist))
    return 0


def cmd_run(args) -> int:
    m, ctx = _load(args)
    if args.stage and args.stage not in S.STAGE_BY_NAME:
        print(f"[ERROR] unknown stage. Choices: {', '.join(S.STAGE_BY_NAME)}")
        return 2
    bags = [args.bag] if args.bag else S.discover_bags(ctx, m)
    if not bags:
        print("No bags found.")
        return 1
    only = {args.stage} if args.stage else None
    rc = 0
    for bag in bags:
        print(f"\n=== {bag} ===")
        r = S.run_bag(ctx, bag, m, dry_run=args.dry_run, only=only)
        if r["ran"]:
            print(f"  ran: {', '.join(r['ran'])}")
        for name, msg in r["attention"]:
            print(paint(S.ATTENTION, f"  attention: {name} - {msg}"))
        for name, cmd in r["manual"]:
            print(paint(S.READY, f"  manual gate '{name}':\n      {cmd}"))
        for name, msg in r["blocked"]:
            print(paint(S.BLOCKED, f"  blocked: {name} - {msg}"))
        if r["failed"]:
            sname, frc = r["failed"]
            print(paint(S.BLOCKED, f"  FAILED at '{sname}' (rc={frc})"))
            rc = 1
            if not args.keep_going:
                print("  stopping (use --keep-going to continue)")
                break
        elif r["complete"]:
            print(paint(S.DONE, "  complete"))
    m.save()
    return rc


def cmd_track(args) -> int:
    m, _ = _load(args)
    if not args.bag:
        print("[ERROR] --bag is required for 'track'")
        return 2
    for track, val in (("needle", args.needle), ("checkerboard", args.checkerboard)):
        if val is not None:
            m.set_track(args.bag, track, val == "on")
            print(f"{args.bag}: {track} -> {val}")
    m.save()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="needle_pipeline",
                                description="Track-aware needle-annotation orchestrator")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--ann-dir", required=True)
        sp.add_argument("--scripts-dir", default=None)
        sp.add_argument("--bag", default=None)
        sp.add_argument("--sam2-repo", default=None)
        sp.add_argument("--out-dir", default=None)
        sp.add_argument("--model", default=None,
                        choices=["tiny", "small", "base", "large"])
        sp.add_argument("--bag-source-dir", action="append",
                help="Directory containing many bag files/directories; names use file/folder stem")
        sp.add_argument("--every-n", type=int, default=None)
        sp.add_argument("--camera-yaml", default=None)
        sp.add_argument("--bag-source", action="append", metavar="STEM=PATH")
        sp.add_argument("--stage-arg", action="append", metavar="STAGE='--flag v'")

    sp = sub.add_parser("status"); common(sp)
    sp.add_argument("--brief", action="store_true"); sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("run"); common(sp)
    sp.add_argument("--stage", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--keep-going", action="store_true"); sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("track"); common(sp)
    sp.add_argument("--needle", choices=["on", "off"], default=None)
    sp.add_argument("--checkerboard", choices=["on", "off"], default=None)
    sp.set_defaults(func=cmd_track)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
