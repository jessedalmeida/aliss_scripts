#!/usr/bin/env python3
"""Compute RMS reprojection statistics from poses.json files.

Usage:
  python3 scripts/compute_pose_rms_stats.py --annotations-dir auto-annotating/annotations
  python3 scripts/compute_pose_rms_stats.py --annotations-dir auto-annotating/annotations --scale-covariances 0.6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Dict, List


def gather_poses(ann_root: Path) -> Dict[Path, List[float]]:
    results: Dict[Path, List[float]] = {}
    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations root not found: {ann_root}")
    for sub in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        pj = sub / "poses.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        rms_vals: List[float] = []
        frames = data.get("frames", {})
        for frame in frames.values():
            if frame.get("status") != "ok":
                continue
            r = frame.get("rms_reprojection_error")
            if r is None:
                continue
            try:
                rms_vals.append(float(r))
            except Exception:
                continue
        results[pj] = rms_vals
    return results


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    cnt = len(values)
    mean = statistics.mean(values)
    med = statistics.median(values)
    mn = min(values)
    mx = max(values)
    stdev = statistics.stdev(values) if cnt > 1 else 0.0
    try:
        q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
    except Exception:
        q1, q3 = 0.0, 0.0
    return {
        "count": cnt,
        "mean": mean,
        "median": med,
        "stdev": stdev,
        "min": mn,
        "max": mx,
        "q1": q1,
        "q3": q3,
    }


def scale_covariances_in_file(pj: Path, new_sigma: float, old_sigma: float | None = None) -> None:
    data = json.loads(pj.read_text(encoding="utf-8"))
    param_old = data.get("parameters", {}).get("pixel_noise_sigma")
    if old_sigma is None:
        if param_old is None:
            raise ValueError(f"No old sigma found in {pj}; provide --old-sigma")
        old_sigma = float(param_old)
    scale = (float(new_sigma) / float(old_sigma)) ** 2
    bak = pj.with_suffix(".json.bak")
    if not bak.exists():
        bak.write_bytes(pj.read_bytes())
    changed = False
    for frame in data.get("frames", {}).values():
        pose = frame.get("pose")
        if not pose:
            continue
        cov = pose.get("covariance")
        if not cov or len(cov) != 36:
            continue
        pose["covariance"] = [float(v) * scale for v in cov]
        changed = True
    data.setdefault("parameters", {})["pixel_noise_sigma"] = float(new_sigma)
    if changed:
        pj.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", default="auto-annotating/annotations", help="Annotations root directory")
    parser.add_argument("--scale-covariances", type=float, default=None, help="If set, update covariances to match this new sigma (will backup original files)")
    parser.add_argument("--old-sigma", type=float, default=None, help="If poses.json lacks parameters.pixel_noise_sigma, supply the original sigma here for scaling")
    args = parser.parse_args()

    root = Path(args.annotations_dir)
    per_file = gather_poses(root)

    overall: List[float] = []
    print("Per-bag RMS reprojection statistics:")
    for pj, vals in per_file.items():
        s = summarize(vals)
        overall.extend(vals)
        try:
            rel = pj.relative_to(Path.cwd())
        except ValueError:
            rel = pj
        if s.get("count", 0) == 0:
            print(f" - {rel}: no valid RMS values")
            continue
        print(f" - {rel}: n={s['count']} mean={s['mean']:.3f}px median={s['median']:.3f}px stdev={s['stdev']:.3f}px min={s['min']:.3f}px max={s['max']:.3f}px")

    print("\nOverall stats:")
    osum = summarize(overall)
    if osum.get("count", 0) == 0:
        print(" No RMS values found in annotations")
        return 0
    print(f" - total_frames={osum['count']} mean={osum['mean']:.3f}px median={osum['median']:.3f}px stdev={osum['stdev']:.3f}px min={osum['min']:.3f}px max={osum['max']:.3f}px")
    print(f" - suggested pixel_noise_sigma ~= median = {osum['median']:.3f} px")

    if args.scale_covariances is not None:
        new_sigma = float(args.scale_covariances)
        print(f"\nScaling covariances to new sigma {new_sigma}")
        for pj in per_file.keys():
            try:
                scale_covariances_in_file(pj, new_sigma, args.old_sigma)
                print(f" - updated {pj}")
            except Exception as e:
                print(f" - failed {pj}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
