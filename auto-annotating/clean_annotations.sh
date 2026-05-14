#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 <bag_dir>            # clean a single bag folder"
  echo "  $0 --all <ann_root>     # clean all bags under annotation root"
  exit 1
}

if [[ $# -lt 1 ]]; then usage; fi

if [[ "$1" == "--all" ]]; then
  ann_root="${2:-./annotations}"
  echo "Cleaning all bags under: $ann_root"
  # remove seeds.json
  find "$ann_root" -maxdepth 2 -type f -name 'seeds.json' -print -exec rm -v {} \;
  # remove propagation_done files
  find "$ann_root" -type f -name 'propagation_done' -print -exec rm -v {} \;
  # remove masks directories
  find "$ann_root" -type d -name 'masks' -print -exec rm -rv {} \;
  # remove frames jpg
  find "$ann_root" -type d -name 'frames_jpg' -print -exec rm -rv {} \;
else
  bag_dir="$1"
  echo "Cleaning bag: $bag_dir"
  rm -v "${bag_dir}/seeds.json" 2>/dev/null || true
  rm -v "${bag_dir}/propagation_done" 2>/dev/null || true
  rm -rv "${bag_dir}/masks" 2>/dev/null || true
  rm -rv "${bag_dir}/frames_jpg" 2>/dev/null || true
fi