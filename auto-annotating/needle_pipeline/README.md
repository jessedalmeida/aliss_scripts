# Needle annotation — pipeline + GUI

A track-aware orchestrator and a local web console over your existing
annotation scripts. Heavy stages (SAM2 propagation, pose estimation, smoothing,
repack, NPZ) run as background subprocess jobs; interactive fixes (mask paint,
keypoint drag, checkerboard corner-fix) happen in the browser. Everything reads
and writes the same on-disk files your scripts already use, so the CLI and the
GUI stay fully interoperable.

## Layout

Drop this beside your existing scripts (the repo root with `seed_annotator.py`,
`propagate.py`, `estimate_checkerboard_pose_offline.py`, etc.):

```
repo/
├── needle_pipeline/        # the new package
│   ├── config.py           # manifest + per-bag tracks + board/camera/model settings
│   ├── extract.py          # NEW headless frame extraction (decoupled from seeding)
│   ├── stages.py           # the track-aware stage DAG + auto runner
│   ├── pose_ops.py         # recompute pose from manually-fixed corners
│   ├── server.py           # FastAPI backend
│   ├── __main__.py         # CLI  (python -m needle_pipeline ...)
│   └── static/index.html   # the GUI
├── correct_poses.py        # NEW: the corner-picker review_failed_frames.py imported (was missing)
└── seed_annotator.py, propagate.py, ...   # your existing scripts (unchanged)
```

## Install

```bash
pip install fastapi uvicorn          # GUI server (the box already has torch/cv2/rosbags)
```

## The two tracks

Each bag has a **needle** track and a **checkerboard** track, independently
toggleable; both default on. Stages in a disabled track show as `--` (n/a) and
never block downstream work.

```
extract ─┬─ seed ─ propagate ─ keypoints ─────────┐
         │                                          ├─ repack ─ npz
         └─ pose ─┬─ review ┐                       │
                  └─────────┴─ smooth ──────────────┘
```

`seed` and `review` are the only manual gates; `review` blocks only when
`poses.json` actually has failed frames.

## CLI

```bash
# First run: register paths (saved to annotations/manifest.json)
python -m needle_pipeline status --ann-dir ./annotations \
    --sam2-repo /path/to/sam2 --out-dir ./annotated_bags \
    --model large --every-n 3 \
    --bag-source suture1=/data/bags/suture1

# Worklist
python -m needle_pipeline status --ann-dir ./annotations

# Run the autonomous chain for all bags (stops at manual gates)
python -m needle_pipeline run --ann-dir ./annotations

# One stage / one bag
python -m needle_pipeline run --ann-dir ./annotations --bag suture1 --stage propagate

# Per-bag track control
python -m needle_pipeline track --ann-dir ./annotations --bag suture1 --checkerboard off
```

Status glyphs: `OK` done · `!!` attention (QC / failures) · `>>` ready ·
`XX` blocked (missing path) · `..` waiting · `--` n/a.

## GUI

```bash
python -m needle_pipeline.server --ann-dir ./annotations --scripts-dir .
# open http://localhost:8000   (remote box: ssh -L 8000:localhost:8000 user@host)
```

- **Left:** bags with track badges (click **N**/**C** to toggle a track) + a live stage strip.
- **Stage rail:** click a ready auto stage to launch it as a background job;
  watch logs in the **Jobs** panel (the worklist refreshes when a job ends).
- **Zoom/pan:** scroll to zoom toward the cursor, `space+drag` (or middle-drag)
  to pan, and the `+ / – / ⤢` controls bottom-right (⤢ = fit). Zoom persists
  while you scrub, so you can work zoomed in on the needle or the board.
- **Modes:**
  - *Seed* — place SAM2 prompts: left-click = positive (needle), right-click or
    `shift` = negative (background); seed as many frames as you like. **Save
    seeds** writes `seeds.json`; **save + run SAM2** also launches `propagate`.
    **borrow from bag ⇄** opens the cross-bag teacher picker (see below).
    (SAM2 itself runs as the propagate job — the browser only places prompts.)
  - *Mask* — paint/erase brush, clear, copy-from-previous, save, for fixing up
    propagated masks.
  - *Keypts* — click places tip then tail; drag to move. **suggest ⌖** fills
    tip/tail for the current frame from its mask (skeleton endpoints; a tip you've
    already placed disambiguates which end is which); **suggest all** does every
    masked frame at once. **track ↔** runs Lucas-Kanade tracking across all frames
    from the current frame; **re-anchor →** re-tracks forward only from a corrected
    frame (leaving earlier frames intact); occlusion is **sticky** — marking tip or
    tail occluded persists onto the following frames (auto-saved) until toggled back
    off, so a long occlusion only needs one click; save.
  - *Corners* — the detected checkerboard is drawn as a numbered grid; drag any
    drifted corner to fix it (or **reset to detected**), **reflow from neighbor ↺**
    re-seeds the corners by optical-flow from the nearest *good* frame when this
    frame's detection is bad (it loads the flowed corners for review — adjust, then
    **re-solve pose** to commit), and **re-solve pose** recomputes the pose
    (covariance + RMS). Click to place corners from scratch if none were detected.
  - *View* — read-only; overlays mask, keypoints, and the checkerboard grid plus
    its **solved coordinate frame** (X=red, Y=green, Z=blue axes drawn from the
    board origin), with pose status + RMS in the top-right pill.

### Playback

The scrubber has a **▶ play** button (or press `p`) that runs through the video
and redraws each frame with all current overlays — masks, keypoints, and the
checkerboard pose axes. Pick a speed (4–30 fps); frames and axes are prefetched
just ahead of the playhead so it stays smooth. Playback pauses automatically if
you step, scrub, or switch bags. Use **View** mode to watch all overlays at once.

The pose axes are reprojected server-side from each frame's solved pose using the
camera intrinsics (preferring the stored corners for an exact solve, falling back
to the stored quaternion), so what you see is the actual `poses.json` pose, not an
approximation.

### Keyboard shortcuts

Press `?` in the app for an in-GUI cheatsheet. Shortcuts are ignored while typing in
a field, and Ctrl/Cmd/Alt combos pass through to the browser (so Cmd-S etc. behave
normally).

Global:

| Key | Action |
| --- | --- |
| `a` / `←` | previous frame |
| `d` / `→` | next frame |
| `q` / `e` | jump back / forward 10 frames |
| `s` | save (context-aware: mask / keypoints / seeds, or re-solve pose in Corners) |
| `p` | play / pause |
| `f` | fit image to view |
| `[` / `]` | zoom out / in |
| `space` + drag | pan (scroll wheel also zooms to cursor) |
| `1` … `5` | switch mode: View / Seed / Mask / Keypts / Corners |
| `?` | toggle the shortcut cheatsheet (`Esc` closes it) |

Keypoints mode:

| Key | Action |
| --- | --- |
| `t` | track ↔ (Lucas-Kanade both directions) |
| `r` | re-anchor → (re-track forward from this frame) |
| `g` | suggest tip/tail for this frame from its mask |
| `z` / `x` | toggle occlusion on tip / tail |

Mask mode:

| Key | Action |
| --- | --- |
| `e` | toggle paint / erase |
| `c` | clear the mask |
| `v` | copy the previous frame's mask |

Corners mode:

| Key | Action |
| --- | --- |
| `r` | reset corners to detected |
| `u` | undo last placed corner |
| `s` | re-solve pose from current corners |

The mode-specific keys (`r`, `e`, etc.) only act when that mode is active, so they
don't collide with the global jumps — e.g. `e` toggles erase in Mask mode but jumps
forward 10 frames everywhere else.

### Annotating several similar bags from one (cross-bag seeding)

If you have several near-identical bags, you can annotate one and let the rest
borrow it instead of clicking each. In **Seed** mode, **borrow from bag ⇄**:

1. Pick a *source* bag and the *frame* whose clicks you want to teach from — the
   preview shows that frame with its + / − clicks so you can confirm the needle
   looks like the target's opening.
2. Check which bags to **apply to** (the current bag is pre-checked; tick others
   to fan one teacher out to many at once).
3. **apply reference** writes `reference_seed.json` into each target.

When a bag has a borrowed reference, its `propagate` stage automatically uses the
cross-bag path (`needle_pipeline.crossprop`): it splices the teacher frame in
front of the target's frames, places the clicks on it, and lets SAM2 carry the
learned needle *appearance* into the target — so it works even when the needle
sits in a different position than the source. Run propagation as usual (stage
rail or `run`). **clear this bag's reference** removes it (the bag falls back to
its own seeds). Always eyeball the target's `sanity_check.png` afterward: tracking
quality across the splice depends on how similar the videos actually are.

## Fixes folded into the refactor

- **Headless `extract`** split out from `seed_annotator.py` (which fused
  extraction with the click UI), with the displaced-`return` decode bug fixed —
  extraction now runs unattended.
- **`correct_poses.py`** supplied so `review_failed_frames.py` runs (it imported
  a module that wasn't in the set).
- **One canonical `keypoints.json` schema** owned by a single editor instead of
  three tools writing slightly different shapes.
- Board params, camera YAML, SAM2 model, and frame stride all live in the
  manifest `config` and flow to the right stages.

## Repack output modes & tool-tip pixels

The `repack` stage can write either **snapshot** mode (one `/needle_tracking/snapshot`
message bundling everything per frame) or **topics** mode (camera stream kept, each
annotation component on its own topic — good for Foxglove). Toggle it per bag in the
sidebar (the `repack: snapshot|topics` pill), or set a global default with
`--stage-arg` / the manifest. The annotated bag is written to
`<out>/<bag>_annotated_<mode>` and the NPZ stage reads that mode-specific directory.

### Smoothed vs raw poses

The **poses ⟿** button (top toolbar) opens trajectory plots for the selected bag:
X/Y/Z checkerboard position vs frame, raw (amber) overlaid with smoothed (cyan),
plus a smoothing-deviation strip and mean/max shift stats. From that panel — or the
sidebar `poses: auto|smooth|raw` pill — you choose what `repack` packs:

- **auto** — smoothed when `poses_smooth.json` exists, else raw (the old behavior).
- **smooth** — force smoothed (falls back to raw with a warning if not yet smoothed).
- **raw** — force the unsmoothed `poses.json`.

The choice is per bag (persisted in the manifest) and threads to `repack_bag.py`'s
new `--poses {auto,smooth,raw}` flag.

The same panel has tools to fix and re-smooth poses without leaving the GUI:

- **recompute all ↻** — re-solves every frame's pose from its stored corners using
  the current camera YAML and the **square (m)** field. This is how you fix a wrong
  `ves_camera.yaml` *or* a wrong `square_size` after the fact — the corners are kept,
  only the metric solve changes. It writes `poses_recomputed.json` for review (the
  plots refresh to show it) and does **not** touch `poses.json` until you press
  **commit ✓**, which backs the old file up to `poses_prev.json` first.
- **re-smooth ∿** — re-runs SE(3) smoothing on the current `poses.json` (so you no
  longer delete `poses_smooth.json` by hand). **Z-downweight** (>1) tells the
  smoother to trust the depth axis less and lean on the temporal prior — useful when
  the board is small in-frame and Z is noisy, though its effect plateaus quickly
  because the smoother is already covariance-weighted.

In **Corners** mode, **mark no board** tags a frame as intentionally having no
checkerboard (`status="no_board"`). Such frames don't count as detection failures,
don't block `review`, and are simply skipped by repack — so a clip where the board
leaves view won't stall the pipeline. Click again to clear the mark.

Note on small boards: if the checkerboard is only ~10–15% of the frame, depth (Z)
is inherently ill-conditioned — a fraction of a pixel of corner noise moves Z by
millimeters even at sub-pixel RMS. Smoothing roughly halves the residual jitter but
can't remove it; the real levers are a larger board or closer capture. Getting the
`square_size` right (via recompute) is the bigger correctness fix, since a wrong
square size rescales every translation.

If the source bag has `ves_smoother/{left,right}/tool_tip_pixels` topics
(`PoseWithCovarianceStamped` with pixel x/y in `position`), those pixel points are
packed into the keypoints message's `left_arm_tip` / `right_arm_tip` fields. They
take precedence over the 3D tip-pose topics when present, and fall back to the old
behavior when absent. (Requires the patched `repack_bag.py` below.)

## Patched legacy scripts (replace your originals)

Three of your scripts are shipped here with fixes — drop them in over the originals:

- **`repack_bag.py`** — adds tool-tip-pixel packing and keeps the snapshot/topics
  modes wired to the orchestrator's per-bag setting.
- **`propagate.py`** — the explicit seed-frame click mask is now *authoritative* and
  is never overwritten by another seed's propagation (fixes the brief wrong-mask
  flash on seeded frames), plus per-seed-frame provenance logging so any remaining
  discontinuity prints its source.
- **`quick_compare.py`** — fixed the undefined variable and wrong arguments.

Still external: `smooth_poses_se3.py --compare` shells out to `compare_poses.py`
(you provided it).
