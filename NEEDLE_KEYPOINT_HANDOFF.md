# Handoff — Needle keypoint head (mask-conditioned) on top of YOLO segmentation

This document hands off an in-progress ML pipeline so a coding agent can continue
**directly in the repo** (`~/aliss_core/src/aliss_scripts`, training code under
`training/`). Read it fully before changing code. It captures the goal, the data
quirks (several are landmines), the files already built and their verification
status, the decisions already made (with rationale, so you don't relitigate
them), and the concrete next steps.

A core working principle throughout this project, inherited from the repo's
`CLAUDE.md`: **state assumptions, surface tradeoffs, don't silently pick; make
surgical changes; and define a verifiable success gate for every step and check
it before moving on.** Every module below was built with a paired verification
script and a pass/fail gate run on real data. Keep doing that.

---

## 1. Objective and the pivot

**Original goal:** train a from-scratch multi-task net (shared encoder → needle
mask head + tip/tail keypoint heatmap head) for real-time use.

**The pivot (current objective):** a **YOLO11m-seg** model is already trained and
produces good needle segmentation most (not all) of the time. So we are **no
longer training our own segmentation as the deliverable**. The job now is a
**learned keypoint head that detects needle tip and tail (with occlusion/visibility
flags), conditioned on the YOLO mask** — i.e. the model takes RGB **plus the YOLO
mask** as input and predicts tip/tail heatmaps + per-keypoint visibility.

Why a learned head rather than reading keypoints off the mask geometry: the
keypoints do **not** reliably sit at mask endpoints, and the mask fragments under
occlusion (see §3). A learned, mask-conditioned head can place tip/tail correctly
even when the mask is broken or the point is buried, and can flag occlusion. This
was verified visually on real occlusion frames (the mask split into disconnected
slivers while the true tip/tail were elsewhere).

**Deployment target:** single RTX 4070 Mobile (Max-Q) workstation, ~8 GB VRAM.
Real-time bar is a loose **10–15 FPS**. Not edge-constrained.

---

## 2. Hardware, image, and constraints

- GPU: **RTX 4070 Laptop / Max-Q**, ~8 GB. AMP/half precision works and is the
  deployment precision. The Step-2 throughput probe measured **163 FPS at 640²**
  for the equivalent-cost ResNet-34 U-Net — i.e. compute is NOT the binding
  constraint; we have ~10× headroom over the FPS bar.
- Images: **1080×1080**, endoscopic, grainy/noisy, strong lens distortion
  (plumb_bob k1 ≈ −0.376), **circular field of view** with black corners.
- Needle: thin but not razor-thin — median mask width ≈ **10.3 px** at 1080²,
  p10 ≈ 8.5 px. It survives downscaling to 640 with no loss (this drove the
  resolution decision).
- **Input resolution is locked at 640×640** (see §6, Decision D2). Heatmaps are
  produced at **stride 4 → 160×160**.

---

## 3. Data — layout, inventory, and the quirks that matter

### 3.1 Layout
Per bag: `auto-annotating/annotations/<bag>/` contains
- `poses.json` — checkerboard pose per frame. **Annotation tooling only — NOT a
  model target.** Used only to derive a bag's board/no-board domain.
- `keypoints.json` — `{frames: {"NNNNNN": {needle_tip:[x,y], needle_tail:[x,y],
  occluded:{...}, status, method, left_arm_tip, right_arm_tip}}}`. Pixel coords
  in native 1080 space.
- `masks/` — SAM2 ground-truth needle masks, `frame_NNNNNN_needle_mask.png`.
- `yolo_mask/` — **NEW, to be populated by the user** with YOLO **predicted**
  masks (any `*.png` whose filename contains the frame index; the exporter
  parses the index out). This is the conditioning input for the keypoint head.
- `frames/` — `frame_NNNNNN.jpg` (a few alternate layouts are handled by
  `resolve_image` in the exporter).

### 3.2 Inventory and the domain split
- Exported so far: **32 bags, 10,618 frames (29 board, 3 no-board; 9,688 board /
  930 no-board frames).** The full annotated set is larger (~42 board + ~13
  no-board exist; +10 board, +8 no-board the user can still annotate). Not all
  bags are exported yet.
- **No-board is the DEPLOYMENT DOMAIN** (the model ships on checkerboard-free
  scenes). It is the scarce, important data. **The single highest-leverage data
  action is annotating more no-board bags** — it gates whether a credible
  bag-level val/test split is even possible.
- Board bags contain a checkerboard AND a **green 3D-printed piece at the needle
  tail**. This is a shortcut hazard: a head could learn "green = tail." Mitigated
  by hue/saturation jitter (already in augmentation) + keeping board frames a
  minority via the sampler. The needle mask ends at the metal/board interface, so
  the green is NOT in the mask — the shortcut risk lives only in the image
  channel for the keypoint head, not the mask.
- `has_board` is a **bag-level** property. Auto-detected from `poses.json`
  (any frame with status `ok`/`high_rms` ⇒ board bag), overridable via
  `--no-board-bags` / `--board-bags`. Prefer passing the no-board list explicitly.

### 3.3 LANDMINE: occlusion key inconsistency in `keypoints.json`
The `occluded` dict uses **two key conventions** that coexist and sometimes
**disagree**:
- older short keys: `tip`, `tail`
- newer long keys: `needle_tip`, `needle_tail` (a correction pass, NOT applied
  everywhere)

**Resolution decision (confirmed with the user): the long key wins when present;
the short key is the fallback where the long form is absent.** Implemented in
`export_dataset.py :: _resolve_occlusion`. On `ch_chicken_1` this matters: 7
frames disagree and must follow `needle_tip`.

**This bug previously corrupted a dataset-wide count.** An earlier breakdown that
read only the long key reported ~10,221/13,770 tip frames as `occluded: None`
(unlabeled), which nearly drove a bad design decision (discarding most keypoint
data). With correct key resolution, far more frames are genuinely labeled.
**ACTION: re-run the label breakdown across all bags with the fixed normalizer**
before trusting any "how much is labeled" number:
```bash
python3 -c "
import json, glob, sys; sys.path.insert(0,'training')
from collections import Counter
from export_dataset import normalize_keypoints
tip=Counter(); tail=Counter()
for f in glob.glob('auto-annotating/annotations/*/keypoints.json'):
    for fr in json.load(open(f)).get('frames',{}).values():
        n=normalize_keypoints(fr); tip[n['needle_tip']['state']]+=1; tail[n['needle_tail']['state']]+=1
print('tip :', dict(tip)); print('tail:', dict(tail))
"
```
Note: the **annotation tooling is the root cause** (writing inconsistent keys, and
occasionally contradictory values). Worth fixing at the source so this stops
recurring; the normalizer defends against it regardless.

### 3.4 Occlusion / visibility convention (confirmed with user)
Three per-keypoint states, emitted by `normalize_keypoints`:
- **`visible`** — populated `xy` AND `occluded` resolves to **False** →
  supervise BOTH the heatmap (render Gaussian at `xy`) AND visibility (target =
  visible).
- **`occluded`** — `occluded` resolves to **True** → supervise visibility ONLY
  (target = not-visible); **mask the heatmap loss**; the `xy` of an occluded
  point is **NOT trustworthy and must be ignored** (do NOT try to predict the
  hidden location — the user explicitly rejected that). The exporter sets
  `xy=None` for occluded points for this reason.
- **`unlabeled`** — anything else (occlusion None/missing, or null `xy`) →
  supervise nothing; drop the point.

On `ch_chicken_1` (233 frames) after correct resolution: tip 133 visible / 100
occluded; tail 233 visible. So occlusion has a **real training signal** now
(the visibility head is not decorative).

### 3.5 LANDMINE: mask fragmentation under occlusion
When the needle is buried/occluded, the SAM2 mask breaks into **disconnected
fragments**, and the true tip/tail are not at any fragment endpoint (verified on
frames 62–67). Consequence: skeleton-endpoint derivation of keypoints fails
exactly here; this is the core justification for the learned head. **The held-out
TEST set MUST include these occlusion / fragmented-mask frames**, or eval will not
measure the capability that justifies the model.

### 3.6 tip vs tail — the ONLY differentiator is the thread
Tip = sharp point; tail = swaged end **with the suture thread next to it**. That
thread is the sole appearance cue distinguishing the two. This makes tip/tail
**identity fragile → swap risk** (see §6 Decision D5). Separate `tip` and `tail`
heatmap channels force the head to learn the appearance distinction; full-rotation
augmentation prevents it from cheating via orientation.

---

## 4. The pipeline — files, status, what each produces

All modules live in (or should be copied to) `training/`. They import each other
by bare module name, so run from `training/` or have it on `sys.path`.

**IMPORTANT — paths in generated data files are NOT portable.** Any
`manifest.jsonl` / `splits.json` produced in a different environment contains
absolute paths from that environment. **Always regenerate them in-repo** with the
exporter. The `.py` modules are portable; the `.json`/`.jsonl` are not.

| File | Purpose | Status |
|---|---|---|
| `export_dataset.py` | bags → `manifest.jsonl` + `dataset_summary.json` | **Updated** for 3-way keypoint state, `yolo_mask` path, occlusion-key resolution. Verified on real keypoints.json. |
| `make_splits.py` | bag-level train/val/test | Verified on real manifest. |
| `needle_dataset.py` | manifest → tensors (image, mask, heatmaps, masks) | Verified — **but NEEDS UPDATE for the pivot** (see §7). |
| `needle_augment.py` | geometric + photometric aug, FOV-aware visibility | Verified on real frames (two bugs found+fixed). |
| `needle_sampler.py` | domain-balanced, bag-equalized weighted sampler | Verified on real manifest. |
| `needle_model.py` | ResNet-34 U-Net, 3 heads, soft-argmax | Verified shapes/grad/sub-pixel — **NEEDS UPDATE: 4-channel input.** |
| `needle_loss.py` | BCE+Dice mask, masked MSE heatmap, masked BCE vis | Verified incl. occlusion-masking gate. |
| `train.py` | loop: AMP, cosine LR, val metrics, checkpoints | Verified mechanics + checkpoint reproducibility. Real GPU overfit PASSED. |
| `overfit_one_batch.py` | end-to-end stack check | Verified — PASSED on GPU (98.2% loss drop). |

Test/utility scripts that exist and are worth keeping: `test_export.py`,
`test_dataset.py`, `test_augment.py`, `test_sampler.py`, `test_model.py`,
`test_train.py`, `build_overfit_manifest.py`.

### Key module details
- **`export_dataset.py`**: `normalize_keypoints` (3-way state via
  `_resolve_occlusion`, long-key-wins), `index_masks(bag_dir, subdir=...)`
  (handles both `masks/` and `yolo_mask/`; globs `*.png`, parses frame index
  from filename), per-frame record includes `image`, `mask`, `yolo_mask`,
  `has_board`, `board_detected`, `pose_status`, `keypoints` (each with
  `xy`/`state`/`visible`). Summary reports per-bag 3-way coverage and
  `yolo_masks_found`.
- **`needle_dataset.py`**: produces image `(3,640,640)` ImageNet-normalized, mask
  `(1,640,640)`, heatmaps `(K,160,160)`, `hm_mask (K,)`, `vis_target (K,)`,
  `vis_mask (K,)`. Has a `transform` hook that operates on **raw arrays BEFORE
  heatmap rendering** (so geometric aug stays exact). `_build_raw` →
  `transform` → `_finalize` (renders heatmaps). **Currently infers state from
  xy-null-ness — must switch to the explicit `state` field (see §7).**
- **`needle_augment.py`**: `NeedleAugmentation` — horizontal flip, **full 360°
  rotation about center** (data shows needle appears at all orientations),
  scale/translate jitter, hue/sat/brightness/contrast/noise (photometric, image
  only). **FOV-aware visibility is parameterizable (`fov_aware`)**: a keypoint
  that lands outside the circular FOV after transform is marked not-visible
  (heatmap unsupervised). `fov_aware` exists because rotation about the image
  center only preserves the FOV if the FOV is centered — the user has a planned
  (not yet built) auto-crop/center step; **once that exists, set
  `fov_aware=False`.** FOV detected from the image (threshold black border →
  largest connected component; NO aggressive morphology — that was a bug).
- **`needle_sampler.py`**: per-frame weight = `domain_mass(D)/num_bags_in_D/
  frames_in_bag`, so each bag contributes equal mass within its domain (large
  bags can't dominate) and the two domains hit a target ratio.
  `target_noboard_frac` controls the ratio (0.5 ⇒ 1:1). **Use a gentle value
  (~0.2–0.25) until the no-board set grows**, otherwise the ~930 no-board frames
  get oversampled ~10× and risk memorization.
- **`needle_model.py`**: `NeedleNet` — ResNet-34 encoder (ImageNet pretrained),
  U-Net decoder with skips to stride 4, three heads: `mask_logits`
  `(B,1,640,640)`, `heatmaps` `(B,K,160,160)`, `vis_logits` `(B,K)`.
  `soft_argmax` gives differentiable sub-pixel coords (verified exact on known
  peaks). 24.4M params. **Input conv is 3-channel — must become 4-channel.**
- **`needle_loss.py`**: `NeedleLoss` — mask = BCE + (1−softDice); heatmap = MSE
  **masked by `hm_mask`** (occluded/absent contribute exactly zero — unit-tested);
  visibility = BCE **masked by `vis_mask`** (only labeled points). Weights
  `w_mask, w_hm, w_vis` (default 1, 1, 0.5).

---

## 5. How to run (in-repo)

```bash
cd ~/aliss_core/src/aliss_scripts

# 1. export (point at the bags you want; pass the no-board bag names)
python training/export_dataset.py \
    --ann-dir auto-annotating/annotations \
    --out dataset --no-board-bags chicken_1,chicken_2,chicken_3

# inspect dataset/dataset_summary.json: keypoint_coverage (visible/occluded/
# unlabeled per kp), yolo_masks_found per bag, board vs no-board counts.

# 2. splits (needs >=1 no-board bag for val and for test)
python training/make_splits.py \
    --manifest dataset/manifest.jsonl --out dataset/splits.json \
    --test-bags chicken_1 --val-bags chicken_2

# 3. overfit one batch (stack sanity; expect all loss terms -> ~0)
head -n 4 dataset/manifest.jsonl > dataset/overfit.jsonl
python training/overfit_one_batch.py --manifest dataset/overfit.jsonl \
    --input-size 640 --steps 300 --pretrained --device cuda

# 4. train
python training/train.py \
    --manifest dataset/manifest.jsonl --splits dataset/splits.json \
    --out runs/exp1 --epochs 50 --batch-size 8 --target-noboard-frac 0.25
```

---

## 6. Decisions already made (do not relitigate without reason)

- **D1 — Learned, mask-conditioned keypoint head, not mask-derived keypoints.**
  Rationale: §3.5, §3.6. Skeleton endpoints break under occlusion and can't tell
  tip from tail.
- **D2 — Input 640×640, heatmaps 160×160 (stride 4).** Rationale: needle survives
  downscaling (p10 IoU plateaus at 640), 163 FPS leaves huge headroom, 160 grid +
  soft-argmax gives sub-pixel keypoints. 512 is an acceptable faster fallback
  (0.013 IoU behind).
- **D3 — Bag-level splits; val/test are no-board only; board bags train-only;
  surplus no-board → train.** Frame-level splitting leaks (adjacent frames are
  near-duplicates).
- **D4 — Domain-balanced sampler, gentle no-board fraction until data grows.**
- **D5 — Treat tip/tail SWAPS as a first-class risk.** User: swaps are
  catastrophic **only if constant**; transient swaps are recoverable. Design to
  make swaps transient + detectable: (a) report **swap rate** separately from
  pixel error, sliced by clean-vs-occluded and by orientation bin; (b) temporal
  consistency + one-euro filter at inference to damp transient swaps; (c) a
  *constant* swap in some pose regime is a data/appearance problem → add labeled
  examples of that regime, don't just filter.
- **D6 — Occlusion: predict visibility, NOT hidden location.** `xy` of occluded
  points is untrustworthy; mask the heatmap there.
- **D7 — Full 360° rotation augmentation.** Justified by measured orientation
  histogram (needle spans all angles within a single bag).
- **D8 — Train the conditioning channel on YOLO's PREDICTED masks, not SAM2 GT.**
  (Recommended; pending user confirmation — see §8.) So the head learns YOLO's
  real error modes and treats the mask as a fallible prior, matching inference.

---

## 7. Updates required for the pivot (the immediate work)

These are the concrete code changes to wire in mask conditioning and the new
3-way labels. Make them surgically and re-run the paired tests.

1. **`needle_dataset.py` — consume the explicit `state` field.** Currently the
   dataset infers state from whether `xy` is null, which now **conflates
   `occluded` with `unlabeled`** (the exporter sets `xy=None` for occluded
   points). Fix `_build_raw` to read `rec["keypoints"][name]["state"]`:
   - `state=="visible"` → `labeled=1, vis=1`, render heatmap (set kp xy).
   - `state=="occluded"` → `labeled=1` (supervise visibility), `vis=0`,
     `hm_mask=0` (no heatmap). NOTE: this needs a representation where a point can
     be "visibility-supervised but heatmap-unsupervised" even with no xy — today
     `vis_mask` comes from `labeled` and `hm_mask` from `labeled & vis`. Verify
     the occluded case yields `vis_mask=1, hm_mask=0, vis_target=0`.
   - `state=="unlabeled"` → supervise nothing (`labeled=0`).
   Add a verification (extend `test_dataset.py`) asserting the three states map to
   the right `(hm_mask, vis_target, vis_mask)` triples.

2. **`needle_dataset.py` — load the YOLO mask as a 4th input channel.** Read
   `rec["yolo_mask"]` (may be `None` — handle missing: either skip the frame for
   training the conditioned head, or feed a zero channel and flag it). Resize to
   640 (nearest), stack behind the 3 RGB channels → image tensor `(4,640,640)`.
   The augmentation must transform the YOLO mask channel in lockstep with the
   image (it's part of the geometric warp — treat like the mask). Decide and
   document whether the YOLO channel is binary {0,1} or soft.

3. **`needle_model.py` — 4-channel stem.** Change `enc.conv1` to accept 4 input
   channels. Standard trick: create a new `Conv2d(4, 64, ...)`, copy the
   pretrained 3-channel weights into the first 3 input channels, and init the 4th
   channel (e.g. zeros or the mean of the RGB filters) so the pretrained features
   are preserved at init. Verify the mask channel actually influences the output
   (perturb it, confirm logits change).

4. **Decide the fate of the model's own mask head.** YOLO now provides
   segmentation, so `mask_head` may be (a) dropped entirely (keypoints-only net),
   (b) kept as an auxiliary task that regularizes the shared encoder, or (c) kept
   to *refine* YOLO's mask. Recommendation: **keep it as a cheap auxiliary head
   initially** (it shares the encoder and the overfit showed it trains fine), and
   drop it only if it doesn't help. This is a measure-don't-assume call — flag it
   to the user; see §8.

5. **YOLO mask generation.** The user will populate `yolo_mask/<frame>.png` per
   bag. If a script is needed to dump YOLO predictions across all bags, build one
   from the existing `infer_yolo.py` (it already loads the weights and iterates a
   bag's `frames/`); write each predicted mask to `yolo_mask/` at full 1080 res,
   filename containing the frame index.

---

## 8. Open decisions that need the USER (ask before assuming)

1. **Training-time conditioning mask: YOLO predicted (D8, recommended) vs SAM2 GT.**
   Needs `yolo_mask/` populated for training bags. Confirm.
2. **Keep, drop, or repurpose the model's mask head** (§7.4).
3. **Whether to keep the loss-weighting as-is.** A known issue (below) may require
   reweighting once real multi-bag training starts.

---

## 9. Verification status and known issues

**Verified (gate passed on real data or real GPU):** exporter 3-way state +
occlusion-key resolution; bag-level splits; dataset tensor shapes + coordinate
round-trip (sub-pixel after soft-argmax); augmentation lockstep + FOV visibility;
sampler domain ratio + bag-equalization; model shapes + soft-argmax exactness +
gradient flow; loss occlusion-masking (occluded kp contributes exactly 0 to
heatmap loss); training loop mechanics + **checkpoint reload reproduces metrics
exactly**; **overfit-one-batch on GPU drove total loss down 98.2%, mask Dice
≈0.04** (this also closed the "is the stride-4 mask decoder good enough" question —
it is; no extra decoder stage needed).

**Known issue to watch — heatmap loss scale.** Target heatmaps are mostly zeros
with a tiny Gaussian bump, so raw MSE can either (a) be dominated by a few
positive pixels or (b) collapse toward predicting all-zeros. The overfit didn't
expose it (memorizing a few frames is easy). **On the first real multi-epoch run,
watch that `val_kp_err_px` actually descends from its ~380px random-init start and
that the `hm` term neither dominates nor flatlines.** If it misbehaves: raise
`w_hm`, or normalize the heatmap loss by the number of positive pixels, or switch
to a soft-argmax coordinate loss. Don't pre-change it; measure first.

**Thin-mask Dice converges slower than the other terms** (BCE shapes the region
fast, Dice refines the thin boundary slowly). Expected, not a bug.

---

## 10. Remaining build steps (post-pivot)

- **Step 1 (now):** §7 updates — 4-channel mask-conditioned input + 3-way labels.
  Gate: model ingests 4 channels, mask channel influences output, dataset maps the
  three states to correct supervision masks.
- **Step 2:** retrain on the real data with corrected labels + conditioning,
  reusing the existing loss/sampler/augmentation/loop. Gate: clean training curves
  (watch §9 heatmap-scale issue), val metrics improving on the no-board val bag(s).
- **Step 3 (eval):** the metrics that matter — localization error (within the
  user's tolerance), **tip/tail swap rate sliced by clean-vs-occluded and by
  orientation bin** (the D5 concern), and visibility accuracy. Test set MUST
  include occlusion/fragmented-mask frames (§3.5). Optionally compare against the
  cheap skeleton-endpoint baseline to confirm the learned head wins under
  occlusion.
- **Step 4 (inference + deploy):** single-frame predict → soft-argmax →
  **temporal consistency + one-euro filter** (damps jitter and transient swaps;
  the user explicitly does not want jittery keypoints) → export in the
  workstation's deployment format (traced/compiled, half precision). At inference
  the conditioning channel is YOLO's predicted mask, produced by running YOLO
  first (seg + keypoint head = two models; the 10–15 FPS bar easily affords both).

---

## 11. Data-growth dependency (flag to user repeatedly)

The no-board (deployment-domain) set is currently ~3 bags / 930 frames, one of
which (chicken_3) had keypoints still in progress at last check. **A trustworthy
bag-level val/test split needs ~5 no-board bags.** Until the user's planned +8
no-board bags are annotated, val/test metrics are noisy and the aggressive
sampler ratio risks memorizing the few no-board frames. The new no-board
annotations are the single biggest lever on final quality — keep surfacing this.
