# Pre-registration: cross-dataset confirmation layer — ORB-SLAM3 vs the monocular pipeline on EndoSLAM (ex-vivo porcine)

Committed BEFORE any ORB-SLAM3 run on EndoSLAM exists, and BEFORE any pipeline-v2 run on
EndoSLAM exists. Positioning: the C3VD head-to-head (`results/orbslam3_baseline_gate.md`,
run 2026-10-17) answered the accuracy question on the rigid, clean phantom — a draw. It left
open the question Act I raised only qualitatively: real tissue, with its glare, texture and
fluid confounds. EndoSLAM is the project's existing cross-dataset layer with OptiTrack ground
truth; this study adds (i) the out-of-the-box baseline on the same five trajectories and
(ii) the adopted pipeline v2's first cross-dataset measurement. It is a confirmation layer on
top of the published zero-shot result, which stands as committed — not a redo of it.

## Benchmark choice (fixed now)

**EndoSLAM** Colon-IV Tumor-free trajectories 1–5 (HighCam), already processed on disk —
identical frames to those the published pipeline consumed. **C3VD** is already covered by the
first gate. **SCARED** is excluded: it requires registered access not on hand, targets
stereo/depth evaluation, and offers no pose-ground-truth capability EndoSLAM does not already
provide here. No disk constraint binds (all data local; outputs are kilobyte trajectories).

## Data (fixed now)

`data/endoslam/proc/colon4_traj{1..5}`: 568/850/885/457/942 undistorted 640×480 frames
(`image01/%06d.jpg`, nominal 30 fps indexing), intrinsics in each trajectory's `K.txt`,
ground truth `results/endoslam/colon4_traj{N}_gt.txt` (OptiTrack, TUM format, as committed).
TUM `rgb.txt` listings are generated deterministically (timestamp i/30) — mechanical prep,
no choices. No other sequence enters this study.

## Baseline system (fixed now)

The same containerized ORB-SLAM3 build and the same two frozen configurations (B-stock,
B-boost) as the C3VD gate, with the camera block from each trajectory's `K.txt`
(fx 957.411, fy 959.386, cx 282.192, cy 170.731; the shipped calibration's 5.62 px skew term
is unsupported by ORB-SLAM3 and dropped — a disclosed approximation of ~0.6% of fx; zero
distortion, the frames are already undistorted). **N = 5 runs** per trajectory per
configuration; per-trajectory summary = median over runs (range reported); the headline uses
whichever configuration is better for the baseline, per trajectory, as before.

## Argus side (fixed now)

- **Comparator 1 — the published zero-shot result (frozen):** both committed conditions,
  `results/endoslam/colon4_traj{N}_plain.txt` (C3VD-identical settings) and
  `colon4_traj{N}_invivo.txt` (glare-aware + illumination compensation, the Act I-recommended
  real-tissue mode). Metric-provenance note, established during prep and stated here rather
  than discovered later: the audited per-trajectory figures 26.6 / 32.0 / 47.1 / 26.5 /
  69.2 mm in `baseline_check.json` are the **in-vivo-mode** trajectories' Sim(3) ATE RMSE
  (recomputation matches all five exactly), while the older commit message's "mean ATE
  39.6 mm" is the **plain** condition's RMSE averaged over the five trajectories — a labeling
  inconsistency in the record that this study surfaces for the author rather than papers
  over. Verification before any run: recomputed in-vivo-mode RMSEs must reproduce
  26.6 / 32.0 / 47.1 / 26.5 / 69.2 mm to within 0.1 mm, **or the study halts**. Both
  statistics (RMSE headline, mean beside it) are reported for every system identically, each
  beside its own-flavor constant-point baseline.
- **Comparator 2 — the adopted pipeline v2 (new, deterministic, one run per trajectory):**
  `track_pair(robust=True, illum=True)` — exactly the re-validated C3VD configuration — on
  the same frames and the same zero-shot fine-tuned depths. This is the adopted tracker's
  first cross-dataset measurement. Pre-stated: if v2 is WORSE here than the v1-era published
  plain condition, that is published as-is — iteration 21 already established that the
  adoption does not blindly generalize across configurations, and this tests transfer across
  data.

No re-tuning of any kind on either comparator.

## Metrics (identical to the C3VD gate, pre-specified)

1. **Tracking completion / tracking-loss** — baseline: tracked-span fraction (keyframe span /
   sequence duration; full-frame fraction if the build emits CameraTrajectory), a
   never-initializing run scoring 0. Argus poses every frame by construction (completion 1);
   its loss analogue is the divergence-guard rate, reported if any guard fires (none is
   expected at these frame rates; the published runs used none).
2. **ATE RMSE (mm)**, Sim(3)/Umeyama on each system's own timestamps, **beside the
   constant-point baseline on the same timestamp subset**. Published context stated up front,
   not discovered later: the Argus plain-condition ratios are already 0.62/0.89/0.76/0.63/0.98
   — near-baseline on trajectories 2 and 5 — and the restated claim ("informative on 3 of 5")
   carries over. ATE is secondary for any run with completion < 80%.
3. **RPE, translational (mm)** — position-based as defined in the C3VD gate, Δ = 1 and 30
   frames, medians and RMSE.
4. **Uncertainty** — bootstrap 95% CIs (10,000 resamples): across trajectories for aggregate
   means; within-trajectory across per-pair RPE values; baseline run spread as median
   [min–max] (n = 5, not dressed up as a CI).

## Protocol (frozen)

- Runner/analysis: `speqslam/baseline_orbslam3_endoslam.py`, committed with this gate before
  any run; shared geometry/metric helpers imported from `speqslam/baseline_orbslam3.py`
  unchanged. Run order: all B-stock then all B-boost, trajectories 1→5; one discarded
  warm-up (traj1, stock) first. Raw trajectories retained under
  `results/orbslam3_endoslam/`; outputs `results/orbslam3_endoslam.csv`,
  `results/orbslam3_endoslam_summary.csv`, figures under `results/figures/`.
- GT association by nearest timestamp within 1/60 s, as before.

## Pre-stated outcomes (all published, whichever occurs)

- **Baseline fragments or DNFs on real tissue while the dense tracker completes:** Act I's
  qualitative robustness claim acquires its quantitative form; completion is the primary
  finding and no accuracy claim is made from fragments.
- **Baseline completes and matches or beats on ATE:** published plainly; the cross-dataset
  conclusion becomes "the standard tool transfers too," and the project's tracking case rests
  on every-frame density, the measured envelope, and integration — exactly as restated after
  the C3VD draw.
- **Pipeline v2 vs the published plain condition:** published either way, better or worse —
  a transfer measurement, not a defense of the adoption.
- **Mixed per-trajectory (likely):** completion first; the table carries both configurations,
  both comparators, and every metric.

No sequence list, configuration, metric, threshold, or outcome statement changes after this
commit.

---

## VERDICT (results section, added after the run — Oct 20 2026)

**Pre-stated outcome 2 applies: the baseline completes and matches — "the standard tool
transfers too," published plainly.** All 50 scored runs produced trajectories (plus the
discarded warm-up); run provenance: gate committed 23:01:22 Oct 18, warm-up artifact created
23:03:50, stock runs 23:03–23:20 Oct 18, v2 + boost runs and analysis Oct 19–20 — gate
before run holds throughout.

**Accuracy is a four-way statistical tie.** Mean Sim(3) ATE RMSE across the five
trajectories: Argus plain 39.6 mm, Argus in-vivo mode 40.3 mm, Argus v2 40.4 mm, ORB-SLAM3
(better config per trajectory) 41.4 mm — every bootstrap CI overlapping (roughly 28–57 mm
for each system). Per-trajectory medians sit within a few mm of one another everywhere
(e.g. traj1: 28.3 v2 vs 27.8 baseline; traj3: 47.8 vs 48.9; traj4: 25.1 vs 27.0).

**Completion is the differentiator that did appear.** ORB-SLAM3 median tracked span:
0.95 / 0.85 / 0.91 / 0.65 / 0.95 — it loses a third of trajectory 4 and poses only sparse
keyframes (63–79 poses on traj1's 568 frames) where Argus poses every frame with no loss.
Act I's robustness hypothesis gets partial quantitative support: the standard tool does not
fragment on ex-vivo tissue at these frame rates, but it is not dense and not complete.

**Comparator 2 (v2 transfer) is neutral, published as measured:** v2 vs the published plain
condition is mixed per trajectory (better on 1, worse by 0.7–3.3 mm on 4), mean 40.4 vs
39.6 mm — the phantom adoption's 0.91 → 0.42 mm gain does not transfer to this data,
consistent with iteration 21's finding that the adoption is per-configuration, never generic.

**Informativeness caveat carries over as pre-disclosed:** traj2 and traj5 ATEs sit near
their constant-point baselines (36.0 and 70.7 mm) for every system including the baseline,
so cross-dataset accuracy conclusions rest on trajectories 1, 3, 4.

**Mechanism:** on this data, a dense direct tracker and a feature-based tracker with
fundamentally different machinery land at the same per-trajectory error level — the error
floor looks data-set, not tracker-set (zero-shot depth quality, OptiTrack ground-truth
registration, and trajectory geometry), which is itself the cross-dataset finding.
