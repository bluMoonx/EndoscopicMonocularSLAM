# Pre-registration: out-of-the-box baseline head-to-head (ORB-SLAM3 vs the adopted pipeline)

Committed BEFORE any ORB-SLAM3 run on C3VD exists. Motivation: the judge question "why build a
from-scratch direct tracker instead of using the standard open-source SLAM?" currently has a
qualitative answer (Act I measured ORB-SLAM3's glare robustness on in-vivo Hamlyn footage by
keyframe survival only — Hamlyn ships no pose ground truth). No pose-accuracy comparison
between Argus and any off-the-shelf SLAM exists anywhere in this project. This study measures
it once, on the project's headline benchmark, with the baseline configured out-of-the-box and
given every stated advantage. This is a CHARACTERIZATION, not an adoption trial: there is no
pass bar, and every outcome below is published.

## Data (fixed now)

The five C3VD headline sequences, exactly the frames the Argus tracker consumes (undistorted
pinhole JPEGs, `data/c3vd/<seq>_undist/image01`, 30 fps TUM listings in `rgb.txt`, intrinsics
in `K.txt`, GT trajectories `results/c3vd/<seq>_gt.txt`):

1. cecum_t1_a   2. desc_t4_a   3. trans_t1_a   4. trans_t2_a   5. trans_t2_b

No other sequence enters this study. (Extension to the complete screening withdrawals is named
future work, not smuggled in later.)

## Baseline system (fixed now)

ORB-SLAM3 monocular via `mono_tum`, the same containerized build used in the Act I severity
experiments (`argus_slam` Docker container, `/ORB_SLAM3`); if that container no longer exists
it is rebuilt from the stock ORB-SLAM3 v1.0 release and the version actually run is recorded
in the results. Because monocular ORB-SLAM3 is non-deterministic, **N = 5 runs** per sequence
per configuration; the per-sequence summary is the **median** over runs (range reported).

Two configurations, both frozen here, and the **headline comparison uses whichever is better
for the baseline** (per-sequence):

- **B-stock** — the stock TUM monocular example settings (nFeatures 1000, scaleFactor 1.2,
  nLevels 8, iniThFAST 20, minThFAST 7), with only the camera block replaced by each
  sequence's own `K.txt` intrinsics (fx=fy=250, cx=320, cy=256, zero distortion, 30 fps).
- **B-boost** — the endoscopy-boosted extractor the project already used on Hamlyn in Act I
  (nFeatures 4000, nLevels 10, iniThFAST 5, minThFAST 2), same camera block.

No other parameter differs from the shipped examples; no per-sequence tuning; no parameter
changes after this commit.

## Argus side (fixed now)

The adopted pipeline-v2 tracker (in-loop Huber), unchanged. Its trajectories are re-emitted by
the existing frozen harness solely so RPE can be computed; the re-emitted ATEs must match the
published red-team-verified values (`results/redteam_retrack.csv`: 0.153 / 0.620 / 0.274 /
0.155 / 0.893 mm, mean 0.419) to within 0.01 mm, **or the study halts and the discrepancy is
investigated before anything else is reported**. No Argus re-tuning of any kind.

## Metrics (pre-specified, in priority order)

1. **Tracking completion** — the fraction of the sequence the system actually tracked. For
   the baseline: if the build emits a full-frame `CameraTrajectory.txt`, completion = frames
   with a pose / total frames; if only `KeyFrameTrajectory.txt` exists, completion = the
   time span [first keyframe, last keyframe] / sequence duration, with keyframe count
   reported beside it. A run that never initializes scores completion 0 (ATE/RPE undefined
   for that run). The Argus analogue reported beside it: its divergence-guard rate on these
   sequences (the tracker estimates every frame by construction).
2. **ATE RMSE (mm)**, Sim(3)/Umeyama alignment of each system's trajectory against GT on the
   timestamps that system produced, **reported beside the constant-point baseline computed on
   the same timestamp subset** (standing policy, retraction 2). Where a run's completion
   < 80%, its ATE is labeled secondary: a fragment's ATE is not comparable to a full-sequence
   ATE, and the alignment-collapse caveat applies.
3. **RPE, translational (mm)** — defined on positions (both systems' retained artifacts are
   position trajectories): after Sim(3)-aligning the estimate to GT (the same alignment as the
   ATE), RPE(Δ) at time t = ‖(p̂(t+Δ) − p̂(t)) − (p_gt(t+Δ) − p_gt(t))‖ over all tracked
   timestamp pairs (t, t+Δ), at Δ = 1 frame and Δ = 30 frames (1 s); median and RMSE
   reported. Computed identically for both systems by one script.
4. **Uncertainty** — bootstrap 95% CIs (10,000 resamples, percentile method): within-sequence
   over per-frame RPE values; across-sequence over the 5 per-sequence values for any
   aggregate mean. Baseline per-sequence spread over its 5 runs is reported as median
   [min–max] (n = 5 is too small for a CI and is not dressed up as one).

## Protocol (frozen)

- Run order: all B-stock runs, then all B-boost runs, sequences in the listed order; one
  warm-up run of cecum_t1_a is discarded before timing-irrelevant scoring begins (ORB
  vocabulary load noise) — its trajectory is NOT scored.
- The runner + analysis script is `speqslam/baseline_orbslam3.py`, written to this spec and
  committed alongside it, before any run. Outputs: `results/orbslam3_baseline.csv` (one row
  per run), `results/orbslam3_baseline_summary.csv`, and figures under `results/figures/`
  (per-sequence ATE beside constant-point baseline; RPE distributions; completion bars).
- GT association by nearest timestamp within 1/60 s.
- Every raw trajectory file is retained under `results/orbslam3_baseline/`.

## Pre-stated outcomes (all of them published, whichever occurs)

- **Baseline fails to initialize or completes <50% (median) on a sequence:** the finding is
  feature-based SLAM's texture requirement on endoscopic wall — completion is the primary
  metric there, and no ATE victory is claimed from a fragment the baseline couldn't track.
- **Baseline completes and its ATE beats pipeline v2:** reported plainly in the README beside
  the headline table; the project's tracking claim is restated (dense, glare-aware,
  every-frame frontend with measured operating envelope — not SOTA accuracy), and adopting a
  feature-based frontend becomes named future work.
- **Pipeline v2 is more accurate:** claimed only on these phantom sequences; no generality
  beyond C3VD is asserted, and the Act I glare result is NOT retroactively inflated into
  "ORB-SLAM3 doesn't work in endoscopy."
- **Mixed (likely):** reported per-sequence with completion first; the comparison table
  carries both configurations and both metrics, and the README headline links here.

No sequence list, configuration, metric, threshold, or outcome statement changes after this
commit.
