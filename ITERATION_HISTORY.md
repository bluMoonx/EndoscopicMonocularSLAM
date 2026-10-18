# Argus — Engineering Design History

**Status: reconstructed and artifact-checked; awaiting the author's attestation.** Every entry
below is reconstructed from the project's own records — the 160-commit git history
(2026-07-13 → 2026-10-17), the manuscript draft (`Argus_Manuscript.docx`, §4), and the
pre-registration documents in `results/*.md` — and nothing has been invented to fill a gap.
Quantitative claims were verified against the raw artifacts (`results/*.csv`) where they
exist; where the records are silent, this document says so rather than guessing. The short
attestation checklist at the end holds the only items that need the author's word.

---

## 1. Engineering goal

Build an instrument for the colonoscopy failure mode no product measures: the wall the camera
never saw. From a **standard, unmodified endoscope's monocular video alone**, produce a 3D
reconstruction of the colon, count how often each surface patch was properly observed, flag
under-inspected regions, and localize and size detected polyps in the map — deployable on a
consumer phone.

## 2. Design criteria and constraints (held for the whole project)

| # | Constraint | Consequence in the record |
|---|---|---|
| C1 | **No new hardware** — input is monocular video only | no depth sensor, no EM tracking, no stereo; scale is recovered by alignment and labeled an estimate on uncalibrated video |
| C2 | **Auditable honesty** — every claim validated against ground truth where it exists, labeled an estimate where it doesn't, and reported beside the trivial baseline that could counterfeit it | two self-caught retractions (§6 of the manuscript); constant-point baseline printed beside every ATE; flag-nothing baseline beside every agreement figure |
| C3 | **Consumer compute budget** — one Windows workstation, single RTX A1000 (8 GB), one Android phone; no cloud | all fine-tuning local (~40 min/fold); acceleration only when a measurement earns it |
| C4 | *(from 2026-09-16, hardened 2026-09-26)* **Pre-registration** — pass/fail criteria committed to git before the test runs; failures published with autopsies | 11 pre-registration documents/commits; 22 iterations, 2 adoptions; frozen tests read once |

## 3. How iterations were run

Each iteration poses one engineering question, states its quantitative requirement, builds the
candidate, and records the verdict — including negative ones — which dictates the next
iteration. From the batch study onward (2026-09-16) protocols were frozen before test data was
read; from the numbers audit onward (2026-09-26) every experiment's gates were **committed to
git before the test ran** (the commit timestamps are the evidence). Standing audit machinery:
a 35-figure numbers audit (`audit_numbers.py`, 35/35 reproduce from raw artifacts) and a
red-team audit (`redteam_audit.py`, 15/15) that re-derives the headline results with
independently written code (Horn alignment vs the pipeline's Umeyama, agreement to 0.0005 mm).

## 4. Timeline at a glance

| Phase | Dates (git) | What happened |
|---|---|---|
| Act I — the glare hypothesis | committed 2026-08-15 (repo created 07-13) | original hypothesis tested and rejected with a mechanism; pivot to direct VO |
| Act II — the monocular instrument | 2026-08-15 → 09-16 | from-scratch tracker, ablation, depth fine-tune, detection loop, deployment |
| Reboot into frozen protocols | 2026-09-16 → 09-26 | batch study discipline; retraction 2; numbers audit 35/35 |
| Act III — pre-registration campaigns | 2026-09-26 → 10-11 | 11 campaigns under committed gates; 2 adoptions |
| Open | 2026-10-17 → | site re-identification experiment pre-registered, not yet run |

## 5. The iteration ledger

Numbering 1–11 follows the manuscript's §4. The README's running tally first appears on
2026-10-08 at "twenty iterations" (`51fc3aa`, after pipeline v2) and advances to 21 at the
Huber app-port trial (`afd2c5c`) and 22 at dense-400 (`331b8c9`) — so those three anchors are
fixed by the record, but **no document in the project ever assigns numbers 12–19 to specific
campaigns**. This ledger therefore deliberately labels the Act III work as lettered campaigns
rather than inventing a numbering the records don't contain.

### Act I

**1 — The glare hypothesis, rejected with a mechanism** *(committed 2026-08-15, `9a872ba`…`4231ed5`;
the 07-13 initial commit is only the repository's creation — a one-line README)*
Hypothesis: removing specular glare improves endoscopic SLAM. Multi-condition experiments on
in-vivo Hamlyn footage (ORB-SLAM3 in Docker, N=5 repeats per condition, keyframe-survival
metric; `run_matrix.sh`, `run_ai_matrix.sh`, `run_temporal_matrix.sh`) showed the opposite
structure: feature-based tracking is already glare-robust (RANSAC discards corrupted
correspondences) and naive inpainting *hurts* by destroying texture. The hypothesis survived
only for photometric methods, which minimize raw brightness. **Verdict: rejected → pivot** to
(i) build a direct VO where glare handling can matter, (ii) evaluate at trajectory level on
data with pose ground truth (Hamlyn has none — verified early).

### Act II

**2 — A direct VO frontend, built and validated** *(2026-08-15; `9b9b428`, `121fae7`)*
Dense photometric VO from scratch: SE(3) Gauss–Newton on a 4-level pyramid, per-pixel depth
warping, glare-pixel exclusion, DSO-style affine illumination compensation. With ground-truth
depth, the tracker validated at **0.38 mm ATE** on the first GT sequence (trans_t1_a, 0.383 mm)
and **0.32–0.99 mm (mean 0.56)** across all five (`results/ablation.csv`, condition B) — the
tracking math sound before any learned component entered. A later stress study (`robustness.py`, 08-18→08-24) found a flicker weakness (±30%
brightness → 12.3 mm); enabling affine compensation restored 0.77 mm (`26120b4`, `4837ada`).

**3 — The blind-spot instrument on ideal inputs** *(2026-08-16; `4a9950c`)*
TSDF fusion + per-patch qualifying-view counting (frustum, front-facing, unoccluded; <3 views
= blind). Driven by GT inputs: **97.8–99.6% agreement** with true blind-spot maps; the view
threshold swept 2–5 with conclusions unmoved (`497d110`).

**4 — Going fully monocular, failing informatively** *(2026-08-16)*
GT inputs replaced by estimates (off-the-shelf Depth-Anything-V2 + own tracker). Agreement
fell to 74.5%; **7–41% of the wall falsely flagged** as never inspected. Failure demanded
attribution before any fix.

**5 — The input-substitution ablation** *(2026-08-16; `stage_ablation.py`)*
Each estimated quantity substituted independently by its GT counterpart: GT-depth+est-poses
holds 99.2%; est-depth+GT-poses falls to 71%. **Depth, and only depth, is the bottleneck.**

**6 — Domain fine-tuning under leave-one-sequence-out discipline** *(2026-08-16 → 08-18;
`train_depth.py`)*
Five LOSO folds on the 8 GB GPU. Held-out δ<1.25: 64–83% → **92–99.7%**; false alarms collapse
to **0.5–3.9%**; fully-monocular ATE 0.17–1.59 mm (mean 0.91); surfaces 1.3–2.4 mm median on
4/5 sequences; the fifth (28 mm travel, scale drift) reported as a characterized failure, not
excluded.

**7 — Closing the detection loop under a frozen operating point** *(2026-08-24 → 09-14;
`38b6e39`, `f89fa3e`, `8ad0e21`, `e60f855`)*
Pretrained DeepLabV3+ segmenter; masks back-projected, 3D-clustered, persistence-gated. Gates
calibrated on control footage and frozen before evaluation footage was processed; one revision
(confidence 0.9→0.7) documented with its control-set justification. Sizing validated at
**0.60 mm median** on 3–30 mm targets. Real-footage closed loop: a ~6 mm polyp found, pinned,
sized — n=1 and labeled as such.

**8 — The batch trial finds the domain gap again; the same medicine cures it again**
*(2026-09-16 → 09-17; `7875536`, `909002d`, `005b2a2`, `14ca2cd`)*
30-clip frozen trial: borrowed segmenter fires on 82.5% of held-out control frames; test read
0/6. Hard-negative fine-tune (documented 800/200 split + 480 hard negatives): control fire
5.0%, Dice unchanged 0.959. Protocol re-frozen, extended to **n=41 never-seen clips, read
once: 11/21 polyp clips, 3/20 controls (all three genuine non-polyp pathology), Fisher
p=0.013**.

**9 — Measuring the rigidity assumption** *(2026-09-19; `59d36e0`)*
C3VDv2 deforming phantoms: **~9× ATE cost** vs speed-matched rigid control (1.37→2.9–13.4 mm);
stationary-camera sequence isolates wall-motion-as-apparent-trajectory (9.3 mm); fast rigid
motion degrades comparably (25.2 mm) — deformation is one of several same-order stressors;
degradation graceful, never collapse.

**10 — Whole-procedure scale: a retraction and a replicated speed limit** *(2026-09-20 →
10-04; `b940671`…`9834f36`, `e62839f`, `28f3510`)*
First full-withdrawal figure (ATE 150 mm, "3.5% of path") **retracted** after audit: silent
divergence on 24 frame pairs + alignment collapse (constant-point scores 168 mm on the same
data). Physical divergence guard added; guard rate became the primary metric. The honest
finding, then replicated: tracking **fails at ~2 effective fps** (guard 16–23%, n=3), **holds
at ~9** (1.5–3.1%, ATE 0.60–0.73× baseline, n=3, 20.3 m GT path), **native rate solves it**
(0.9–2.1%, n=3, 28 m). Coverage at screening pace: ~37–40% of mapped wall under-seen (carries
the tracking caveat).

**11 — Deployment, with acceleration decided by measurement** *(2026-09-01 → 09-23;
`511f147`, `0aacbef`, `1350e7a`, `5802de0`)*
Full pipeline ported to Android (ONNX Runtime). Depth int8: accuracy-identical, shipped
(250→109 ms/frame). Detector int8: Dice 0.959→0.942, control fires 5.0→17.5% — **rejected**.
NNAPI: 8× slower — **reverted**. One shipped optimization, two documented rejections.

### Act III — the pre-registration campaigns (gates committed before each run)

**Campaign A — Loop closure, three generations, closed as a question** *(2026-09-27 → 10-03;
`141c133`, `d22a718`, `1f15415`, `03a2c4d`)*
v1 rejected by its own frozen test. v2 (feature-based closures) frozen before its tests — FAIL.
v3 (translation-only PnP closures) clean but bounded by a ground-truth oracle: **even perfect
closures recover ≤14%** on screening paths. Finding: screening drift is not a loop-closure
problem — *a withdrawal is topologically a line*.

**Campaign B — Deformation attempt 1: residual masking** *(2026-09-28; `586b1ac`)* —
rejected by its pre-stated criterion.

**Campaign C — Detector distillation → the screen-then-confirm cascade** *(2026-09-29 →
10-01; `d3c9ca5`, `f9f4424`, `169d919`, `0ea2116`, `a349e1b`, `8bcda7b`; `results/distill_protocol_v2.md`)*
Distillation v1 rejected by its gates (Dice saturates 0.888; speed 2.29× vs 2.5× bar).
Protocol v2 committed before any external evaluation; external pilot shows teacher advantage
vanishes off-Kvasir; adoption trial passes gates — **cascade adopted** (student screens 3.2–3.7×
faster, measured in-app; teacher confirms; full replacement withheld by a pre-stated clip-level
check).

**Campaign D — Deformation attempts 2–3** *(2026-10-04; `fa5abc4`, `37159b5`)* — in-loop
Huber and oscillation smoothing, pre-registered together, **both FAIL** their bar, each with a
mechanism.

**Campaign E — Deformation attempt 4: explicit wall-displacement model** *(2026-10-04;
`054a671`…`53d9475`)* — joint optimization with pose; a synthetic test exposed along-ray
displacement as projectively invisible; hyperparameters frozen from dev; **FAIL at +18.0% vs
the 20% bar** — "the narrowest and richest failure yet."

**Campaign F — Deformation attempt 5: adaptive gated field** *(2026-10-04; `3d2b27a`,
`65116d5`)* — dead at dev: no gate threshold beats always-on; mid-range gating harms controls.

**Campaign G — Tracker characterization pair** *(2026-10-04 → 10-05)*
*Huber characterization* (`398b2cc`, `cba83da`): the dormant robust toggle would drop the
headline mean 0.90 → 0.42 mm — measured as a characterization, explicitly not an adoption.
*Keyframe study* (`a70b104`, `990c325`, `2da85c5`): threshold frozen from dev (2 mm); frozen
test **FAIL** at median +5.6% vs the 15% bar, inconsistent across withdrawals.

**Campaign H — Deformation attempt 6: learned correspondence weighting** *(2026-10-05 →
10-06; `c0850ff`…`f9f6bac`)* — GT-pose-supervised labels; α frozen from dev (0.5 disqualified
by its own dev control); a judge amendment disclosed mid-campaign (one sequence ships no pose
file); frozen verdict **FAIL — the learned weights do not generalize to unseen segments**.
*Deformation total: six attempts, six rejections, each with a mechanism.*

**Campaign I — Depth speed push** *(2026-10-06; `300210b`, `7fe023d`)* — two resolution
candidates (196/168 px) under three gates; **both rejected**.

**20 — Pipeline v2: in-loop Huber, adopted** *(2026-10-06 → 10-08; `c2fc103`, `f15d2c5`,
`98d05a0`)*
Staged re-validation of every tracking benchmark, stop at first failed gate — all four stages
pass; **ADOPTED** (headline mean 0.91 → 0.42 mm). Native postscript: 2 of 3 improve under v2;
one apparent regression did not replicate; one post-adoption regression disclosed in the README.

**21 — Huber app-port trial** *(2026-10-10; `150bc56`, `7820271`;
`results/huber_port_gate.md`)*
Does the adopted upgrade transfer to the phone's own tracker? Gates P/A/S committed before any
fixture existed. **Gate A FAIL (−24% on one of two pre-registered windows)** — the phone keeps
its fixed-k kernel. The adoption was validated per configuration, never blindly generalized.

**22 — The dense-400 screening protocol, adopted** *(2026-10-10 → 10-11; `86ca8c5`,
`1f98c3e`, `e5e272e`; `results/dense_protocol_gate.md`)*
Dense autopsy of the first frozen test's 10 misses traced 8/10 to sparse frame sampling, not
blind detection. Candidate frozen (maxC≥14, calibration-derived), gates G1/G2/G3 committed
before any test3 clip was downloaded. Second frozen test on 48 fresh clips: **19/25 vs 13/25
paired on identical clips (p=0.0012)**, 2 clean false alarms within the pre-stated allowance,
other flagged "controls" containing labeled real pathology — **ADOPTED**.

### Verification layer (not iterations, standing machinery)

- **Numbers audit** *(2026-09-26; `54efd89`)* — 35/35 published figures reproduce from raw artifacts.
- **Red-team audit** *(2026-10-10; `b378d32`, `6146599`)* — 15/15 headline checks reproduce
  under independently written code; a fresh end-to-end v2 re-track reproduces all five headline
  ATEs and the 0.419 mm mean exactly.

**Baseline head-to-head characterization** *(pre-registered and run 2026-10-17; `e0e3117`;
`results/orbslam3_baseline_gate.md`)* — out-of-the-box ORB-SLAM3 (monocular, two frozen
configs, 5 runs each) vs the adopted pipeline v2 on the five C3VD headline sequences, answering
the judge question "why not just use the standard tool?" with a measurement. Result (the
pre-stated "mixed" outcome): **a draw on accuracy** — per-sequence 2–2–1, means 0.42 (Argus)
vs 0.39 mm (baseline, better config), overlapping bootstrap CIs — with the baseline tracking
93–99% of each sequence on sparse keyframes vs Argus posing every frame. Published per the
gate's outcome statement: the tracking claim is density + glare/illumination handling + the
measured envelope, not SOTA accuracy.

### Open

**23 (open) — Cross-session site re-identification** *(pre-registered 2026-10-17; `ed8a3a5`;
`results/site_reid_gate.md`)* — 5 qualified cross-session pairs, oracle registration
disclosed, gates G1 (median ≤10 mm) and G2 (80% discrimination) committed. **Not yet run.**

## 6. What was adopted vs. what the evidence rejected

Adopted (2, per the README tally): pipeline v2 (in-loop Huber), dense-400 screening protocol.
Also shipped, outside that tally's wording (see attestation item 4): the screen-then-confirm
cascade (Campaign C), depth int8 quantization (iteration 11).
Rejected by pre-stated criteria (partial list): glare removal as a SLAM aid, naive and AI
inpainting, six deformation fixes, three loop-closure generations, detector int8, NNAPI,
distillation v1 as replacement, keyframe tracking, two depth-resolution candidates, Huber
app-port, synthetic dye augmentation.
Retracted by internal audit (2): the padded agreement headline; the collapsed-alignment
full-withdrawal ATE.

## 7. What was checked, and what still needs the author's word

Checked against raw artifacts (no action needed): iteration 2's tracker-validation ATEs
(`results/ablation.csv` condition B: 0.32–0.99 mm, first sequence 0.38); iteration 6's
fully-monocular ATEs (`results/huber_study.csv` ate_recorded: 0.17–1.59 mm, mean 0.91);
the view-threshold sweep (`results/thr_sensitivity.csv`: 97.8%+ agreement at every setting);
pipeline v2's per-sequence ATEs (`results/redteam_retrack.csv`, red-team reproduced); the
README tally anchors (20 → 21 → 22 at `51fc3aa`, `afd2c5c`, `331b8c9`). The records contain
no numbering of campaigns 12–19; this ledger says so instead of inventing one.

Author attestation (the only open items):

- [ ] Nothing in Act III ran before its gate commit — the commit order is the evidence
      offered to judges, and only the author can attest the runs followed it.
- [ ] Any work done before 2026-08-15 outside git (the repo's first content commit): date it
      here or confirm there was none.
- [ ] The manuscript draft (`~/Downloads/Argus_Manuscript.docx`) covers iterations 1–11 only
      and predates all of Act III; the README cites "the manuscript's twenty-two-iteration
      design sequence." Update the manuscript or re-point the reference.
- [ ] Tally wording: the README counts "two adoptions" (pipeline v2, dense-400) while the
      cascade (Campaign C) and depth int8 (iteration 11) were also shipped — confirm the
      narrower sense of "adoption" is intentional.

---

*Reconstructed 2026-10-17 from the repository's own records. This file documents the project's
design process; the numbers themselves live in the README, the manuscript, and `results/`.*
