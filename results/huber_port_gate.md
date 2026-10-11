# Pre-registration: pipeline-v2 (median-scaled Huber) port into the Argus Lite Kotlin tracker

Committed BEFORE any fixture is exported or any A/B number exists. Task chip task_36313b82.

## Background

The adopted workstation pipeline v2 weights residuals with a median-scaled Huber kernel
(`s = 1.4826*median|r| + 1e-6; a = |r|/(1.345 s); w = min(1, 1/a)`, direct_vo.py:93-96).
The app's Kotlin tracker (StudyEngine.trackPair) was found to already apply a FIXED-threshold
Huber (`w = 1 if |r| <= 0.08 else 0.08/|r|`). So the on-device question is not "robust vs
none" but "fixed-k vs median-scaled" — it is possible the fixed-k scheme already banks the
v2 benefit, and that outcome would keep fixed-k as the shipped default.

## Change under test

`DirectVo` weighting mode: FIXED_K (current shipped behavior, unchanged math) vs
MEDIAN_SCALED (v2 formula above, two-pass per Gauss-Newton iteration). All other tracker
code identical between arms.

## Fixtures (inputs, not results — may be generated after this commit)

Two 31-frame windows, stride 4 from frame 0 (stride 4 ≈ the app's video-streaming sampling
of 150 ms steps), from sequences never used to tune any tracker parameter:
- `data/c3vdv2/c1_ascending_t4_v1_s1`  (410 native frames on disk)
- `data/c3vdv2/c1_sigmoid1_t1_v1_s1`   (191 native frames on disk)

Each frame is converted exactly as the app does: 126x126 gray (0.299/0.587/0.114 luma),
depth from the app's own depth_q.onnx at 252 box-pooled to 126, one global scale
35 mm / median(1/disp) from frame 0 scene pixels, static mask all-false (C3VD frames carry
no burned-in overlay). Both arms consume identical fixtures, so any exporter/Android
resampling mismatch cancels in the A/B.

## Gates (decision rule: adopt MEDIAN_SCALED as shipped default iff P AND A AND S pass)

- **Gate P (parity):** the Kotlin median-scaled weight function reproduces Python
  reference weights on an exported residual vector, max abs difference < 1e-5.
- **Gate A (accuracy, JVM unit test on the fixtures):** Sim(3)-Umeyama mean-free RMSE ATE
  (umeyama_ate, the project-standard metric, computed by the Python judge on trajectories
  the test writes out) must improve by >= 5% under MEDIAN_SCALED on BOTH windows.
- **Gate S (on-device speed, benchmark card on the demo phone):** MEDIAN_SCALED trackPair
  median time <= 2.0x FIXED_K, AND <= 15% of the measured depth-stage time on the same
  device.

## Pre-stated outcomes

- All pass → MEDIAN_SCALED becomes the app default; fixed-k numbers disclosed beside it.
- Gate A fails (either window) → fixed-k stays the default; the flag ships off; the
  verdict is recorded as "the app's fixed-threshold Huber already captures the robustness
  benefit on app-like inputs" or "median-scaled does not transfer to the app's 126 px /
  3-level configuration", whichever the numbers support.
- Gate S fails → no adoption regardless of Gate A; cost documented.
- n=2 windows is small; requiring BOTH to improve is the conservative direction and is
  stated before any number exists.

---

## VERDICT (added after the trial ran — results section, Oct 10 2026)

- **Gate P: PASS.** Max weight difference Kotlin vs Python reference: 1.2e-7 (< 1e-5).
- **Gate A: FAIL.** Sim(3) Umeyama ATE on the two windows (constant-point baseline beside
  each, per project rule):
  - c1_ascending_t4_v1_s1: fixed-k **7.34 mm** vs median-scaled 9.09 mm
    (constant 15.36 mm) — gain **−23.8%**
  - c1_sigmoid1_t1_v1_s1: fixed-k **12.32 mm** vs median-scaled 12.18 mm
    (constant 12.76 mm) — gain +1.2%, below the 5% bar
- Gate S not run: the decision rule requires all three gates, so Gate A's failure ends
  the trial. **FIXED_K remains the shipped default**; MEDIAN_SCALED ships implemented but
  off (Vo.Mode), exactly as pre-stated.
- Tracker health identical in both arms (0 bails / 720 solves, ~5000 px/iteration), so
  the loss is in the weighting, not in starvation.
- Reading: consistent with the c1_t4 native-rate regression mechanism recorded in
  §4.20's postscript — at the app's 126 px / 3-level / small-motion operating point,
  residuals sit near the noise floor and a median-derived scale down-weights signal;
  the app's fixed 0.08 threshold is calibrated to that regime and keeps its job. The
  workstation adoption stands where it was validated; it does not transfer here, and
  the port trial is what caught that. Note the sigmoid window's honest footnote: at
  this short length the tracker only narrowly beats the constant-point baseline.
- Fixture caveat (stated pre-hoc): windows are 30 pairs each, n=2, exporter resampling
  approximates Android's; both arms shared identical inputs so the A/B comparison is
  unaffected, but the absolute mm values are not comparable to the workstation benchmarks.
