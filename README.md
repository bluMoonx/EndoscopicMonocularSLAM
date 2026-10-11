<img src="assets/argus_mark.svg" width="80" align="left" alt="Argus">

# Argus

**A fully monocular endoscopy system: raw video in → 3D colon reconstruction + blind-spot map out.**
Named for Argus Panoptes, the hundred-eyed watchman — this system watches every patch of colon
wall so nothing goes uninspected.

<br clear="left">

## The problem

Colonoscopies miss roughly **1 in 4 precancerous polyps**, largely because parts of the colon wall
are never actually looked at — hidden behind folds, passed too quickly, seen at the wrong angle —
and the doctor has no way of knowing *which* parts. Argus answers that question from the video of a
**standard, unmodified endoscope**: no depth sensor, no external tracking, no new hardware.

## The system

```
raw endoscope video → learned depth → direct visual odometry → TSDF fusion → blind-spot map
                     (fine-tuned         (from-scratch dense       (3D colon      (per-patch
                      Depth-Anything-V2)  photometric tracker)      mesh)          view counting)
```

Every stage is validated independently against the C3VD phantom's 6-DOF ground truth via a staged
ablation (swap each estimated input for ground truth to isolate error sources).

## Headline results (all vs. ground truth; every ATE printed beside its constant-point baseline)

| what | result |
|---|---|
| Tracking (fully monocular, pipeline v2) | **0.42 mm** mean ATE across 5 GT sequences (v1: 0.91; v2 = in-loop Huber, adopted after a four-stage re-validation of every tracking benchmark) |
| False blind-spot alarms | **7–41 % of wall → 0.5–3.9 %** after depth fine-tuning (the headline: generic AI fails in-body, targeted fine-tuning rescues it) |
| Depth accuracy (δ<1.25, LOSO held-out) | 68 % off-the-shelf → **92–99.7 %** fine-tuned |
| 3D reconstruction (median Chamfer) | **1.3–2.4 mm** on 4/5 sequences, from video alone |
| Frame-rate requirement (complete withdrawals) | tracking **fails at ~2 effective fps** (guard 16–23 %), **holds at ~9** (1.5–4.2 %, n=5, ~43 m GT path), **solved at native rate** (0.7–2.1 %, n=3) — real scopes stream 25+ |
| Coverage at screening pace | **~40 % of mapped wall genuinely under-seen**, validated against a perfect-tracking twin instrument (×3 withdrawals) |
| Unseen clinical footage, first frozen test (n=41) | **11/21 polyp clips · 3/20 controls flagged, all real pathology** (p=0.020) — the miss autopsy traced 8/10 misses to sparse frame sampling, not blind detection |
| Unseen clinical footage, second frozen test (n=48, fresh clips) | dense-sampling protocol **adopted**: **19/25 polyp clips (76%) vs 13/25 (52%)** for the old protocol, paired on identical clips (p=0.0012); 2 clean false alarms (pre-stated allowance), and the other flagged "controls" contained a labeled **cancer, tumor, anastomotic-leak cavity, and EMR site** |
| Polyp detection on-device | screen-then-confirm **cascade**: student screens every frame **3.2–3.7× faster** (measured in-app), teacher confirms — findings stay teacher-grade |
| Monocular measurement | **0.60 mm** median sizing error on 3–30 mm targets |
| Loop closure | **closed as a question**: a ground-truth oracle proves even perfect closures recover ≤14 % on screening paths — a withdrawal is topologically a line |
| Deformation | 9× cost measured; **six fixes attempted under pre-registered pass bars, six rejected with mechanisms** (best miss: −18 % vs a 20 % bar) |

**Method before results:** every experiment since the mid-project reboot carries a pass/fail
criterion committed to git *before* the test runs; failures are published with autopsies (see the
manuscript's twenty-two-iteration design sequence). Twenty-two iterations, **two adoptions** —
pipeline v2 (in-loop Huber, four-stage re-validation, one disclosed post-adoption regression) and
the dense screening protocol (adopted only after a second frozen test on 48 fresh clips passed
three pre-registered gates) — two self-caught retractions, a 35-figure audit that reproduces
every published number from raw artifacts, and a red-team audit (`speqslam/redteam_audit.py`, 15/15)
that re-derives the headline results with independently written code — a different alignment
algorithm (Horn, not Umeyama) matches every tracking ATE to 0.0005 mm, the clinical operating
point/tally/Fisher p re-derive from the frozen CSVs, and a fresh end-to-end re-track reproduces
the 0.42 mm headline mean exactly. Iteration 21 tested whether the adopted tracker
upgrade transfers to the phone's own tracker: it does not (−24% on one of two pre-registered
windows), so the phone keeps its fixed-threshold kernel — the adoption was validated per
configuration, never blindly generalized.

The scientific arc: the original hypothesis (glare breaks SLAM; removing it helps) was **tested
and largely disproven** (feature-based SLAM is RANSAC-robust; naive inpainting hurts). The
investigation produced an accurate from-scratch tracker, the project pivoted to coverage mapping,
a controlled ablation identified learned depth as the sole bottleneck, domain fine-tuning closed
it — and the same disease-and-cure then repeated in the polyp detector on unseen clinical footage,
making "competence ends at the edge of the training data" the project's central, twice-demonstrated
finding.

## Argus Lite — the whole lab on an Android phone

The complete pipeline (quantized depth, guarded tracking, fusion, cascade polyp detection, the 3D
map viewer with blind-spot and vs-ground-truth overlays) runs on-device in `arguslite/`. The app's
**"Measure this phone"** card re-benchmarks the models live on whatever device it's running on —
no number is quoted that the judge can't re-measure in hand.

## Argus Studio — every demo in one place

```
python speqslam/argus_studio.py        →  http://localhost:7860  (LAN: http://<hostname>:7860)
```

A local web console with ten stations, fully offline once models are on disk:

1. **Live SLAM pipeline** — launch the real-time demo (9 sequences: 5 phantom + 4 in-vivo);
   native windows show raw | glare mask | learned depth with a LIVE COVERAGE readout, plus the
   growing 3D reconstruction (press **B** for the blind-spot reveal)
2. **Polyp finder + auto-sizing** — sample gallery or upload; detection outline + estimated size
3. **Click-to-measure** — click two points on a frame, get real millimeters from monocular depth
4. **Glare lab** — interactive specular-detection threshold on real in-vivo frames
5. **Depth X-ray** — off-the-shelf vs. fine-tuned depth side by side (the key result, visually)
6. **Trajectory viewer** — estimated vs. ground-truth camera path in interactive 3D
7. **Evidence graphs** — ablation, robustness, withdrawal-speed, and depth charts (interactive)
8. **Interactive 3D models** — rotatable colon reconstructions with blind-spot heatmaps
9. **Procedure report cards** — per-sequence inspection quality reports (coverage grade, blind
   regions in mm², camera speed)
10. **Results dashboard** — the full evidence, one page

## Repository layout

| Path | Purpose |
|---|---|
| `speqslam/argus_studio.py` | The Studio (Flask, port 7860) |
| `speqslam/demo_app.py` | Live pipeline demo (native windows, CPU fallback) |
| `speqslam/direct_vo.py` | From-scratch dense photometric VO (SE(3) Gauss–Newton, glare-aware + illumination-compensation options) |
| `speqslam/train_depth.py` | Depth-Anything-V2 fine-tuning (leave-one-sequence-out) |
| `speqslam/fuse_monocular.py` | TSDF fusion + Chamfer evaluation |
| `speqslam/stage_ablation.py` | The A/B/C/D + C′/D′ ablation ladder |
| `speqslam/end2end_coverage.py` | Blind spots on the self-built mesh (vertex + region metrics) |
| `speqslam/measure.py` / `polyp_measure.py` | Monocular measurement + polyp find-and-size |
| `speqslam/polyp_detect.py` | Pre-trained polyp segmentation (DeepLabV3+/EfficientNet-B4) |
| `speqslam/robustness.py` / `withdrawal_speed.py` | Stress envelope + cost-of-rushing experiments |
| `speqslam/report_card.py` | Procedure report generator |
| `speqslam/endoslam_pipeline.py` | Cross-dataset evaluation (EndoSLAM ex-vivo porcine) |
| `speqslam/glare_removal.py`, `ai_inpaint.py`, `temporal_inpaint.py` | Act I: the glare study |
| `PROJECT_GUIDE.html` | The two-layer (plain + technical) field guide |
| `LAPTOP_SETUP.md` | Fair-day migration plan |
| `data/`, `results/`, `models/` | Datasets / outputs / weights — **not tracked** |

## Honest limitations

Validated on the C3VD phantom (the only place trajectory ground truth exists); residual
whole-colon drift (~100 mm) is proven unfixable by loop closure and resists the per-meter levers
tested; tissue deformation costs ~9× and six candidate fixes are excluded by their own pre-stated
criteria; blind-spot localization is region-level, not vertex-level; processing is near-real-time
on-device but not yet at video rate (the ViT depth network is the measured bottleneck, and the
resolution lever was evaluated and rejected); cross-patient/scope generalization needs data that
doesn't publicly exist. Each limitation ships with its mechanism — that's the point.

---

*Everything runs on one Windows workstation with a single RTX A1000 (8 GB) — including all
neural-network fine-tuning — plus one ordinary Android phone. GSDSEF 2027.*
