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

## Headline results (all vs. ground truth; depth numbers on held-out sequences)

| what | result |
|---|---|
| Tracking (fully monocular) | **0.91 mm** mean ATE across 5 sequences (0.17–1.59) |
| Blind-spot map agreement | **98.6 %** mean (97.6–99.5) after depth fine-tuning — was 74.5 % before |
| Depth accuracy (δ<1.25, LOSO held-out) | 68 % off-the-shelf → **92–99.7 %** fine-tuned |
| 3D reconstruction (median Chamfer) | **1.3–2.4 mm** on 4/5 sequences, from video alone |
| Monocular measurement (click-to-measure) | **0.4–1.3 mm** median error on 4–30 mm distances |
| Polyp segmentation (pre-trained, Kvasir-SEG) | Dice **0.96** (in-distribution; see docstring caveats) |
| Robustness | blur/frame-drop tolerant; flicker weakness found (7–16×) and **fixed** with illumination compensation |
| Cost of rushing | blind spots grow monotonically with withdrawal speed on all 5 sequences (e.g. 2.9 %→14.9 % at 6×) |

The scientific arc: the original hypothesis (specular glare breaks SLAM; removing it helps) was
**tested and largely disproven** (Act I — feature-based SLAM is RANSAC-robust; naive inpainting
hurts). The investigation produced an accurate from-scratch tracker, the project pivoted to
coverage mapping, a controlled ablation identified learned depth as the sole bottleneck, and
domain fine-tuning (leave-one-sequence-out validated) closed it.

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

Validated on the C3VD phantom (the only place trajectory ground truth exists); blind-spot
localization is region-level (65 % recall / 53 % precision at 5 mm), not vertex-level; very short
camera motions sit at the tracker's noise floor and drift in scale; depth fine-tuning generalizes
across sequences (LOSO-proven) but cross-patient/scope generalization needs broader data.

---

*Everything runs on one Windows workstation with a single RTX A1000 (8 GB) — including the
neural-network fine-tuning. GSDSEF 2026.*
