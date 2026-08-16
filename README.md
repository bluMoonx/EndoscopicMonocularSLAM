# Argus: End-to-End Specular Removal for Endoscopic Monocular SLAM

**Adaptive specular-glare removal for monocular endoscopic SLAM.**

## What Argus is

Argus is a **preprocessing layer** that sits between raw endoscopic video and a SLAM system
(ORB-SLAM3). Inside the body, the endoscope's own light reflects off wet tissue as bright
**specular glare**. That glare corrupts the visual landmarks a SLAM system relies on to figure
out where the camera is and to build a 3-D map of the surface — leaving gaps where lesions or
polyps could hide. Argus detects that glare and removes it *before* the SLAM system sees it, and
then rigorously measures whether doing so actually improves camera tracking.

## The goal

Determine **whether — and how — removing specular glare affects monocular endoscopic SLAM**, by
treating glare removal as a controlled variable in an ablation study across video sequences that
span a range of glare severity. The headline output is a **severity curve**: how the benefit (or
harm) of glare removal scales with how much glare a sequence contains.

## Design

1. **Glare detection** — a per-frame *adaptive* brightness threshold produces a specularity mask
   (isolating the bright, view-dependent highlights without eating real tissue).
2. **Glare removal — the conditions being compared:**
   - `baseline` — raw frames, no removal (control)
   - `Telea` — classical inpainting (smooth fill)
   - `LaMa` — deep/AI inpainting (fills the hole with realistic texture)
3. **Evaluation** — run monocular ORB-SLAM3 on each condition and measure *tracking health*
   (keyframes, tracking survival), with **repeated runs per condition** to control for the
   tracker's run-to-run randomness.
4. **Severity curve** — correlate each sequence's glare severity with the measured effect of
   removal. A rising line means removal helps more when there's more glare.

## Intended outcome

A quantified, mechanistic answer to *"does removing endoscopic glare help feature-based monocular
SLAM, and if so, which removal strategy works?"* — explicitly including the honest possibility
that naive removal is counterproductive (because it can starve an already feature-poor tracker),
and testing whether a feature-preserving AI approach changes that.

## Repository layout

| Path | Purpose |
|---|---|
| `speqslam/` | The Argus layer + experiment tooling (glare detection & removal, AI inpainting, run matrix, analysis). |
| `configs/` | Per-sequence ORB-SLAM3 monocular camera configs. |
| `PROJECT_GUIDE.html` | Plain-English + technical project guide. |
| `data/` | Datasets — **not tracked** (large; downloaded separately). |
| `results/` | Trajectories, plots, experiment output — **not tracked** (generated). |

## Running (outline)

1. Detect + remove glare on a sequence: `python speqslam/glare_removal.py --in_seq … --out_seq …`
   (or `ai_inpaint.py` for the LaMa condition).
2. Run monocular ORB-SLAM3 on the original and cleaned sequences (Dockerized ORB-SLAM3).
3. Aggregate and plot: `python speqslam/analyze_severity.py` → the severity curve.

*Argus runs entirely on a single Windows workstation with one GPU — no cluster required.*
