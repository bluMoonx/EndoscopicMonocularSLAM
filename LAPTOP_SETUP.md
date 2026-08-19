# Taking Argus to the fair — laptop migration plan

The workstation stays home; the laptop runs the demo. This is the complete checklist.
**Do a full dry run on the laptop at least a week before GSDSEF.**

## What lives where

| Thing | On GitHub? | Size | How it gets to the laptop |
|---|---|---|---|
| All code (`speqslam/`, `configs/`) | ✅ yes | ~1 MB | `git clone` |
| Field guide + README | ✅ yes | ~1 MB | comes with the clone |
| Python environment | ❌ no | ~2–8 GB | reinstall from `requirements-argus.txt` (below) |
| Demo depth model (`models/depth_c3vd_demo/`) | ❌ no | ~100 MB | USB drive or cloud drive |
| Demo footage (`data/c3vd/cecum_t1_a_undist/image01/` + `K.txt`) | ❌ no | ~35 MB | USB drive |
| Result pages (`results/*.html` — dashboard, 3D models) | ❌ no | ~25 MB | USB drive (they also live at the claude.ai artifact links) |
| Raw datasets, depth maps, meshes, .npz | ❌ no | ~20 GB | **NOT needed** — leave home |

**Total on the laptop: under 1 GB plus the Python environment.** Any laptop made this decade has room.

## The demo app only needs three things
1. The cloned repo
2. `data/c3vd/cecum_t1_a_undist/image01/` (the frames) + `K.txt` — it computes everything else live
3. `models/depth_c3vd_demo/` for the fine-tuned depth (without it, it falls back to downloading
   the off-the-shelf model on first run — needs internet once)

## Setup steps on the laptop

```
# 1. install Miniconda (or Anaconda), then:
conda create -n argus python=3.10 -y
conda activate argus

# 2. PyTorch — pick ONE line:
#    laptop HAS an NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124
#    laptop has NO NVIDIA GPU (Intel/AMD graphics):
pip install torch

# 3. everything else:
pip install -r requirements-argus.txt

# 4. clone + copy
git clone https://github.com/bluMoonx/EndoscopicMonocularSLAM
#    then copy from USB into the repo folder:
#      models/depth_c3vd_demo/
#      data/c3vd/cecum_t1_a_undist/   (image01/ + K.txt is enough)
#      results/  (the .html pages, optional)

# 5. run it
cd EndoscopicMonocularSLAM
python speqslam/demo_app.py --seq cecum_t1_a
```

## Performance expectations
- **With an NVIDIA GPU:** ~2 fps, same as the workstation.
- **Without one (CPU only):** the depth network is the slow part; expect ~0.5–1 fps.
  Still fine for a booth — judges see it computing. If it feels too slow, run with
  `--stride 2` (every other frame, twice the apparent speed).

## The safety net (do all three)
1. **Screen-record a full demo run** on the workstation (Win+G → record). Play the video if the
   live demo misbehaves at the booth. A judge watching a recording of your real system is fine;
   a frozen window is not.
2. The **hosted pages** work from any browser, no laptop needed:
   dashboard, interactive 3D colon, monocular-vs-GT overlay (claude.ai artifact links; put QR
   codes on the poster).
3. Keep the USB drive with the full kit (model + footage + HTMLs + the recording) in your pocket.

## Dry-run checklist (a week early)
- [ ] Clone + env install completes on the laptop
- [ ] `python speqslam/demo_app.py --selftest` passes
- [ ] Full demo run: three panels + 3D window + blind-spot reveal all work
- [ ] Result HTMLs open in the laptop's browser
- [ ] Screen recording plays
