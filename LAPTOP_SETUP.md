# Taking Argus to the fair — laptop migration plan

The workstation stays home; the laptop runs everything through **Argus Studio** (one local web page
with every demo). **Golden rule: nothing happens for the first time on fair day — full dry run at
least a week early.**

## Step 0 — know your laptop (2 min)
Device Manager → Display adapters:
- **NVIDIA anything** → you have a GPU: demos run fast (~2 fps live pipeline), install CUDA torch.
- **Intel / AMD graphics** → CPU mode: everything still works, live pipeline ~0.5–1 fps
  (use `--stride 2` to double apparent speed), polyp finder ~2–3 s per image. Fine for a booth.

## Storage budget (laptop needs FAR less than the workstation)
| item | with NVIDIA GPU | without |
|---|---|---|
| Miniconda + env | ~7 GB | ~3 GB |
| repo clone | 0.1 GB | 0.1 GB |
| USB kit (below) | 1.2 GB | 1.2 GB |
| **total** | **~8.5 GB** | **~4.5 GB** |

## Step 1 — install (needs internet, ~45 min)
```
# install Miniconda first (miniconda.com), then:
conda create -n argus python=3.10 -y
conda activate argus

# torch — pick ONE:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch                                                      # no NVIDIA GPU

git clone https://github.com/bluMoonx/EndoscopicMonocularSLAM
cd EndoscopicMonocularSLAM
pip install -r requirements-argus.txt
```

## Step 2 — the USB kit (~1.2 GB), copied into the same paths inside the clone
| folder | size | powers |
|---|---|---|
| `models/depth_c3vd_demo/` | 95 MB | fine-tuned depth (live demo + Studio) |
| `models/polyp_seg/` | 72 MB | polyp finder (else auto-downloads, internet once) |
| `data/c3vd/cecum_t1_a_undist/` — `image01/`, `depth01_ftc/`, `K.txt` | ~150 MB | live phantom demo + click-to-measure |
| `data/c3vd/desc_t4_a_undist/` + `trans_t2_a_undist/` — same three items | ~250 MB | other click-to-measure frames |
| `data/hamlyn/rectified01/image01/` + `data/hamlyn/intrinsics_rectified01.txt` | 39 MB | in-vivo live demo |
| `data/kvasir_seg/` | 56 MB | polyp finder samples |
| `results/` (all .html + report cards + pngs) | ~150 MB | 3D models, dashboard, reports |

## Step 3 — run everything with ONE command
```
python speqslam/argus_studio.py
```
→ open **http://localhost:7860**. Every demo is behind that page:
live SLAM launcher (phantom + in-vivo) · polyp finder + auto-sizing · click-to-measure ·
interactive 3D models · procedure report cards · results dashboard. Works with WiFi OFF.

Native live-demo keys: SPACE pause · B blind-spot reveal · ESC quit (work in both windows).

## Step 4 — dry run checklist (a week early, on the actual laptop)
- [ ] `python speqslam/demo_app.py --selftest` passes (HUD line says which depth model loaded)
- [ ] Studio opens; ALL six stations clicked through like a judge would
- [ ] Live demo runs to the blind-spot reveal in both modes (phantom + in-vivo)
- [ ] Polyp finder works on a sample AND an upload
- [ ] Click-to-measure returns mm on all four frames
- [ ] 3D models rotate smoothly in the laptop browser; report cards + dashboard open
- [ ] Everything re-tested once with WiFi OFF

## Step 5 — the safety net (non-negotiable)
1. **Screen-record one full live-demo run on the workstation** (Win+G) including the blind-spot
   reveal. Copy to laptop + USB. If the live demo misbehaves mid-judging, play the recording.
2. **QR codes on the poster** to the hosted pages (dashboard, 3D colon, field guide artifacts) —
   they work from any phone, no laptop needed.
3. The USB stick carries: the whole kit, the recording, and a copy of the results/ pages.

## Fair morning
Charge the laptop. WiFi off. Run the Studio once before judging. Pin http://localhost:7860.
