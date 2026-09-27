"""Run the frozen polyp protocol over the full withdrawal (stride-3 frames).

Same deployed pieces as the batch study: fine-tuned detector (dlv3p_ft_ep8),
frozen gates (mask conf >= 0.7, size 2-22 mm), camera-frame 10 mm-ball
persistence clustering. Calibrated intrinsics (undistorted 250/250/320/256).

Honest framing printed with the result: the maxC >= 14 rule was frozen on
80-frame clip sampling; this sequence is sampled ~35x denser, so dwell inflates
cluster sizes. We therefore report the raw evidence (fired / gated / cluster
sizes and their frame spans) and export overlays of the strongest clusters for
visual verdicts, rather than declaring detections by the clip-rule alone.
Outputs: results/screening_polyp.json + results/screening_polyp/*.jpg
"""
import os, sys, glob, json, time
import numpy as np, cv2, torch
sys.path.insert(0, "speqslam")
import demo_app
from polyp_detect import load_model, segment, DEV
from polyp_measure import diameter_mm

BASE = (sys.argv[1].rstrip("/\\") if len(sys.argv) > 1
        else "data/c3vdv2/c1_full_t4_v1_s3")
sys.argv = sys.argv[:1]
SEQ = os.path.basename(BASE).replace("_s3", "")
OUT = f"results/screening_polyp_{SEQ}" if SEQ != "c1_full_t4_v1" \
    else "results/screening_polyp"
WEIGHTS = "models/polyp_ft/dlv3p_ft_ep8.pth"
K = np.array([[250.0, 0, 320.0], [0, 250.0, 256.0], [0, 0, 1]])

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
net = demo_app.DepthNet()
model = load_model()
model.load_state_dict(torch.load(WEIGHTS, map_location=DEV, weights_only=True))
print(f"{n} frames | detector {WEIGHTS}", flush=True)

os.makedirs(OUT, exist_ok=True)
fired = 0
det = []   # (frame, camera-frame xyz m, size mm, conf, u0, v0)
t0 = time.time()
for i in range(n):
    img = cv2.imread(und[i])
    mask = segment(img, model)
    binm = mask > 0.5
    if binm.mean() < 0.005:
        continue
    fired += 1
    conf = float(mask[binm].mean())
    if conf < 0.7:
        continue
    dep = net.predict(img)
    ys, xs = np.nonzero(binm)
    u0, v0 = int(xs.mean()), int(ys.mean())
    vals = dep[max(0, v0 - 2):v0 + 3, max(0, u0 - 2):u0 + 3]
    vals = vals[vals > 0]
    if len(vals) == 0:
        continue
    z = float(np.median(vals))
    size = diameter_mm(mask.astype(np.float32), dep, K) or 0.0
    if not (2.0 <= size <= 22.0):
        continue
    det.append((i, np.array([(u0 - K[0, 2]) / K[0, 0] * z,
                             (v0 - K[1, 2]) / K[1, 1] * z, z]), size, conf, u0, v0))
    if i % 400 == 0:
        print(f"{i}/{n} fired {fired} gated {len(det)} ({time.time()-t0:.0f}s)",
              flush=True)
print(f"scan done: fired {fired}/{n} frames, {len(det)} gated detections", flush=True)

np.savez(f"{OUT}_det.npz",
         frames=np.array([d[0] for d in det]),
         pos=np.array([d[1] for d in det]),
         size=np.array([d[2] for d in det]),
         conf=np.array([d[3] for d in det]),
         uv=np.array([[d[4], d[5]] for d in det]))

# DWELL EVENTS, not whole-video camera-frame clusters: a long moving sequence
# aliases "same position relative to the lens" across distant anatomy, so
# membership additionally requires temporal contiguity (<= 10 frames ~ 1 s gap).
pos = [d[1] for d in det]
used = [False] * len(det)
clusters = []
for a in range(len(det)):
    if used[a]:
        continue
    grp = [a]; used[a] = True
    last = det[a][0]
    for b2 in range(a + 1, len(det)):
        if used[b2] or det[b2][0] - last > 10:
            if det[b2][0] - last > 10:
                break
            continue
        if min(np.linalg.norm(pos[g] - pos[b2]) for g in grp) < 0.010:
            grp.append(b2); used[b2] = True
            last = det[b2][0]
    clusters.append(grp)
clusters.sort(key=len, reverse=True)

report = []
for ci, grp in enumerate(clusters[:6]):
    fr = sorted(det[j][0] for j in grp)
    sizes = [det[j][2] for j in grp]
    confs = [det[j][3] for j in grp]
    report.append(dict(members=len(grp), frame_first=fr[0], frame_last=fr[-1],
                       span_frames=fr[-1] - fr[0] + 1,
                       size_mm=round(float(np.median(sizes)), 1),
                       conf=round(float(np.mean(confs)), 2)))
    # overlay montage: up to 4 member frames
    for k2, j in enumerate(sorted(grp, key=lambda j: -det[j][3])[:4]):
        i, _, size, conf, u0, v0 = det[j]
        img = cv2.imread(und[i])
        mask = segment(img, model)
        edge = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        img[edge > 0] = (90, 255, 90)
        cv2.putText(img, f"cluster {ci} frame {i} conf {conf:.2f} ~{size:.0f}mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 255, 90), 2)
        cv2.imwrite(f"{OUT}/c{ci}_{k2}_f{i:04d}.jpg", img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

strong = sum(1 for g in clusters if len(g) >= 15)
res = dict(frames=n, fired=fired, gated=len(det),
           dwell_events_ge15=strong,
           clusters_top=report,
           note="dwell-event clustering: camera-frame 10mm ball + temporal "
                "contiguity (<=10-frame gap); frozen gates conf>=0.7 size 2-22mm; "
                ">=15 members ~ the live pipeline's support bar at this ~10fps "
                "sampling. Phantom contains deliberate debris; no polyp "
                "annotations exist, so events are verdicts-by-eye")
json.dump(res, open(f"{OUT}.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
