"""Batch-study runner: workstation-grade pipeline over the manifest clips, resumable.
Each clip: 80 frames -> static-overlay mask -> zero-shot fine-tuned depth -> direct VO
(in-vivo mode) -> frozen detection gates (conf>=0.7, size 2-22mm) -> persistence measured
under THREE candidate geometries, all recorded:
  W  world-frame 10mm ball (the original frozen rule)
  C  camera-frame 10mm ball
  T  2D image-track ball, radius 0.12 of image width
Nothing is decided here; the calibration analysis picks ONE rule on the calib split,
freezes it, and only then reads the test split.

Usage: python speqslam/batch_run.py [minutes] [results_csv] [detector_weights]
       (default 8 minutes, results/batch_study/results.csv, original detector;
        resumes where it left off)
"""
import sys, os, csv, time, json
import numpy as np, cv2, torch
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
from polyp_detect import load_model, segment, DEV
from polyp_measure import diameter_mm

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 8.0) * 60
RES = sys.argv[2] if len(sys.argv) > 2 else "results/batch_study/results.csv"
ALT_WEIGHTS = sys.argv[3] if len(sys.argv) > 3 else None
COLS = ["file", "label", "split", "frames", "fired", "gated",
        "maxW", "maxC", "maxT", "path_mm", "secs"]

manifest = list(csv.DictReader(open("results/batch_study/manifest.csv")))
done = set()
if os.path.exists(RES):
    done = {r["file"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(COLS)

net = demo_app.DepthNet()
model = load_model()
if ALT_WEIGHTS:
    model.load_state_dict(torch.load(ALT_WEIGHTS, map_location=DEV, weights_only=True))
    print(f"detector weights: {ALT_WEIGHTS}", flush=True)
t_start = time.time()

def biggest(positions, radius):
    used = [False] * len(positions)
    best = 0
    for a in range(len(positions)):
        if used[a]:
            continue
        grp = 1; used[a] = True
        for b in range(a + 1, len(positions)):
            if not used[b] and np.linalg.norm(positions[a] - positions[b]) < radius:
                grp += 1; used[b] = True
        best = max(best, grp)
    return best

for row in manifest:
    if row["file"] in done:
        continue
    if time.time() - t_start > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    t0 = time.time()
    cap = cv2.VideoCapture(row["file"])
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(1, N) / 80))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if len(frames) < 20:
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([row["file"], row["label"], row["split"],
                                    len(frames), 0, 0, 0, 0, 0, 0, 0])
        print(row["file"], "unreadable", flush=True)
        continue
    n = len(frames)
    h, w = frames[0].shape[:2]
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], float)

    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    static = (stack.std(0) < 4.0).astype(np.uint8) * 255
    static = cv2.dilate(static, np.ones((7, 7), np.uint8))
    if (static > 0).mean() > 0.9:
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([row["file"], row["label"], row["split"],
                                    n, 0, 0, 0, 0, 0, 0, round(time.time()-t0, 1)])
        print(row["file"], "no scene", flush=True)
        continue

    deps = []
    for f in frames:
        d = net.predict(f)
        d[static > 0] = 0
        deps.append(d)

    pose = np.eye(4); poses = [pose.copy()]
    for i in range(1, n):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        bad = np.maximum(glare, static)
        T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0,
                          glare=bad, illum=True)
        pose = pose @ np.linalg.inv(T)
        poses.append(pose.copy())
    path_mm = 1000*sum(np.linalg.norm(poses[i][:3,3]-poses[i-1][:3,3]) for i in range(1, n))

    fired = 0
    posW, posC, posT = [], [], []
    for i, f in enumerate(frames):
        mask = segment(f, model)
        mask[static > 0] = 0
        binm = mask > 0.5
        if binm.mean() < 0.005:
            continue
        fired += 1
        conf = float(mask[binm].mean())
        if conf < 0.7:
            continue
        ys, xs = np.nonzero(binm)
        u0, v0 = int(xs.mean()), int(ys.mean())
        vals = deps[i][max(0, v0-2):v0+3, max(0, u0-2):u0+3]
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        z = float(np.median(vals))
        size = diameter_mm(mask.astype(np.float32), deps[i], K) or 0.0
        if not (2.0 <= size <= 22.0):
            continue
        Pc = np.array([(u0-K[0,2])/K[0,0]*z, (v0-K[1,2])/K[1,1]*z, z])
        T = poses[i]
        posW.append(T[:3, :3] @ Pc + T[:3, 3])
        posC.append(Pc)
        posT.append(np.array([u0 / w, v0 / w]))
    secs = time.time() - t0
    out = [row["file"], row["label"], row["split"], n, fired, len(posW),
           biggest(posW, 0.010), biggest(posC, 0.010), biggest(posT, 0.12),
           round(path_mm, 1), round(secs, 1)]
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow(out)
    print(f'{row["file"]} [{row["label"]}/{row["split"]}] fired {fired} gated {len(posW)} '
          f'W {out[6]} C {out[7]} T {out[8]}  ({secs:.0f}s)', flush=True)

pending = sum(1 for r in manifest if r["file"] not in done)
print(f"\nsession done; results -> {RES}", flush=True)
