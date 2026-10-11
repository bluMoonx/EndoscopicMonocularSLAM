"""DENSE SCREENING PROTOCOL - development candidate for the detection campaign.

The 41-clip frozen result (11/21, 3/20, maxC>=14 at 80 frames/clip) stands as
published. This runner evaluates one candidate improvement: sample up to 400
frames per clip instead of 80, keep every detection gate identical (fire bar
0.5% of frame, conf>=0.7, size 2-22mm, static-overlay zeroing, maxC = largest
camera-frame cluster within 10mm), and RE-DERIVE the operating threshold on the
same calibration split under the same zero-false-alarm rule.

STATUS OF THIS DATA: the original test/test2 clips are hereby DEVELOPMENT data
for the improvement campaign (the 11/21 burned them as a frozen result; nothing
measured here is publishable as test evidence). The campaign's publishable test
is a pre-registered gate on fresh, never-downloaded clips (test3).

maxC was the frozen operating geometry and needs NO tracking (camera-frame
clustering), so this runs segmentation on every sampled frame and depth only on
firing frames. Checkpointed; relaunch until ALL DONE.
Writes results/detect_dense.csv.
"""
import csv, os, sys, time
import numpy as np, cv2, torch

sys.path.insert(0, "speqslam")
import demo_app
from polyp_detect import load_model, segment, DEV
from polyp_measure import diameter_mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 27.0) * 60
RES = sys.argv[2] if len(sys.argv) > 2 else "results/detect_dense.csv"
NFR = 400

MANIFESTS = sys.argv[3].split(",") if len(sys.argv) > 3 else [
    "results/batch_study/manifest.csv", "results/batch_study/manifest_ext.csv"]
rows = []
for f in MANIFESTS:
    rows += list(csv.DictReader(open(f)))

done = set()
if os.path.exists(RES):
    done = {r["file"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["file", "label", "split", "sampled", "fired",
                                "gated", "maxC_dense", "secs"])

model = load_model()
model.load_state_dict(torch.load("models/polyp_ft/dlv3p_ft_ep8.pth",
                                 map_location=DEV, weights_only=True))
model.eval()
net = demo_app.DepthNet()

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

t_start = time.time()
for row in rows:
    if row["file"] in done:
        continue
    if time.time() - t_start > BUDGET_S:
        print("budget reached; relaunch to continue", flush=True)
        sys.exit(0)
    t0 = time.time()
    cap = cv2.VideoCapture(row["file"])
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = min(NFR, max(1, N))
    frames = []
    for i in range(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(1, N) / want))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if len(frames) < 20:
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([row["file"], row["label"], row["split"],
                                    len(frames), 0, 0, 0, 0])
        continue
    h, w = frames[0].shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    sub = frames[:: max(1, len(frames) // 80)]
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sub]).astype(np.float32)
    static = (stack.std(0) < 4.0).astype(np.uint8) * 255
    static = cv2.dilate(static, np.ones((7, 7), np.uint8))
    if (static > 0).mean() > 0.9:
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([row["file"], row["label"], row["split"],
                                    len(frames), 0, 0, 0, round(time.time() - t0, 1)])
        print(row["file"], "no scene", flush=True)
        continue

    fired = 0; posC = []; fidx = []
    for fi, f in enumerate(frames):
        mask = segment(f, model)
        mask[static > 0] = 0
        binm = mask > 0.5
        if binm.mean() < 0.005:
            continue
        fired += 1
        conf = float(mask[binm].mean())
        if conf < 0.7:
            continue
        dep = net.predict(f)
        dep[static > 0] = 0
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
        posC.append(np.array([(u0 - K[0, 2]) / K[0, 0] * z,
                              (v0 - K[1, 2]) / K[1, 1] * z, z]))
        fidx.append(fi)
    mc = biggest(posC, 0.010)
    # every gated fire, for offline support-rule analysis (time-binned variants etc.)
    os.makedirs("results/detect_dense_fires", exist_ok=True)
    stem = os.path.splitext(os.path.basename(row["file"]))[0]
    np.savez(f"results/detect_dense_fires/{stem}.npz",
             fidx=np.array(fidx, np.int32),
             pos=np.array(posC) if posC else np.zeros((0, 3)),
             sampled=len(frames))
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([row["file"], row["label"], row["split"], len(frames),
                                fired, len(posC), mc, round(time.time() - t0, 1)])
    print(f'{os.path.basename(row["file"])} [{row["label"]}/{row["split"]}] '
          f'fired {fired}/{len(frames)} gated {len(posC)} maxC {mc} '
          f'({time.time() - t0:.0f}s)', flush=True)

print("ALL DONE", flush=True)
