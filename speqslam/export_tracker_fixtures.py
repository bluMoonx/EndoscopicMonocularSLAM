"""Export JVM-test fixtures for the pre-registered Huber app-port trial
(results/huber_port_gate.md). Replicates the app's input conventions exactly:
126x126 gray (0.299/0.587/0.114 luma), depth from the app's own depth_q.onnx at
252 box-pooled 2x2 to 126, one global scale 35 / median(1/disp) from the first
sampled frame's scene pixels (Kotlin's inv[size/2] median, no even averaging).
Also writes the golden residual/weight vectors for the parity gate (Gate P).

Inputs only - produces no A/B number. Fixtures land in data/trackerab_fixtures/.
"""
import json, os, struct, sys
import numpy as np
import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "trackerab_fixtures")
MODEL = os.path.join(ROOT, "arguslite", "app", "src", "main", "assets", "depth_q.onnx")
SEQS = ["c1_ascending_t4_v1_s1", "c1_sigmoid1_t1_v1_s1"]
STRIDE, NFRAMES = 4, 31          # 30 pairs per window, from frame 0
W, SZ, GLARE = 126, 252, 0.94

os.makedirs(OUT, exist_ok=True)
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def app_gray(img_rgb):
    s = cv2.resize(img_rgb, (W, W), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (0.299 * s[:, :, 0] + 0.587 * s[:, :, 1] + 0.114 * s[:, :, 2]) / 255.0

def app_disp(img_rgb):
    s = cv2.resize(img_rgb, (SZ, SZ), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((s - MEAN) / STD).transpose(2, 0, 1)[None]
    d = sess.run(None, {"pixels": x})[0][0]        # 252 x 252 disparity
    # box-pool 2x2 -> 126, exactly DepthEngine.disparityGrid
    return d.reshape(W, 2, W, 2).mean(axis=(1, 3))

for seq in SEQS:
    sdir = os.path.join(ROOT, "data", "c3vdv2", seq)
    names = sorted(os.listdir(os.path.join(sdir, "rgb")))
    gt_all = [l for l in open(os.path.join(sdir, "pose.txt")) if l.strip()]
    nums = [int(os.path.splitext(n)[0]) for n in names]
    off = 1 if max(nums) >= len(gt_all) else 0     # the project's 1-based GT rule
    idx = list(range(0, STRIDE * NFRAMES, STRIDE))
    assert idx[-1] < len(names), f"{seq}: window exceeds {len(names)} frames"

    grays, disps, gts = [], [], []
    for k, i in enumerate(idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(sdir, "rgb", names[i])), cv2.COLOR_BGR2RGB)
        grays.append(app_gray(img))
        disps.append(app_disp(img))
        v = [float(x) for x in gt_all[nums[i] - off].split(",") if x.strip()]
        gts.append(v[12:15])                        # camera position, mm (row-major 4x4)
        print(f"{seq} {k+1}/{NFRAMES}", flush=True)

    # one global scale from the first sampled frame's scene pixels (Kotlin rule)
    d0, g0 = disps[0].ravel(), grays[0].ravel()
    inv = np.sort(1.0 / d0[(d0 > 1e-6) & (g0 < GLARE)])
    scale = 35.0 / inv[len(inv) // 2]
    depths = [np.where(d > 1e-6, scale / d, 0.0).astype(np.float32) for d in disps]

    sd = os.path.join(OUT, seq)
    os.makedirs(sd, exist_ok=True)
    np.stack(grays).astype("<f4").tofile(os.path.join(sd, "gray.bin"))
    np.stack(depths).astype("<f4").tofile(os.path.join(sd, "depth.bin"))
    with open(os.path.join(sd, "gt.csv"), "w") as f:
        for t in gts:
            f.write(f"{t[0]},{t[1]},{t[2]}\n")
    print(f"{seq}: scale {scale:.2f}, GT offset {off}, wrote {sd}")

# Gate P golden vectors: inlier-shaped residuals plus a heavy-tail minority
rng = np.random.default_rng(0)
r = np.concatenate([rng.normal(0, 0.02, 900), rng.uniform(-0.5, 0.5, 100)]).astype(np.float64)
s = 1.4826 * np.median(np.abs(r)) + 1e-6
w = np.minimum(1.0, 1.0 / (np.abs(r) / (1.345 * s)))
r.astype("<f4").tofile(os.path.join(OUT, "residuals.bin"))
w.astype("<f4").tofile(os.path.join(OUT, "weights.bin"))
json.dump({"seqs": SEQS, "stride": STRIDE, "nframes": NFRAMES, "w": W},
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("golden residual/weight vectors written; export complete")
