"""Re-track the full withdrawal with a divergence guard.

Forensics on the first pass: 24 of 699 frame pairs at screening speed (12x temporal
subsampling) made the photometric solver diverge - single steps of up to 457 METERS
against a 22 mm median scene depth - teleporting the rest of an otherwise sane
trajectory (median step 6.1 mm). Umeyama alignment then collapsed the exploded
cloud onto the GT centroid, which is what made the first-pass "ATE 3.5%" figure
meaningless (a constant-point predictor scores the same; that number is retracted).

The guard: a frame-to-frame step is rejected as physically impossible if its
translation exceeds 60 mm (10x the median step, ~3x the median scene depth) or its
rotation exceeds 35 degrees; the pose then coasts on the previous step (constant
velocity) instead. The guard is a deployment-grade sanity check, disclosed wherever
the numbers are shown; the tracker itself is untouched.

Outputs poses_est2.npy + honest evaluation: guarded ATE, rejected-pair count, and
the constant-point baseline every ATE must beat to mean anything.
"""
import os, glob, sys, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
from c3vdv2_deform import load_poses, umeyama_ate

BASE = "data/c3vdv2/c1_full_t4_v1"
OUT = f"{BASE}/poses_est2.npy"
K = np.array([[250.0, 0, 320.0], [0, 250.0, 256.0], [0, 0, 1]])
MAX_STEP_M = 0.060
MAX_ROT_DEG = 35.0

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
net = demo_app.DepthNet()

start = 1
poses = [np.eye(4)]
rejected = []
if os.path.exists(OUT):
    poses = list(np.load(OUT))
    start = len(poses)
prev_img = cv2.imread(und[start - 1])
prev_dep = net.predict(prev_img)
prev_delta = np.eye(4)
t0 = time.time()
for i in range(start, n):
    img = cv2.imread(und[i])
    dep = net.predict(img)
    g0 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    glare = specular_mask(prev_img, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
    T = dv.track_pair(g0, prev_dep, g1, K, 4, robust=False, depth_scale=1.0,
                      glare=glare, illum=True)
    step = float(np.linalg.norm(T[:3, 3]))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    if step > MAX_STEP_M or ang > MAX_ROT_DEG:
        T = prev_delta.copy()          # coast: constant velocity through the bad pair
        rejected.append(i)
    else:
        prev_delta = T.copy()
    poses.append(poses[-1] @ np.linalg.inv(T))
    prev_img, prev_dep = img, dep
    if i % 50 == 0:
        np.save(OUT, np.array(poses))
        print(f"tracked {i}/{n} ({time.time()-t0:.0f}s, rejected {len(rejected)})",
              flush=True)
np.save(OUT, np.array(poses))
print(f"done: {len(rejected)} pairs guarded at {sorted(rejected)[:30]}", flush=True)

nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]
est = np.array([p[:3, 3] for p in poses])
steps = np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000
path = float(steps.sum())
ate = umeyama_ate(gt, est) * 1000
base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
es = np.linalg.norm(np.diff(est, axis=0), axis=1)
print(f"est step m: p50 {np.percentile(es,50):.4f} p99 {np.percentile(es,99):.4f} "
      f"max {es.max():.3f}", flush=True)
print(f"ATE {ate:.1f} mm over {path:.0f} mm GT path ({100*ate/path:.1f}%) | "
      f"constant-point baseline {base:.1f} mm | guarded pairs {len(rejected)}/{n-1}",
      flush=True)
