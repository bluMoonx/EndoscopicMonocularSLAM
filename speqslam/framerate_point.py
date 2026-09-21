"""One point on the frame-rate requirement curve: track a subsampled view of an
already-undistorted full-withdrawal frame set with the frozen pipeline + guard,
then report guarded ATE, the constant-point baseline, and the guard rate.

Usage: python speqslam/framerate_point.py <base> <sub> [minutes]
  base: data/c3vdv2/c1_full_t4_v1     (stride-12 frames)  or  ..._s3 (stride-3)
  sub:  integer subsample of that set (base stride x sub = effective stride)
Checkpoints per (base, sub); budget-resumable.
"""
import os, sys, glob, time
import numpy as np, cv2

BASE = sys.argv[1]
SUB = int(sys.argv[2])
BUDGET_S = (float(sys.argv[3]) if len(sys.argv) > 3 else 9.0) * 60
sys.argv = sys.argv[:1]        # c3vdv2_deform parses sys.argv at import time
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
from c3vdv2_deform import load_poses, umeyama_ate
K = np.array([[250.0, 0, 320.0], [0, 250.0, 256.0], [0, 0, 1]])
MAX_STEP_M = 0.060
MAX_ROT_DEG = 35.0
POSE_CK = f"{BASE}/poses_sub{SUB}.npy"
REJ_CK = f"{BASE}/rejected_sub{SUB}.npy"

frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
nums_all = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
und_all = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
assert len(und_all) == len(frames_raw), "run the base study first (undistort)"
und = und_all[::SUB]
nums = nums_all[::SUB]
n = len(und)
gt_all = load_poses(f"{BASE}/pose.txt")
print(f"{n} frames (sub {SUB} of {BASE})", flush=True)

net = demo_app.DepthNet()
poses = [np.eye(4)]
rejected = []
if os.path.exists(POSE_CK):
    poses = list(np.load(POSE_CK))
    if os.path.exists(REJ_CK):
        rejected = list(np.load(REJ_CK))
start = len(poses)
t0 = time.time()
if start < n:
    prev_img = cv2.imread(und[start - 1])
    prev_dep = net.predict(prev_img)
    prev_delta = np.eye(4)
    for i in range(start, n):
        img = cv2.imread(und[i])
        dep = net.predict(img)
        g0 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        glare = specular_mask(prev_img, method="adaptive", k=2.0, lo=225, hi=245,
                              dilate=1)
        T = dv.track_pair(g0, prev_dep, g1, K, 4, robust=False, depth_scale=1.0,
                          glare=glare, illum=True)
        step = float(np.linalg.norm(T[:3, 3]))
        ang = float(np.degrees(np.arccos(
            np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
        if step > MAX_STEP_M or ang > MAX_ROT_DEG:
            T = prev_delta.copy()
            rejected.append(i)
        else:
            prev_delta = T.copy()
        poses.append(poses[-1] @ np.linalg.inv(T))
        prev_img, prev_dep = img, dep
        if i % 25 == 0:
            np.save(POSE_CK, np.array(poses))
            np.save(REJ_CK, np.array(rejected))
            print(f"tracked {i}/{n} (rejected {len(rejected)})", flush=True)
        if time.time() - t0 > BUDGET_S:
            np.save(POSE_CK, np.array(poses))
            np.save(REJ_CK, np.array(rejected))
            print(f"budget at pair {i}; resume me", flush=True)
            sys.exit(0)
    np.save(POSE_CK, np.array(poses))
    np.save(REJ_CK, np.array(rejected))

poses = list(np.load(POSE_CK))
rejected = list(np.load(REJ_CK)) if os.path.exists(REJ_CK) else []
est = np.array([p[:3, 3] for p in poses])
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]
path = float((np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000).sum())
ate = umeyama_ate(gt, est) * 1000
base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
print(f"POINT {BASE} sub{SUB}: ATE {ate:.1f} mm ({100*ate/path:.2f}% of "
      f"{path:.0f} mm) | baseline {base:.1f} mm | ATE/baseline "
      f"{ate/base:.2f} | guarded {len(rejected)}/{len(poses)-1} "
      f"({100*len(rejected)/max(1, len(poses)-1):.1f}%)", flush=True)
