"""The frame-rate experiment: the same full withdrawal, same frozen pipeline, same
divergence guard - at stride 3 (every 3rd frame, ~9 fps effective) instead of the
stride 12 that broke tracking. The question: at what frame rate does the tracker
come back inside its envelope? Success criterion is stated in advance: the guarded
ATE must clearly beat the constant-point baseline (printed alongside, per Trap 5),
and the guard-rejection rate must fall from 16% toward zero.

Budget-resumable like the original runner (undistortion and poses checkpointed).
Usage: python speqslam/c3vdv2_screening_s3.py [minutes]
"""
import os, sys, glob, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL, load_poses, umeyama_ate

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
BASE = "data/c3vdv2/c1_full_t4_v1_s3"
UND = f"{BASE}/undist"
POSE_CK = f"{BASE}/poses_est.npy"
REJ_CK = f"{BASE}/rejected.npy"
MAX_STEP_M = 0.060
MAX_ROT_DEG = 35.0

t0 = time.time()
frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
gt_all = load_poses(f"{BASE}/pose.txt")
print(f"{len(frames_raw)} frames (stride 3), {len(gt_all)} GT poses", flush=True)

os.makedirs(UND, exist_ok=True)
mapu, mapv, K = su.build_map(V2CAL)
if len(glob.glob(f"{UND}/*.jpg")) < len(frames_raw):
    for i, p in enumerate(frames_raw):
        out = f"{UND}/{i:04d}.jpg"
        if os.path.exists(out):
            continue
        cv2.imwrite(out, su.undistort(cv2.imread(p), mapu, mapv),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if time.time() - t0 > BUDGET_S:
            print("budget during undistort; resume me", flush=True)
            sys.exit(0)
    print("undistort done", flush=True)

und = sorted(glob.glob(f"{UND}/*.jpg"))
n = len(und)
net = demo_app.DepthNet()

poses = [np.eye(4)]
rejected = []
if os.path.exists(POSE_CK):
    poses = list(np.load(POSE_CK))
    if os.path.exists(REJ_CK):
        rejected = list(np.load(REJ_CK))
start = len(poses)
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
    print("tracking done", flush=True)

poses = list(np.load(POSE_CK))
rejected = list(np.load(REJ_CK)) if os.path.exists(REJ_CK) else []
est = np.array([p[:3, 3] for p in poses])
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]
path = float((np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000).sum())
ate = umeyama_ate(gt, est) * 1000
base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
es = np.linalg.norm(np.diff(est, axis=0), axis=1)
print(f"est step m: p50 {np.percentile(es, 50):.4f} p99 {np.percentile(es, 99):.4f} "
      f"max {es.max():.3f}", flush=True)
print(f"STRIDE-3 RESULT: ATE {ate:.1f} mm over {path:.0f} mm GT path "
      f"({100*ate/path:.2f}%) | constant-point baseline {base:.1f} mm | "
      f"guarded pairs {len(rejected)}/{len(poses)-1} "
      f"({100*len(rejected)/max(1, len(poses)-1):.1f}%)", flush=True)
