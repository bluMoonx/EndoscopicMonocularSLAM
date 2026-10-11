"""RED-TEAM stage 2: re-track the five headline C3VD sequences under pipeline v2
(robust=True) with a freshly written driver and the Horn-quaternion metric, and
compare against results/huber_study.csv's ate_huber_mm. Reuses direct_vo's tracker
core (disclosed; a second tracker implementation is out of scope) - this validates
data loading, GT handling, the metric, and that the published per-sequence numbers
reproduce end to end from raw frames.

Self-terminating budget cycles (default 27 min), per-sequence checkpoints in
data/redteam/<seq>_traj.npy, results appended to results/redteam_retrack.csv.
Relaunch until it prints ALL DONE.
"""
import csv, glob, os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "speqslam")
import direct_vo as dv
from redteam_metric import horn_ate, tum_xyz

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 27.0) * 60
DS = 100.0 / 65535.0 / 1000.0
SEQS = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]
RES = "results/redteam_retrack.csv"
os.makedirs("data/redteam", exist_ok=True)

done = set()
if os.path.exists(RES):
    done = {r["seq"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["seq", "published_v2_mm", "redteam_v2_mm"])

t0 = time.time()
for seq in SEQS:
    if seq in done:
        continue
    if time.time() - t0 > BUDGET_S:
        print("budget reached; relaunch to continue", flush=True)
        sys.exit(0)
    base = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{base}/K.txt")
    imgs = sorted(glob.glob(f"{base}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{base}/depth01_ftc/*.png"))
    ck = f"data/redteam/{seq}_traj.npy"
    if os.path.exists(ck):
        traj = list(np.load(ck))
        pose = np.eye(4); pose[:3, 3] = traj[-1]           # position-only resume marker
        start = len(traj)
        # positions alone can't resume mid-chain faithfully; restart if partial
        if start < len(imgs):
            traj = []; start = 1
    else:
        traj = []; start = 1
    if not traj:
        pose = np.eye(4); traj = [pose[:3, 3].copy()]
        g0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        d0 = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
        for i in range(1, len(imgs)):
            g1 = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            T = dv.track_pair(g0, d0, g1, K, 4, robust=True, depth_scale=DS, illum=True)
            pose = pose @ np.linalg.inv(T)
            traj.append(pose[:3, 3].copy())
            g0 = g1
            d0 = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            if i % 100 == 0:
                print(f"{seq} {i}/{len(imgs)} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if time.time() - t0 > BUDGET_S:
                print(f"{seq}: budget mid-sequence at {i}; restarting it next launch", flush=True)
                sys.exit(0)
        np.save(ck, np.array(traj))
    gt = tum_xyz(f"results/c3vd/{seq}_gt.txt")
    n = min(len(gt), len(traj))
    a = horn_ate(gt[:n], np.array(traj)[:n]) * 1000
    pub = {r["seq"]: float(r["ate_huber_mm"]) for r in csv.DictReader(open("results/huber_study.csv"))}[seq]
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([seq, f"{pub:.3f}", f"{a:.3f}"])
    print(f"{seq}: published v2 {pub:.3f} mm | red-team fresh {a:.3f} mm", flush=True)

print("ALL DONE", flush=True)
rows = list(csv.DictReader(open(RES)))
pub = np.mean([float(r["published_v2_mm"]) for r in rows])
new = np.mean([float(r["redteam_v2_mm"]) for r in rows])
print(f"means: published {pub:.3f} vs red-team {new:.3f} mm", flush=True)
