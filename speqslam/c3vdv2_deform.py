"""The deformation experiment: what does a moving, deforming colon cost the tracker?

C3VDv2 provides matched pairs: v4 videos of a phantom segment being actively deformed
(GT camera poses, no pixel GT) and v1/v2 rigid videos of the SAME segment (full GT).
Both run through the deployed pipeline configuration (fine-tuned-on-v1 depth, zero-shot
on v2; direct VO with glare exclusion + illumination compensation), and trajectories are
scored against GT poses with standard Sim(3)-aligned ATE. Controls additionally get a
depth-generalization score (fine-tuned v1 model vs v2 GT depth on a new phantom
generation). One fixed configuration; no per-sequence tuning.

Usage: python speqslam/c3vdv2_deform.py [minutes]   (resumable via the results CSV)
"""
import sys, os, csv, glob, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
import scaramuzza_undistort as su
from scipy.spatial.transform import Rotation

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
RES = "results/c3vdv2_deform.csv"
MAXF = 120

V2CAL = dict(W=1350, H=1080, cx=677.739464094188, cy=543.057997844875,
             a=[767.733695862103, -0.000592506426558248,
                -2.69440266600040e-07, -2.16380341010063e-10],
             c=0.9999, d=1.10e-4, e=-1.83e-4)

SEQS = [
    ("c1_cecum_t1_v4", "deform"), ("c1_cecum_t1_v1", "control"),
    ("c1_descending_t4_v4", "deform"), ("c1_descending_t4_v1", "control"),
    ("c1_transverse1_t1_v4", "deform"), ("c1_transverse1_t1_v2", "control"),
    ("c2_cecum_t1_v4", "deform"), ("c2_cecum_t1_v1", "control"),
    ("c1_ascending_t4_v4", "deform"),
    ("c1_sigmoid2_t4_v4", "deform"),
]

def load_poses(path):
    out = []
    for line in open(path):
        v = [float(x) for x in line.strip().split(",") if x]
        if len(v) != 16:
            continue
        M = np.array(v).reshape(4, 4)
        R = M[:3, :3].T                 # row-vector convention -> R_cw
        t = M[3, :3] / 1000.0           # mm -> m
        out.append((R, t))
    return out

def umeyama_ate(gt, est):
    """Sim(3)-aligned RMSE between two position arrays (meters). Umeyama 1991:
    est is the source, gt the target; returns RMSE after optimal s,R,t alignment."""
    mg, me = gt.mean(0), est.mean(0)
    G, E = gt - mg, est - me
    Sig = G.T @ E / len(gt)
    U, S, Vt = np.linalg.svd(Sig)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    var = (E ** 2).sum() / len(gt)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    aligned = s * (R @ E.T).T + mg
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))

done = set()
if os.path.exists(RES):
    done = {r["seq"] for r in csv.DictReader(open(RES))}
else:
    os.makedirs("results", exist_ok=True)
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["seq", "kind", "frames", "stride", "ate_mm", "path_mm",
                                "step_p50_mm", "step_p90_mm", "d1_pct", "medAbsRel"])

mapu, mapv, K = su.build_map(V2CAL)
net = demo_app.DepthNet()
t_start = time.time()

for seq, kind in SEQS:
    if seq in done:
        continue
    if time.time() - t_start > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    base = f"data/c3vdv2/{seq}"
    rgbs = sorted(glob.glob(f"{base}/rgb/*.png")) or \
        sorted(glob.glob(f"{base}/*/rgb/*.png"))
    pose_candidates = glob.glob(f"{base}/pose.txt") + glob.glob(f"{base}/*/pose.txt")
    if not rgbs or not pose_candidates:
        print(f"{seq}: missing {'rgb' if not rgbs else 'pose.txt'} — recording skip",
              flush=True)
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([seq, kind, 0, 0, "", "", "", "", "", ""])
        continue
    poses_gt = load_poses(pose_candidates[0])
    n_all = min(len(rgbs), len(poses_gt))
    stride = max(1, int(np.ceil(n_all / MAXF)))
    idx = list(range(0, n_all, stride))
    print(f"{seq} [{kind}]: {n_all} frames, stride {stride} -> {len(idx)}", flush=True)

    frames, deps = [], []
    for i in idx:
        img = su.undistort(cv2.imread(rgbs[i]), mapu, mapv)
        frames.append(img)
        deps.append(net.predict(img))

    # depth generalization vs v2 GT (controls only)
    d1 = medrel = ""
    if kind == "control" and os.path.isdir(f"{base}/depth"):
        r1, rm = [], []
        for j, i in enumerate(idx[::4]):
            gtd = cv2.imread(f"{base}/depth/{i:04d}_depth.tiff", cv2.IMREAD_UNCHANGED)
            if gtd is None:
                continue
            gtd = su.undistort(gtd, mapu, mapv, nearest=True).astype(np.float32)
            gtm = gtd / 65535.0 * 0.1               # meters
            est = deps[j * 4]
            ok = (gtm > 0.003) & (gtm < 0.099) & (est > 0)
            if ok.sum() < 1000:
                continue
            # global affine (scale) per frame is NOT used; one global scale per sequence,
            # matching the monocular protocol: fit scale on first eval frame only
            if not r1:
                s0 = np.median(gtm[ok] / est[ok])
            e = est * s0
            rel = np.abs(e[ok] - gtm[ok]) / gtm[ok]
            r1.append(np.mean(rel < 0.25) * 100)
            rm.append(np.median(rel))
        if r1:
            d1 = f"{np.mean(r1):.1f}"
            medrel = f"{np.mean(rm):.3f}"

    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    for i in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0,
                          glare=glare, illum=True)
        pose = pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
    est = np.array(traj)
    gt = np.array([poses_gt[i][1] for i in idx])
    gt = gt - gt[0]
    steps = np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000
    ate = umeyama_ate(gt, est) * 1000
    path = steps.sum()
    row = [seq, kind, len(idx), stride, f"{ate:.2f}", f"{path:.0f}",
           f"{np.percentile(steps, 50):.2f}", f"{np.percentile(steps, 90):.2f}",
           d1, medrel]
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"  ATE {ate:.2f}mm over {path:.0f}mm GT path | d1 {d1}", flush=True)

print(f"done -> {RES}", flush=True)
