"""Deformation-aware tracking, attempts 2 and 3 (attempt 1, post-hoc residual
masking, was rejected by its pre-stated criterion - see c3vdv2_deform_robust.py
and manuscript 4.13).

Both variants attack the measured ~9x deformation cost (deforming median
~10 mm ATE vs 1.4 mm speed-matched rigid control) with mechanisms distinct
from the rejected attempt:

  H  in-loop Huber weighting (robust=True in direct_vo). The deployed config
     runs robust=False (chosen in the glare era: OFF exposes glare sensitivity,
     which that study needed). Deforming-wall pixels are outliers DURING the
     optimization, so Huber down-weights them before the pose can absorb wall
     motion - unlike 4.13's post-convergence masking, which its own diagnosis
     showed acts too late.
  S  trajectory smoothing (Savitzky-Golay, window 9 samples, polyorder 2,
     applied to the baseline robust=False positions). Tests the hypothesis
     that deformation injects zero-mean oscillatory pose error at the
     breathing/peristalsis timescale (~1-2 s =~ 9 samples at these strides)
     while true camera motion is smooth. Window and order are set from that
     timescale, stated here before any result, not tuned.

PRE-REGISTERED CRITERION (same family as 4.13, stated before any result):
a variant PASSES iff median ATE improvement across the six deforming
sequences is >= 20% AND no rigid control worsens by more than 10%.
Each variant is judged independently. Sequences, strides, frame budget,
calibration and evaluation are bit-identical to c3vdv2_deform.py.

Usage: python speqslam/deform_aware.py [minutes]   (resumable via CSV)
"""
import sys, os, csv, glob, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
import scaramuzza_undistort as su
from scipy.signal import savgol_filter
from c3vdv2_deform import V2CAL, SEQS, load_poses, umeyama_ate

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
RES = "results/deform_aware.csv"
MAXF = 120
SG_WIN, SG_POLY = 9, 2

done = set()
if os.path.exists(RES):
    done = {r["seq"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["seq", "kind", "frames",
                                "ate_base_mm", "ate_huber_mm", "ate_smooth_mm"])

mapu, mapv, K = su.build_map(V2CAL)
net = demo_app.DepthNet()
t_start = time.time()

def track(frames, deps, robust):
    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    for i in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245,
                              dilate=1)
        T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=robust, depth_scale=1.0,
                          glare=glare, illum=True)
        pose = pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
    return np.array(traj)

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
        continue
    poses_gt = load_poses(pose_candidates[0])
    n_all = min(len(rgbs), len(poses_gt))
    stride = max(1, int(np.ceil(n_all / MAXF)))
    idx = list(range(0, n_all, stride))
    print(f"{seq} [{kind}]: {len(idx)} frames", flush=True)
    frames, deps = [], []
    for i in idx:
        img = su.undistort(cv2.imread(rgbs[i]), mapu, mapv)
        frames.append(img)
        deps.append(net.predict(img))
    gt = np.array([poses_gt[i][1] for i in idx])
    gt = gt - gt[0]

    est_base = track(frames, deps, robust=False)
    est_hub = track(frames, deps, robust=True)
    est_sm = savgol_filter(est_base, SG_WIN, SG_POLY, axis=0)

    a_base = umeyama_ate(gt, est_base) * 1000
    a_hub = umeyama_ate(gt, est_hub) * 1000
    a_sm = umeyama_ate(gt, est_sm) * 1000
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([seq, kind, len(idx), f"{a_base:.2f}",
                                f"{a_hub:.2f}", f"{a_sm:.2f}"])
    print(f"  base {a_base:.2f} | huber {a_hub:.2f} | smooth {a_sm:.2f} mm",
          flush=True)

rows = list(csv.DictReader(open(RES)))
if len(rows) == len(SEQS):
    print("\nVERDICT (pre-registered: deform median improvement >=20%, "
          "no control worse than 10%):", flush=True)
    for name, key in (("Huber", "ate_huber_mm"), ("Smoothing", "ate_smooth_mm")):
        d_imp = [1 - float(r[key]) / float(r["ate_base_mm"])
                 for r in rows if r["kind"] == "deform"]
        c_wor = [float(r[key]) / float(r["ate_base_mm"]) - 1
                 for r in rows if r["kind"] == "control"]
        ok = np.median(d_imp) >= 0.20 and max(c_wor) <= 0.10
        print(f"  {name}: deform median {100*np.median(d_imp):+.1f}% | worst "
              f"control {100*max(c_wor):+.1f}% -> {'PASS' if ok else 'FAIL'}",
              flush=True)
