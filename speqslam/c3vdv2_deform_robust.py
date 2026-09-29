"""Deformation-robust tracking: first attempted fix for the measured 9x cost.

Mechanism targeted: a deforming wall violates the tracker's rigidity
assumption, and those pixels drag the pose estimate. The variant is two-pass
per frame pair: pass 1 tracks normally; the post-alignment photometric
residual is computed; the highest-residual quartile of pixels (the parts of
the image the rigid model could not explain - deforming wall, plus any
leftover glare or debris motion) is masked out; pass 2 re-tracks with that
mask merged into the existing glare mask. The validated tracker itself is
untouched; the mask rides the same input the glare mask uses.

SUCCESS CRITERION, stated before running: median ATE across the six deforming
sequences improves by >= 20% versus the recorded baseline
(results/c3vdv2_deform.csv), while no rigid control worsens by more than 10%.
Anything less is reported as a negative or mixed result.

Warp-convention self-check: the residual is computed under both T and inv(T)
on the first pair of each sequence and the lower-median convention is used
throughout that sequence (printed for the record).

Budget-resumable per sequence. Writes results/c3vdv2_deform_robust.csv.
Usage: python speqslam/c3vdv2_deform_robust.py [minutes]
"""
import os, sys, glob, csv, time, json
import numpy as np, cv2

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL, load_poses, umeyama_ate, SEQS, MAXF

RES = "results/c3vdv2_deform_robust.csv"
BASE_CSV = "results/c3vdv2_deform.csv"
MASK_Q = 75          # exclude the top quartile of post-alignment residual
DILATE = 5

done = set()
if os.path.exists(RES):
    done = {r["seq"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["seq", "kind", "frames", "ate_mm", "baseline_ate_mm",
                                "change_pct", "convention_inv", "masked_pct_mean"])
base_rows = {r["seq"]: r for r in csv.DictReader(open(BASE_CSV))}

mapu, mapv, K = su.build_map(V2CAL)
net = demo_app.DepthNet()
t_start = time.time()

def residual_map(g0, d0, g1, T, half=True):
    """Photometric |g0 - g1(warp)| per ref pixel under transform T."""
    if half:
        g0 = cv2.resize(g0, None, fx=0.5, fy=0.5)
        g1 = cv2.resize(g1, None, fx=0.5, fy=0.5)
        d0 = cv2.resize(d0, (g0.shape[1], g0.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        Kl = K.copy(); Kl[:2, :] *= 0.5
    else:
        Kl = K
    h, w = g0.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    z = d0
    X = (xs - Kl[0, 2]) / Kl[0, 0] * z
    Y = (ys - Kl[1, 2]) / Kl[1, 1] * z
    P = np.stack([X, Y, z], -1) @ T[:3, :3].T + T[:3, 3]
    zc = np.maximum(P[..., 2], 1e-6)
    u = (P[..., 0] / zc * Kl[0, 0] + Kl[0, 2]).astype(np.float32)
    v = (P[..., 1] / zc * Kl[1, 1] + Kl[1, 2]).astype(np.float32)
    warped = cv2.remap(g1, u, v, cv2.INTER_LINEAR, borderValue=-1)
    res = np.abs(g0 - warped)
    res[(warped < 0) | (z <= 0)] = np.nan
    return res

for seq, kind in SEQS:
    if seq in done:
        continue
    if time.time() - t_start > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    br = base_rows.get(seq)
    if not br or not br["ate_mm"]:
        continue
    base = f"data/c3vdv2/{seq}"
    rgbs = sorted(glob.glob(f"{base}/rgb/*.png")) or \
        sorted(glob.glob(f"{base}/*/rgb/*.png"))
    pose_candidates = glob.glob(f"{base}/pose.txt") + glob.glob(f"{base}/*/pose.txt")
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

    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    conv_inv = None
    masked_pcts = []
    for i in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245,
                              dilate=1)
        T1 = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0,
                           glare=glare, illum=True)
        if conv_inv is None:
            ra = residual_map(g0, deps[i-1], g1, T1)
            rb = residual_map(g0, deps[i-1], g1, np.linalg.inv(T1))
            conv_inv = np.nanmedian(rb) < np.nanmedian(ra)
            print(f"  warp convention: {'inv(T)' if conv_inv else 'T'}", flush=True)
        Tw = np.linalg.inv(T1) if conv_inv else T1
        res = residual_map(g0, deps[i-1], g1, Tw)
        thr = np.nanpercentile(res, MASK_Q)
        m_small = (res > thr).astype(np.uint8) * 255
        m_small[np.isnan(res)] = 0
        mask = cv2.resize(m_small, (g0.shape[1], g0.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask = cv2.dilate(mask, np.ones((DILATE, DILATE), np.uint8))
        masked_pcts.append(100 * (mask > 0).mean())
        both = np.maximum(glare, mask)
        T2 = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0,
                           glare=both, illum=True)
        pose = pose @ np.linalg.inv(T2)
        traj.append(pose[:3, 3].copy())
        if time.time() - t_start > BUDGET_S * 1.3:
            print(f"{seq}: budget mid-sequence; will redo this sequence", flush=True)
            sys.exit(0)

    gt = np.array([poses_gt[i][1] for i in idx])[:len(traj)]
    gt = gt - gt[0]
    ate = umeyama_ate(gt, np.array(traj)) * 1000
    b_ate = float(br["ate_mm"])
    ch = 100 * (ate - b_ate) / b_ate
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([seq, kind, len(traj), f"{ate:.2f}", f"{b_ate:.2f}",
                                f"{ch:+.1f}", int(bool(conv_inv)),
                                f"{np.mean(masked_pcts):.1f}"])
    print(f"  {seq}: ATE {ate:.2f} mm vs baseline {b_ate:.2f} ({ch:+.1f}%)", flush=True)

rows = list(csv.DictReader(open(RES)))
if len(rows) == sum(1 for s, k in SEQS if base_rows.get(s, {}).get("ate_mm")):
    dch = [float(r["change_pct"]) for r in rows if r["kind"] == "deform"]
    cch = [float(r["change_pct"]) for r in rows if r["kind"] == "control"]
    print(f"\nCOMPLETE. deform median change {np.median(dch):+.1f}% | "
          f"worst control change {max(cch, default=0):+.1f}%", flush=True)
    ok = np.median(dch) <= -20 and max(cch, default=0) <= 10
    print("SUCCESS CRITERION MET" if ok else
          "criterion NOT met - report as negative/mixed", flush=True)
