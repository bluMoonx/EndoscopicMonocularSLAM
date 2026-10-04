"""Deformation attempt 4: an EXPLICIT deformation model, estimated jointly
with camera pose (the first remaining candidate named since 4.13).

Model: each reference-frame point may move along its viewing ray by a
fractional depth perturbation b(x), parameterized on a coarse GX x GY
bilinear grid (wall "breathing" is radial at colon scale). The photometric
Gauss-Newton state becomes [xi (6) | b (GX*GY)] and both are optimized
together on the half-resolution level after the standard rigid pyramid
initializes the pose. Two priors resolve the camera-vs-wall gauge ambiguity
(the stationary-camera sequence proved a rigid tracker absorbs wall motion
into pose; here the ridge makes pose the cheaper explanation wherever both
fit): LAM * ||b||^2 pulls deformation to zero, MU * ||L b||^2 (grid
Laplacian) keeps it spatially smooth. The pose is then read out and the
deformation discarded - b is a nuisance parameter that exists to keep wall
motion OUT of the pose.

Mechanism vs the three excluded families: masking (4.13) and Huber (4.17)
could only REMOVE pixels, which fails when most of the image deforms;
smoothing acted after the damage. This variant lets every pixel participate
but gives the optimizer a second place to PUT non-rigid motion.

PROTOCOL (develop-freeze-test, stated before any result):
  dev pair     c1_cecum_t1_v4 + its control c1_cecum_t1_v1, ONLY sequences
               on which LAM/MU may be chosen (grid fixed 8x6 a priori).
  frozen test  the remaining 5 deforming + 3 control sequences, untouched.
  criterion    median ATE improvement over the robust=False baseline across
               the 5 judged deforming sequences >= 20%, AND no judged
               control worsened by > 10% (same family as 4.13/4.17).

Usage: python speqslam/deform_model.py dev            (sweep on the dev pair)
       python speqslam/deform_model.py run [minutes]  (frozen evaluation)
"""
import sys, os, csv, glob, time
import numpy as np, cv2
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
BUDGET_S = (float(sys.argv[2]) if len(sys.argv) > 2 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from direct_vo import se3_exp, _bilinear
from glare_removal import specular_mask
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL, SEQS, load_poses, umeyama_ate

GX, GY = 8, 6
# FROZEN after the dev sweep (values recorded by `dev` mode; see results/deform_model_dev.txt)
LAM, MU = 3000.0, 3000.0
PX_STRIDE = 2          # pixel subsampling at the joint level (half resolution)
JOINT_ITERS = 12
DEV_PAIR = {"c1_cecum_t1_v4", "c1_cecum_t1_v1"}
RES = "results/deform_model.csv"
MAXF = 120

def grid_basis(xs, ys, w, h):
    """Bilinear basis matrix Phi (N, GX*GY) for pixel coords."""
    gx = xs / (w - 1) * (GX - 1)
    gy = ys / (h - 1) * (GY - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, GX - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, GY - 2)
    wx = gx - x0; wy = gy - y0
    Phi = np.zeros((len(xs), GX * GY), np.float32)
    idx = np.arange(len(xs))
    Phi[idx, y0 * GX + x0] = (1 - wx) * (1 - wy)
    Phi[idx, y0 * GX + x0 + 1] = wx * (1 - wy)
    Phi[idx, (y0 + 1) * GX + x0] = (1 - wx) * wy
    Phi[idx, (y0 + 1) * GX + x0 + 1] = wx * wy
    return Phi

def grid_laplacian():
    n = GX * GY
    L = np.zeros((n, n), np.float32)
    for y in range(GY):
        for x in range(GX):
            i = y * GX + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < GX and 0 <= yy < GY:
                    L[i, i] += 1
                    L[i, yy * GX + xx] -= 1
    return L

LTL = grid_laplacian().T @ grid_laplacian()

def joint_refine(ref, dep, cur, K, T, glare, lam, mu):
    """One joint [pose | deformation] Gauss-Newton at this level. Returns T."""
    h, w = ref.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = (dep > 0) & (glare == 0)
    valid[::PX_STRIDE, :] = valid[::PX_STRIDE, :]
    sub = np.zeros_like(valid)
    sub[::PX_STRIDE, ::PX_STRIDE] = True
    valid &= sub
    ys, xs = np.nonzero(valid)
    if len(xs) < 500:
        return T
    d = dep[ys, xs].astype(np.float32)
    P0 = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], 1)
    Iref = ref[ys, xs].astype(np.float32)
    Phi = grid_basis(xs.astype(np.float32), ys.astype(np.float32), w, h)
    gxi = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gyi = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    nb = GX * GY
    b = np.zeros(nb, np.float32)
    Aprior = lam * np.eye(nb, dtype=np.float32) + mu * LTL
    for _ in range(JOINT_ITERS):
        bp = Phi @ b                            # per-pixel fractional depth
        P = P0 * (1.0 + bp)[:, None]
        R, t = T[:3, :3], T[:3, 3]
        Pc = P @ R.T + t
        Zc = Pc[:, 2]
        u = fx * Pc[:, 0] / Zc + cx
        v = fy * Pc[:, 1] / Zc + cy
        m = (Zc > 1e-6) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if m.sum() < 500:
            break
        uu, vv = u[m], v[m]
        Pcv, Zv = Pc[m], Zc[m]
        Icur = _bilinear(cur, uu, vv)
        Ir = Iref[m]
        A_ = np.stack([Ir, np.ones_like(Ir)], 1)
        try:
            ab = np.linalg.lstsq(A_, Icur, rcond=None)[0]
        except np.linalg.LinAlgError:
            ab = np.array([1.0, 0.0])
        r = Icur - (ab[0] * Ir + ab[1])
        gxv, gyv = _bilinear(gxi, uu, vv), _bilinear(gyi, uu, vv)
        iZ = 1.0 / Zv
        Jpp = np.stack([gxv * fx * iZ, gyv * fy * iZ,
                        -(gxv * fx * Pcv[:, 0] + gyv * fy * Pcv[:, 1]) * iZ * iZ], 1)
        N = int(m.sum())
        dP = np.zeros((N, 3, 6), np.float32)
        dP[:, 0, 0] = dP[:, 1, 1] = dP[:, 2, 2] = 1.0
        dP[:, 0, 4] = Zv;        dP[:, 0, 5] = -Pcv[:, 1]
        dP[:, 1, 3] = -Zv;       dP[:, 1, 5] = Pcv[:, 0]
        dP[:, 2, 3] = Pcv[:, 1]; dP[:, 2, 4] = -Pcv[:, 0]
        Jxi = np.einsum('nk,nkj->nj', Jpp, dP)
        # dPc/db_k = R @ P0 * phi_k  ->  scalar s = Jpp . (P0 R^T), Jb = s * Phi
        w3 = (P0[m] @ R.T)
        s = np.einsum('nk,nk->n', Jpp, w3)
        Jb = s[:, None] * Phi[m]
        J = np.concatenate([Jxi, Jb], 1)        # (N, 6+nb)
        H = J.T @ J
        g = J.T @ r
        H[6:, 6:] += Aprior
        g[6:] += Aprior @ b
        try:
            delta = -np.linalg.solve(H + 1e-6 * np.eye(6 + nb), g)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta[:6]) @ T
        b = b + delta[6:].astype(np.float32)
        if np.linalg.norm(delta) < 1e-6:
            break
    return T

def track_pair_dm(g0, dep0, g1, K, glare, lam=LAM, mu=MU):
    T = dv.track_pair(g0, dep0, g1, K, 4, robust=False, depth_scale=1.0,
                      glare=glare, illum=True)
    half = lambda im: cv2.pyrDown(im)
    K1 = K.copy(); K1[:2, :] *= 0.5
    dep1 = cv2.resize(dep0, (g0.shape[1] // 2, g0.shape[0] // 2),
                      interpolation=cv2.INTER_NEAREST)
    gl1 = cv2.resize(glare, (g0.shape[1] // 2, g0.shape[0] // 2),
                     interpolation=cv2.INTER_NEAREST)
    return joint_refine(half(g0), dep1, half(g1), K1, T, gl1, lam, mu)

def run_seq(seq, net, mapu, mapv, K, lam, mu):
    base = f"data/c3vdv2/{seq}"
    rgbs = sorted(glob.glob(f"{base}/rgb/*.png")) or \
        sorted(glob.glob(f"{base}/*/rgb/*.png"))
    pose_candidates = glob.glob(f"{base}/pose.txt") + glob.glob(f"{base}/*/pose.txt")
    poses_gt = load_poses(pose_candidates[0])
    n_all = min(len(rgbs), len(poses_gt))
    stride = max(1, int(np.ceil(n_all / MAXF)))
    idx = list(range(0, n_all, stride))
    frames, deps = [], []
    for i in idx:
        img = su.undistort(cv2.imread(rgbs[i]), mapu, mapv)
        frames.append(img)
        deps.append(net.predict(img))
    gt = np.array([poses_gt[i][1] for i in idx])
    gt = gt - gt[0]
    out = {}
    for name, fn in (("base", None), ("dm", (lam, mu))):
        pose = np.eye(4); traj = [pose[:3, 3].copy()]
        for i in range(1, len(frames)):
            g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
            glare = specular_mask(frames[i-1], method="adaptive", k=2.0,
                                  lo=225, hi=245, dilate=1)
            if fn is None:
                T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False,
                                  depth_scale=1.0, glare=glare, illum=True)
            else:
                T = track_pair_dm(g0, deps[i-1], g1, K, glare, *fn)
            pose = pose @ np.linalg.inv(T)
            traj.append(pose[:3, 3].copy())
        out[name] = umeyama_ate(gt, np.array(traj)) * 1000
    return out

mapu, mapv, K = su.build_map(V2CAL)
net = demo_app.DepthNet()

if MODE == "dev":
    t0 = time.time()
    with open("results/deform_model_dev.txt", "a") as f:
        for lam in (300.0, 3000.0, 30000.0):
            for seq in ("c1_cecum_t1_v4", "c1_cecum_t1_v1"):
                r = run_seq(seq, net, mapu, mapv, K, lam, lam)
                line = (f"lam=mu={lam:g} {seq}: base {r['base']:.2f} "
                        f"dm {r['dm']:.2f} mm")
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
    print(f"dev sweep done ({time.time()-t0:.0f}s)", flush=True)
else:
    done = set()
    if os.path.exists(RES):
        done = {r["seq"] for r in csv.DictReader(open(RES))}
    else:
        with open(RES, "w", newline="") as f:
            csv.writer(f).writerow(["seq", "kind", "ate_base_mm", "ate_dm_mm"])
    t0 = time.time()
    for seq, kind in SEQS:
        if seq in done or seq in DEV_PAIR:
            continue
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        r = run_seq(seq, net, mapu, mapv, K, LAM, MU)
        with open(RES, "a", newline="") as f:
            csv.writer(f).writerow([seq, kind, f"{r['base']:.2f}", f"{r['dm']:.2f}"])
        print(f"{seq} [{kind}]: base {r['base']:.2f} | dm {r['dm']:.2f} mm",
              flush=True)
    rows = list(csv.DictReader(open(RES)))
    if len(rows) == len(SEQS) - 2:
        d_imp = [1 - float(r["ate_dm_mm"]) / float(r["ate_base_mm"])
                 for r in rows if r["kind"] == "deform"]
        c_wor = [float(r["ate_dm_mm"]) / float(r["ate_base_mm"]) - 1
                 for r in rows if r["kind"] == "control"]
        ok = np.median(d_imp) >= 0.20 and max(c_wor) <= 0.10
        print(f"\nVERDICT (pre-registered): deform median "
              f"{100*np.median(d_imp):+.1f}% | worst control "
              f"{100*max(c_wor):+.1f}% -> {'PASS' if ok else 'FAIL'}", flush=True)
