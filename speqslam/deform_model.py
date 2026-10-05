"""Deformation attempt 4: an EXPLICIT deformation model, estimated jointly
with camera pose (the first remaining candidate named since 4.13).

Model (v2): each reference-frame point may be displaced by a 3D vector
field D(x), parameterized on a coarse GX x GY bilinear grid (3 components
per node). v1 parameterized displacement ALONG THE VIEWING RAY, and a
synthetic recovery test exposed that as projectively invisible: scaling a
point along its own ray does not move its projection, so the model spanned
almost exactly the motion subspace a monocular camera cannot see, and
correctly absorbed nothing. Real wall motion appears as tangential flow,
which needs the full 3D field. The state is [xi (6) | D (3*GX*GY)], solved
jointly at the half-resolution level after a COARSE-levels-only rigid init
(a fully converged rigid init leaves zero pose gradient and re-imports the
4.13 failure - learned in dev round 1). Gauge handling: a uniform field is
exactly a camera translation, so the per-axis MEAN of D is constrained to
zero (heavy quadratic penalty), pushing global motion into the pose;
LAM * ||D||^2 (scaled to the data term) pulls residual deformation to zero
and MU * ||L D||^2 keeps it smooth. The pose is read out; D is discarded -
a nuisance parameter that exists to keep non-rigid motion OUT of the pose.

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

ATTEMPT 5 (adaptive field) - PROTOCOL, stated before its test data was
downloaded. Attempt 4 failed its bar at +18.0% with a clean structure: the
field helps in proportion to deformation present and costs noise where none
exists. The adaptive variant gates per frame-pair on the recovered field's
RMS displacement (the field is its own deformation detector: the prior pins
it near zero on rigid pairs, deformation inflates it): rms < THETA -> use
the rigid pose, else the joint pose. Because the gate idea exists BECAUSE
of attempt 4's test results, those eight sequences are burned as a judge
and become the dev set: THETA is chosen there (mode dev2, from per-pair
logs produced by mode log). The frozen test runs on sequences untouched by
every prior attempt: deforming c1_sigmoid1_t4_v4, c1_transverse1_t4_v4,
c2_transverse1_t1_v4, c2_ascending_t1_v4; rigid controls c1_sigmoid1_t1_v1,
c1_ascending_t4_v1, c2_transverse1_t1_v1. Criterion unchanged: median
deforming improvement >= 20%, no control worse than 10%.

Usage: python speqslam/deform_model.py dev             (attempt-4 dev-pair sweep)
       python speqslam/deform_model.py run [minutes]   (attempt-4 frozen eval)
       python speqslam/deform_model.py log [minutes]   (attempt-5 dev logging)
       python speqslam/deform_model.py dev2            (attempt-5 THETA sweep)
       python speqslam/deform_model.py test2 [minutes] (attempt-5 frozen test)
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
# LAM/MU are RELATIVE prior strengths: the solver scales them by the data
# term's own magnitude (mean diagonal of H_bb), so "1.0" means prior and data
# compete at parity. The first dev sweep used absolute values (300-30000) that
# turned out to be numerically negligible against H_bb ~ 1e5, AND initialized
# the joint solve from the fully rigid-converged pose - where the pose
# gradient is already zero, so wall motion absorbed during init stayed in the
# pose (the 4.13 "too late" failure, rediscovered inside attempt 4). Fixed:
# rigid init uses only the coarse pyramid levels; the joint solve runs at the
# two finer levels while the pose is still rough.
# FROZEN after the dev re-sweep (see results/deform_model_dev.txt): the
# loosest prior won on dev (deform 13.43 -> 13.16, control 25.17 -> 23.91),
# an effect of only -2% on the deforming sequence vs the 20% bar
LAM, MU = 0.03, 0.03
PX_STRIDE = 2          # pixel subsampling at the joint level (half resolution)
JOINT_ITERS = 12
DEV_PAIR = {"c1_cecum_t1_v4", "c1_cecum_t1_v1"}
RES = "results/deform_model.csv"
MAXF = 120
# ---- attempt 5 ----
THETA_MM = 0.15        # FROZEN after the dev2 sweep (results/deform_adaptive_dev.txt)
LOGD = "results/deform_logs"
TEST2 = [  # fresh sequences, untouched by attempts 1-4 (dirs are *_s1 from fetch_full)
    ("c1_sigmoid1_t4_v4_s1", "deform"), ("c1_transverse1_t4_v4_s1", "deform"),
    ("c2_transverse1_t1_v4_s1", "deform"), ("c2_ascending_t1_v4_s1", "deform"),
    ("c1_sigmoid1_t1_v1_s1", "control"), ("c1_ascending_t4_v1_s1", "control"),
    ("c2_transverse1_t1_v1_s1", "control"),
]
RES2 = "results/deform_adaptive.csv"

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
    """One joint [pose | 3D deformation field] Gauss-Newton at this level."""
    h, w = ref.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = (dep > 0) & (glare == 0)
    sub = np.zeros_like(valid)
    sub[::PX_STRIDE, ::PX_STRIDE] = True
    valid &= sub
    ys, xs = np.nonzero(valid)
    if len(xs) < 500:
        return T, 0.0
    d = dep[ys, xs].astype(np.float32)
    P0 = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], 1)
    Iref = ref[ys, xs].astype(np.float32)
    Phi = grid_basis(xs.astype(np.float32), ys.astype(np.float32), w, h)
    gxi = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gyi = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    nb = GX * GY
    nd = 3 * nb                       # D stored axis-major: [Dx(nb), Dy(nb), Dz(nb)]
    D = np.zeros(nd, np.float32)
    Aprior = None
    for _ in range(JOINT_ITERS):
        disp = np.stack([Phi @ D[0:nb], Phi @ D[nb:2*nb], Phi @ D[2*nb:]], 1)
        P = P0 + disp
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
        Jxi = np.einsum('nk,nkj->nj', Jpp, dP).astype(np.float32)
        # dPc/dD_(k,axis) = R[:,axis] * phi_k  ->  Jd_axis = (Jpp @ R)[:,axis] * Phi
        JR = (Jpp @ R).astype(np.float32)     # (N,3)
        Phim = Phi[m]
        J = np.concatenate([Jxi,
                            JR[:, 0:1] * Phim,
                            JR[:, 1:2] * Phim,
                            JR[:, 2:3] * Phim], 1)   # (N, 6+nd)
        H = (J.T @ J).astype(np.float64)
        g = (J.T @ r).astype(np.float64)
        if Aprior is None:
            scale = float(np.trace(H[6:, 6:]) / nd)
            Aprior = np.zeros((nd, nd))
            ones = np.ones((nb, nb)) / nb
            for a in range(3):
                s_, e_ = a * nb, (a + 1) * nb
                Aprior[s_:e_, s_:e_] = (lam * scale) * np.eye(nb) \
                    + (mu * scale) * LTL \
                    + (1e3 * scale) * ones   # zero-mean gauge: uniform D = camera translation
        H[6:, 6:] += Aprior
        g[6:] += Aprior @ D
        try:
            delta = -np.linalg.solve(H + 1e-6 * np.eye(6 + nd), g)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta[:6]) @ T
        D = (D + delta[6:]).astype(np.float32)
        if np.linalg.norm(delta) < 1e-6:
            break
    disp = np.stack([Phi @ D[0:nb], Phi @ D[nb:2*nb], Phi @ D[2*nb:]], 1)
    rms_mm = float(np.sqrt((disp ** 2).sum(1).mean())) * 1000.0
    return T, rms_mm

def track_pair_dm(g0, dep0, g1, K, glare, lam=LAM, mu=MU):
    # rigid init from the COARSE pyramid levels only (quarter res and below),
    # so the joint solve sees a pose that has not yet absorbed wall motion
    grefs, gcurs, gdeps, gls, Ks = [g0], [g1], [dep0], [glare], [K]
    for _ in range(1, 4):
        grefs.append(cv2.pyrDown(grefs[-1]))
        gcurs.append(cv2.pyrDown(gcurs[-1]))
        gdeps.append(cv2.resize(gdeps[-1], (grefs[-1].shape[1], grefs[-1].shape[0]),
                                interpolation=cv2.INTER_NEAREST))
        gls.append(cv2.resize(gls[-1], (grefs[-1].shape[1], grefs[-1].shape[0]),
                              interpolation=cv2.INTER_NEAREST))
        Kl = Ks[-1].copy(); Kl[:2, :] *= 0.5; Ks.append(Kl)
    T = np.eye(4)
    for l in (3, 2):
        T = dv._align_level(grefs[l], gdeps[l], gcurs[l], Ks[l], T, 15,
                            False, 1.0, glare=gls[l], illum=True)
    T, rms_mm = joint_refine(grefs[1], gdeps[1], gcurs[1], Ks[1], T, gls[1], lam, mu)
    return T, rms_mm

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
                T, _ = track_pair_dm(g0, deps[i-1], g1, K, glare, *fn)
            pose = pose @ np.linalg.inv(T)
            traj.append(pose[:3, 3].copy())
        out[name] = umeyama_ate(gt, np.array(traj)) * 1000
    return out

def run_seq_log(seq, net, mapu, mapv, K):
    """Track a sequence once, logging per-pair rigid delta, joint delta and
    the recovered field RMS - the raw material for the theta sweep."""
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
    rig, joi, rms = [], [], []
    for i in range(1, len(frames)):
        g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        glare = specular_mask(frames[i-1], method="adaptive", k=2.0,
                              lo=225, hi=245, dilate=1)
        Tr = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False,
                           depth_scale=1.0, glare=glare, illum=True)
        Tj, r = track_pair_dm(g0, deps[i-1], g1, K, glare)
        rig.append(Tr); joi.append(Tj); rms.append(r)
    np.savez(f"{LOGD}/{seq}.npz", rigid=np.array(rig), joint=np.array(joi),
             rms=np.array(rms), gt=gt - gt[0])

def assemble_ate(log, theta):
    """ATE of the adaptive trajectory (rigid where rms < theta, else joint)."""
    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    for Tr, Tj, r in zip(log["rigid"], log["joint"], log["rms"]):
        T = Tr if r < theta else Tj
        pose = pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
    return umeyama_ate(log["gt"], np.array(traj)) * 1000

if __name__ != "__main__":
    MODE = "imported"      # importing this module must never run the experiment

if MODE in ("dev", "run", "log", "test2"):
    mapu, mapv, K = su.build_map(V2CAL)
    net = demo_app.DepthNet()

if MODE == "dev":
    t0 = time.time()
    devlog = "results/deform_model_dev.txt"
    have = set()
    if os.path.exists(devlog):
        for ln in open(devlog):
            if ln.startswith("lam"):
                have.add(ln.split(":")[0])
    with open(devlog, "a") as f:
        for lam in (0.03, 0.3, 3.0):
            for seq in ("c1_cecum_t1_v4", "c1_cecum_t1_v1"):
                key = f"lam=mu={lam:g} {seq}"
                if key in have:
                    continue
                if time.time() - t0 > BUDGET_S:
                    print("budget reached; resume me", flush=True)
                    sys.exit(0)
                r = run_seq(seq, net, mapu, mapv, K, lam, lam)
                line = f"{key}: base {r['base']:.2f} dm {r['dm']:.2f} mm"
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
    print(f"dev sweep done ({time.time()-t0:.0f}s)", flush=True)
elif MODE == "run":
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

elif MODE == "log":
    os.makedirs(LOGD, exist_ok=True)
    t0 = time.time()
    for seq, kind in SEQS:
        if os.path.exists(f"{LOGD}/{seq}.npz"):
            continue
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        print(f"logging {seq} [{kind}]", flush=True)
        run_seq_log(seq, net, mapu, mapv, K)
    print("log session done", flush=True)

elif MODE == "dev2":
    with open("results/deform_adaptive_dev.txt", "w") as f:
        for theta in (0.05, 0.10, 0.15, 0.25, 0.40, 1e9):
            d_imp, c_wor, gates = [], [], []
            for seq, kind in SEQS:
                log = np.load(f"{LOGD}/{seq}.npz")
                base = assemble_ate(log, 1e9)      # always-rigid
                ada = assemble_ate(log, theta)
                gates.append(float((log["rms"] >= theta).mean()))
                if kind == "deform":
                    d_imp.append(1 - ada / base)
                else:
                    c_wor.append(ada / base - 1)
            line = (f"theta {theta:g} mm: deform median {100*np.median(d_imp):+.1f}% "
                    f"| worst control {100*max(c_wor):+.1f}% "
                    f"| joint engaged on {100*np.mean(gates):.0f}% of pairs")
            print(line, flush=True)
            f.write(line + "\n")

elif MODE == "test2":
    done = set()
    if os.path.exists(RES2):
        done = {r["seq"] for r in csv.DictReader(open(RES2))}
    else:
        with open(RES2, "w", newline="") as f:
            csv.writer(f).writerow(["seq", "kind", "ate_base_mm", "ate_adaptive_mm",
                                    "joint_pct"])
    t0 = time.time()
    for seq, kind in TEST2:
        if seq in done:
            continue
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        os.makedirs(LOGD, exist_ok=True)
        if not os.path.exists(f"{LOGD}/{seq}.npz"):
            print(f"logging {seq} [{kind}]", flush=True)
            run_seq_log(seq, net, mapu, mapv, K)
        log = np.load(f"{LOGD}/{seq}.npz")
        base = assemble_ate(log, 1e9)
        ada = assemble_ate(log, THETA_MM)
        jp = 100 * float((log["rms"] >= THETA_MM).mean())
        with open(RES2, "a", newline="") as f:
            csv.writer(f).writerow([seq, kind, f"{base:.2f}", f"{ada:.2f}",
                                    f"{jp:.0f}"])
        print(f"{seq} [{kind}]: base {base:.2f} | adaptive {ada:.2f} mm "
              f"(joint on {jp:.0f}%)", flush=True)
    rows = list(csv.DictReader(open(RES2)))
    if len(rows) == len(TEST2):
        d_imp = [1 - float(r["ate_adaptive_mm"]) / float(r["ate_base_mm"])
                 for r in rows if r["kind"] == "deform"]
        c_wor = [float(r["ate_adaptive_mm"]) / float(r["ate_base_mm"]) - 1
                 for r in rows if r["kind"] == "control"]
        ok = np.median(d_imp) >= 0.20 and max(c_wor) <= 0.10
        print(f"\nVERDICT attempt 5 (pre-registered): deform median "
              f"{100*np.median(d_imp):+.1f}% | worst control "
              f"{100*max(c_wor):+.1f}% -> {'PASS' if ok else 'FAIL'}", flush=True)
