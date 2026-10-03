"""Loop closure v2: feature-based wide-baseline closures, the fix v1's own
failure diagnosis named.

v1 (loop_closure.py) verdict: revisit RECOGNITION is true (39/41 audited
closures), but the photometric relative-pose MEASUREMENT is biased at low
parallax - revisit alignments systematically underestimate separation, and
even position-only soft constraints contracted the map (-5% dev, +17%/+1%
frozen tests). v2 keeps v1's stage-1 candidates and v1's odometry-consistency
alias gate, and replaces the measurement:

  stage 2: SIFT features (glare-masked) matched across each candidate pair
     (Lowe ratio), lifted to 3D through the fine-tuned depth, relative pose
     by PnP+RANSAC both directions with forward-backward consistency - a
     geometric wide-baseline measurement where photometric alignment was the
     bias source. Each closure carries a PARALLAX-AWARE sigma: translation
     uncertainty grows with reprojection spread and median scene depth and
     shrinks with inlier count, so a weakly-conditioned closure is a weak
     constraint instead of a wrong one.
  stage 3: v1's Sim(3) pose graph, soft_L1, same odometry edges and alias
     gate - but closures now carry rotation AND translation with their
     per-closure sigmas (v1 had disabled rotation entirely).
  stage 4: v1's evaluation, unchanged (GT only here).

PROTOCOL (stated before any v2 result exists): develop and tune on
c1_full_t4_v1_s3 ONLY. Freeze every constant in this header block. Then run
the two other withdrawals untouched. PASS iff BOTH frozen-test ATEs improve
by >= 10%; any frozen test worsening by > 2% is a FAIL regardless of the
other. Native-rate application only follows a PASS.

V2 VERDICT (recorded after its frozen tests): FAIL - c2 +16.6%, c1_t1 +2.9%.
Post-hoc audit: the PnP translations are accurate (median closure error
5.2 mm vs 31.5 mm odometry drift at the same pairs on c2); the harm came
from the rotation and scale terms v2 newly enabled on closure edges (post-hoc
ablation on c2: same closures translation-only = -9.2%, vs oracle -13.5%).
The oracle run on all three withdrawals bounds ANY closure method at
-4.6% / -13.5% / -7.5%: screening withdrawals are near-linear, so most drift
accumulates on wall seen exactly once.

V3 (stage 13 below), frozen before its test data was even downloaded:
stages 1 and 2 bit-identical to v2's frozen versions; the pose graph uses
TRANSLATION-ONLY closure edges (rotation and scale sigmas 10.0, i.e. off)
with each closure's own parallax-aware sig_t - the one combination neither
v1 (biased translations) nor v2 (good translations, poisoned rotation/scale)
tested. Test sequences: c1_full_t2_v1 and c2_full_t2_v1, untouched by every
prior stage of this project. PASS iff both improve, mean improvement >= 5%
(the oracle showed the achievable ceiling is sequence-dependent, 5-14%);
any worsening > 2% is a FAIL. The two v2-burned sequences are reported as
development evidence only.

Usage: python speqslam/loop_closure2.py <base_dir> <stage> [minutes]
"""
import os, sys, glob, time, json
import numpy as np, cv2

BASE = sys.argv[1].rstrip("/\\")
STAGE = int(sys.argv[2])
BUDGET_S = (float(sys.argv[3]) if len(sys.argv) > 3 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")

KF_STRIDE = 8
# ---- v2 stage-1 constants: wider candidate net than v1 (sim 0.82/top3), safe
# because v2's geometric chain (ratio test -> RANSAC -> forward-backward ->
# alias gate) rejects lookalikes that v1's photometric verification passed ----
SIM_T2 = 0.78
TOPK2 = 5
MIN_SEP = 60
D = 40
# ---- v2 stage-2 constants (dev: c1_full_t4_v1_s3; frozen before the tests) ----
N_FEAT = 3000          # SIFT features per frame
SIFT_CT = 0.01         # contrast threshold: the default 0.04 finds ~50 keypoints
                       # on this smooth mucosal texture; CLAHE + 0.01 finds ~2000
CLAHE_CLIP = 3.0
RATIO = 0.75           # Lowe ratio
MIN_MATCH = 40         # matches needed to attempt PnP
MIN_INLIER = 25        # RANSAC inliers to accept
MIN_INLIER_FRAC = 0.35
REPROJ_PX = 4.0        # RANSAC reprojection threshold
FB_ROT_DEG = 4.0       # forward-backward consistency bounds (as v1)
FB_T = 0.006
SIG_T_FLOOR = 0.03     # translation sigma floor (depth units)
SIG_R = 0.08           # rotation sigma (rad), fixed
# ---- v1 alias gate, frozen as-is ----
DRIFT_BUDGET = 0.10
DRIFT_FLOOR = 0.18

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
KFS = list(range(0, n, KF_STRIDE))
OUTD = f"{BASE}/loopclosure"
K = np.array([[250.0, 0, 320.0], [0, 250.0, 256.0], [0, 0, 1]])

if STAGE == 1:
    def descriptor(path):
        g = cv2.imread(path, 0)
        g = cv2.resize(g, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        g -= g.mean()
        nrm = np.linalg.norm(g)
        return (g / max(nrm, 1e-6)).ravel()
    t0 = time.time()
    desc_f = f"{OUTD}/desc.npy"
    desc = np.load(desc_f) if os.path.exists(desc_f) else \
        np.stack([descriptor(p) for p in und[::KF_STRIDE]])
    np.save(desc_f, desc)
    S = desc @ desc.T
    nk = len(KFS)
    cands = []
    for i in range(nk):
        best = []
        for j in range(i):
            if (KFS[i] - KFS[j]) < MIN_SEP:
                continue
            if S[i, j] >= SIM_T2:
                best.append((S[i, j], j))
        best.sort(reverse=True)
        for s, j in best[:TOPK2]:
            cands.append((KFS[j], KFS[i], float(s)))
    json.dump(cands, open(f"{OUTD}/candidates2.json", "w"))
    print(f"stage 1 (v2): {nk} keyframes | {len(cands)} candidates "
          f"(sim >= {SIM_T2}, top {TOPK2}) ({time.time()-t0:.0f}s)", flush=True)

elif STAGE == 2:
    import demo_app
    from glare_removal import specular_mask
    net = demo_app.DepthNet()
    sift = cv2.SIFT_create(N_FEAT, contrastThreshold=SIFT_CT)
    clahe = cv2.createCLAHE(CLAHE_CLIP, (8, 8))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    c2f = f"{OUTD}/candidates2.json"
    cands = json.load(open(c2f if os.path.exists(c2f) else f"{OUTD}/candidates.json"))
    done_f = f"{OUTD}/verified2.json"
    verified = json.load(open(done_f)) if os.path.exists(done_f) else []
    prog_f = f"{OUTD}/verify2_progress.json"
    start = json.load(open(prog_f))["i"] if os.path.exists(prog_f) else 0
    t0 = time.time()

    feat_cache, dep_cache = {}, {}
    def feats(idx):
        if idx not in feat_cache:
            if len(feat_cache) > 80:
                feat_cache.clear()
            img = cv2.imread(und[idx])
            g = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            glare = specular_mask(img, method="adaptive", k=2.0, lo=225, hi=245,
                                  dilate=3)
            mask = (glare == 0).astype(np.uint8) * 255
            kp, des = sift.detectAndCompute(g, mask)
            feat_cache[idx] = (kp, des)
        return feat_cache[idx]
    def dep(idx):
        if idx not in dep_cache:
            if len(dep_cache) > 40:
                dep_cache.clear()
            dep_cache[idx] = net.predict(cv2.imread(und[idx]))
        return dep_cache[idx]

    why = {"feat": 0, "match": 0, "depth": 0, "pnp": 0, "inlier": 0, "fb": 0}
    def pnp(src, dst):
        """3D from src's depth -> matched 2D in dst. Returns (T, inl, rms) or None.
        T maps src-camera coords to dst-camera coords (same convention as v1 T_ab)."""
        kps, dess = feats(src)
        kpd, desd = feats(dst)
        if dess is None or desd is None or len(kps) < MIN_MATCH or len(kpd) < MIN_MATCH:
            why["feat"] += 1
            return None
        pairs = bf.knnMatch(dess, desd, k=2)
        good = [m for m, s in (p for p in pairs if len(p) == 2)
                if m.distance < RATIO * s.distance]
        if len(good) < MIN_MATCH:
            why["match"] += 1
            return None
        D = dep(src)
        obj, img = [], []
        for m in good:
            u, v = kps[m.queryIdx].pt
            z = D[int(round(v)), int(round(u))]
            if z <= 0:
                continue
            obj.append([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
            img.append(kpd[m.trainIdx].pt)
        if len(obj) < MIN_MATCH:
            why["depth"] += 1
            return None
        obj = np.array(obj, np.float64); img = np.array(img, np.float64)
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, K, None, iterationsCount=300,
                reprojectionError=REPROJ_PX, confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            why["pnp"] += 1
            return None
        if not ok or inl is None or len(inl) < MIN_INLIER \
           or len(inl) / len(obj) < MIN_INLIER_FRAC:
            why["inlier"] += 1
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
        # inlier reprojection rms
        P = (R @ obj[inl.ravel()].T + tvec).T
        uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
        rms = float(np.sqrt(((uv - img[inl.ravel()]) ** 2).sum(1).mean()))
        return T, len(inl), rms

    for ci in range(start, len(cands)):
        a, b, sim = cands[ci]
        fa, fb = pnp(a, b), pnp(b, a)
        acc = False
        if fa is not None and fb is not None:
            T_ab, inl_ab, rms_ab = fa
            T_ba, inl_ba, rms_ba = fb
            E = T_ab @ T_ba
            r_err = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
            t_err = np.linalg.norm(E[:3, 3])
            if not (r_err < FB_ROT_DEG and t_err < FB_T):
                why["fb"] += 1
            if r_err < FB_ROT_DEG and t_err < FB_T:
                med_a = float(np.median(dep(a)[dep(a) > 0]))
                med_b = float(np.median(dep(b)[dep(b) > 0]))
                # parallax-aware translation sigma: reprojection spread of REPROJ-px
                # at median depth, deflated by inlier count
                inl = min(inl_ab, inl_ba)
                rms = max(rms_ab, rms_ba)
                sig_t = max(SIG_T_FLOOR,
                            2.0 * rms / K[0, 0] * med_a * (30.0 / inl) ** 0.5)
                verified.append([a, b, sim, T_ab.tolist(), med_a, med_b,
                                 inl, rms, sig_t])
                acc = True
        if ci % 20 == 0 or acc:
            json.dump(verified, open(done_f, "w"))
            json.dump({"i": ci + 1}, open(prog_f, "w"))
            print(f"verify2 {ci}/{len(cands)}: accepted {len(verified)} | "
                  f"rejects {why}", flush=True)
        if time.time() - t0 > BUDGET_S:
            json.dump(verified, open(done_f, "w"))
            json.dump({"i": ci + 1}, open(prog_f, "w"))
            print(f"budget at candidate {ci}; resume me", flush=True)
            sys.exit(0)
    json.dump(verified, open(done_f, "w"))
    json.dump({"i": len(cands)}, open(prog_f, "w"))
    print(f"stage 2 (v2) done: {len(verified)} geometric closures of "
          f"{len(cands)} candidates", flush=True)

elif STAGE == 3:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    poses = list(np.load(f"{BASE}/poses_est.npy"))
    verified = json.load(open(f"{OUTD}/verified2.json"))
    if not verified:
        print("stage 3: no v2 closures; nothing to do", flush=True)
        sys.exit(0)
    nodes = sorted(set(KFS) | {0, n - 1})
    nid = {f: i for i, f in enumerate(nodes)}
    nn = len(nodes)
    x0 = np.zeros(nn * 7)
    for f, i in nid.items():
        T = poses[f]
        x0[7*i:7*i+3] = Rot.from_matrix(T[:3, :3]).as_rotvec()
        x0[7*i+3:7*i+6] = T[:3, 3]
        x0[7*i+6] = 0.0
    edges = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        Trel = np.linalg.inv(poses[a]) @ poses[b]
        edges.append((nid[a], nid[b], Trel, 0.0, 1.0, 0.02, 0.02, 0.05))
    steps = np.array([np.linalg.norm((np.linalg.inv(poses[i-1]) @ poses[i])[:3, 3])
                      for i in range(1, len(poses))])
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    kept, dropped = 0, 0
    for a, b, sim, T_ab_l, med_a, med_b, inl, rms, sig_t in verified:
        T_ab = np.array(T_ab_l)
        Trel = np.linalg.inv(T_ab)
        odo_rel = (np.linalg.inv(poses[a]) @ poses[b])[:3, 3]
        implied_drift = np.linalg.norm(odo_rel - Trel[:3, 3])
        chain = max(cum[b] - cum[a], 1e-6)
        if implied_drift > max(DRIFT_FLOOR, DRIFT_BUDGET * chain):
            dropped += 1
            continue
        kept += 1
        dlogs = float(np.log(max(med_b, 1e-6) / max(med_a, 1e-6)))
        # v2: rotation enabled with fixed sigma, translation with the closure's
        # own parallax-aware sigma, scale still soft (depth-median based)
        edges.append((nid[a], nid[b], Trel, dlogs, 1.0, SIG_R, sig_t, 0.5))
    print(f"stage 3 (v2): alias gate kept {kept}, dropped {dropped}", flush=True)
    def unpack(x, i):
        R = Rot.from_rotvec(x[7*i:7*i+3]).as_matrix()
        t = x[7*i+3:7*i+6]
        s = np.exp(x[7*i+6])
        return R, t, s
    def residuals(x):
        res = np.zeros(len(edges) * 7 + 7)
        for e, (i, j, Tij, dls, w, sr, stt, ss) in enumerate(edges):
            Ri, ti, si = unpack(x, i)
            Rj, tj, sj = unpack(x, j)
            Rrel = Ri.T @ Rj
            trel = Ri.T @ (tj - ti) / si
            res[7*e:7*e+3] = w * (Rot.from_matrix(
                Tij[:3, :3].T @ Rrel).as_rotvec()) / sr
            res[7*e+3:7*e+6] = w * (trel - Tij[:3, 3]) / stt
            res[7*e+6] = w * ((x[7*j+6] - x[7*i+6]) - dls) / ss
        res[-7:] = 50.0 * (x[:7] - x0[:7])
        return res
    sp = lil_matrix((len(edges) * 7 + 7, nn * 7), dtype=int)
    for e, (i, j, *_r) in enumerate(edges):
        sp[7*e:7*e+7, 7*i:7*i+7] = 1
        sp[7*e:7*e+7, 7*j:7*j+7] = 1
    sp[-7:, :7] = 1
    t0 = time.time()
    sol = least_squares(residuals, x0, jac_sparsity=sp, method="trf",
                        loss="soft_l1", f_scale=3.0, max_nfev=60, verbose=0)
    print(f"stage 3 (v2): {nn} nodes, {len(edges)} edges in {time.time()-t0:.0f}s | "
          f"cost {np.sum(residuals(x0)**2):.1f} -> {2*sol.cost:.1f}", flush=True)
    xs = sol.x
    corr = {}
    for f, i in nid.items():
        R0, t0_, _ = unpack(x0, i)
        R1, t1, s1 = unpack(xs, i)
        corr[f] = (R1 @ R0.T, t1 - (R1 @ R0.T) @ t0_, s1)
    frames_out = np.zeros((n, 3))
    node_arr = np.array(nodes)
    for f in range(n):
        k = node_arr[np.searchsorted(node_arr, f, side="right") - 1]
        Rg, tg, sg = corr[k]
        p = poses[f][:3, 3]
        anchor = poses[k][:3, 3]
        frames_out[f] = (Rg @ (sg * (p - anchor))) + (Rg @ anchor + tg)
    np.save(f"{OUTD}/traj_optimized2.npy", frames_out)
    print("stage 3 (v2): corrected trajectory saved", flush=True)

elif STAGE == 13:
    # v3 pose graph: translation-only closures, parallax-aware sigmas.
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    poses = list(np.load(f"{BASE}/poses_est.npy"))
    verified = json.load(open(f"{OUTD}/verified2.json"))
    nodes = sorted(set(KFS) | {0, n - 1})
    nid = {f: i for i, f in enumerate(nodes)}
    nn = len(nodes)
    x0 = np.zeros(nn * 7)
    for f, i in nid.items():
        T = poses[f]
        x0[7*i:7*i+3] = Rot.from_matrix(T[:3, :3]).as_rotvec()
        x0[7*i+3:7*i+6] = T[:3, 3]
    edges = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        Trel = np.linalg.inv(poses[a]) @ poses[b]
        edges.append((nid[a], nid[b], Trel, 0.0, 1.0, 0.02, 0.02, 0.05))
    steps = np.array([np.linalg.norm((np.linalg.inv(poses[i-1]) @ poses[i])[:3, 3])
                      for i in range(1, len(poses))])
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    kept, dropped = 0, 0
    for a, b, sim, T_ab_l, med_a, med_b, inl, rms, sig_t in verified:
        T_ab = np.array(T_ab_l)
        Trel = np.linalg.inv(T_ab)
        odo_rel = (np.linalg.inv(poses[a]) @ poses[b])[:3, 3]
        if np.linalg.norm(odo_rel - Trel[:3, 3]) > \
           max(DRIFT_FLOOR, DRIFT_BUDGET * max(cum[b] - cum[a], 1e-6)):
            dropped += 1
            continue
        kept += 1
        edges.append((nid[a], nid[b], Trel, 0.0, 1.0, 10.0, sig_t, 10.0))
    print(f"stage 13 (v3): alias gate kept {kept}, dropped {dropped}", flush=True)
    def unpack(x, i):
        R = Rot.from_rotvec(x[7*i:7*i+3]).as_matrix()
        return R, x[7*i+3:7*i+6], np.exp(x[7*i+6])
    def residuals(x):
        res = np.zeros(len(edges) * 7 + 7)
        for e, (i, j, Tij, dls, w, sr, stt, ss) in enumerate(edges):
            Ri, ti, si = unpack(x, i)
            Rj, tj, sj = unpack(x, j)
            Rrel = Ri.T @ Rj
            trel = Ri.T @ (tj - ti) / si
            res[7*e:7*e+3] = w * (Rot.from_matrix(
                Tij[:3, :3].T @ Rrel).as_rotvec()) / sr
            res[7*e+3:7*e+6] = w * (trel - Tij[:3, 3]) / stt
            res[7*e+6] = w * ((x[7*j+6] - x[7*i+6]) - dls) / ss
        res[-7:] = 50.0 * (x[:7] - x0[:7])
        return res
    sp = lil_matrix((len(edges) * 7 + 7, nn * 7), dtype=int)
    for e, (i, j, *_r) in enumerate(edges):
        sp[7*e:7*e+7, 7*i:7*i+7] = 1
        sp[7*e:7*e+7, 7*j:7*j+7] = 1
    sp[-7:, :7] = 1
    t0 = time.time()
    sol = least_squares(residuals, x0, jac_sparsity=sp, method="trf",
                        loss="soft_l1", f_scale=3.0, max_nfev=60)
    print(f"stage 13 (v3): {nn} nodes, {len(edges)} edges in {time.time()-t0:.0f}s",
          flush=True)
    xs = sol.x
    corr = {}
    for f, i in nid.items():
        R0, t0_, _ = unpack(x0, i)
        R1, t1, s1 = unpack(xs, i)
        corr[f] = (R1 @ R0.T, t1 - (R1 @ R0.T) @ t0_, s1)
    frames_out = np.zeros((n, 3))
    node_arr = np.array(nodes)
    for f in range(n):
        k = node_arr[np.searchsorted(node_arr, f, side="right") - 1]
        Rg, tg, sg = corr[k]
        p = poses[f][:3, 3]
        anchor = poses[k][:3, 3]
        frames_out[f] = (Rg @ (sg * (p - anchor))) + (Rg @ anchor + tg)
    np.save(f"{OUTD}/traj_optimized3.npy", frames_out)
    print("stage 13 (v3): corrected trajectory saved", flush=True)

elif STAGE == 14:
    from c3vdv2_deform import load_poses, umeyama_ate
    frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
    gt_all = load_poses(f"{BASE}/pose.txt")
    gt = np.array([gt_all[k][1] for k in nums[:n]])
    gt = gt - gt[0]
    est0 = np.array([p[:3, 3] for p in np.load(f"{BASE}/poses_est.npy")])[:n]
    est1 = np.load(f"{OUTD}/traj_optimized3.npy")[:n]
    base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
    a0 = umeyama_ate(gt, est0) * 1000
    a1 = umeyama_ate(gt, est1) * 1000
    print(f"stage 14 (v3) [{os.path.basename(BASE)}]:", flush=True)
    print(f"  odometry only : ATE {a0:.1f} mm ({a0/base:.2f}x baseline {base:.0f} mm)",
          flush=True)
    print(f"  + v3 closure  : ATE {a1:.1f} mm ({a1/base:.2f}x baseline)", flush=True)
    print(f"  change: {100*(a1-a0)/a0:+.1f}%", flush=True)

elif STAGE == 5:
    # ORACLE (dev diagnostic only, disclosed): rerun stage 3 with each kept
    # closure's relative pose replaced by GROUND TRUTH. This bounds what any
    # closure measurement could achieve at these 13 revisit locations: if the
    # oracle also improves only ~5%, the cap is revisit sparsity (a property
    # of screening withdrawals), not measurement error. GT touches nothing
    # that feeds the real pipeline.
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    from c3vdv2_deform import load_poses, umeyama_ate
    poses = list(np.load(f"{BASE}/poses_est.npy"))
    verified = json.load(open(f"{OUTD}/verified2.json"))
    frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
    gt_all = load_poses(f"{BASE}/pose.txt")
    gt_T = {}
    for k in nums[:n]:
        q = gt_all[k]
        T = np.eye(4); T[:3, 3] = q[1]
        T[:3, :3] = Rot.from_quat(q[0]).as_matrix() if len(q) > 1 and \
            hasattr(q[0], "__len__") and len(q[0]) == 4 else np.eye(3)
        gt_T[nums.index(k)] = T
    # GT scale differs from depth-network scale; estimate one global scale from
    # path lengths so oracle closures live in the odometry's units
    est_steps = sum(np.linalg.norm((np.linalg.inv(poses[i-1]) @ poses[i])[:3, 3])
                    for i in range(1, n))
    gt_pos = np.array([gt_all[k][1] for k in nums[:n]])
    gt_steps = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()
    s_gt2est = est_steps / max(gt_steps, 1e-9)
    nodes = sorted(set(KFS) | {0, n - 1})
    nid = {f: i for i, f in enumerate(nodes)}
    nn = len(nodes)
    x0 = np.zeros(nn * 7)
    for f, i in nid.items():
        T = poses[f]
        x0[7*i:7*i+3] = Rot.from_matrix(T[:3, :3]).as_rotvec()
        x0[7*i+3:7*i+6] = T[:3, 3]
    edges = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        Trel = np.linalg.inv(poses[a]) @ poses[b]
        edges.append((nid[a], nid[b], Trel, 0.0, 1.0, 0.02, 0.02, 0.05))
    steps = np.array([np.linalg.norm((np.linalg.inv(poses[i-1]) @ poses[i])[:3, 3])
                      for i in range(1, len(poses))])
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    kept = 0
    for a, b, sim, T_ab_l, med_a, med_b, inl, rms, sig_t in verified:
        T_ab = np.array(T_ab_l)
        Trel = np.linalg.inv(T_ab)
        odo_rel = (np.linalg.inv(poses[a]) @ poses[b])[:3, 3]
        if np.linalg.norm(odo_rel - Trel[:3, 3]) > \
           max(DRIFT_FLOOR, DRIFT_BUDGET * max(cum[b] - cum[a], 1e-6)):
            continue
        # oracle: GT relative position (scaled), tight sigma
        rel_gt = (gt_pos[b] - gt_pos[a]) * s_gt2est
        T_o = np.eye(4); T_o[:3, 3] = rel_gt
        kept += 1
        edges.append((nid[a], nid[b], T_o, 0.0, 1.0, 10.0, 0.01, 10.0))
    print(f"oracle: {kept} GT closures (scale {s_gt2est:.4f})", flush=True)
    def unpack(x, i):
        R = Rot.from_rotvec(x[7*i:7*i+3]).as_matrix()
        return R, x[7*i+3:7*i+6], np.exp(x[7*i+6])
    def residuals(x):
        res = np.zeros(len(edges) * 7 + 7)
        for e, (i, j, Tij, dls, w, sr, stt, ss) in enumerate(edges):
            Ri, ti, si = unpack(x, i)
            Rj, tj, sj = unpack(x, j)
            Rrel = Ri.T @ Rj
            trel = Ri.T @ (tj - ti) / si
            res[7*e:7*e+3] = w * (Rot.from_matrix(
                Tij[:3, :3].T @ Rrel).as_rotvec()) / sr
            res[7*e+3:7*e+6] = w * (trel - Tij[:3, 3]) / stt
            res[7*e+6] = w * ((x[7*j+6] - x[7*i+6]) - dls) / ss
        res[-7:] = 50.0 * (x[:7] - x0[:7])
        return res
    sp = lil_matrix((len(edges) * 7 + 7, nn * 7), dtype=int)
    for e, (i, j, *_r) in enumerate(edges):
        sp[7*e:7*e+7, 7*i:7*i+7] = 1
        sp[7*e:7*e+7, 7*j:7*j+7] = 1
    sp[-7:, :7] = 1
    sol = least_squares(residuals, x0, jac_sparsity=sp, method="trf",
                        loss="soft_l1", f_scale=3.0, max_nfev=60)
    xs = sol.x
    corr = {}
    for f, i in nid.items():
        R0, t0_, _ = unpack(x0, i)
        R1, t1, s1 = unpack(xs, i)
        corr[f] = (R1 @ R0.T, t1 - (R1 @ R0.T) @ t0_, s1)
    frames_out = np.zeros((n, 3))
    node_arr = np.array(nodes)
    for f in range(n):
        k2 = node_arr[np.searchsorted(node_arr, f, side="right") - 1]
        Rg, tg, sg = corr[k2]
        p = poses[f][:3, 3]
        anchor = poses[k2][:3, 3]
        frames_out[f] = (Rg @ (sg * (p - anchor))) + (Rg @ anchor + tg)
    gt_c = gt_pos - gt_pos[0]
    est0 = np.array([p[:3, 3] for p in poses])[:n]
    base = float(np.sqrt(((gt_c - gt_c.mean(0)) ** 2).sum(1).mean())) * 1000
    a0 = umeyama_ate(gt_c, est0) * 1000
    a1 = umeyama_ate(gt_c, frames_out[:n]) * 1000
    print(f"oracle [{os.path.basename(BASE)}]: odometry {a0:.1f} mm -> "
          f"GT-closures {a1:.1f} mm ({100*(a1-a0)/a0:+.1f}%) | baseline {base:.0f}",
          flush=True)

elif STAGE == 4:
    from c3vdv2_deform import load_poses, umeyama_ate
    frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
    gt_all = load_poses(f"{BASE}/pose.txt")
    gt = np.array([gt_all[k][1] for k in nums[:n]])
    gt = gt - gt[0]
    est0 = np.array([p[:3, 3] for p in np.load(f"{BASE}/poses_est.npy")])[:n]
    est1 = np.load(f"{OUTD}/traj_optimized2.npy")[:n]
    base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
    a0 = umeyama_ate(gt, est0) * 1000
    a1 = umeyama_ate(gt, est1) * 1000
    print(f"stage 4 (v2) [{os.path.basename(BASE)}]:", flush=True)
    print(f"  odometry only : ATE {a0:.1f} mm ({a0/base:.2f}x baseline {base:.0f} mm)",
          flush=True)
    print(f"  + loop closure: ATE {a1:.1f} mm ({a1/base:.2f}x baseline)", flush=True)
    print(f"  change: {100*(a1-a0)/a0:+.1f}%", flush=True)
