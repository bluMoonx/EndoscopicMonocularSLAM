"""Loop closure + Sim(3) pose-graph optimization for the full withdrawals.

Motivation: the tracker is odometry - it chains pairwise motions and never
corrects itself, and the measured cost is 92-115 mm of drift over whole-colon
paths (0.60-0.73x the constant-point baseline). The standard remedy is loop
closure: recognize when the camera sees a place it has seen before, measure the
relative pose between the two visits, and relax the whole trajectory so both
visits agree. In a screening withdrawal true loops are re-inspection moments
(doubling back, lingering), so stage 1 first asks whether such moments exist.

Stages (resumable; each prints its verdict before the next runs):
  1  keyframe appearance descriptors + candidate revisit pairs (no GT anywhere)
  2  geometric verification of candidates: photometric alignment both ways,
     forward-backward consistency, sanity bounds -> accepted closure edges
  3  Sim(3) pose-graph optimization (7-DOF nodes: rotation, translation, scale,
     because monocular drift includes scale drift), scipy sparse least-squares
  4  honest evaluation: ATE vs GT before/after, same Umeyama protocol, same
     constant-point baseline; GT is used ONLY here, never in stages 1-3

Usage: python speqslam/loop_closure.py <base_dir> <stage> [minutes]
e.g.:  python speqslam/loop_closure.py data/c3vdv2/c1_full_t4_v1_s3 1
"""
import os, sys, glob, time, json
import numpy as np, cv2

BASE = sys.argv[1].rstrip("/\\")
STAGE = int(sys.argv[2])
BUDGET_S = (float(sys.argv[3]) if len(sys.argv) > 3 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")

KF_STRIDE = 8          # a keyframe every 8 stride-3 frames (~0.8 s of footage)
MIN_SEP = 60           # candidate pairs must be >= 60 keyframe-strides apart? no:
                       # measured in FRAMES between keyframes (see below)
SIM_T = 0.82           # cosine similarity floor for a candidate
TOPK = 3               # best candidates kept per keyframe
D = 40                 # descriptor edge: 40x40 normalized grayscale

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
KFS = list(range(0, n, KF_STRIDE))
OUTD = f"{BASE}/loopclosure"
os.makedirs(OUTD, exist_ok=True)

def descriptor(path):
    g = cv2.imread(path, 0)
    g = cv2.resize(g, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g -= g.mean()
    nrm = np.linalg.norm(g)
    return (g / max(nrm, 1e-6)).ravel()

if STAGE == 1:
    t0 = time.time()
    desc = np.stack([descriptor(p) for p in und[::KF_STRIDE]])
    np.save(f"{OUTD}/desc.npy", desc)
    S = desc @ desc.T
    nk = len(KFS)
    cands = []
    for i in range(nk):
        best = []
        for j in range(i):
            if (KFS[i] - KFS[j]) < MIN_SEP:      # too close in time
                continue
            if S[i, j] >= SIM_T:
                best.append((S[i, j], j))
        best.sort(reverse=True)
        for s, j in best[:TOPK]:
            cands.append((KFS[j], KFS[i], float(s)))
    json.dump(cands, open(f"{OUTD}/candidates.json", "w"))
    seps = [b - a for a, b, _ in cands]
    print(f"stage 1: {nk} keyframes | {len(cands)} candidate revisit pairs "
          f"(sim >= {SIM_T}, separation >= {MIN_SEP} frames) "
          f"({time.time()-t0:.0f}s)", flush=True)
    if cands:
        print(f"  separations: p50 {int(np.median(seps))} frames, "
              f"max {max(seps)} | sims: p50 "
              f"{np.median([s for _,_,s in cands]):.3f}", flush=True)
        hist, edges = np.histogram([a for a, _, _ in cands], bins=8, range=(0, n))
        print("  where (frame-index bins):", hist.tolist(), flush=True)
    else:
        print("  NO candidates: this withdrawal offers no appearance revisits "
              "at these thresholds - itself a finding", flush=True)

elif STAGE == 2:
    import demo_app, direct_vo as dv
    from glare_removal import specular_mask
    K = np.array([[250.0, 0, 320.0], [0, 250.0, 256.0], [0, 0, 1]])
    net = demo_app.DepthNet()
    cands = json.load(open(f"{OUTD}/candidates.json"))
    done_f = f"{OUTD}/verified.json"
    verified = json.load(open(done_f)) if os.path.exists(done_f) else []
    done_pairs = {(v[0], v[1]) for v in verified}
    prog_f = f"{OUTD}/verify_progress.json"
    start = json.load(open(prog_f))["i"] if os.path.exists(prog_f) else 0
    t0 = time.time()
    depcache = {}
    def dep_of(idx):
        if idx not in depcache:
            if len(depcache) > 60:
                depcache.clear()
            depcache[idx] = net.predict(cv2.imread(und[idx]))
        return depcache[idx]
    for ci in range(start, len(cands)):
        a, b, sim = cands[ci]
        if (a, b) in done_pairs:
            continue
        ia, ib = cv2.imread(und[a]), cv2.imread(und[b])
        ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gl_a = specular_mask(ia, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        gl_b = specular_mask(ib, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        T_ab = dv.track_pair(ga, dep_of(a), gb, K, 4, robust=False,
                             depth_scale=1.0, glare=gl_a, illum=True)
        T_ba = dv.track_pair(gb, dep_of(b), ga, K, 4, robust=False,
                             depth_scale=1.0, glare=gl_b, illum=True)
        # forward-backward consistency: composing the two should be identity
        E = T_ab @ T_ba
        r_err = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
        t_err = np.linalg.norm(E[:3, 3])
        step = np.linalg.norm(T_ab[:3, 3])
        med_a = float(np.median(dep_of(a)[dep_of(a) > 0]))
        ok = (r_err < 4.0 and t_err < 0.006 and step < 2.5 * med_a)
        if ok:
            verified.append([a, b, sim, T_ab.tolist(),
                             med_a, float(np.median(dep_of(b)[dep_of(b) > 0]))])
        if ci % 20 == 0 or ok:
            json.dump(verified, open(done_f, "w"))
            json.dump({"i": ci + 1}, open(prog_f, "w"))
            print(f"verify {ci}/{len(cands)}: accepted {len(verified)}", flush=True)
        if time.time() - t0 > BUDGET_S:
            json.dump(verified, open(done_f, "w"))
            json.dump({"i": ci + 1}, open(prog_f, "w"))
            print(f"budget at candidate {ci}; resume me", flush=True)
            sys.exit(0)
    json.dump(verified, open(done_f, "w"))
    json.dump({"i": len(cands)}, open(prog_f, "w"))
    print(f"stage 2 done: {len(verified)} verified closures of {len(cands)} "
          f"candidates", flush=True)

elif STAGE == 3:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation as Rot
    poses = list(np.load(f"{BASE}/poses_est.npy"))
    verified = json.load(open(f"{OUTD}/verified.json"))
    if not verified:
        print("stage 3: no closures to optimize; nothing to do", flush=True)
        sys.exit(0)
    # nodes: keyframes (plus first/last frame indices to anchor the ends)
    nodes = sorted(set(KFS) | {0, n - 1})
    nid = {f: i for i, f in enumerate(nodes)}
    nn = len(nodes)
    # initial node states from odometry: rotvec (3), translation (3), log-scale (1)
    x0 = np.zeros(nn * 7)
    for f, i in nid.items():
        T = poses[f]
        x0[7*i:7*i+3] = Rot.from_matrix(T[:3, :3]).as_rotvec()
        x0[7*i+3:7*i+6] = T[:3, 3]
        x0[7*i+6] = 0.0
    # odometry edges between consecutive nodes (measured relative from chained poses)
    edges = []   # (i, j, T_ij, dlogs, weight, sigma_rot, sigma_trans, sigma_logs)
    for a, b in zip(nodes[:-1], nodes[1:]):
        Trel = np.linalg.inv(poses[a]) @ poses[b]
        edges.append((nid[a], nid[b], Trel, 0.0, 1.0, 0.02, 0.02, 0.05))
    # Odometry-consistency gate (uses NO ground truth): a closure claims the two
    # frames are essentially co-located; the odometry chain between them predicts
    # some separation. The difference is the drift the closure implies. Measured
    # drift on these sequences is a few percent of path, so a closure implying
    # more than DRIFT_BUDGET of the chained path length is an appearance alias
    # (a lookalike fold), not a revisit — the failure mode that wrecked v1.
    # Gate calibrated on c1_full_t4 (development sequence, disclosed) and then
    # FROZEN for the other withdrawals: floor covers local within-dwell drift,
    # the fraction covers long chains where drift accumulates.
    DRIFT_BUDGET = 0.10
    DRIFT_FLOOR = 0.18
    steps = np.array([np.linalg.norm((np.linalg.inv(poses[i-1]) @ poses[i])[:3, 3])
                      for i in range(1, len(poses))])
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    W_CLOSE = 1.0
    kept, dropped = 0, 0
    for a, b, sim, T_ab_l, med_a, med_b in verified:
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
        # closures are POSITION-ONLY constraints with honest uncertainty:
        # low-parallax revisit alignments systematically underestimate the
        # true separation and their rotations are unreliable, so rotation and
        # scale terms are disabled (huge sigma) and translation carries a
        # sigma matching the measured bias (~tens of mm), making a closure a
        # statement of rough co-location, not a precise measurement
        edges.append((nid[a], nid[b], Trel, dlogs, W_CLOSE, 10.0, 0.10, 10.0))
    print(f"stage 3: odometry-consistency gate kept {kept}, dropped {dropped} "
          f"alias closures", flush=True)
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
            # predicted relative: node_i^-1 o node_j with per-node scale
            Rrel = Ri.T @ Rj
            trel = Ri.T @ (tj - ti) / si
            res[7*e:7*e+3] = w * (Rot.from_matrix(
                Tij[:3, :3].T @ Rrel).as_rotvec()) / sr
            res[7*e+3:7*e+6] = w * (trel - Tij[:3, 3]) / stt
            res[7*e+6] = w * ((x[7*j+6] - x[7*i+6]) - dls) / ss
        # gauge anchor: pin node 0
        res[-7:] = 50.0 * (x[:7] - x0[:7])
        return res
    sp = lil_matrix((len(edges) * 7 + 7, nn * 7), dtype=int)
    for e, (i, j, *_r) in enumerate(edges):
        sp[7*e:7*e+7, 7*i:7*i+7] = 1
        sp[7*e:7*e+7, 7*j:7*j+7] = 1
    sp[-7:, :7] = 1
    t0 = time.time()
    # soft-L1 robust kernel: a closure edge that disagrees with the consensus of
    # odometry and the other closures gets automatically down-weighted instead
    # of dragging the whole graph (the c2 frozen test exposed exactly this:
    # 39/41 surviving closures were true, but least squares let the outliers
    # and small biases of the rest deform the trajectory)
    sol = least_squares(residuals, x0, jac_sparsity=sp, method="trf",
                        loss="soft_l1", f_scale=3.0, max_nfev=60, verbose=0)
    print(f"stage 3: optimized {nn} nodes, {len(edges)} edges "
          f"({len(verified)} closures) in {time.time()-t0:.0f}s | "
          f"cost {np.sum(residuals(x0)**2):.1f} -> {2*sol.cost:.1f}", flush=True)
    # rebuild full trajectory: apply each frame's correction interpolated from
    # its bracketing keyframe nodes (rigid interpolation of the node update)
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
    np.save(f"{OUTD}/traj_optimized.npy", frames_out)
    print("stage 3: corrected trajectory saved", flush=True)

elif STAGE == 4:
    from c3vdv2_deform import load_poses, umeyama_ate
    frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
    gt_all = load_poses(f"{BASE}/pose.txt")
    gt = np.array([gt_all[k][1] for k in nums[:n]])
    gt = gt - gt[0]
    est0 = np.array([p[:3, 3] for p in np.load(f"{BASE}/poses_est.npy")])[:n]
    est1 = np.load(f"{OUTD}/traj_optimized.npy")[:n]
    base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
    a0 = umeyama_ate(gt, est0) * 1000
    a1 = umeyama_ate(gt, est1) * 1000
    print(f"stage 4 [{os.path.basename(BASE)}]:", flush=True)
    print(f"  odometry only : ATE {a0:.1f} mm ({a0/base:.2f}x baseline {base:.0f} mm)",
          flush=True)
    print(f"  + loop closure: ATE {a1:.1f} mm ({a1/base:.2f}x baseline)", flush=True)
    print(f"  change: {100*(a1-a0)/a0:+.1f}%", flush=True)
