"""Deformation attempt 6: LEARNED correspondence weighting - the last named
candidate. A small network predicts, per pixel, how much the observed motion
disagrees with any rigid explanation; the tracker down-weights those pixels
DURING optimization.

Why a network can see what the optimizer cannot: attempts 1-5 established
that deforming pixels cannot be found post-hoc (residuals absorbed), cannot
be out-voted (they are often the majority), and move partly invisibly. But
their APPEARANCE and local flow structure are distinctive, and we can
manufacture dense supervision: on sequences with GT camera poses, the rigid
flow implied by (GT rotation + predicted depth + a single per-pair
translation-magnitude fit along the GT direction, which removes monocular
scale) is the best rigid story for that pair - per-pixel disagreement
between observed optical flow and that story IS the deformation, labeled
without any pixel annotations. The GT pose is the teacher; at inference the
network sees only images, flow and predicted depth.

PROTOCOL (stated before any training or result):
  training data   ONLY the ten burned attempt-4/5 sequences (6 deforming +
                  4 controls). Pose-supervised labels as above; convention
                  self-check on a rigid control pair is printed and fixed.
  model           smp Unet / mobilenet_v2, in_channels=5 (gray_i, gray_j,
                  flow_u, flow_v, depth), one output channel, L1 on
                  log1p(residual flow px), half resolution.
  inference       per-pixel weights w = 1 / (1 + (score/ALPHA)^2) multiply
                  the photometric weights in the tracker; ALPHA is tuned on
                  the burned deforming sequences (dev), then FROZEN.
  judge           the seven fresh sequences named in deform_model.py
                  (4 deforming, 3 controls; FIDs 14196/14300/14181/14240 +
                  14291/14280/14329), downloaded only after the freeze.
  criterion       median ATE improvement across the 4 fresh deforming
                  sequences >= 20%, no fresh control worse than 10% - the
                  same bar every deformation attempt has faced.

Usage: python speqslam/deform_learned.py gen   [minutes]  (training tensors)
       python speqslam/deform_learned.py train [minutes]
       python speqslam/deform_learned.py dev   [minutes]  (ALPHA sweep, burned set)
       python speqslam/deform_learned.py test  [minutes]  (frozen judge)
"""
import os, sys, csv, glob, json, time
import numpy as np, cv2
MODE = sys.argv[1] if len(sys.argv) > 1 else "gen"
BUDGET_S = (float(sys.argv[2]) if len(sys.argv) > 2 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
from c3vdv2_deform import V2CAL, SEQS, load_poses, umeyama_ate

GEN_DIR = "data/_deform_learned"
MAXF = 120
HALF = (320, 256)        # training/inference resolution (full undistort is 640x512)

def dis_flow(g0, g1):
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return d.calc((g0 * 255).astype(np.uint8), (g1 * 255).astype(np.uint8), None)

def rigid_flow_fit(dep, K, R_rel, t_dir, flow_obs, valid):
    """Rigid flow with GT rotation and translation = s * t_dir; s fit by 1-D
    search minimizing median |flow_obs - flow_rigid| over valid pixels.
    Returns (residual map px, best rigid flow)."""
    h, w = dep.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    Z = dep
    X = (xx - cx) / fx * Z
    Y = (yy - cy) / fy * Z
    P = np.stack([X, Y, Z], -1)
    Pr = P @ R_rel.T
    best = None
    for s in np.linspace(-0.02, 0.02, 81):
        Pc = Pr + s * t_dir
        u = fx * Pc[..., 0] / np.maximum(Pc[..., 2], 1e-6) + cx
        v = fy * Pc[..., 1] / np.maximum(Pc[..., 2], 1e-6) + cy
        fr = np.stack([u - xx, v - yy], -1)
        res = np.linalg.norm(flow_obs - fr, axis=-1)
        m = float(np.median(res[valid]))
        if best is None or m < best[0]:
            best = (m, res, fr, s)
    return best

def pair_tensors(img0, img1, dep0, K, R_rel, t_dir, glare):
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    flow = dis_flow(g0, g1)
    valid = (dep0 > 0) & (glare == 0)
    med_fit, res, fr, s = rigid_flow_fit(dep0, K, R_rel, t_dir, flow, valid)
    rs = lambda a: cv2.resize(a, HALF, interpolation=cv2.INTER_AREA)
    x = np.stack([rs(g0), rs(g1),
                  rs(flow[..., 0]) / 8.0, rs(flow[..., 1]) / 8.0,
                  rs(dep0) / max(1e-6, float(np.median(dep0[dep0 > 0])))], 0)
    y = rs(res)
    vm = rs(valid.astype(np.float32)) > 0.5
    return x.astype(np.float16), np.log1p(y).astype(np.float16), vm, med_fit

if __name__ != "__main__":
    MODE = "imported"

if MODE == "gen":
    import demo_app
    import scaramuzza_undistort as su
    from glare_removal import specular_mask
    os.makedirs(GEN_DIR, exist_ok=True)
    mapu, mapv, K = su.build_map(V2CAL)
    net = demo_app.DepthNet()
    t0 = time.time()
    convention = None
    for seq, kind in SEQS:
        outf = f"{GEN_DIR}/{seq}.npz"
        if os.path.exists(outf):
            continue
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        base = f"data/c3vdv2/{seq}"
        rgbs = sorted(glob.glob(f"{base}/rgb/*.png")) or \
            sorted(glob.glob(f"{base}/*/rgb/*.png"))
        pc = glob.glob(f"{base}/pose.txt") + glob.glob(f"{base}/*/pose.txt")
        poses_gt = load_poses(pc[0])
        n_all = min(len(rgbs), len(poses_gt))
        stride = max(1, int(np.ceil(n_all / MAXF)))
        idx = list(range(0, n_all, stride))
        Xs, Ys, Vs = [], [], []
        fits = []
        for a, b in zip(idx[:-1], idx[1:]):
            i0 = su.undistort(cv2.imread(rgbs[a]), mapu, mapv)
            i1 = su.undistort(cv2.imread(rgbs[b]), mapu, mapv)
            dep = net.predict(i0)
            glare = specular_mask(i0, method="adaptive", k=2.0, lo=225, hi=245,
                                  dilate=1)
            Ra, ta = poses_gt[a]; Rb, tb = poses_gt[b]
            # stored R maps camera->world (row-vector file convention transposed)
            if convention is None and kind == "control":
                cands = {}
                for name, Rr in (("c2w", Ra.T @ Rb), ("w2c", Ra @ Rb.T)):
                    td = Ra.T @ (tb - ta) if name == "c2w" else Ra @ (ta - tb)
                    nrm = np.linalg.norm(td)
                    td = td / nrm if nrm > 1e-9 else np.array([0, 0, 1.0])
                    g0 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                    g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                    fl = dis_flow(g0, g1)
                    m, *_ = rigid_flow_fit(dep, K, Rr, td,
                                           fl, (dep > 0) & (glare == 0))
                    cands[name] = m
                convention = min(cands, key=cands.get)
                print(f"pose convention self-check: {cands} -> {convention}",
                      flush=True)
            conv = convention or "c2w"
            if conv == "c2w":
                R_rel = Ra.T @ Rb
                td = Ra.T @ (tb - ta)
            else:
                R_rel = Ra @ Rb.T
                td = Ra @ (ta - tb)
            nrm = np.linalg.norm(td)
            td = td / nrm if nrm > 1e-9 else np.array([0, 0, 1.0])
            x, y, vm, med_fit = pair_tensors(i0, i1, dep, K, R_rel, td, glare)
            Xs.append(x); Ys.append(y); Vs.append(vm); fits.append(med_fit)
        np.savez_compressed(outf, X=np.array(Xs), Y=np.array(Ys),
                            V=np.array(Vs), kind=kind)
        print(f"{seq} [{kind}]: {len(Xs)} pairs, median rigid-fit residual "
              f"{np.median(fits):.2f}px", flush=True)
    print("gen session done", flush=True)

elif MODE == "train":
    import torch
    import segmentation_models_pytorch as smp
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7); np.random.seed(7)
    os.makedirs("models/deform_weight", exist_ok=True)
    Xs, Ys, Vs = [], [], []
    for f in sorted(glob.glob(f"{GEN_DIR}/*.npz")):
        z = np.load(f)
        Xs.append(z["X"]); Ys.append(z["Y"]); Vs.append(z["V"])
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    V = np.concatenate(Vs).astype(np.float32)
    n = len(X)
    rng = np.random.RandomState(7)
    order = rng.permutation(n)
    X, Y, V = X[order], Y[order], V[order]
    nval = max(32, n // 10)
    net = smp.Unet("mobilenet_v2", encoder_weights="imagenet", in_channels=5,
                   classes=1, activation=None).to(DEV)
    ck = "models/deform_weight/ckpt.pth"
    start_ep = 0
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    if os.path.exists(ck):
        z = torch.load(ck, map_location=DEV, weights_only=True)
        net.load_state_dict(z["model"]); opt.load_state_dict(z["opt"])
        start_ep = z["ep"]
    t0 = time.time()
    for ep in range(start_ep, 15):
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        net.train()
        losses = []
        for i in range(nval, n, 8):
            xb = torch.from_numpy(X[i:i+8]).to(DEV)
            yb = torch.from_numpy(Y[i:i+8]).to(DEV)
            vb = torch.from_numpy(V[i:i+8]).to(DEV)
            opt.zero_grad()
            p = net(xb)[:, 0]
            loss = (torch.abs(p - yb) * vb).sum() / vb.sum().clamp(min=1)
            loss.backward(); opt.step()
            losses.append(float(loss))
        net.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, nval, 8):
                xb = torch.from_numpy(X[i:i+8]).to(DEV)
                yb = torch.from_numpy(Y[i:i+8]).to(DEV)
                vb = torch.from_numpy(V[i:i+8]).to(DEV)
                p = net(xb)[:, 0]
                vl.append(float((torch.abs(p - yb) * vb).sum() / vb.sum()))
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "ep": ep + 1}, ck)
        torch.save(net.state_dict(), "models/deform_weight/unet_mbv2.pth")
        print(f"epoch {ep+1}/15: train L1 {np.mean(losses):.4f} | "
              f"val L1 {np.mean(vl):.4f}", flush=True)
    print("train session done", flush=True)

elif MODE in ("dev", "test"):
    import torch
    import segmentation_models_pytorch as smp
    import demo_app, direct_vo as dv
    import scaramuzza_undistort as su
    from glare_removal import specular_mask
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    wnet = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=5,
                    classes=1, activation=None).to(DEV).eval()
    wnet.load_state_dict(torch.load("models/deform_weight/unet_mbv2.pth",
                                    map_location=DEV, weights_only=True))
    mapu, mapv, K = su.build_map(V2CAL)
    net = demo_app.DepthNet()
    ALPHA = 1.2    # FROZEN after dev (results/deform_learned_dev.txt)

    def score_map(g0, g1, dep):
        flow = dis_flow(g0, g1)
        rs = lambda a: cv2.resize(a, HALF, interpolation=cv2.INTER_AREA)
        x = np.stack([rs(g0), rs(g1), rs(flow[..., 0]) / 8.0,
                      rs(flow[..., 1]) / 8.0,
                      rs(dep) / max(1e-6, float(np.median(dep[dep > 0])))],
                     0).astype(np.float32)
        with torch.no_grad():
            p = wnet(torch.from_numpy(x[None]).to(DEV))[0, 0].cpu().numpy()
        return cv2.resize(p, (g0.shape[1], g0.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def run_weighted(seq, alpha, with_base):
        base = f"data/c3vdv2/{seq}"
        rgbs = sorted(glob.glob(f"{base}/rgb/*.png")) or \
            sorted(glob.glob(f"{base}/*/rgb/*.png"))
        pc = glob.glob(f"{base}/pose.txt") + glob.glob(f"{base}/*/pose.txt")
        poses_gt = load_poses(pc[0])
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
        arms = (["base"] if with_base else []) + ["weighted"]
        for arm in arms:
            pose = np.eye(4); traj = [pose[:3, 3].copy()]
            for i in range(1, len(frames)):
                g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                glare = specular_mask(frames[i-1], method="adaptive", k=2.0,
                                      lo=225, hi=245, dilate=1)
                wm = None
                if arm == "weighted":
                    s = score_map(g0, g1, deps[i-1])
                    wm = 1.0 / (1.0 + (np.maximum(s, 0) / alpha) ** 2)
                T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False,
                                  depth_scale=1.0, glare=glare, illum=True,
                                  wmap=wm)
                pose = pose @ np.linalg.inv(T)
                traj.append(pose[:3, 3].copy())
            out[arm] = umeyama_ate(gt, np.array(traj)) * 1000
        return out

    if MODE == "dev":
        baselines = {r["seq"]: float(r["ate_base_mm"])
                     for r in csv.DictReader(open("results/deform_aware.csv"))}
        t0 = time.time()
        donelog = "results/deform_learned_dev.txt"
        have = set()
        if os.path.exists(donelog):
            for ln in open(donelog):
                have.add(ln.split(":")[0])
        with open(donelog, "a") as f:
            for alpha in (0.7, 1.2, 2.0):
                for seq, kind in SEQS:
                    key = f"a={alpha:g} {seq}"
                    if key in have:
                        continue
                    if time.time() - t0 > BUDGET_S:
                        print("budget reached; resume me", flush=True)
                        sys.exit(0)
                    r = run_weighted(seq, alpha, with_base=False)
                    line = (f"{key}: base {baselines[seq]:.2f} weighted "
                            f"{r['weighted']:.2f} mm [{kind}]")
                    print(line, flush=True)
                    f.write(line + "\n"); f.flush()
        print("dev sweep done", flush=True)
    else:
        from deform_model import TEST2
        res = "results/deform_learned_test.csv"
        done = set()
        if os.path.exists(res):
            done = {r["seq"] for r in csv.DictReader(open(res))}
        else:
            with open(res, "w", newline="") as f:
                csv.writer(f).writerow(["seq", "kind", "ate_base_mm",
                                        "ate_weighted_mm"])
        t0 = time.time()
        for seq, kind in TEST2:
            if seq in done:
                continue
            if time.time() - t0 > BUDGET_S:
                print("budget reached; resume me", flush=True)
                break
            r = run_weighted(seq, ALPHA, with_base=True)
            with open(res, "a", newline="") as f:
                csv.writer(f).writerow([seq, kind, f"{r['base']:.2f}",
                                        f"{r['weighted']:.2f}"])
            print(f"{seq} [{kind}]: base {r['base']:.2f} | weighted "
                  f"{r['weighted']:.2f} mm", flush=True)
        rows = list(csv.DictReader(open(res)))
        if len(rows) == len(TEST2):
            d_imp = [1 - float(r["ate_weighted_mm"]) / float(r["ate_base_mm"])
                     for r in rows if r["kind"] == "deform"]
            c_wor = [float(r["ate_weighted_mm"]) / float(r["ate_base_mm"]) - 1
                     for r in rows if r["kind"] == "control"]
            ok = np.median(d_imp) >= 0.20 and max(c_wor) <= 0.10
            print(f"\nVERDICT attempt 6 (pre-registered): deform median "
                  f"{100*np.median(d_imp):+.1f}% | worst control "
                  f"{100*max(c_wor):+.1f}% -> {'PASS' if ok else 'FAIL'}",
                  flush=True)
