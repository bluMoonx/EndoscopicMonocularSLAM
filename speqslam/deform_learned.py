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
