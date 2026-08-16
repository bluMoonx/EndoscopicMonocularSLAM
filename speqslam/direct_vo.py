"""
Argus - minimal dense DIRECT (photometric) visual odometry.

This is the ECC alignment experiment promoted to a full 6-DOF trajectory. For each consecutive
frame pair it estimates the camera motion T (SE(3)) that minimizes the PHOTOMETRIC residual
  sum_x [ I_cur( warp(x, T, depth) ) - I_ref(x) ]^2
via Gauss-Newton on a coarse-to-fine image pyramid, using per-pixel depth to warp in 3D. It
chains the per-frame motions into a camera trajectory and writes it in TUM format for `evo`.

Because it tracks by raw brightness (which specular glare violates), it is GLARE-SENSITIVE -
the opposite of feature-based ORB-SLAM3. A `robust` toggle down-weights large residuals
(Huber): ON approximates ORB-SLAM3's outlier immunity, OFF exposes the glare sensitivity.
"""
import cv2, numpy as np, glob, os, argparse
from scipy.spatial.transform import Rotation


def se3_exp(xi):
    """Exponential map: 6-vector [vx,vy,vz, wx,wy,wz] (trans, rot) -> 4x4 SE(3)."""
    v, w = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        R = np.eye(3) + W
        V = np.eye(3)
    else:
        A, B, C = np.sin(th) / th, (1 - np.cos(th)) / th**2, (th - np.sin(th)) / th**3
        R = np.eye(3) + A * W + B * (W @ W)
        V = np.eye(3) + B * W + C * (W @ W)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = V @ v
    return T


def _bilinear(img, x, y):
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    wx = x - x0; wy = y - y0
    return (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)


def _align_level(ref, dep, cur, K, T, iters, robust, depth_scale, glare=None):
    h, w = ref.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = dep > 0
    if glare is not None:                 # glare-aware: drop specular pixels from the photometric cost
        valid = valid & (glare == 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 50:
        return T
    d = dep[ys, xs].astype(np.float32) * depth_scale
    P = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], 1)  # ref-frame 3D points (N,3)
    Iref = ref[ys, xs].astype(np.float32)
    gx = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    for _ in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        Pc = P @ R.T + t
        Zc = Pc[:, 2]
        u = fx * Pc[:, 0] / Zc + cx
        v = fy * Pc[:, 1] / Zc + cy
        m = (Zc > 1e-6) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if m.sum() < 50:
            break
        uu, vv, Pcv, Zv = u[m], v[m], Pc[m], Zc[m]
        r = _bilinear(cur, uu, vv) - Iref[m]
        gxv, gyv = _bilinear(gx, uu, vv), _bilinear(gy, uu, vv)
        iZ = 1.0 / Zv
        # dI/dPc (N,3): image gradient composed with projection Jacobian
        Jpp = np.stack([gxv * fx * iZ, gyv * fy * iZ,
                        -(gxv * fx * Pcv[:, 0] + gyv * fy * Pcv[:, 1]) * iZ * iZ], 1)
        # dPc/dxi = [I | -skew(Pc)] (N,3,6)
        N = len(uu); dP = np.zeros((N, 3, 6))
        dP[:, 0, 0] = dP[:, 1, 1] = dP[:, 2, 2] = 1.0
        dP[:, 0, 4] = Zv;        dP[:, 0, 5] = -Pcv[:, 1]
        dP[:, 1, 3] = -Zv;       dP[:, 1, 5] = Pcv[:, 0]
        dP[:, 2, 3] = Pcv[:, 1]; dP[:, 2, 4] = -Pcv[:, 0]
        J = np.einsum('nk,nkj->nj', Jpp, dP)  # (N,6)
        if robust:  # Huber weights
            s = 1.4826 * np.median(np.abs(r)) + 1e-6
            a = np.abs(r) / (1.345 * s)
            wt = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-6))
        else:
            wt = np.ones_like(r)
        H = np.einsum('ni,n,nj->ij', J, wt, J)
        g = np.einsum('ni,n->i', J, wt * r)
        try:
            delta = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta) @ T
        if np.linalg.norm(delta) < 1e-6:
            break
    return T


def track_pair(ref, dep, cur, K, levels=4, iters=15, robust=True, depth_scale=1.0, glare=None):
    grefs, gdeps, gcurs, Ks = [ref], [dep], [cur], [K]
    glrs = None if glare is None else [glare]
    for _ in range(1, levels):
        grefs.append(cv2.pyrDown(grefs[-1]))
        gcurs.append(cv2.pyrDown(gcurs[-1]))
        gdeps.append(cv2.resize(gdeps[-1], (grefs[-1].shape[1], grefs[-1].shape[0]),
                                interpolation=cv2.INTER_NEAREST))
        if glrs is not None:
            glrs.append(cv2.resize(glrs[-1], (grefs[-1].shape[1], grefs[-1].shape[0]),
                                   interpolation=cv2.INTER_NEAREST))
        Kl = Ks[-1].copy(); Kl[:2, :] *= 0.5; Ks.append(Kl)
    T = np.eye(4)
    for l in reversed(range(levels)):
        g = None if glrs is None else glrs[l]
        T = _align_level(grefs[l], gdeps[l], gcurs[l], Ks[l], T, iters, robust, depth_scale, glare=g)
    return T


def load_K(seq):
    toks = open(f"data/hamlyn/intrinsics_{seq}.txt").read().split()
    fx, cx, fy, cy = float(toks[0]), float(toks[2]), float(toks[5]), float(toks[6])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def run(seq_dir, depth_dir, out_txt, K, levels=4, robust=False, depth_scale=1.0, stride=1):
    imgs = sorted(glob.glob(f"{seq_dir}/image01/*.jpg"))[::stride]
    deps = sorted(glob.glob(f"{depth_dir}/depth01/*.png"))[::stride]
    pose = np.eye(4)  # camera-to-world
    lines = []
    prev_g = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    prev_d = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
    lines.append(f"0.0 0 0 0 0 0 0 1")
    for i in range(1, len(imgs)):
        cur_g = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        T_rel = track_pair(prev_g, prev_d, cur_g, K, levels, robust=robust, depth_scale=depth_scale)
        pose = pose @ np.linalg.inv(T_rel)  # accumulate camera-to-world
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()  # xyzw
        t = pose[:3, 3]
        lines.append(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        prev_g = cur_g
        prev_d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        if i % 100 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    open(out_txt, "w").write("\n".join(lines) + "\n")
    print(f"trajectory -> {out_txt}  ({len(lines)} poses)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="e.g. rectified05")
    ap.add_argument("--depth_seq", default=None, help="depth source seq (default: same as --seq; use raw seq for depth even when images are glare-removed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--depth_scale", type=float, default=0.01)
    a = ap.parse_args()
    seq_img_dir = f"data/hamlyn/{a.seq}"
    depth_src = a.depth_seq or a.seq
    K = load_K(depth_src.replace("_masked", "").replace("_ai", "").replace("_temporal", ""))
    run(seq_img_dir, f"data/hamlyn/{depth_src}", a.out, K,
        levels=a.levels, robust=a.robust, depth_scale=a.depth_scale)
