"""
Validate Argus blind-spot detection against ground truth (fixed-surface method).

Build ONE surface (from GT depth + poses). Then, for each surface patch, count how many frames
VIEWED it (project into the frame, check it's in-view and not occluded) -- once using ground-truth
camera poses, once using the monocular VO poses (Sim(3)-aligned to GT). This isolates the effect
of the monocular tracking on coverage: if the VO poses are accurate enough, the blind-spot map
matches ground truth. Reports agreement, blind-spot IoU, and coverage correlation.
"""
import sys, glob, cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = 100.0 / 65535.0 / 1000.0
SURF_VOXEL = 0.0015   # dedup surface to ~1.5mm points
THR = 3               # 'blind' = viewed in fewer than THR frames
OCC_TOL = 0.003       # occlusion tolerance (m), absolute floor
OCC_REL = 0.0         # occlusion tolerance as a FRACTION of depth (scales with depth uncertainty)


def load_poses(path):
    P = {}
    for l in open(path):
        v = l.split()
        if len(v) < 8:
            continue
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(np.array(v[4:8], float)).as_matrix()
        T[:3, 3] = np.array(v[1:4], float)
        P[round(float(v[0]), 6)] = T
    return P


def umeyama(src, dst):
    n = len(src); mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((Dc.T @ Sc) / n)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / ((Sc ** 2).sum() / n)
    A = np.eye(4); A[:3, :3] = s * R; A[:3, 3] = mu_d - s * (R @ mu_s)
    return A, s


def align_fullpose(gt, vo, common):
    """Rotation-aware (full-pose) Sim(3) alignment of VO onto GT. Position-only Umeyama leaves the
    roll about the motion axis unconstrained on near-straight trajectories, which corrupts the
    coverage projection. Here the global rotation Rg is estimated from the ORIENTATIONS (the mean
    of R_gt @ R_vo^T), scale from positions, translation from centroids. Returns the aligned VO
    poses (rotation un-scaled, translation scaled) ready for coverage projection."""
    Rg = Rot.from_matrix(np.array([gt[t][:3, :3] @ vo[t][:3, :3].T for t in common])).mean().as_matrix()
    Pv = np.array([vo[t][:3, 3] for t in common]); Pg = np.array([gt[t][:3, 3] for t in common])
    _, s = umeyama(Pv, Pg)
    tg = Pg.mean(0) - s * (Rg @ Pv.mean(0))
    out = {}
    for t in vo:
        M = np.eye(4)
        M[:3, :3] = Rg @ vo[t][:3, :3]
        M[:3, 3] = s * (Rg @ vo[t][:3, 3]) + tg
        out[t] = M
    return out, s


def build_surface(deps, poses, K, samples=15000):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vox = {}
    for i in range(len(deps)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        T = poses[ts]
        d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        ys, xs = np.nonzero(d > 0)
        if len(xs) == 0:
            continue
        step = max(1, len(xs) // samples); ys, xs = ys[::step], xs[::step]
        z = d[ys, xs]
        Pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        Pw = Pc @ T[:3, :3].T + T[:3, 3]
        for p in Pw:
            vox[tuple(np.floor(p / SURF_VOXEL).astype(np.int64))] = p
    return np.array(list(vox.values()))


def coverage_on_surface(S, deps, poses, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    counts = np.zeros(len(S))
    for i in range(len(deps)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        T = poses[ts]
        Pc = (S - T[:3, 3]) @ T[:3, :3]           # world -> camera
        z = Pc[:, 2]
        d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        h, w = d.shape
        u = fx * Pc[:, 0] / np.where(z > 1e-6, z, 1e-6) + cx
        v = fy * Pc[:, 1] / np.where(z > 1e-6, z, 1e-6) + cy
        inb = (z > 1e-4) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        idx = np.where(inb)[0]
        if len(idx) == 0:
            continue
        dz = d[v[idx].astype(int), u[idx].astype(int)]
        tol = OCC_TOL + OCC_REL * z[idx]                         # depth-scaled occlusion tolerance
        seen = idx[(dz > 0) & (np.abs(dz - z[idx]) < tol)]       # in view AND not occluded
        counts[seen] += 1
    return counts


def main(seq):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    gt, vo = load_poses(f"results/c3vd/{seq}_gt.txt"), load_poses(f"results/c3vd/{seq}_raw.txt")
    common = sorted(set(gt) & set(vo))
    vo_al, s = align_fullpose(gt, vo, common)   # rotation-aware alignment (fixes straight-path roll)

    S = build_surface(deps, gt, K)
    cg = coverage_on_surface(S, deps, gt, K)
    cv_ = coverage_on_surface(S, deps, vo_al, K)
    gt_blind, vo_blind = cg < THR, cv_ < THR
    agree = 100 * (gt_blind == vo_blind).mean()
    iou = (gt_blind & vo_blind).sum() / max((gt_blind | vo_blind).sum(), 1)
    corr = np.corrcoef(cg, cv_)[0, 1]
    print(f"{seq}: surface patches={len(S)}  align scale={s:.3f}")
    print(f"  monocular vs ground-truth: agreement={agree:.1f}%  blind-spot IoU={iou:.3f}  coverage corr r={corr:.3f}")
    print(f"  true blind={100*gt_blind.mean():.1f}%   monocular-detected blind={100*vo_blind.mean():.1f}%")

    fig = plt.figure(figsize=(9, 4.4))
    for j, (bl, ttl) in enumerate([(gt_blind, "ground truth"), (vo_blind, "monocular (Argus)")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.scatter(S[~bl, 0], S[~bl, 1], S[~bl, 2], c="#2C7B74", s=2, alpha=0.4)
        ax.scatter(S[bl, 0], S[bl, 1], S[bl, 2], c="red", s=8, alpha=0.9)
        ax.view_init(elev=20, azim=-60); ax.set_axis_off(); ax.set_title(f"blind spots: {ttl}", fontsize=10)
    fig.suptitle(f"Argus validation - {seq}: monocular vs ground-truth blind spots  "
                 f"({agree:.1f}% agree, r={corr:.2f})", fontsize=12)
    fig.tight_layout(); fig.savefig("results/blindspot_validation.png", dpi=130)
    print("saved results/blindspot_validation.png")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        OCC_TOL = float(sys.argv[2])
    main(sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a")
