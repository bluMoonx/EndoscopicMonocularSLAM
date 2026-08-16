"""
Direct-VO severity experiment. Runs the direct visual odometry (glare-sensitive, robust OFF) on
raw vs glare-removed (temporal) frames for each sequence, records the mean photometric tracking
residual (lower = cleaner brightness signal = better direct tracking) and saves the trajectory.
Depth always comes from the RAW sequence (glare removal doesn't change depth), so only the image
intensity differs -> the effect is purely the glare.
"""
import sys, glob, os, csv, cv2, numpy as np
sys.path.insert(0, "speqslam")
from direct_vo import track_pair, load_K, _bilinear
from scipy.spatial.transform import Rotation

DS = 0.01
STRIDE = 2
SEQS = [("rectified01", 0.49), ("rectified04", 3.51), ("rectified05", 5.55)]


def photo_resid(g0, d0, g1, T, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; h, w = g0.shape
    ys, xs = np.nonzero(d0 > 0); d = d0[ys, xs] * DS
    P = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], 1) @ T[:3, :3].T + T[:3, 3]
    Z = P[:, 2]; u = fx * P[:, 0] / Z + cx; v = fy * P[:, 1] / Z + cy
    m = (Z > 1e-6) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    if m.sum() < 50:
        return np.nan
    return float(np.sqrt(np.mean((_bilinear(g1, u[m], v[m]) - g0[ys, xs][m]) ** 2)))


def run_vo(img_dir, depth_dir, K, out_txt):
    imgs = sorted(glob.glob(f"{img_dir}/image01/*.jpg"))[::STRIDE]
    deps = sorted(glob.glob(f"{depth_dir}/depth01/*.png"))[::STRIDE]
    n = min(len(imgs), len(deps))
    pose = np.eye(4); lines = ["0.0 0 0 0 0 0 0 1"]; resids = []
    pg = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    pd = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
    for i in range(1, n):
        cg = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        T = track_pair(pg, pd, cg, K, robust=False, depth_scale=DS)
        resids.append(photo_resid(pg, pd, cg, T, K))
        pose = pose @ np.linalg.inv(T)
        q = Rotation.from_matrix(pose[:3, :3]).as_quat(); t = pose[:3, 3]
        lines.append(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        pg = cg; pd = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    open(out_txt, "w").write("\n".join(lines) + "\n")
    return float(np.nanmean(resids))


def main():
    rows = [("seq", "glare", "cond", "mean_residual")]
    print(f"{'seq':<12}{'glare':>7}{'raw_resid':>12}{'temporal':>12}{'reduction':>11}")
    for seq, g in SEQS:
        K = load_K(seq)
        r_raw = run_vo(f"data/hamlyn/{seq}", f"data/hamlyn/{seq}", K, f"results/vo/{seq}_raw.txt")
        r_tmp = run_vo(f"data/hamlyn/{seq}_temporal", f"data/hamlyn/{seq}", K, f"results/vo/{seq}_temporal.txt")
        rows += [(seq, g, "raw", f"{r_raw:.5f}"), (seq, g, "temporal", f"{r_tmp:.5f}")]
        print(f"{seq:<12}{g:>7.2f}{r_raw:>12.5f}{r_tmp:>12.5f}{r_raw - r_tmp:>+11.5f}", flush=True)
    with open("results/vo_experiment.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved results/vo_experiment.csv + trajectories in results/vo/", flush=True)


if __name__ == "__main__":
    main()
