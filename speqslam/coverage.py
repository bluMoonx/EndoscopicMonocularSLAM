"""
Argus blind-spot mapping. Reconstructs the colon surface from a camera trajectory + per-frame
depth, and computes a COVERAGE map: how many frames observed each little patch of surface. Patches
seen few/no times are 'blind spots' - regions a clinician may never have gotten a clean look at
(where missed polyps hide). Works with either ground-truth or VO-estimated poses.
"""
import sys, glob, cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = 100.0 / 65535.0 / 1000.0  # C3VD uint16 depth -> meters


def load_traj(path):
    poses = {}
    for l in open(path):
        v = l.split()
        if len(v) < 8:
            continue
        poses[round(float(v[0]), 6)] = (np.array(v[1:4], float), np.array(v[4:8], float))
    return poses


def compute_coverage(undist, traj_path, K, voxel=0.0015, ds=DS, samples=15000):
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    poses = load_traj(traj_path)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    counts = {}
    for i in range(len(imgs)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        trans, q = poses[ts]
        Rw = Rot.from_quat(q).as_matrix()
        d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * ds
        ys, xs = np.nonzero(d > 0)
        if len(xs) == 0:
            continue
        step = max(1, len(xs) // samples)
        ys, xs = ys[::step], xs[::step]
        z = d[ys, xs]
        Pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        Pw = Pc @ Rw.T + trans                      # surface points in world coords
        vox = np.floor(Pw / voxel).astype(np.int64)
        for v in set(map(tuple, vox)):              # each voxel counts once per frame
            counts[v] = counts.get(v, 0) + 1
    keys = np.array(list(counts.keys()))
    cnt = np.array([counts[tuple(k)] for k in keys])
    centers = (keys + 0.5) * voxel
    return centers, cnt


def main(seq):
    undist = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{undist}/K.txt")
    centers, cnt = compute_coverage(undist, f"results/c3vd/{seq}_gt.txt", K)
    thr = 3                                          # 'blind' = observed in fewer than 3 frames
    blind = cnt < thr
    print(f"{seq}: surface patches={len(cnt)}  median coverage={int(np.median(cnt))} views  "
          f"blind (<{thr} views)={100*blind.mean():.1f}%")

    # 3D coverage heatmap (blind spots in red)
    fig = plt.figure(figsize=(9, 4.2))
    for j, (az, el) in enumerate([(-60, 20), (120, 20)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        good = ~blind
        ax.scatter(centers[good, 0], centers[good, 1], centers[good, 2], c=np.clip(cnt[good], 0, 40),
                   cmap="viridis", s=2, alpha=0.5)
        ax.scatter(centers[blind, 0], centers[blind, 1], centers[blind, 2], c="red", s=6, alpha=0.9)
        ax.view_init(elev=el, azim=az); ax.set_axis_off()
        ax.set_title("coverage (red = blind spot)" if j == 0 else "", fontsize=10)
    fig.suptitle(f"Argus blind-spot map - {seq}  ({100*blind.mean():.1f}% of surface under-observed)", fontsize=12)
    fig.tight_layout(); fig.savefig("results/blindspot_map.png", dpi=130)
    print("saved results/blindspot_map.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a")
