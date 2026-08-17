"""
Dense colored 3D reconstruction of the colon surface from ground-truth poses + depth + color.
Accumulates every frame's back-projected points into a voxel grid, averaging real tissue COLOR
and counting COVERAGE (how many frames saw each patch). Saves points/colors/coverage for the
interactive viewer.
"""
import sys, glob, cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot

DS = 100.0 / 65535.0 / 1000.0


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


def reconstruct(seq, voxel=0.0012, per_frame=12000):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    poses = load_poses(f"results/c3vd/{seq}_gt.txt")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    allP, allC = [], []
    traj = []
    for i in range(len(imgs)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        T = poses[ts]; traj.append(T[:3, 3])
        col = cv2.imread(imgs[i]); d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        ys, xs = np.nonzero(d > 0)
        if len(xs) == 0:
            continue
        step = max(1, len(xs) // per_frame); ys, xs = ys[::step], xs[::step]
        z = d[ys, xs]
        Pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        allP.append(Pc @ T[:3, :3].T + T[:3, 3])
        allC.append(col[ys, xs][:, ::-1])            # BGR -> RGB
    P = np.vstack(allP); C = np.vstack(allC).astype(np.float32)
    vox = np.floor(P / voxel).astype(np.int64)
    uniq, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    sumP = np.zeros((len(uniq), 3)); np.add.at(sumP, inv, P)
    sumC = np.zeros((len(uniq), 3)); np.add.at(sumC, inv, C)
    pts = sumP / counts[:, None]
    colors = (sumC / counts[:, None]).clip(0, 255).astype(np.uint8)
    np.savez(f"results/{seq}_recon.npz", pts=pts, colors=colors, coverage=counts,
             traj=np.array(traj))
    print(f"{seq}: {len(pts)} surface points, {len(traj)} camera poses -> results/{seq}_recon.npz")


if __name__ == "__main__":
    reconstruct(sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a")
