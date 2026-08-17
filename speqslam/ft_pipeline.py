"""
Downstream pipeline for the FINE-TUNED depth (per held-out sequence):
  1. global affine correction of depth01_ft -> depth01_ftc  (one scale+shift per sequence, the
     standard monocular scale-recovery protocol -- same as used for the off-the-shelf depth)
  2. direct VO on the corrected fine-tuned depth -> results/c3vd/{seq}_ft.txt
  3. TSDF-fuse the monocular mesh (ft depth + ft-VO poses) -> results/mesh/{seq}_ftmono.ply
Run after train_depth.py has produced depth01_ft for the sequence.
"""
import sys, os, glob, numpy as np, cv2
sys.path.insert(0, "speqslam")
from direct_vo import run as vo_run

DS = 100.0 / 65535.0 / 1000.0


def affine_correct(seq, n=40):
    u = f"data/c3vd/{seq}_undist"
    pr = sorted(glob.glob(f"{u}/depth01_ft/*.png")); gt = sorted(glob.glob(f"{u}/depth01/*.png"))
    idx = np.linspace(0, len(gt) - 1, min(n, len(gt))).astype(int)
    X, Y = [], []
    for i in idx:
        P = cv2.imread(pr[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        G = cv2.imread(gt[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        m = (G > 0) & (P > 0)
        s = max(1, m.sum() // 4000)
        X.append(P[m][::s]); Y.append(G[m][::s])
    X, Y = np.concatenate(X), np.concatenate(Y)
    a, b = np.linalg.lstsq(np.stack([X, np.ones_like(X)], 1), Y, rcond=None)[0]
    out = f"{u}/depth01_ftc"; os.makedirs(out, exist_ok=True)
    for p in pr:
        P = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        D = np.where(P > 0, a * P + b, 0.0)
        D[(D < 0.003) | (D > 0.11)] = 0.0
        cv2.imwrite(f"{out}/{os.path.basename(p)}", np.clip(D / DS, 0, 65535).astype(np.uint16))
    print(f"{seq}: affine a={a:.3f} b={b*1000:.2f}mm -> {out}", flush=True)


def run_seq(seq):
    u = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{u}/K.txt")
    affine_correct(seq)
    print(f"[ft-VO] {seq}", flush=True)
    vo_run(u, u, f"results/c3vd/{seq}_ft.txt", K, robust=False, depth_scale=DS,
           depth_sub="depth01_ftc")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["cecum_t1_a"]):
        run_seq(s)
