"""
PROPER end-to-end monocular coverage: compute blind spots on the FUSED surface itself, the way the
system would actually run. For each surface vertex, count frames whose camera saw it (in frustum +
front-facing) -- using the fused geometry + normals, NOT a per-pixel comparison to a raw depth map.
This is robust to per-frame depth noise (already averaged out by TSDF fusion).

  mono system : mono mesh (learned depth+VO) + learned-VO poses  -> blind_mono
  ground truth: GT mesh (GT depth+pose)      + GT poses          -> blind_gt
Align mono->GT, then compare blind labels by nearest neighbour. Reports agreement + blind fractions.
"""
import sys, glob, csv, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "speqslam")
from validate_coverage import load_poses, umeyama

THR = 3
MODES = {"mono": "learned", "ftmono": "ft"}   # mesh mode -> trajectory suffix


def coverage_on_mesh(V, Nrm, poses, K, hw):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; h, w = hw
    cnt_a = np.zeros(len(V)); cnt_b = np.zeros(len(V))
    for T in poses.values():
        c, R = T[:3, 3], T[:3, :3]
        Pc = (V - c) @ R                       # world -> camera
        z = Pc[:, 2]
        u = fx * Pc[:, 0] / np.where(z > 1e-6, z, 1e-6) + cx
        v = fy * Pc[:, 1] / np.where(z > 1e-6, z, 1e-6) + cy
        inb = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dot = (Nrm * (c - V)).sum(1)           # normal vs view direction
        cnt_a += inb & (dot > 0)
        cnt_b += inb & (dot < 0)               # opposite normal orientation
    return cnt_a, cnt_b


def load_mesh(seq, mode):
    m = o3d.io.read_triangle_mesh(f"results/mesh/{seq}_{mode}.ply"); m.compute_vertex_normals()
    return np.asarray(m.vertices), np.asarray(m.vertex_normals)


def evaluate(seq, mode="mono"):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    h, w = __import__("cv2").imread(sorted(glob.glob(f"{undist}/depth01/*.png"))[0], -1).shape[:2]
    gt = load_poses(f"results/c3vd/{seq}_gt.txt")
    ln = load_poses(f"results/c3vd/{seq}_{MODES[mode]}.txt")
    Vg, Ng = load_mesh(seq, "gt"); Vm, Nm = load_mesh(seq, mode)

    def pick(V, N, poses):                     # choose normal orientation that yields more coverage
        a, b = coverage_on_mesh(V, N, poses, K, (h, w))
        return a if a.mean() >= b.mean() else b
    cg = pick(Vg, Ng, gt); cm = pick(Vm, Nm, ln)
    bg, bm = cg < THR, cm < THR

    # align mono verts -> GT frame (rotation-aware Sim(3) from camera trajectories)
    common = sorted(set(gt) & set(ln))
    Pv = np.array([ln[t][:3, 3] for t in common]); Pg = np.array([gt[t][:3, 3] for t in common])
    Rg = Rot.from_matrix(np.array([gt[t][:3, :3] @ ln[t][:3, :3].T for t in common])).mean().as_matrix()
    _, s = umeyama(Pv, Pg)
    Vm_al = (s * (Rg @ Vm.T).T) + (Pg.mean(0) - s * (Rg @ Pv.mean(0)))
    # compare: for each GT vertex, nearest mono vertex's blind label
    nn = cKDTree(Vm_al).query(Vg)[1]
    bm_on_g = bm[nn]
    cm_on_g = cm[nn]
    agree = 100 * (bg == bm_on_g).mean()
    iou = (bg & bm_on_g).sum() / max((bg | bm_on_g).sum(), 1)
    inter = (bg & bm_on_g).sum()
    recall = 100 * inter / max(bg.sum(), 1)          # of true blind, fraction flagged
    prec = 100 * inter / max(bm_on_g.sum(), 1)       # of flagged, fraction truly blind
    ccorr = np.corrcoef(cg, cm_on_g)[0, 1]           # continuous coverage-count correlation
    # REGION-level detection (clinical framing: "re-inspect this area"), tolerance = REG_TOL.
    # a true-blind vertex is CAUGHT if any flagged vertex lies within REG_TOL of it; a flagged
    # vertex is a TRUE ALARM if any true-blind vertex lies within REG_TOL of it.
    REG_TOL = 0.005                                   # 5 mm localization tolerance, fixed
    if bg.sum() and bm_on_g.sum():
        d_t2f = cKDTree(Vg[bm_on_g]).query(Vg[bg])[0]
        d_f2t = cKDTree(Vg[bg]).query(Vg[bm_on_g])[0]
        rrec = 100 * (d_t2f < REG_TOL).mean()
        rprec = 100 * (d_f2t < REG_TOL).mean()
    else:
        rrec = rprec = 0.0
    print(f"{seq:<12} agree={agree:5.1f}%  cov_corr={ccorr:5.2f}  IoU={iou:4.2f}  "
          f"vert R/P={recall:3.0f}/{prec:3.0f}%  region(5mm) R/P={rrec:3.0f}/{rprec:3.0f}%  "
          f"(true_bl {100*bg.mean():.1f}% / mono_bl {100*bm_on_g.mean():.1f}%)", flush=True)
    return dict(seq=seq, agree=agree, cov_corr=ccorr, iou=iou, recall=recall, prec=prec,
                reg_recall=rrec, reg_prec=rprec,
                true_blind=100 * bg.mean(), mono_blind=100 * bm_on_g.mean())


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in MODES else "mono"
    seqs = args or ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
    rows = [evaluate(s, mode) for s in seqs]
    print(f"\nMEAN agree={np.mean([r['agree'] for r in rows]):.1f}%  "
          f"cov_corr={np.mean([r['cov_corr'] for r in rows]):.2f}  "
          f"vert R/P={np.mean([r['recall'] for r in rows]):.0f}/{np.mean([r['prec'] for r in rows]):.0f}%  "
          f"region R/P={np.mean([r['reg_recall'] for r in rows]):.0f}/"
          f"{np.mean([r['reg_prec'] for r in rows]):.0f}%", flush=True)
    with open(f"results/end2end_coverage_{mode}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
