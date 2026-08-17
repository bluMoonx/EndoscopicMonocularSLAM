"""
Evaluate a depth folder against C3VD ground truth, two ways:
  metric : the prediction used as-is (tests absolute scale -> matters for fully-monocular use)
  affine : after a single global scale+shift to GT (tests structure -> the standard relative-depth
           protocol, and what the global-affine pipeline consumes)
Reports median AbsRel, delta<1.25, and pixel depth correlation.

Usage: python eval_depth.py <seq> <depth_subfolder>   e.g. cecum_t1_a depth01_ft
"""
import sys, glob, numpy as np, cv2
DS = 100.0 / 65535.0 / 1000.0


def metrics(P, G):
    m = (G > 0) & (P > 0) & np.isfinite(P)
    if m.sum() < 100:
        return None
    r = np.maximum(P[m] / G[m], G[m] / P[m])
    return np.median(np.abs(P[m] - G[m]) / G[m]), (r < 1.25).mean(), np.corrcoef(P[m], G[m])[0, 1]


def evaluate(seq, sub):
    u = f"data/c3vd/{seq}_undist"
    pr = sorted(glob.glob(f"{u}/{sub}/*.png")); gt = sorted(glob.glob(f"{u}/depth01/*.png"))
    idx = np.linspace(0, len(gt) - 1, min(25, len(gt))).astype(int)
    met, aff = [], []
    for i in idx:
        P = cv2.imread(pr[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        G = cv2.imread(gt[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        m = metrics(P, G)
        if m:
            met.append(m)
        # affine align P->G (scale+shift in depth) on valid pixels
        mm = (G > 0) & (P > 0)
        if mm.sum() > 100:
            A = np.stack([P[mm], np.ones(mm.sum())], 1)
            a, b = np.linalg.lstsq(A, G[mm], rcond=None)[0]
            m2 = metrics(np.where(P > 0, a * P + b, 0), G)
            if m2:
                aff.append(m2)
    met, aff = np.array(met), np.array(aff)
    print(f"{seq:<12} {sub:<15} metric: AbsRel={met[:,0].mean():.3f} d1={100*met[:,1].mean():4.1f}% "
          f"corr={met[:,2].mean():.3f}  | affine: AbsRel={aff[:,0].mean():.3f} "
          f"d1={100*aff[:,1].mean():4.1f}% corr={aff[:,2].mean():.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        for s in ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]:
            evaluate(s, "depth01_ft")
