"""
Find + size: polyp detection fused with metric depth = automatic lesion sizing.

Clinically, polyp SIZE drives care (the 5/10mm category boundaries decide surveillance intervals),
and eyeballed size through a scope is unreliable. Here the segmentation mask's boundary is
back-projected to 3D with the depth map and the polyp's max caliper diameter is computed in mm.

Two honesty tiers, kept separate on purpose:
  --validate : the SIZING GEOMETRY is validated on C3VD (real calibration + GT depth): random
               Kvasir mask SHAPES are placed on C3VD frames and their max diameter is computed
               from the system's fine-tuned depth vs from ground-truth depth -> error in mm.
  --image    : the full chain demonstrated on a real Kvasir polyp image. Kvasir has NO camera
               calibration, so a nominal K + relative depth are used and the size is labeled
               ILLUSTRATIVE -- the geometry is validated, the absolute scale here is not.
"""
import sys, os, glob, csv, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))

DS = 100.0 / 65535.0 / 1000.0
SEQS = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]


def _boundary_pts(mask, thr=0.5, n=200):
    m = (mask > thr).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    step = max(1, len(c) // n)
    return c[::step]


def _backproject(pts, depth, K, win=2):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = []
    h, w = depth.shape
    for (u, v) in pts:
        patch = depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        vals = patch[patch > 0]
        if len(vals) == 0:
            continue
        d = np.median(vals)
        out.append([(u - cx) / fx * d, (v - cy) / fy * d, d])
    return np.array(out)


def diameter_mm(mask, depth, K, thr=0.5):
    """Max caliper diameter (mm) of the masked region, from its 3D boundary."""
    b = _boundary_pts(mask, thr)
    if b is None or len(b) < 3:
        return None
    P = _backproject(b, depth, K)
    if len(P) < 3:
        return None
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()) * 1000.0)


def validate(n_trials=150, seed=0):
    """Place real polyp mask SHAPES onto C3VD frames; diameter from system depth vs GT depth."""
    rng = np.random.default_rng(seed)
    kv_masks = sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/masks/*.jpg"))
    errs, gts = [], []
    rows = [("seq", "frame", "gt_diam_mm", "sys_diam_mm", "err_mm")]
    while len(errs) < n_trials:
        seq = SEQS[int(rng.integers(0, len(SEQS)))]
        u = f"data/c3vd/{seq}_undist"
        imgs = sorted(glob.glob(f"{u}/image01/*.jpg"))
        i = int(rng.integers(0, len(imgs)))
        dsys = cv2.imread(sorted(glob.glob(f"{u}/depth01_ftc/*.png"))[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        dgt = cv2.imread(sorted(glob.glob(f"{u}/depth01/*.png"))[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        K = np.loadtxt(f"{u}/K.txt")
        h, w = dgt.shape
        km = cv2.imread(kv_masks[int(rng.integers(0, len(kv_masks)))], cv2.IMREAD_GRAYSCALE)
        # scale the polyp shape to a plausible on-screen size and place it randomly
        s = int(rng.integers(60, 160))
        km = cv2.resize(km, (s, s), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros((h, w), np.float32)
        y0 = int(rng.integers(0, h - s)); x0 = int(rng.integers(0, w - s))
        mask[y0:y0 + s, x0:x0 + s] = km / 255.0
        gt = diameter_mm(mask, dgt, K)
        if gt is None or not (3 <= gt <= 30):        # clinical polyp size range
            continue
        est = diameter_mm(mask, dsys, K)
        if est is None:
            continue
        errs.append(abs(est - gt)); gts.append(gt)
        rows.append((seq, i, f"{gt:.2f}", f"{est:.2f}", f"{abs(est-gt):.2f}"))
    errs = np.array(errs); gts = np.array(gts)
    pct = 100 * errs / gts
    print(f"region-diameter sizing vs GT depth ({len(errs)} trials, 3-30mm regions):", flush=True)
    print(f"  median err {np.median(errs):.2f} mm ({np.median(pct):.1f}%)   "
          f"mean {errs.mean():.2f} mm   within 2mm: {100*(errs < 2).mean():.0f}%", flush=True)
    with open("results/polyp_size_validation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows + [("MEDIAN", "", "", "", f"{np.median(errs):.2f}")])
    print("saved results/polyp_size_validation.csv", flush=True)


def demo_image(path, out=None):
    """Full chain on a real polyp image: detect -> size (ILLUSTRATIVE scale; Kvasir has no calib)."""
    from polyp_detect import load_model, segment, overlay
    import demo_app
    bgr = cv2.imread(path); h, w = bgr.shape[:2]
    model = load_model()
    mask = segment(bgr, model)
    net = demo_app.DepthNet()
    depth = net.predict(bgr, relative=True)
    # nominal scale for illustration: typical close-inspection distances (1.2-4.5cm). Kvasir ships
    # no calibration, so mm here are ILLUSTRATIVE; the sizing geometry itself is validated on C3VD.
    m = depth > 0
    depth[m] = 0.012 + (depth[m] - depth[m].min()) / max(depth[m].ptp(), 1e-6) * (0.045 - 0.012)
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)   # nominal calibration
    dia = diameter_mm(mask, depth, K)
    vis = overlay(bgr, mask)
    if dia:
        cv2.putText(vis, f"est. size ~{dia:.0f} mm (illustrative scale)", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, f"est. size ~{dia:.0f} mm (illustrative scale)", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 255, 90), 2, cv2.LINE_AA)
    out = out or "results/polyp_find_size_demo.png"
    cv2.imwrite(out, vis)
    print(f"detected + sized -> {out}  (diameter ~{dia:.1f} mm, illustrative)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        demo_image(a.image or sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/images/*.jpg"))[13], a.out)
