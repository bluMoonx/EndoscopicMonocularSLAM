"""
Click-to-measure: monocular metric measurement of anatomy (e.g. polyp size).

Because the system's depth is METRIC, two clicked pixels can be back-projected to 3D and the real
distance between them computed -- turning a plain endoscope into a measuring device. Clinically,
lesion size drives surveillance intervals, and eyeballed size through a scope is unreliable.

Interactive:  python speqslam/measure.py --seq cecum_t1_a --frame 40
              click two points -> distance in mm overlays; C clears; ESC quits.
              Uses the SYSTEM's depth (fine-tuned, depth01_ftc) -- the honest monocular path.
Validation:   python speqslam/measure.py --validate
              Random point pairs on all 5 sequences: distance from SYSTEM depth vs distance from
              GROUND-TRUTH depth. Reports the measurement error (mm + %) -> results/measure_validation.csv
"""
import sys, os, glob, csv, argparse
import numpy as np, cv2

DS = 100.0 / 65535.0 / 1000.0
SEQS = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]


def load(seq, i, sub):
    u = f"data/c3vd/{seq}_undist"
    imgs = sorted(glob.glob(f"{u}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{u}/{sub}/*.png"))
    K = np.loadtxt(f"{u}/K.txt")
    return cv2.imread(imgs[i]), cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS, K, len(imgs)


def backproject(u, v, d, K):
    return np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])


def dist_mm(p1, p2, depth, K, win=2):
    """3D distance between two pixels, using a small median window for depth robustness."""
    out = []
    for (u, v) in (p1, p2):
        patch = depth[max(0, v - win):v + win + 1, max(0, u - win):u + win + 1]
        vals = patch[patch > 0]
        if len(vals) == 0:
            return None
        out.append(backproject(u, v, np.median(vals), K))
    return float(np.linalg.norm(out[0] - out[1]) * 1000.0)


def interactive(seq, frame):
    bgr, dep, K, n = load(seq, frame, "depth01_ftc")
    clicks, canvas = [], bgr.copy()

    def redraw():
        c = bgr.copy()
        for i in range(0, len(clicks) - 1, 2):
            p, q = clicks[i], clicks[i + 1]
            mm = dist_mm(p, q, dep, K)
            cv2.line(c, p, q, (60, 220, 60), 2)
            cv2.circle(c, p, 4, (60, 220, 60), -1); cv2.circle(c, q, 4, (60, 220, 60), -1)
            if mm:
                mid = ((p[0] + q[0]) // 2 + 8, (p[1] + q[1]) // 2 - 8)
                cv2.putText(c, f"{mm:.1f} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(c, f"{mm:.1f} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 255, 90), 2, cv2.LINE_AA)
        if len(clicks) % 2 == 1:
            cv2.circle(c, clicks[-1], 4, (60, 220, 60), -1)
        cv2.putText(c, f"ARGUS measure  |  {seq} frame {frame}/{n}  |  click 2 points = distance  "
                    f"|  C clear  ESC quit", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, f"ARGUS measure  |  {seq} frame {frame}/{n}  |  click 2 points = distance  "
                    f"|  C clear  ESC quit", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 235, 225), 1, cv2.LINE_AA)
        return c

    def on_mouse(ev, x, y, flags, param):
        nonlocal canvas
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            canvas = redraw()

    cv2.namedWindow("Argus - measure"); cv2.setMouseCallback("Argus - measure", on_mouse)
    canvas = redraw()
    while True:
        cv2.imshow("Argus - measure", canvas)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k in (ord('c'), ord('C')):
            clicks.clear(); canvas = redraw()
    cv2.destroyAllWindows()


def validate(pairs_per_seq=250, dmin_mm=4, dmax_mm=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = [("seq", "n_pairs", "median_err_mm", "mean_err_mm", "median_err_pct", "within_1mm_pct")]
    print(f"{'seq':<12}{'pairs':>6}{'med_err_mm':>11}{'mean_err_mm':>12}{'med_err_%':>10}{'<1mm %':>8}", flush=True)
    for seq in SEQS:
        u = f"data/c3vd/{seq}_undist"
        n = len(glob.glob(f"{u}/image01/*.jpg"))
        errs, errpct = [], []
        tries = 0
        while len(errs) < pairs_per_seq and tries < pairs_per_seq * 40:
            tries += 1
            i = int(rng.integers(0, n))
            _, dsys, K, _ = load(seq, i, "depth01_ftc")
            _, dgt, _, _ = load(seq, i, "depth01")
            h, w = dgt.shape
            (u1, v1), (u2, v2) = rng.integers([8, 8], [w - 8, h - 8], size=(2, 2))
            gt = dist_mm((u1, v1), (u2, v2), dgt, K)
            if gt is None or not (dmin_mm <= gt <= dmax_mm):
                continue
            est = dist_mm((u1, v1), (u2, v2), dsys, K)
            if est is None:
                continue
            errs.append(abs(est - gt)); errpct.append(100 * abs(est - gt) / gt)
        errs, errpct = np.array(errs), np.array(errpct)
        rows.append((seq, len(errs), f"{np.median(errs):.2f}", f"{errs.mean():.2f}",
                     f"{np.median(errpct):.1f}", f"{100*(errs < 1.0).mean():.1f}"))
        print(f"{seq:<12}{len(errs):>6}{np.median(errs):>11.2f}{errs.mean():>12.2f}"
              f"{np.median(errpct):>10.1f}{100*(errs < 1.0).mean():>8.1f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/measure_validation.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved results/measure_validation.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="cecum_t1_a")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        interactive(a.seq, a.frame)
