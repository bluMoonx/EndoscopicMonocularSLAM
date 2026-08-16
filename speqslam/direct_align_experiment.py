"""
Direct/photometric severity experiment. Feature-based ORB-SLAM3 is immune to glare (RANSAC),
but DIRECT methods track by raw pixel brightness, which glare violates. This measures direct
frame-to-frame alignment quality (ECC correlation, higher=better) over each sequence, for the
raw frames vs each glare-removal method, and plots the severity curve: does removal help direct
alignment, scaling with glare?
"""
import cv2, numpy as np, glob, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQS = [("rectified06", 0.47), ("rectified01", 0.49), ("rectified04", 3.51), ("rectified05", 5.55)]
METHODS = {"telea": ("_masked", "Telea (smooth)", "#2C7B74"),
           "lama": ("_ai", "LaMa (AI)", "#BC5346"),
           "temporal": ("_temporal", "Temporal (real)", "#B67F26")}
STEP = 3  # sample every Nth consecutive pair for speed


def align_quality(seqdir):
    files = sorted(glob.glob(f"{seqdir}/image01/*.jpg"))
    ccs = []
    for i in range(0, len(files) - 1, STEP):
        a = cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(cv2.imread(files[i + 1]), cv2.COLOR_BGR2GRAY).astype(np.float32)
        w = np.eye(2, 3, dtype=np.float32)
        try:
            cc, _ = cv2.findTransformECC(
                a, b, w, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), None, 5)
        except cv2.error:
            cc = 0.0
        ccs.append(cc)
    return float(np.mean(ccs))


def main():
    rows = [("seq", "glare", "method", "raw_align", "removed_align", "improvement")]
    results = {m: ([], []) for m in METHODS}  # method -> (glare_xs, improvement_ys)
    print(f"{'seq':<12}{'glare%':>7}{'raw':>10}" + "".join(f"{m:>12}" for m in METHODS))
    for seq, g in SEQS:
        raw = align_quality(f"data/hamlyn/{seq}")
        line = f"{seq:<12}{g:>7.2f}{raw:>10.4f}"
        for m, (suf, _, _) in METHODS.items():
            d = f"data/hamlyn/{seq}{suf}"
            if not os.path.exists(d):
                line += f"{'-':>12}"; continue
            val = align_quality(d)
            rows.append((seq, g, m, f"{raw:.5f}", f"{val:.5f}", f"{val - raw:+.5f}"))
            results[m][0].append(g); results[m][1].append(val - raw)
            line += f"{val - raw:>+12.4f}"
        print(line)

    with open("results/direct_align.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # severity curve
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="#888", lw=1, ls="--")
    for m, (suf, label, color) in METHODS.items():
        xs, ys = results[m]
        if not xs:
            continue
        order = np.argsort(xs); xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ax.plot(xs, ys, "o-", ms=8, color=color, lw=2, label=label)
    ax.set_xlabel("Glare severity  (% of pixels specular)")
    ax.set_ylabel("Direct-alignment improvement  (ECC, removed - raw)")
    ax.set_title("Direct/photometric SLAM: glare removal helps, scaling with glare")
    ax.grid(alpha=0.25); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig("results/direct_severity_curve.png", dpi=130)
    print("\nsaved results/direct_severity_curve.png and results/direct_align.csv")


if __name__ == "__main__":
    main()
