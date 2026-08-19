"""
Robustness envelope: how does tracking degrade under controlled video corruptions?

Three realistic failure modes, three severities each, applied to the IMAGES (depth stays the
system's fine-tuned depth from clean frames -- this isolates the TRACKER's sensitivity):
  flicker : per-frame random brightness gain U(1-a, 1+a)   (auto-exposure hunting, light changes)
  blur    : horizontal box blur of k pixels                (fast camera motion)
  drop    : keep every Nth frame                           (dropped frames / low fps)
Metric: trajectory ATE vs ground truth, per condition -> results/robustness.csv + degradation plot.
Deterministic (fixed seed). One fixed sequence (trans_t2_a: mid-length, well-tracked baseline).
"""
import sys, os, glob, csv, subprocess, shutil
import numpy as np, cv2
sys.path.insert(0, "speqslam")
from direct_vo import run as vo_run
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
DS = 100.0 / 65535.0 / 1000.0
SEQ = "trans_t2_a"
SCRATCH = os.environ.get("ROBUST_SCRATCH", "data/_robust_tmp")

CONDITIONS = [("baseline", None, 0)] + \
    [("flicker", a, a) for a in (0.10, 0.20, 0.35)] + \
    [("blur", k, k) for k in (3, 7, 13)] + \
    [("drop", s, s) for s in (2, 3, 5)]


def ate_mm(est):
    o = subprocess.run([EVO, "tum", f"results/c3vd/{SEQ}_gt.txt", est, "-as", "--no_warnings"],
                       capture_output=True, text=True).stdout
    for l in o.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000
    return float("nan")


def corrupt_seq(kind, level, rng):
    """Write a corrupted copy of the sequence's images; return its dir."""
    src = f"data/c3vd/{SEQ}_undist"
    dst = f"{SCRATCH}/{kind}_{level}"
    os.makedirs(f"{dst}/image01", exist_ok=True)
    for p in sorted(glob.glob(f"{src}/image01/*.jpg")):
        img = cv2.imread(p)
        if kind == "flicker":
            g = rng.uniform(1 - level, 1 + level)
            img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
        elif kind == "blur":
            img = cv2.blur(img, (int(level), 1))
        cv2.imwrite(f"{dst}/image01/{os.path.basename(p)}", img)
    return dst


def main():
    undist = f"data/c3vd/{SEQ}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    rng = np.random.default_rng(0)
    rows = [("condition", "level", "ate_mm", "vs_baseline_pct")]
    results = []
    base_ate = None
    for kind, level, lv in CONDITIONS:
        out = f"results/c3vd/{SEQ}_rob_{kind}_{lv}.txt"
        if kind == "baseline":
            vo_run(undist, undist, out, K, robust=False, depth_scale=DS, depth_sub="depth01_ftc")
        elif kind == "drop":
            vo_run(undist, undist, out, K, robust=False, depth_scale=DS, depth_sub="depth01_ftc",
                   stride=int(level))
        else:
            cdir = corrupt_seq(kind, level, rng)
            vo_run(cdir, undist, out, K, robust=False, depth_scale=DS, depth_sub="depth01_ftc")
            shutil.rmtree(cdir, ignore_errors=True)
        a = ate_mm(out)
        if kind == "baseline":
            base_ate = a
        d = 100 * (a - base_ate) / base_ate if base_ate else 0.0
        rows.append((kind, level if level is not None else "-", f"{a:.3f}", f"{d:+.1f}"))
        results.append((kind, lv, a))
        print(f"{kind:<10}{str(level):>6}  ATE {a:.3f} mm  ({d:+.1f}% vs baseline)", flush=True)

    with open("results/robustness.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, kind, xlab in zip(axes, ["flicker", "blur", "drop"],
                              ["brightness flicker (±fraction)", "blur kernel (px)", "keep every Nth frame"]):
        xs = [0] + [lv for k, lv, _ in results if k == kind]
        ys = [base_ate] + [a for k, _, a in results if k == kind]
        ax.plot(xs, ys, "o-", color="#BC5346", lw=2)
        ax.axhline(base_ate, color="#2C7B74", ls="--", lw=1, label=f"baseline {base_ate:.2f}mm")
        ax.set_xlabel(xlab); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axes[0].set_ylabel("tracking error ATE (mm)")
    fig.suptitle(f"Argus robustness envelope — {SEQ} (tracker stress, system depth)")
    fig.tight_layout(); fig.savefig("results/robustness.png", dpi=130)
    print("saved results/robustness.csv + results/robustness.png", flush=True)


if __name__ == "__main__":
    main()
