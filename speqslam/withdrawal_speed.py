"""
The coverage cost of rushing: how do blind spots grow with withdrawal speed?

Colonoscopy's one famous quality rule is SLOW withdrawal (the "6-minute rule") -- but the cost of
rushing has never been visible. Here we simulate an N-times-faster withdrawal by keeping every Nth
camera pose of the real trajectory, and measure the blind-spot fraction on the ground-truth surface
with the validated coverage machinery. One deterministic curve per sequence: blind% vs speed.
Uses GT poses/depth to isolate the question (what does SPEED cost?) from tracking error.
"""
import sys, glob, csv, numpy as np
sys.path.insert(0, "speqslam")
import validate_coverage as vc
from validate_coverage import load_poses, build_surface, coverage_on_surface
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vc.OCC_TOL = 0.008; vc.OCC_REL = 0.0     # GT depth -> no depth-uncertainty widening needed
SEQS = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
SPEEDS = [1, 2, 3, 4, 6]


def main():
    rows = [("seq", "speed_x", "n_poses", "blind_pct")]
    curves = {}
    for seq in SEQS:
        u = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{u}/K.txt")
        deps = sorted(glob.glob(f"{u}/depth01/*.png"))
        gt = load_poses(f"results/c3vd/{seq}_gt.txt")
        S = build_surface(deps, gt, K)
        ts_sorted = sorted(gt.keys())
        curves[seq] = []
        for sp in SPEEDS:
            sub = {t: gt[t] for t in ts_sorted[::sp]}
            # coverage_on_surface indexes deps by frame -> pass poses dict with only kept frames
            c = coverage_on_surface(S, deps, sub, K)
            blind = 100 * (c < vc.THR).mean()
            rows.append((seq, sp, len(sub), f"{blind:.2f}"))
            curves[seq].append(blind)
            print(f"{seq:<12} speed x{sp}  poses {len(sub):>4}  blind {blind:5.1f}%", flush=True)
    with open("results/withdrawal_speed.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for seq in SEQS:
        ax.plot(SPEEDS, curves[seq], "o-", lw=2, label=seq.replace("_t", "-t"))
    ax.set_xlabel("withdrawal speed (x real speed)")
    ax.set_ylabel("blind spots (% of colon surface)")
    ax.set_title("The coverage cost of rushing: blind spots vs withdrawal speed")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("results/withdrawal_speed.png", dpi=130)
    print("saved results/withdrawal_speed.csv + results/withdrawal_speed.png", flush=True)


if __name__ == "__main__":
    main()
