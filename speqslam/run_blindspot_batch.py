"""
Rigorous blind-spot validation across all C3VD sequences, ONE fixed setting (no per-sequence
tuning). For each sequence: measure tracking ATE, and how well the monocular blind-spot map
agrees with the ground-truth one. Then quantify the relationship between tracking accuracy and
blind-spot agreement -> "detection quality is predictable from tracking error."
"""
import sys, glob, subprocess, csv, numpy as np
sys.path.insert(0, "speqslam")
import validate_coverage as vc
from validate_coverage import load_poses, umeyama, build_surface, coverage_on_surface, align_fullpose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
vc.OCC_TOL = 0.008   # ONE fixed, physically-justified occlusion tolerance for every sequence
SEQS = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]


def ate_mm(seq):
    o = subprocess.run([EVO, "tum", f"results/c3vd/{seq}_gt.txt", f"results/c3vd/{seq}_raw.txt",
                        "-as", "--no_warnings"], capture_output=True, text=True).stdout
    for l in o.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000
    return float("nan")


def validate(seq):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    gt, vo = load_poses(f"results/c3vd/{seq}_gt.txt"), load_poses(f"results/c3vd/{seq}_raw.txt")
    common = sorted(set(gt) & set(vo))
    vo_al, s = align_fullpose(gt, vo, common)   # rotation-aware alignment (fixes straight-path roll)
    S = build_surface(deps, gt, K)
    cg, cv_ = coverage_on_surface(S, deps, gt, K), coverage_on_surface(S, deps, vo_al, K)
    gb, vb = cg < vc.THR, cv_ < vc.THR
    return 100 * (gb == vb).mean(), np.corrcoef(cg, cv_)[0, 1], 100 * gb.mean(), 100 * vb.mean()


def main():
    rows = [("seq", "ate_mm", "agreement_pct", "coverage_corr", "true_blind_pct", "mono_blind_pct")]
    print(f"{'seq':<12}{'ATE(mm)':>9}{'agree%':>8}{'corr':>7}{'true_bl':>9}{'mono_bl':>9}", flush=True)
    for seq in SEQS:
        a = ate_mm(seq)
        agr, cor, tb, mb = validate(seq)
        rows.append((seq, f"{a:.3f}", f"{agr:.1f}", f"{cor:.3f}", f"{tb:.1f}", f"{mb:.1f}"))
        print(f"{seq:<12}{a:>9.3f}{agr:>8.1f}{cor:>7.2f}{tb:>9.1f}{mb:>9.1f}", flush=True)
    with open("results/blindspot_batch.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    d = [r for r in rows[1:]]
    ate = np.array([float(r[1]) for r in d]); corr = np.array([float(r[3]) for r in d])
    rel = np.corrcoef(ate, corr)[0, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ate, corr, s=90, color="#BC5346", zorder=3)
    for r in d:
        ax.annotate(r[0].replace("_t", "-t"), (float(r[1]), float(r[3])),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    if len(ate) >= 2:
        m, b = np.polyfit(ate, corr, 1)
        xr = np.linspace(ate.min(), ate.max(), 30)
        ax.plot(xr, m * xr + b, color="#2C7B74", lw=2, label=f"trend r={rel:+.2f}")
        ax.legend()
    ax.set_xlabel("Tracking error  ATE (mm)"); ax.set_ylabel("Blind-spot map agreement with GT (coverage corr)")
    ax.set_title("Blind-spot detection quality is predictable from tracking accuracy")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig("results/blindspot_summary.png", dpi=130)
    print(f"\nrelationship (ATE vs blind-spot agreement): r={rel:.3f}", flush=True)
    print("saved results/blindspot_batch.csv + results/blindspot_summary.png", flush=True)


if __name__ == "__main__":
    main()
