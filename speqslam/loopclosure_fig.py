"""fig15 + results record for the loop-closure study (honest negative).

Final frozen configuration (appearance candidates -> two-way photometric
verification -> odometry-consistency gate -> position-only closures, robust
Sim(3) pose graph), developed on c1_t4 and tested untouched on c2_t1 and c1_t1.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEQS = [
    ("model 1, texture 4\n(development)", 101.4, 96.4, 168.3),
    ("model 2, texture 1\n(frozen test)", 92.1, 107.5, 133.4),
    ("model 1, texture 1\n(frozen test)", 114.5, 115.8, 157.6),
]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(SEQS))
w = 0.32
b1 = ax.bar(x - w/2, [s[1] for s in SEQS], w, color="#5A6062", label="odometry only")
b2 = ax.bar(x + w/2, [s[2] for s in SEQS], w, color="#1E6E66",
            label="+ loop closure (final config)")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width()/2,
                    r.get_height()), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5)
for i, s in enumerate(SEQS):
    ax.hlines(s[3], i - 0.45, i + 0.45, color="#9A4A3D", ls=":", lw=1.5)
ax.annotate("dotted: constant-point baseline", (0.02, 0.95),
            xycoords="axes fraction", fontsize=8.5, color="#9A4A3D")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in SEQS])
ax.set_ylabel("ATE (mm), Sim(3)-aligned")
ax.set_ylim(0, 185)
ax.legend(fontsize=9, loc="lower right")
ax.set_title("Appearance-based loop closure: a diagnosed negative\n"
             "(-5% on development, +17% / +1% on frozen tests)")
fig.tight_layout()
fig.savefig("results/poster/fig15_loopclosure.png", dpi=170)
print("wrote results/poster/fig15_loopclosure.png")

res = {}
res["config"] = ("keyframe appearance candidates (40x40 normalized gray, cos>=0.82, "
                 "sep>=60 frames) -> two-way photometric verification (fwd-bwd "
                 "consistency <4deg/6mm) -> odometry-consistency gate (implied "
                 "drift <= max(0.18u, 10% of chained path)) -> position-only "
                 "closure edges (sigma 0.10u; rotation/scale disabled) -> "
                 "soft-L1 robust Sim(3) pose graph; developed on c1_t4, frozen "
                 "for the two test sequences")
res["sequences"] = {
    "c1_full_t4_v1 (development)": dict(closures_verified=16, closures_kept=9,
        ate_before=101.4, ate_after=96.4, change_pct=-5.0),
    "c2_full_t1_v1 (frozen test)": dict(closures_verified=44, closures_kept=41,
        ate_before=92.1, ate_after=107.5, change_pct=16.7),
    "c1_full_t1_v1 (frozen test)": dict(closures_verified=46, closures_kept=46,
        ate_before=114.5, ate_after=115.8, change_pct=1.1),
}
res["diagnosis"] = ("recognition works (GT audit: 39/41 gate-surviving closures on "
                    "c2 are true revisits within 60 mm) but low-parallax revisit "
                    "alignments systematically UNDERESTIMATE separation (claiming "
                    "2-5 mm where GT says 20-60 mm); sparse closures help (c1_t4), "
                    "dense biased closures contract the dwell region and bend the "
                    "downstream trajectory (c2). Development iterations that were "
                    "tried and their per-variant results are retained in git "
                    "history; the final config was frozen before the c1_t1 test.")
res["verdict"] = ("NEGATIVE: photometric position-only loop closure does not "
                  "reliably improve whole-colon ATE on withdrawal-style motion; "
                  "improving it needs parallax-aware closure covariance or "
                  "feature-based wide-baseline relative poses")
json.dump(res, open("results/loopclosure_study.json", "w"), indent=1)
print("wrote results/loopclosure_study.json")
