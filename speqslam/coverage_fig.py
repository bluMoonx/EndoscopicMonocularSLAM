"""fig16: the coverage claim, validated. Under-seen fraction at screening pace
per withdrawal, from two self-consistent instruments: the deployed pipeline
(estimated poses) and a perfect-tracking twin (ground-truth poses). The GT-pose
instrument lands at ~40% on all three sequences; the pipeline's deviation from
it is the measured tracking-attributable coverage error."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/coverage_validation.json"))
seqs = [("c1_full_t4_v1", "model 1, texture 4"),
        ("c2_full_t1_v1", "model 2, texture 1"),
        ("c1_full_t1_v1", "model 1, texture 1")]
pipe = [d[k]["underseen_pipeline_pct"] for k, _ in seqs]
gtp = [d[k]["underseen_gtpose_pct"] for k, _ in seqs]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(3)
w = 0.32
b1 = ax.bar(x - w/2, pipe, w, color="#5A6062", label="pipeline (estimated poses)")
b2 = ax.bar(x + w/2, gtp, w, color="#1E6E66", label="perfect tracking (GT poses)")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2,
                    r.get_height()), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in seqs])
ax.set_ylabel("wall seen fewer than 3 times (%)\n700-frame screening-pace budget")
ax.set_ylim(0, 62)
ax.legend(fontsize=9)
ax.set_title("Under-inspection at real screening pace, validated:\n"
             "~40% under perfect tracking, on all three withdrawals")
fig.tight_layout()
fig.savefig("results/poster/fig16_coverage_validation.png", dpi=170)
print("wrote results/poster/fig16_coverage_validation.png")

d["summary"] = ("GT-pose instrument: 39.9-42.0% under-seen across all three "
                "withdrawals (two colon models) - the clinical thesis with "
                "tracking error removed. Pipeline deviation from it: -5.1 to "
                "+13.1 points = the measured tracking-attributable coverage "
                "error band.")
json.dump(d, open("results/coverage_validation.json", "w"), indent=1)
print("summary recorded")
