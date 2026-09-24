"""fig14: the frame-rate requirement, replicated on a second colon model.

Guard-rejection rate (frame pairs beyond the tracker's physical limit) at 1/12
vs 1/3 of native frame rate, for two different full withdrawals: c1_full_t4
(colon model 1, the original study) and c2_full_t1 (colon model 2, different
anatomy and texture, 7 m GT path). Records the replication block in
results/c3vdv2_screening.json.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# seq label, stride-12 guard %, stride-3 guard %, s12 (ate, base), s3 (ate, base)
SEQS = [
    ("model 1, texture 4\n(c1_full_t4, 5.5 m)", 16.0, 3.0, (146.3, 168.3), (101.4, 168.3)),
    ("model 2, texture 1\n(c2_full_t1, 7.0 m)", 16.5, 1.5, (103.0, 133.4), (92.1, 133.4)),
    ("model 1, texture 1\n(c1_full_t1, 7.8 m)", 23.0, 3.1, (94.4, 157.6), (114.5, 157.6)),
]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(SEQS))
w = 0.32
b1 = ax.bar(x - w/2, [s[1] for s in SEQS], w, color="#B03A2E",
            label="1/12 of native frame rate (~2 fps): fails")
b2 = ax.bar(x + w/2, [s[2] for s in SEQS], w, color="#2C7B74",
            label="1/3 of native frame rate (~9 fps): holds")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2,
                    r.get_height()), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in SEQS])
ax.set_ylabel("frame pairs beyond tracker limit (%)")
ax.set_ylim(0, 27)
ax.legend(fontsize=9)
ax.set_title("The speed limit replicates: three complete withdrawals,\n"
             "two colon models, one frozen pipeline")
fig.tight_layout()
fig.savefig("results/poster/fig14_replication.png", dpi=170)
print("wrote results/poster/fig14_replication.png")

res = json.load(open("results/c3vdv2_screening.json"))
res["replication_c2_full_t1_v1"] = dict(
    frames_s3=3667, gt_path_mm=7003,
    stride3=dict(ate_mm=92.1, ate_pct=1.32, baseline_mm=133.4,
                 ate_over_baseline=0.69, guarded="55 of 3666 (1.5%)"),
    stride12=dict(ate_mm=103.0, ate_pct=1.79, baseline_mm=133.4,
                  ate_over_baseline=0.77, guarded="151 of 916 (16.5%)"),
    verdict="frame-rate requirement replicates on the second colon model: "
            "guard rate 16.5% at 1/12 vs 1.5% at 1/3 (c1: 16.0% vs 3.0%)")
res["replication_c1_full_t1_v1"] = dict(
    frames_s3=3667, gt_path_mm=7814,
    stride3=dict(ate_mm=114.5, ate_pct=1.47, baseline_mm=157.6,
                 ate_over_baseline=0.73, guarded="112 of 3666 (3.1%)"),
    stride12=dict(ate_mm=94.4, ate_pct=1.46, baseline_mm=157.6,
                  ate_over_baseline=0.60, guarded="211 of 916 (23.0%)",
                  note="23% of the path is guard coasting - per the Trap-5 "
                       "rule its ATE reflects dead-reckoning, not tracking; "
                       "guard rate is the verdict"),
    verdict="third withdrawal (model 1, different texture): fails at 1/12 "
            "(23.0% guarded), holds at 1/3 (3.1%)")
res["replication_summary"] = ("n=3 complete withdrawals, 2 colon models, "
                              "20.3 m combined GT path: guard rate 16-23% at "
                              "1/12 native rate vs 1.5-3.1% at 1/3; all "
                              "stride-3 ATEs 0.60-0.73x their baselines")
json.dump(res, open("results/c3vdv2_screening.json", "w"), indent=1)
print("replication recorded in results/c3vdv2_screening.json")
