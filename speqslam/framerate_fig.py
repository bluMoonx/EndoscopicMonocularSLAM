"""fig12: the frame-rate requirement curve for the full screening withdrawal.

Four points, one frozen pipeline, one divergence guard. Primary metric: guard
rate (fraction of frame pairs the tracker could not physically track) - cleanly
monotone in frame rate. Secondary: Sim(3) ATE / constant-point baseline, which
is only meaningful when the guard is quiet: at 1/24 sampling, 40% of the path is
guard coasting, so its ATE measures dead-reckoning, not tracking (plotted hollow
with that caveat). Also records the curve in results/c3vdv2_screening.json.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# stride, guard %, ATE mm, baseline mm, GT path mm
PTS = [
    (24, 39.5, 120.2, 168.2, 3756),
    (12, 16.0, 146.3, 168.3, 4299),
    (6,  6.7,  132.8, 168.3, 4790),
    (3,  3.0,  101.4, 168.3, 5452),
]

frac = [1.0 / s for s, *_ in PTS]
guard = [g for _, g, *_ in PTS]
ratio = [a / b for _, _, a, b, _ in PTS]

fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
ax1.set_xscale("log")
ax1.set_xticks(frac)
ax1.set_xticklabels(["1/24", "1/12", "1/6", "1/3"])
ax1.minorticks_off()
ax1.set_xlabel("fraction of native frame rate fed to the tracker")
ax1.set_ylabel("frame pairs beyond tracker limit (%)", color="#B03A2E")
ax1.plot(frac, guard, "o-", color="#B03A2E", lw=2, ms=7)
for x, y in zip(frac, guard):
    ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9, color="#B03A2E")
ax1.set_ylim(0, 46)
ax1.tick_params(axis="y", labelcolor="#B03A2E")

ax2 = ax1.twinx()
ax2.set_ylabel("trajectory error / do-nothing baseline", color="#1F618D")
# solid where tracking is real; hollow where coasting dominates the path
ax2.plot(frac[1:], ratio[1:], "s-", color="#1F618D", lw=2, ms=7)
ax2.plot(frac[:2], ratio[:2], "s--", color="#1F618D", lw=1, alpha=0.55)
ax2.plot(frac[0], ratio[0], "s", mfc="white", mec="#1F618D", ms=8)
ax2.axhline(1.0, color="#1F618D", lw=1, ls=":", alpha=0.7)
ax2.annotate("baseline: guessing one fixed point", (frac[0], 1.0),
             textcoords="offset points", xytext=(4, 4), fontsize=8.5,
             color="#1F618D")
ax2.annotate("40% of path is guard coasting —\nATE no longer measures tracking",
             (frac[0], ratio[0]), textcoords="offset points", xytext=(8, -30),
             fontsize=8, color="#1F618D")
for x, y in zip(frac[1:], ratio[1:]):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(0, -14), ha="center", fontsize=9, color="#1F618D")
ax2.set_ylim(0.4, 1.1)
ax2.tick_params(axis="y", labelcolor="#1F618D")

ax1.set_title("The frame-rate requirement: one full screening withdrawal,\n"
              "one frozen pipeline, sampled at four rates")
fig.tight_layout()
fig.savefig("results/poster/fig12_framerate.png", dpi=170)
print("wrote results/poster/fig12_framerate.png")

res = json.load(open("results/c3vdv2_screening.json"))
res["framerate_curve"] = [
    dict(stride=s, guard_pct=g, ate_mm=a, baseline_mm=b, path_mm=p,
         ate_over_baseline=round(a / b, 2))
    for s, g, a, b, p in PTS]
res["framerate_note"] = ("guard rate is the primary metric; ATE/baseline is "
                         "meaningful only when the guard is quiet (at stride 24, "
                         "40% of the path is coasting, so its ATE reflects "
                         "dead-reckoning, not tracking)")
json.dump(res, open("results/c3vdv2_screening.json", "w"), indent=1)
print("curve recorded in results/c3vdv2_screening.json")
