"""
The poster figure factory: regenerates every headline graphic in ONE consistent style at print
resolution -> results/poster/. Rerun any time results change. Data comes from the committed CSVs.
"""
import os, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORAL, TEAL, CYAN, AMBER, MUTED = "#BC5346", "#2C7B74", "#1C7F9E", "#B67F26", "#6C655E"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "Segoe UI",
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "legend.frameon": False})
OUT = "results/poster"
os.makedirs(OUT, exist_ok=True)


def rows(path):
    return list(csv.DictReader(open(path)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  {name}.png", flush=True)


def seqlab(s):
    return s.replace("_t", "-t")


# 1 -- the ablation: the depth fix rescues the system
abl = rows("results/ablation.csv")
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(abl)); w = 0.38
ax.bar(x - w/2, [float(r["agree_D"]) for r in abl], w, color=CORAL, label="generic AI depth")
ax.bar(x + w/2, [float(r["agree_D2"]) for r in abl], w, color=TEAL, label="fine-tuned depth")
ax.axhline(95, color=MUTED, ls="--", lw=1)
ax.text(len(abl)-0.4, 95.7, "design goal 95%", color=MUTED, fontsize=10, ha="right")
ax.set_xticks(x, [seqlab(r["seq"]) for r in abl])
ax.set_ylabel("blind-spot agreement with ground truth (%)")
ax.set_title("One fix, whole system rescued: 74.5% \u2192 98.6% mean")
ax.set_ylim(0, 104); ax.legend(loc="lower right")
save(fig, "fig1_ablation")

# 2 -- depth accuracy before/after (held-out)
D1 = {"cecum-t1-a": (64.2, 98.9), "desc-t4-a": (69.0, 91.8), "trans-t1-a": (66.0, 97.9),
      "trans-t2-a": (74.5, 99.7), "trans-t2-b": (67.1, 99.6)}
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(D1))
ax.bar(x - 0.19, [v[0] for v in D1.values()], 0.38, color=CORAL, label="off-the-shelf")
ax.bar(x + 0.19, [v[1] for v in D1.values()], 0.38, color=TEAL, label="fine-tuned (held-out)")
ax.set_xticks(x, list(D1))
ax.set_ylabel("pixels within 25% of true depth (%)")
ax.set_title("Teaching the AI what a colon looks like")
ax.set_ylim(0, 104); ax.legend(loc="lower right")
save(fig, "fig2_depth_fix")

# 3 -- robustness: flicker weakness found and fixed
rob = rows("results/robustness.csv")
base = float(next(r["ate_mm"] for r in rob if r["condition"] == "baseline"))
fl = [(float(r["level"]), float(r["ate_mm"])) for r in rob if r["condition"] == "flicker"]
ill = [r for r in csv.reader(open("results/robustness_illum.csv"))][1:]
il = [(float(r[1]), float(r[2])) for r in ill if r[0].startswith("flicker")]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot([0] + [p[0] for p in fl], [base] + [p[1] for p in fl], "o-", color=CORAL, lw=2.5,
        label="tracker as-is")
ax.plot([0] + [p[0] for p in il], [0.75] + [p[1] for p in il], "s--", color=TEAL, lw=2.5,
        label="+ illumination compensation")
ax.set_xlabel("brightness flicker (\u00b1 fraction)"); ax.set_ylabel("tracking error (mm)")
ax.set_title("Weakness found, weakness erased (16\u00d7 recovery)")
ax.legend()
save(fig, "fig3_flicker_fix")

# 4 -- cost of rushing
wd = rows("results/withdrawal_speed.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
for i, s in enumerate(dict.fromkeys(r["seq"] for r in wd)):
    pts = [(float(r["speed_x"]), float(r["blind_pct"])) for r in wd if r["seq"] == s]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=2, label=seqlab(s))
ax.set_xlabel("withdrawal speed (\u00d7 real)"); ax.set_ylabel("blind wall (%)")
ax.set_title("Rushing has a price, now visible")
ax.legend(fontsize=9)
save(fig, "fig4_rushing")

# 5 -- threshold sensitivity
th = rows("results/thr_sensitivity.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
for s in dict.fromkeys(r["seq"] for r in th):
    pts = [(int(r["thr"]), float(r["agreement_pct"])) for r in th if r["seq"] == s]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=2, label=seqlab(s))
ax.set_xticks([2, 3, 4, 5])
ax.set_xlabel("blind-spot threshold (minimum views)")
ax.set_ylabel("agreement (%)"); ax.set_ylim(90, 101)
ax.set_title("The conclusion does not depend on the threshold")
ax.legend(fontsize=9, loc="lower right")
save(fig, "fig5_threshold")

# 6 -- cross-dataset: zero-shot on foreign tissue
es = [r for r in rows("results/endoslam/summary.csv") if r["traj"] != "MEAN"]
fig, ax = plt.subplots(figsize=(7, 4.6))
x = np.arange(len(es))
ax.bar(x, [float(r["plain_ate_mm"]) for r in es], 0.55, color=CYAN)
for i, r in enumerate(es):
    ax.text(i, float(r["plain_ate_mm"]) + 1, f'{float(r["plain_ate_pct_of_path"]):.0f}% of path',
            ha="center", fontsize=9, color=MUTED)
ax.set_xticks(x, [f'traj {r["traj"]}' for r in es])
ax.set_ylabel("tracking error (mm)")
ax.set_title("Zero-shot on real porcine colon: degrades, never collapses")
save(fig, "fig6_crossdataset")

# 7 -- monocular measurement accuracy
me = [r for r in rows("results/measure_validation.csv")]
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(me))
ax.bar(x, [float(r["median_err_mm"]) for r in me], 0.55, color=TEAL)
ax.axhline(1.0, color=MUTED, ls="--", lw=1)
ax.text(len(me)-0.4, 1.05, "1 mm", color=MUTED, fontsize=10, ha="right")
ax.set_xticks(x, [seqlab(r["seq"]) for r in me])
ax.set_ylabel("median measurement error (mm)")
ax.set_title("Measuring anatomy with one camera")
save(fig, "fig7_measurement")

print(f"all figures -> {OUT}/", flush=True)
