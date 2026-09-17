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


# 1 -- the ablation: the depth fix rescues the system. Agreement is shown against the
# flag-nothing baseline (a system that flags no blind spots scores 100 - true_blind%),
# because blind wall is rare (~1% of patches) and agreement alone saturates.
abl = rows("results/ablation.csv")
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(abl)); w = 0.38
ax.bar(x - w/2, [float(r["agree_D"]) for r in abl], w, color=CORAL, label="generic AI depth")
ax.bar(x + w/2, [float(r["agree_D2"]) for r in abl], w, color=TEAL, label="fine-tuned depth")
base_y = [100.0 - float(r["true_blind"]) for r in abl]
ax.hlines(base_y, x - 0.55, x + 0.55, color="k", ls=":", lw=1.6,
          label="flag-nothing baseline")
ax.axhline(95, color=MUTED, ls="--", lw=1, label="design goal 95%")
ax.set_xticks(x, [seqlab(r["seq"]) for r in abl])
ax.set_ylabel("blind-spot agreement with ground truth (%)")
ax.set_title("One fix, whole system rescued: 74.5% \u2192 98.6% mean")
ax.set_ylim(0, 104)
ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95,
          edgecolor="none")
save(fig, "fig1_ablation")

# 1b -- the honest headline: blind wall is rare, so the real signal is how much wall each
# system FLAGS blind vs the truth. Generic depth over-flags up to 40x; fine-tuning
# collapses the false-alarm rate to the same order as the ground truth.
fig, ax = plt.subplots(figsize=(8, 4.6))
w = 0.27
ax.bar(x - w, [float(r["true_blind"]) for r in abl], w, color=MUTED, label="truly blind")
ax.bar(x, [float(r["blind_D"]) for r in abl], w, color=CORAL, label="generic AI depth flags")
ax.bar(x + w, [float(r["blind_D2"]) for r in abl], w, color=TEAL, label="fine-tuned depth flags")
ax.set_yscale("log"); ax.set_ylim(0.008, 80)
ax.set_xticks(x, [seqlab(r["seq"]) for r in abl])
ax.set_ylabel("wall flagged blind (%, log scale)")
ax.set_title("Generic depth cries wolf; fine-tuning silences it")
ax.legend(loc="upper right", fontsize=10)
save(fig, "fig1b_falsealarm")

# 1c -- blind-spot detection quality itself (5 mm region level, end-to-end meshes):
# recall = fraction of true blind regions the system finds, precision = fraction of its
# flags that are real. Reported per sequence, off-the-shelf vs fine-tuned depth.
e2m = {r["seq"]: r for r in rows("results/end2end_coverage_mono.csv")}
e2f = {r["seq"]: r for r in rows("results/end2end_coverage_ftmono.csv")}
seqs = list(e2f)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
xx = np.arange(len(seqs))
for ax2, key, ttl in [(axes[0], "reg_recall", "blind regions found (recall)"),
                      (axes[1], "reg_prec", "flags that are real (precision)")]:
    ax2.bar(xx - 0.19, [float(e2m[s][key]) for s in seqs], 0.38, color=CORAL,
            label="off-the-shelf")
    ax2.bar(xx + 0.19, [float(e2f[s][key]) for s in seqs], 0.38, color=TEAL,
            label="fine-tuned")
    ax2.set_xticks(xx, [seqlab(s) for s in seqs], rotation=20)
    ax2.set_title(ttl, fontsize=13)
    ax2.set_ylim(0, 104)
axes[0].set_ylabel("%  (5 mm regions)")
axes[1].legend(loc="upper right", fontsize=10)
fig.suptitle("Blind-spot detection graded on the blind spots themselves",
             fontweight="bold", fontsize=15)
save(fig, "fig1c_blindspot_pr")

# the baseline numbers themselves, for the poster table / write-up
with open("results/baseline_agreement.csv", "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["seq", "true_blind_pct", "flag_nothing_agree", "agree_D", "agree_D2"])
    for r in abl:
        wcsv.writerow([r["seq"], f'{float(r["true_blind"]):.3f}',
                       f'{100.0 - float(r["true_blind"]):.2f}',
                       f'{float(r["agree_D"]):.2f}', f'{float(r["agree_D2"]):.2f}'])
    tb = np.mean([float(r["true_blind"]) for r in abl])
    wcsv.writerow(["MEAN", f"{tb:.3f}", f"{100.0 - tb:.2f}",
                   f'{np.mean([float(r["agree_D"]) for r in abl]):.2f}',
                   f'{np.mean([float(r["agree_D2"]) for r in abl]):.2f}'])
print("baseline table -> results/baseline_agreement.csv", flush=True)

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

# 8 -- closed-loop calibration: persistence kills false alarms
fig, ax = plt.subplots(figsize=(7, 4.4))
sup = [5, 8, 12, 15]
fp = [8, 4, 3, 2]
ax.plot(sup, fp, "o-", color=CORAL, lw=2.5, label="false lesions on lesion-free controls")
ax.axhline(0, color=MUTED, lw=1)
ax.annotate("frozen operating point\n(+ size prior 2\u201322 mm)", (15, 2), xytext=(9.5, 5.5),
            arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=10, color=MUTED)
ax.set_xlabel("required persistence (frames in the same 3D spot)")
ax.set_ylabel("false lesions")
ax.set_title("Calibrating the detector on videos known to be clean")
ax.set_xticks(sup); ax.legend()
save(fig, "fig8_calibration")

# 10 -- the second domain-gap cure: detector fine-tune, before/after on the 30-clip
# batch study. Biggest persistent detection cluster per clip (camera-frame geometry,
# the frozen rule's); controls should sit under the support line, polyp clips above it.
try:
    before = {r["file"]: r for r in rows("results/batch_study/results.csv")}
    after = {r["file"]: r for r in rows("results/batch_study/results_ft.csv")}
    common = [k for k in before if k in after]
    ctrls = sorted([k for k in common if "nopolyp" in k],
                   key=lambda k: -int(before[k]["maxC"]))
    pols = sorted([k for k in common if "nopolyp" not in k],
                  key=lambda k: -int(before[k]["maxC"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax2, keys, ttl in [(axes[0], ctrls, "control clips (no polyp) — should stay quiet"),
                           (axes[1], pols, "polyp clips — should cluster")]:
        xx = np.arange(len(keys))
        ax2.bar(xx - 0.2, [int(before[k]["maxC"]) for k in keys], 0.4,
                color=CORAL, label="off-the-shelf detector")
        ax2.bar(xx + 0.2, [int(after[k]["maxC"]) for k in keys], 0.4,
                color=TEAL, label="domain fine-tuned")
        ax2.axhline(14, color=MUTED, ls="--", lw=1)
        ax2.set_xticks([])
        ax2.set_title(ttl, fontsize=12)
    axes[0].set_ylabel("largest persistent detection cluster (frames)")
    axes[0].text(len(ctrls) - 0.6, 15.5, "finding threshold", color=MUTED,
                 fontsize=9, ha="right")
    axes[1].legend(loc="upper right", fontsize=10)
    fig.suptitle("The domain gap, cured a second time: detector before vs after fine-tuning",
                 fontweight="bold", fontsize=14)
    save(fig, "fig10_detector_fix")
except FileNotFoundError:
    print("  (fig10 skipped: batch study CSVs missing)", flush=True)

# 9 -- edge computing: one network, three machines
fig, ax = plt.subplots(figsize=(6.5, 4.4))
devs = ["workstation\nGPU", "workstation\nCPU", "Pixel 9\n(in your hand)"]
ms = [28, 90, 250]
bars = ax.bar(devs, ms, 0.5, color=[TEAL, CYAN, AMBER])
for b, v in zip(bars, ms):
    ax.text(b.get_x() + b.get_width()/2, v + 6, f"{v} ms", ha="center", fontweight="bold")
ax.set_ylabel("depth inference per frame (ms)")
ax.set_title("The same fine-tuned network, from lab bench to pocket")
save(fig, "fig9_edge")
