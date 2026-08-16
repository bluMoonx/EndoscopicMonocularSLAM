"""
Analyze the Argus severity experiment (3 conditions).

Reads results/severity_experiment.csv (seq,condition,run,keyframes) with conditions:
    baseline  = raw frames
    argus     = Telea (smooth) glare inpainting
    ai        = LaMa (AI, textured) glare inpainting

Plots the SEVERITY CURVE: x = glare severity (%), y = improvement over baseline (keyframes),
one line per treatment. A line rising above 0 = that method helps, and more so with more glare.

Glare severity per sequence comes from data/hamlyn/<seq>_masked/severity.csv (what was removed).
"""
import csv, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREAT = {"argus": ("Telea (smooth fill)", "#2C7B74"),
         "ai":    ("LaMa (AI textured)", "#BC5346")}


def glare_severity(seq):
    path = f"data/hamlyn/{seq}_masked/severity.csv"
    if not os.path.exists(path):
        return None
    vals = [float(r["specular_fraction"]) for r in csv.DictReader(open(path))]
    return 100.0 * sum(vals) / len(vals)


def load():
    data = {}
    for r in csv.DictReader(open("results/severity_experiment.csv")):
        data.setdefault(r["seq"], {}).setdefault(r["condition"], []).append(int(r["keyframes"]))
    return data


def stats(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main():
    data = load()
    seqs = sorted(data, key=lambda s: glare_severity(s) or 0)

    # ---- printed table ----
    hdr = f"{'seq':<12}{'glare%':>7}{'baseline':>14}"
    for t in TREAT:
        hdr += f"{t:>14}{'Δ':>7}"
    print(hdr)
    per = {}
    for s in seqs:
        if "baseline" not in data[s]:
            continue
        sev = glare_severity(s)
        bm, bsd = stats(data[s]["baseline"])
        line = f"{s:<12}{sev:>7.2f}{bm:>10.1f}±{bsd:<3.1f}"
        per[s] = {"sev": sev, "baseline": (bm, bsd)}
        for t in TREAT:
            if t in data[s]:
                tm, tsd = stats(data[s][t])
                per[s][t] = (tm, tsd)
                line += f"{tm:>10.1f}±{tsd:<3.1f}{tm - bm:>+6.1f}"
            else:
                line += f"{'-':>14}{'-':>7}"
        print(line)

    # ---- severity curve ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="#888", lw=1, ls="--", zorder=1)
    for t, (label, color) in TREAT.items():
        xs, ys, ye = [], [], []
        for s in seqs:
            if s in per and t in per[s]:
                bm, bsd = per[s]["baseline"]; tm, tsd = per[s][t]
                xs.append(per[s]["sev"]); ys.append(tm - bm)
                ye.append((bsd**2 + tsd**2) ** 0.5)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=ye, fmt="o", ms=8, color=color, ecolor=color,
                    elinewidth=1.4, capsize=4, zorder=3, label=label)
        if len(xs) >= 2:
            m, b = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 50)
            ax.plot(xr, m * xr + b, color=color, lw=2, alpha=0.8, zorder=2)
    ax.set_xlabel("Glare severity  (% of pixels specular)")
    ax.set_ylabel("Improvement over baseline  (keyframes)")
    ax.set_title("Argus severity curve: does glare removal help, and does AI beat smooth fill?")
    ax.grid(alpha=0.25); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig("results/severity_curve.png", dpi=130)
    print("\nsaved results/severity_curve.png")

    # ---- grouped bars: baseline vs Telea vs LaMa ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    conds = ["baseline"] + list(TREAT)
    colors = {"baseline": "#9A938A", "argus": "#2C7B74", "ai": "#BC5346"}
    labels = {"baseline": "baseline (raw)", "argus": "Telea", "ai": "LaMa (AI)"}
    idx = np.arange(len(seqs)); wd = 0.26
    for j, c in enumerate(conds):
        vals = [st.mean(data[s][c]) if c in data[s] else 0 for s in seqs]
        ax2.bar(idx + (j - 1) * wd, vals, wd, label=labels[c], color=colors[c])
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{s.replace('rectified','seq')}\n{glare_severity(s):.1f}% glare" for s in seqs], fontsize=8)
    ax2.set_ylabel("mean keyframes"); ax2.legend()
    ax2.set_title("Baseline vs Telea vs LaMa keyframes per sequence")
    fig2.tight_layout(); fig2.savefig("results/baseline_vs_argus.png", dpi=130)
    print("saved results/baseline_vs_argus.png")


if __name__ == "__main__":
    main()
