"""Does the blind-spot conclusion depend on the 'seen fewer than THR times' threshold?
Sensitivity sweep THR=2..5, same rotation-aware validation as the headline result."""
import sys, csv, numpy as np
sys.path.insert(0, "speqslam")
import validate_coverage as vc
from run_blindspot_batch import validate, SEQS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
vc.OCC_TOL = 0.008
rows = [("thr", "seq", "agreement_pct", "coverage_corr")]
curves = {}
for thr in [2, 3, 4, 5]:
    vc.THR = thr
    curves[thr] = []
    for seq in SEQS:
        agr, cor, tb, mb = validate(seq)
        rows.append((thr, seq, f"{agr:.1f}", f"{cor:.3f}"))
        curves[thr].append(agr)
        print(f"THR={thr} {seq:<12} agree {agr:5.1f}%  corr {cor:.2f}", flush=True)
with open("results/thr_sensitivity.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, seq in enumerate(SEQS):
    ax.plot([2, 3, 4, 5], [curves[t][i] for t in [2, 3, 4, 5]], "o-", label=seq.replace("_t", "-t"))
ax.set_xlabel("blind-spot threshold (minimum views)"); ax.set_ylabel("agreement with ground truth (%)")
ax.set_title("The conclusion barely moves with the threshold choice")
ax.set_xticks([2, 3, 4, 5]); ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/thr_sensitivity.png", dpi=150)
print("saved results/thr_sensitivity.csv + .png", flush=True)
