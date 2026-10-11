"""Time-binned support analysis for the dense protocol (development, offline).
Support = distinct time-bins (80 bins/clip) represented in the best spatial
cluster (same greedy 10mm camera-frame rule). Restores the 80-frame protocol's
implicit temporal independence while keeping dense sampling's sensitivity."""
import csv, glob, os
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def bin_support(pos, fidx, sampled, radius=0.010, nbins=80):
    n = len(pos)
    if n == 0: return 0
    bins = (np.asarray(fidx) * nbins // max(1, sampled)).astype(int)
    used = [False]*n; best = 0
    for a in range(n):
        if used[a]: continue
        used[a] = True; members = [a]
        for b in range(a+1, n):
            if not used[b] and np.linalg.norm(pos[a]-pos[b]) < radius:
                used[b] = True; members.append(b)
        best = max(best, len(set(bins[m] for m in members)))
    return best

rows = []
for f in ["results/batch_study/manifest.csv", "results/batch_study/manifest_ext.csv"]:
    rows += list(csv.DictReader(open(f)))
sup = {}
for r in rows:
    stem = os.path.splitext(os.path.basename(r["file"]))[0]
    p = f"results/detect_dense_fires/{stem}.npz"
    if not os.path.exists(p):
        sup[r["file"]] = 0; continue
    z = np.load(p)
    sup[r["file"]] = bin_support(z["pos"], z["fidx"], int(z["sampled"]))

calib = [r for r in rows if r["split"] == "calib"]
test = [r for r in rows if r["split"] != "calib"]
topc = max(sup[r["file"]] for r in calib if r["label"] == "nopolyp")
best = None
for t in range(4, 81):
    hits = sum(1 for r in calib if r["label"] == "polyp" and sup[r["file"]] >= t)
    fps = sum(1 for r in calib if r["label"] == "nopolyp" and sup[r["file"]] >= t)
    if fps == 0:
        cand = (hits, t - topc, t)
        if best is None or cand > best: best = cand
hits, margin, thr = best
print(f"calib control max bin-support = {topc}; operating point: >= {thr} ({hits} hits, margin {margin})")
pol = [r for r in test if r["label"] == "polyp"]; ctl = [r for r in test if r["label"] == "nopolyp"]
tp = [r for r in pol if sup[r["file"]] >= thr]; fp = [r for r in ctl if sup[r["file"]] >= thr]
print(f"DEV TALLY binned: {len(tp)}/{len(pol)} polyps | {len(fp)}/{len(ctl)} controls")
print("  (frozen-80: 11/21 | 3/20;  dense-raw: 16/21 | 8/20)")
print("\nflagged controls:", [os.path.basename(r['file'])[:22] for r in fp])
old = {}
for f in ["results/batch_study/results_ft.csv", "results/batch_study/results_ft_ext.csv"]:
    for r in csv.DictReader(open(f)): old[r["file"]] = int(r["maxC"])
print("\npolyp detail (80f maxC -> binned support):")
for r in sorted(pol, key=lambda r: -sup[r["file"]]):
    d = "DET" if sup[r["file"]] >= thr else "mis"
    print(f"  {os.path.basename(r['file'])[:26]:26} {old[r['file']]:>3} -> {sup[r['file']]:>3}  {d}")
