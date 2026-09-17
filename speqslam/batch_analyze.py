"""Batch-study analysis. Selection rule (stated before looking at any number):
on the CALIB split, for each geometry W/C/T and support threshold t in 8..40,
count polyp clips with max>=t (hits) and nopolyp clips with max>=t (false alarms).
Choose the (geometry, t) maximizing calib hits subject to ZERO calib false alarms;
ties broken by the widest margin (distance from t to the highest control cluster),
then by preferring W over C over T (simpler claim). Freeze. Then score TEST split once."""
import sys, csv
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/batch_study/results.csv"
print(f"analyzing {SRC}\n")
rows = list(csv.DictReader(open(SRC)))
for r in rows:
    for k in ("maxW", "maxC", "maxT", "frames", "gated", "fired"):
        r[k] = int(r[k])
calib = [r for r in rows if r["split"] == "calib"]
test = [r for r in rows if r["split"] == "test"]
print(f"clips: {len(rows)} total | calib {len(calib)} "
      f"({sum(1 for r in calib if r['label']=='polyp')}P/"
      f"{sum(1 for r in calib if r['label']=='nopolyp')}N) | test {len(test)} "
      f"({sum(1 for r in test if r['label']=='polyp')}P/"
      f"{sum(1 for r in test if r['label']=='nopolyp')}N)\n")

print("full table (label split fired gated W C T file):")
for r in sorted(rows, key=lambda x: (x["split"], x["label"])):
    print(f'  {r["label"]:8s} {r["split"]:5s} fired {r["fired"]:3d} gated {r["gated"]:3d} '
          f'W {r["maxW"]:3d} C {r["maxC"]:3d} T {r["maxT"]:3d}  {r["file"].split("/")[-1]}')

best = None
print("\ncalibration scan (zero-false-alarm candidates only):")
for geom in ("maxW", "maxC", "maxT"):
    for t in range(8, 41):
        hits = sum(1 for r in calib if r["label"] == "polyp" and r[geom] >= t)
        fps = sum(1 for r in calib if r["label"] == "nopolyp" and r[geom] >= t)
        if fps == 0:
            top_ctrl = max((r[geom] for r in calib if r["label"] == "nopolyp"), default=0)
            margin = t - top_ctrl
            cand = (hits, margin, {"maxW": 3, "maxC": 2, "maxT": 1}[geom], geom, t)
            if best is None or cand > best:
                best = cand
            if hits > 0:
                print(f"  {geom} t={t}: {hits} calib hits, 0 calib FPs (margin {margin})")
if best is None:
    print("  no zero-FP operating point exists on calibration -> report that honestly")
else:
    hits, margin, _, geom, t = best
    npol = sum(1 for r in calib if r["label"] == "polyp")
    print(f"\nFROZEN: geometry {geom}, support >= {t} "
          f"(calib: {hits}/{npol} polyp clips, 0 false alarms)")
    th = sum(1 for r in test if r["label"] == "polyp" and r[geom] >= t)
    tf = sum(1 for r in test if r["label"] == "nopolyp" and r[geom] >= t)
    tp = sum(1 for r in test if r["label"] == "polyp")
    tn = sum(1 for r in test if r["label"] == "nopolyp")
    print(f"TEST SPLIT (opened once): detections {th}/{tp} polyp clips | "
          f"false alarms {tf}/{tn} control clips")
