"""Final unseen-footage score. The operating point was frozen on the calibration split
(camera-frame clustering, support >= 14) BEFORE any test row was read; this script only
counts. Test population = original test split (results_ft.csv) + extended test2
(results_ft_ext.csv). Prints per-clip outcomes with annotations, then the two numbers."""
import csv

GEOM, T = "maxC", 14
ann = {}
with open("data/hyperkvasir/video-annotations.csv") as f:
    for row in csv.reader(f, delimiter=";"):
        if len(row) == 2 and row[0] != "videoID":
            ann[row[0][:8]] = row[1].strip()

rows = []
for src, keep in [("results/batch_study/results_ft.csv", "test"),
                  ("results/batch_study/results_ft_ext.csv", "test2")]:
    for r in csv.DictReader(open(src)):
        if r["split"] == keep:
            rows.append(r)

det = {"polyp": [0, 0], "nopolyp": [0, 0]}   # [flagged, total]
print(f"frozen rule: {GEOM} >= {T}\n")
for r in sorted(rows, key=lambda x: (x["label"], -int(x[GEOM]))):
    vid = r["file"].split("_")[-1].split(".")[0]
    hit = int(r[GEOM]) >= T
    det[r["label"]][0] += int(hit)
    det[r["label"]][1] += 1
    mark = "FLAGGED" if hit else "   -   "
    print(f'  {r["label"]:8s} {mark} cluster {int(r[GEOM]):3d}  '
          f'{r["file"].split("/")[-1]:26s} "{ann.get(vid, "?")}"')
p, np_ = det["polyp"], det["nopolyp"]
print(f"\nFINAL (frozen protocol, n={len(rows)} unseen test clips):")
print(f"  detection:    {p[0]}/{p[1]} polyp clips flagged ({100*p[0]/max(1,p[1]):.0f}%)")
print(f"  false alarms: {np_[0]}/{np_[1]} control clips flagged ({100*np_[0]/max(1,np_[1]):.0f}%)")
