"""Judge for the dense-protocol test3 gate (results/dense_protocol_gate.md).
Applies G1/G2/G3 exactly as pre-registered. No side effects at import."""
import csv, json, math, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THR = 14
man = list(csv.DictReader(open("results/batch_study/manifest_t3.csv")))
ann = {}
for row in csv.reader(open("data/hyperkvasir/video-annotations.csv"), delimiter=";"):
    if len(row) == 2 and row[0] != "videoID":
        ann[row[0]] = row[1].strip()
d80 = {r["file"]: int(r["maxC"]) for r in csv.DictReader(open("results/batch_study/results_t3_80.csv"))}
dde = {r["file"]: int(r["maxC_dense"]) for r in csv.DictReader(open("results/detect_dense_t3.csv"))}

# anatomy-only annotations are CLEAN per the pre-registered rule ("naming only
# anatomy"). First judge run used too narrow a list and misclassified "duodenum"
# and "z-line distal oesophagus" as pathology; corrected to match the rule text.
ANATOMY = ("blind end", "ileocecal", "normal", "anatom", "duodenum", "z-line",
           "oesophagus", "esophagus", "stomach", "pylorus", "cardia", "cecum",
           "rectum")
FINDING = ("cancer", "tumor", "tumour", "ulcer", "polyp", "emr", "resection",
           "leak", "colitis", "dye", "esophagitis", "oesophagitis", "varice",
           "barrett", "bleed", "stenosis")
def is_clean(a):
    a = a.lower()
    if any(k in a for k in FINDING):
        return False
    return a == "nothing" or any(k in a for k in ANATOMY)

pol = [r for r in man if r["label"] == "polyp"]
ctl = [r for r in man if r["label"] == "nopolyp"]
p80 = sum(d80.get(r["file"], 0) >= THR for r in pol)
pde = sum(dde.get(r["file"], 0) >= THR for r in pol)
c80 = [r for r in ctl if d80.get(r["file"], 0) >= THR]
cde = [r for r in ctl if dde.get(r["file"], 0) >= THR]
c80_clean = [r for r in c80 if is_clean(ann.get(r["videoID"], "?"))]
cde_clean = [r for r in cde if is_clean(ann.get(r["videoID"], "?"))]

print(f"test3: {len(pol)} polyp / {len(ctl)} control clips")
print(f"80-frame : {p80}/{len(pol)} polyps | {len(c80)} controls flagged ({len(c80_clean)} clean)")
print(f"dense-400: {pde}/{len(pol)} polyps | {len(cde)} controls flagged ({len(cde_clean)} clean)")
print("\nflagged controls (dense), annotation verbatim:")
for r in cde:
    tag = "CLEAN" if is_clean(ann.get(r["videoID"], "?")) else "pathology"
    print(f'  {os.path.basename(r["file"])[:24]:24} {ann.get(r["videoID"], "?"):38} [{tag}]')

def log_comb(n, k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)
def fisher(a, b, c, d):
    n, r1, c1 = a+b+c+d, a+b, a+c
    pmf = lambda x: math.exp(log_comb(r1, x) + log_comb(n-r1, c1-x) - log_comb(n, c1))
    po = pmf(a)
    lo, hi = max(0, r1+c1-n), min(r1, c1)
    return sum(pmf(x) for x in range(lo, hi+1) if pmf(x) <= po*(1+1e-9))

p = fisher(pde, len(pol)-pde, len(cde), len(ctl)-len(cde))
g1 = pde > p80
g2 = len(cde_clean) <= len(c80_clean) + 2 and len(cde_clean) <= 4
g3 = p < 0.05
print(f"\nG1 recall: dense {pde} > 80f {p80} -> {'PASS' if g1 else 'FAIL'}")
print(f"G2 clean FAs: dense {len(cde_clean)} vs 80f {len(c80_clean)} (+2 allowed, cap 4) -> {'PASS' if g2 else 'FAIL'}")
print(f"G3 Fisher two-sided p = {p:.4f} -> {'PASS' if g3 else 'FAIL'}")
verdict = "ADOPT dense-400 as published screening protocol" if (g1 and g2 and g3) \
          else "KEEP 80-frame protocol (pre-stated outcome)"
print("VERDICT:", verdict)
json.dump({"thr": THR, "n_polyp": len(pol), "n_ctl": len(ctl),
           "p80": p80, "pdense": pde, "c80": len(c80), "cdense": len(cde),
           "c80_clean": len(c80_clean), "cdense_clean": len(cde_clean),
           "flagged_dense": [{"file": r["file"], "annotation": ann.get(r["videoID"], "?"),
                              "clean": is_clean(ann.get(r["videoID"], "?"))} for r in cde],
           "fisher_p": p, "G1": g1, "G2": g2, "G3": g3, "verdict": verdict},
          open("results/dense_t3_verdict.json", "w"), indent=1)
