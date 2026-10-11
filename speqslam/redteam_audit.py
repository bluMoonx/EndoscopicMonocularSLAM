"""RED-TEAM AUDIT: re-derive the project's headline numbers with independently
written code, from raw artifacts, without importing any project module.

Independence claims (stated honestly):
- RT1 (tracking v1 headline): fresh TUM parser + HORN (1987) quaternion absolute
  orientation with scale - a different algorithm from the project's Umeyama SVD -
  applied to the recorded trajectories in results/c3vd/. Validates the metric and
  the recorded v1 numbers end to end.
- RT1b (CSV arithmetic): the v2 headline mean (0.419 mm) re-averaged from
  results/huber_study.csv. The per-sequence v2 ATEs themselves are re-derived by
  speqslam/redteam_retrack.py (fresh driver + Horn metric; it necessarily reuses
  direct_vo's tracker core, which is disclosed - a second tracker implementation
  is out of scope).
- RT3 (clinical 11/21 + 3/20, Fisher p=0.020): threshold re-derived from the calib
  controls, tally re-counted from the frozen batch CSVs, Fisher two-sided exact
  test implemented from the hypergeometric PMF by hand (no scipy).

Writes results/redteam_audit.json.
"""
import csv, json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
report = {"checks": []}

def check(name, expected, actual, tol):
    ok = abs(actual - expected) <= tol
    report["checks"].append({"name": name, "expected": expected,
                             "actual": round(actual, 4), "ok": bool(ok)})
    print(f"[{'OK ' if ok else 'XXX'}] {name}: expected {expected}, got {actual:.4f}")
    return ok

from redteam_metric import horn_ate, tum_xyz

# ---------- RT1: v1 headline from recorded trajectories, Horn metric ----------
print("== RT1: v1 tracking headline, fresh parser + Horn alignment ==")
rows = list(csv.DictReader(open("results/huber_study.csv")))
fresh_v1 = []
for r in rows:
    seq = r["seq"]
    gt = tum_xyz(f"results/c3vd/{seq}_gt.txt")
    ft = tum_xyz(f"results/c3vd/{seq}_ft.txt")
    n = min(len(gt), len(ft))
    a = horn_ate(gt[:n], ft[:n]) * 1000
    fresh_v1.append(a)
    check(f"RT1 {seq} v1 ATE (Horn) vs recorded", float(r["ate_recorded_mm"]), a, 0.01)
# Definitions matter: the published 0.90 is the mean of the huber-study FRESH v1
# re-track (ate_base_mm), not of the recorded August trajectories (ate_recorded_mm,
# mean 0.910) - the 0.008 gap is run-to-run variation between the two v1 runs,
# both on record. Check each mean against its own definition.
check("RT1 recorded-trajectory mean (Horn, fresh code)",
      float(np.mean([float(r["ate_recorded_mm"]) for r in rows])),
      float(np.mean(fresh_v1)), 0.005)
check("RT1 v1 headline mean = mean(ate_base_mm)", 0.902,
      float(np.mean([float(r["ate_base_mm"]) for r in rows])), 0.005)
check("RT1b v2 headline mean (CSV re-average)", 0.419,
      float(np.mean([float(r["ate_huber_mm"]) for r in rows])), 0.0005)

# ---------- RT3: clinical batch tally + hand-rolled Fisher ----------
print("== RT3: clinical n=41 tally + Fisher exact (hand-rolled) ==")
clip_rows = []
for f in ["results/batch_study/results_ft.csv", "results/batch_study/results_ft_ext.csv"]:
    clip_rows += list(csv.DictReader(open(f)))
# Re-derive the frozen operating point with an independent implementation of the
# pre-registered calib rule (batch_analyze.py): among all (geometry, t) with ZERO
# calib-control false alarms, pick max (calib hits, margin over top control), with
# the documented geometry priority maxW > maxC > maxT on ties.
calib = [r for r in clip_rows if r["split"] == "calib"]
best = None
for geom, pri in (("maxW", 3), ("maxC", 2), ("maxT", 1)):
    for t in range(8, 41):
        hits = sum(1 for r in calib if r["label"] == "polyp" and int(r[geom]) >= t)
        fps = sum(1 for r in calib if r["label"] == "nopolyp" and int(r[geom]) >= t)
        if fps == 0:
            top = max((int(r[geom]) for r in calib if r["label"] == "nopolyp"), default=0)
            cand = (hits, t - top, pri, geom, t)
            if best is None or cand > best:
                best = cand
geom, thr = best[3], best[4]
print(f"  independently re-derived operating point: {geom} >= {thr} "
      f"({best[0]} calib hits, margin {best[1]})")
check("RT3 operating geometry is maxC", 1, 1 if geom == "maxC" else 0, 0)
check("RT3 operating threshold", 14, thr, 0)
test = [r for r in clip_rows if r["split"] != "calib"]
pol = [r for r in test if r["label"] == "polyp"]
ctl = [r for r in test if r["label"] == "nopolyp"]
tp = sum(int(r[geom]) >= thr for r in pol)
fp = sum(int(r[geom]) >= thr for r in ctl)
check("RT3 polyp clips in frozen test", 21, len(pol), 0)
check("RT3 control clips in frozen test", 20, len(ctl), 0)
check("RT3 detected polyp clips", 11, tp, 0)
check("RT3 flagged controls", 3, fp, 0)

def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]]: sum of hypergeometric PMFs <= PMF(observed)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def pmf(x):
        return math.exp(log_comb(r1, x) + log_comb(n - r1, c1 - x) - log_comb(n, c1))
    p_obs = pmf(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))

p = fisher_two_sided(tp, len(pol) - tp, fp, len(ctl) - fp)
check("RT3 Fisher two-sided p", 0.020, p, 0.0005)

# ---------- summary ----------
ok = sum(c["ok"] for c in report["checks"]); n = len(report["checks"])
report["pass"] = ok; report["total"] = n
json.dump(report, open("results/redteam_audit.json", "w"), indent=1)
print(f"RED-TEAM: {ok}/{n} checks pass -> results/redteam_audit.json")
