"""Accuracy audit: recompute every published tracking/frame-rate number from the
raw saved artifacts (pose checkpoints, rejection logs, GT files, app binaries,
batch CSVs) and compare against the values carried in results JSON and the
documents. Pure CPU; no model inference. Writes results/audit_report.json.

Guarded-pair counts for runs whose rejection log predates logging (c1_t4
stride-12) are re-derived by counting exactly-repeated step vectors, which the
constant-velocity guard produces by construction; the method is noted in the
report row.
"""
import os, sys, glob, json, struct
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
from c3vdv2_deform import load_poses, umeyama_ate

report = {"checks": [], "pass": 0, "fail": 0}

def check(name, expected, actual, tol):
    ok = abs(actual - expected) <= tol
    report["checks"].append(dict(name=name, expected=expected,
                                 actual=round(float(actual), 3), ok=bool(ok)))
    report["pass" if ok else "fail"] += 1
    print(("PASS " if ok else "FAIL ") + f"{name}: expected {expected}, got {actual:.2f}",
          flush=True)

def gt_positions(base, nums):
    gt_all = load_poses(f"{base}/pose.txt")
    gt = np.array([gt_all[k][1] for k in nums])
    return gt - gt[0]

def eval_traj(gt, est):
    ate = umeyama_ate(gt, est) * 1000
    base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
    return ate, base

def nums_of(base):
    fr = sorted(glob.glob(f"{base}/rgb/*.png"),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return [int(os.path.splitext(os.path.basename(p))[0]) for p in fr]

def coast_count(poses):
    d = [np.linalg.inv(poses[i-1]) @ poses[i] for i in range(1, len(poses))]
    c = 0
    for i in range(1, len(d)):
        if np.allclose(d[i], d[i-1], atol=1e-12):
            c += 1
    return c

S3 = {
    "c1_full_t4_v1": (3.0, 101.4, 168.3),
    "c2_full_t1_v1": (1.5, 92.1, 133.4),
    "c1_full_t1_v1": (3.1, 114.5, 157.6),
}
SUB = {  # (base, sub-file suffix, expected guard %, ATE, baseline)
    "c1_full_t4_v1 stride6":  ("c1_full_t4_v1_s3", "sub2", 6.7, 132.8, 168.3),
    "c2_full_t1_v1 stride12": ("c2_full_t1_v1_s3", "sub4", 16.5, 103.0, 133.4),
    "c1_full_t1_v1 stride12": ("c1_full_t1_v1_s3", "sub4", 23.0, 94.4, 157.6),
}

for seq, (g_exp, ate_exp, base_exp) in S3.items():
    b = f"data/c3vdv2/{seq}_s3"
    poses = list(np.load(f"{b}/poses_est.npy"))
    rej = np.load(f"{b}/rejected.npy") if os.path.exists(f"{b}/rejected.npy") else []
    nums = nums_of(b)
    gt = gt_positions(b, nums[:len(poses)])
    est = np.array([p[:3, 3] for p in poses])
    ate, base = eval_traj(gt, est)
    check(f"{seq} s3 guard%", g_exp, 100 * len(rej) / (len(poses) - 1), 0.15)
    check(f"{seq} s3 ATE mm", ate_exp, ate, 0.5)
    check(f"{seq} s3 baseline mm", base_exp, base, 0.5)

for label, (d, suf, g_exp, ate_exp, base_exp) in SUB.items():
    b = f"data/c3vdv2/{d}"
    poses = list(np.load(f"{b}/poses_{suf}.npy"))
    rej = np.load(f"{b}/rejected_{suf}.npy")
    sub = int(suf[-1])
    nums = nums_of(b)[::sub]
    gt = gt_positions(b, nums[:len(poses)])
    est = np.array([p[:3, 3] for p in poses])
    ate, base = eval_traj(gt, est)
    check(f"{label} guard%", g_exp, 100 * len(rej) / (len(poses) - 1), 0.15)
    check(f"{label} ATE mm", ate_exp, ate, 0.6)

# c1_t4 stride-12 (guarded re-run; rejection log predates logging - derive by
# repeated-step counting) and stride-24
b12 = "data/c3vdv2/c1_full_t4_v1"
p12 = list(np.load(f"{b12}/poses_est2.npy"))
n12 = [i * 12 for i in range(len(p12))] if not glob.glob(f"{b12}/rgb/*.png") else None
s3nums = nums_of("data/c3vdv2/c1_full_t4_v1_s3")
nums12 = s3nums[::4][:len(p12)]
gt12 = gt_positions(b12, nums12)
est12 = np.array([p[:3, 3] for p in p12])
ate12, base12 = eval_traj(gt12, est12)
check("c1_full_t4_v1 stride12 guard% (repeated-step derivation)", 16.0,
      100 * coast_count(p12) / (len(p12) - 1), 1.0)
check("c1_full_t4_v1 stride12 ATE mm", 146.3, ate12, 0.6)
check("c1_full_t4_v1 stride12 baseline mm", 168.3, base12, 0.5)
if os.path.exists(f"{b12}/poses_sub2.npy"):
    p24 = list(np.load(f"{b12}/poses_sub2.npy"))
    r24 = np.load(f"{b12}/rejected_sub2.npy")
    nums24 = nums12[::2][:len(p24)]
    gt24 = gt_positions(b12, nums24)
    ate24, base24 = eval_traj(gt24, np.array([p[:3, 3] for p in p24]))
    check("c1_full_t4_v1 stride24 guard%", 39.5, 100 * len(r24) / (len(p24) - 1), 0.2)
    check("c1_full_t4_v1 stride24 ATE mm", 120.2, ate24, 0.6)

# app binary assets: parse, bounds, cross-consistency
A = "arguslite/app/src/main/assets"
bts = open(f"{A}/mesh_full.bin", "rb").read()
nv, nt = struct.unpack_from("<ii", bts, 0)
V = np.frombuffer(bts, np.float32, 3 * nv, 8).reshape(-1, 3)
W = np.frombuffer(bts, np.uint8, nv, 8 + 27 * nv)
T = np.frombuffer(bts, np.int32, 3 * nt, 8 + 28 * nv)
ok_mesh = (len(bts) == 8 + 28 * nv + 12 * nt and T.max() < nv and T.min() >= 0
           and not np.isnan(V).any())
check("app mesh_full.bin integrity (1=ok)", 1, int(ok_mesh), 0)
check("app mesh under-seen % (views<3)", 36.9, 100 * (W < 3).mean(), 0.2)
span = float((V.max(0) - V.min(0)).max() / 2)
check("app mesh span mm (sane 100-400)", 1, int(100 <= span <= 400), 0)
for name, npts in (("traj_full.bin", 2800), ("gt_full.bin", 2800)):
    tb = open(f"{A}/{name}", "rb").read()
    tn = struct.unpack_from("<i", tb, 0)[0]
    P = np.frombuffer(tb, np.float32, 3 * tn, 4)
    check(f"app {name} points", npts, tn, 0)
    check(f"app {name} normalized (|max|<=1.6)", 1,
          int(np.abs(P).max() <= 1.6 and not np.isnan(P).any()), 0)
bb = open(f"{A}/born_full.bin", "rb").read()
bn = struct.unpack_from("<i", bb, 0)[0]
born = np.frombuffer(bb, np.uint16, bn, 4)
check("app born_full.bin per-vertex", nv, bn, 0)
check("app born indices < traj length", 1, int(born.max() < 2800), 0)

# batch detection stats from the frozen CSVs
import csv as csvmod
from scipy.stats import fisher_exact
rows = []
for f in ("results/batch_study/results_ft.csv", "results/batch_study/results_ft_ext.csv"):
    if os.path.exists(f):
        rows += list(csvmod.DictReader(open(f)))
test = [r for r in rows if r["split"] in ("test", "test2")]
det = lambda r: int(r["maxC"]) >= 14
tp = sum(1 for r in test if r["label"] == "polyp" and det(r))
pn = sum(1 for r in test if r["label"] == "polyp")
fp = sum(1 for r in test if r["label"] == "nopolyp" and det(r))
cn = sum(1 for r in test if r["label"] == "nopolyp")
check("frozen test: polyp clips flagged", 11, tp, 0)
check("frozen test: polyp clips total", 21, pn, 0)
check("frozen test: controls flagged", 3, fp, 0)
check("frozen test: controls total", 20, cn, 0)
# two-sided is the published standard as of the 2026-09-26 audit: the earlier
# 0.013 was the unlabeled one-sided value and was corrected everywhere
odds, p = fisher_exact([[tp, pn - tp], [fp, cn - fp]])
check("frozen test: Fisher p (two-sided)", 0.0203, p, 0.002)
check("frozen test: odds ratio", 6.2, odds, 0.3)

json.dump(report, open("results/audit_report.json", "w"), indent=1)
print(f"\nAUDIT: {report['pass']} passed, {report['fail']} FAILED "
      f"-> results/audit_report.json", flush=True)
