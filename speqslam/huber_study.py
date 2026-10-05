"""Huber characterization study: what does the tracker's dormant robust
toggle do to the VALIDATED core benchmarks?

Motivation (deform attempts, manuscript 4.17): in-loop Huber improved or
matched 9 of 10 C3VDv2 sequences including every rigid control, as a side
observation. The deployed pipeline runs robust=False - a choice made in the
glare era for experimental reasons, never revisited. This study MEASURES
the toggle on the headline benchmark (the five C3VD GT sequences, fine-tuned
depth, the fully-monocular ATE table); it does NOT adopt anything - flipping
the deployed config would require re-validating the entire frozen stack, a
decision recorded for future work.

Pre-stated reporting rule: both arms run fresh under identical settings
(depth01_ftc, illum=True, same pair recipe); the robust=False arm is first
checked against the recorded results/c3vd/<seq>_ft.txt ATE - if the fresh
baseline does not reproduce the recorded number within 15%, the discrepancy
is reported and the comparison is fresh-vs-fresh only. Output: per-sequence
ATE for recorded / fresh-base / fresh-huber -> results/huber_study.csv.

Usage: python speqslam/huber_study.py [minutes]
"""
import os, sys, csv, glob, time
import numpy as np, cv2
BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import direct_vo as dv
from c3vdv2_deform import umeyama_ate

DS = 100.0 / 65535.0 / 1000.0
SEQS = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]
RES = "results/huber_study.csv"

def tum_positions(path):
    out = []
    for l in open(path):
        v = l.split()
        if len(v) >= 8:
            out.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(out)

def track(seq, robust):
    base = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{base}/K.txt")
    imgs = sorted(glob.glob(f"{base}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{base}/depth01_ftc/*.png"))
    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    g0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    d0 = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
    for i in range(1, len(imgs)):
        g1 = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        T = dv.track_pair(g0, d0, g1, K, 4, robust=robust, depth_scale=DS, illum=True)
        pose = pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
        g0 = g1
        d0 = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.array(traj)

done = set()
if os.path.exists(RES):
    done = {r["seq"] for r in csv.DictReader(open(RES))}
else:
    with open(RES, "w", newline="") as f:
        csv.writer(f).writerow(["seq", "ate_recorded_mm", "ate_base_mm",
                                "ate_huber_mm"])
t0 = time.time()
for seq in SEQS:
    if seq in done:
        continue
    if time.time() - t0 > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    gt = tum_positions(f"results/c3vd/{seq}_gt.txt")
    rec = tum_positions(f"results/c3vd/{seq}_ft.txt")
    n = min(len(gt), len(rec))
    a_rec = umeyama_ate(gt[:n], rec[:n]) * 1000
    tb = track(seq, False)
    th = track(seq, True)
    n2 = min(len(gt), len(tb))
    a_b = umeyama_ate(gt[:n2], tb[:n2]) * 1000
    a_h = umeyama_ate(gt[:n2], th[:n2]) * 1000
    with open(RES, "a", newline="") as f:
        csv.writer(f).writerow([seq, f"{a_rec:.3f}", f"{a_b:.3f}", f"{a_h:.3f}"])
    print(f"{seq}: recorded {a_rec:.3f} | fresh base {a_b:.3f} | huber {a_h:.3f} mm",
          flush=True)
rows = list(csv.DictReader(open(RES)))
if len(rows) == len(SEQS):
    ch = [float(r["ate_huber_mm"]) / float(r["ate_base_mm"]) - 1 for r in rows]
    print(f"\nHuber vs fresh baseline: per-seq change "
          f"{['%+.0f%%' % (100*c) for c in ch]} | median {100*np.median(ch):+.1f}%",
          flush=True)
