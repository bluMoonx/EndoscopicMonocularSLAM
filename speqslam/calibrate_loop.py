"""Calibrate the closed-loop filters on the two LESION-FREE videos (negative controls): find the
gentlest (confidence, frame-support) setting that, with a clinical size prior (2-22mm), yields
ZERO false lesions on both. Freeze that setting for the real polyp test."""
import sys, glob, numpy as np, cv2
sys.path.insert(0, "speqslam")
from validate_coverage import load_poses
from polyp_detect import load_model, segment
from polyp_measure import diameter_mm

DS = 100.0 / 65535.0 / 1000.0
MIN_AREA = 0.005


def collect(name):
    if name == "endoslam1":
        p = "data/endoslam/proc/colon4_traj1"
        imgs, deps = sorted(glob.glob(f"{p}/image01/*.jpg")), sorted(glob.glob(f"{p}/depth01_ft/*.png"))
        K, poses = np.loadtxt(f"{p}/K.txt"), load_poses("results/endoslam/colon4_traj1_plain.txt")
    else:
        u = f"data/c3vd/{name}_undist"
        imgs, deps = sorted(glob.glob(f"{u}/image01/*.jpg")), sorted(glob.glob(f"{u}/depth01_ftc/*.png"))
        K, poses = np.loadtxt(f"{u}/K.txt"), load_poses(f"results/c3vd/{name}_ft.txt")
    model = load_model()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hits = []
    for i in range(len(imgs)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        bgr = cv2.imread(imgs[i]); mask = segment(bgr, model)
        binm = mask > 0.5
        if binm.mean() < MIN_AREA:
            continue
        dep = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        ys, xs = np.nonzero(binm)
        u0, v0 = int(xs.mean()), int(ys.mean())
        vals = dep[max(0, v0-2):v0+3, max(0, u0-2):u0+3]; vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        z = float(np.median(vals))
        Pc = np.array([(u0-cx)/fx*z, (v0-cy)/fy*z, z])
        T = poses[ts]; Pw = T[:3, :3] @ Pc + T[:3, 3]
        hits.append((Pw, diameter_mm(mask.astype(np.float32), dep, K) or 0.0, float(mask[binm].mean())))
    return hits


def lesions(hits, conf_min, support, smin=2.0, smax=22.0):
    hits = [h for h in hits if h[2] >= conf_min and smin <= h[1] <= smax]
    used = np.zeros(len(hits), bool); n = 0
    for a in range(len(hits)):
        if used[a]: continue
        grp = [a]
        for b in range(a+1, len(hits)):
            if not used[b] and np.linalg.norm(hits[a][0]-hits[b][0]) < 0.010:
                grp.append(b); used[b] = True
        used[a] = True
        if len(grp) >= support: n += 1
    return n


if __name__ == "__main__":
    all_hits = {n: collect(n) for n in ["cecum_t1_a", "endoslam1"]}
    print({k: len(v) for k, v in all_hits.items()}, "raw hits after size-valid depth", flush=True)
    best = None
    for conf in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for sup in [5, 8, 12, 15]:
            fp = sum(lesions(h, conf, sup) for h in all_hits.values())
            print(f"conf>={conf} support>={sup}: {fp} false lesions", flush=True)
            if fp == 0 and best is None:
                best = (conf, sup)
    print(f"FROZEN OPERATING POINT: size 2-22mm, confidence >= {best[0]}, support >= {best[1]}",
          flush=True)
