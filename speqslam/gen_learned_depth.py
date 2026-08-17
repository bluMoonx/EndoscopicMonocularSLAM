"""
Generate LEARNED metric depth maps for a sequence, so the rest of the pipeline (VO, surface,
coverage) can run WITHOUT the ground-truth depth sensor -- i.e. fully monocular.

Monocular depth is scale-ambiguous, so we recover ONE global affine (a,b): disparity -> metric,
fit once against GT over the sequence (the standard single-scale recovery, exactly like Sim(3)
scale alignment for monocular SLAM). The SAME (a,b) is applied to every frame, giving a
temporally-consistent depth stream. Saved as uint16 PNG in <seq>_undist/depth01_learned/ using
C3VD's own scale, so direct_vo/coverage read them unchanged.
"""
import sys, glob, os, numpy as np, cv2
import mono_depth as md

DS = 100.0 / 65535.0 / 1000.0


def fit_global_affine(imgs, deps, n=40):
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    X, Y = [], []
    for i in idx:
        disp = md.predict_disparity(cv2.imread(imgs[i]))
        dgt = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        m = dgt > 0
        s = max(1, m.sum() // 4000)
        X.append(disp[m][::s]); Y.append(1.0 / dgt[m][::s])
    X, Y = np.concatenate(X), np.concatenate(Y)
    A = np.stack([X, np.ones_like(X)], 1)
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return a, b


def generate(seq, dmin=0.003, dmax=0.11):
    undist = f"data/c3vd/{seq}_undist"
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    a, b = fit_global_affine(imgs, deps)
    out = f"{undist}/depth01_learned"; os.makedirs(out, exist_ok=True)
    print(f"{seq}: global affine a={a:.4f} b={b:.4f} -> generating {len(imgs)} learned depth maps", flush=True)
    for i, ip in enumerate(imgs):
        disp = md.predict_disparity(cv2.imread(ip))
        depth = 1.0 / np.clip(a * disp + b, 1e-6, None)          # metric depth (m)
        depth[(depth < dmin) | (depth > dmax) | ~np.isfinite(depth)] = 0.0
        raw = np.clip(depth / DS, 0, 65535).astype(np.uint16)
        cv2.imwrite(f"{out}/{os.path.basename(ip).replace('.jpg','.png')}", raw)
        if i % 50 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["cecum_t1_a"]):
        generate(s)
