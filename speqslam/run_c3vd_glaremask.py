"""
Glare-AWARE direct VO on C3VD: instead of inpainting the glare (which adds its own errors), the
VO simply DROPS specular pixels from its photometric optimization. Compares ATE of the baseline
VO (all pixels, already computed as {seq}_raw.txt) vs the glare-masked VO, per sequence. Clean,
deterministic, no image editing -> the confound-free test.
"""
import sys, os, glob, subprocess, csv, cv2, numpy as np
sys.path.insert(0, "speqslam")
from direct_vo import track_pair
from glare_removal import specular_mask
from scipy.spatial.transform import Rotation as R

EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
DS = 100.0 / 65535.0 / 1000.0
DET = dict(method="fixed", fixed_thr=248, dilate=2, max_area=1500)


def glaremask_vo(undist, K, out_txt):
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    pose = np.eye(4); lines = ["0.0 0 0 0 0 0 0 1"]
    pc = cv2.imread(imgs[0]); pg = cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    pd = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32); pgl = specular_mask(pc, **DET)
    for i in range(1, len(imgs)):
        cc = cv2.imread(imgs[i]); cg = cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        T = track_pair(pg, pd, cg, K, robust=False, depth_scale=DS, glare=pgl)
        pose = pose @ np.linalg.inv(T)
        q = R.from_matrix(pose[:3, :3]).as_quat(); t = pose[:3, 3]
        lines.append(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        pg, pd, pgl = cg, cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32), specular_mask(cc, **DET)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    open(out_txt, "w").write("\n".join(lines) + "\n")


def ate(seq, est):
    out = subprocess.run([EVO, "tum", f"results/c3vd/{seq}_gt.txt", est, "-as", "--no_warnings"],
                         capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000.0
    return float("nan")


if __name__ == "__main__":
    rows = [("seq", "glare_pct", "baseline_ate_mm", "glaremask_ate_mm", "impr_pct")]
    for seq in (sys.argv[1:] or ["cecum_t1_a"]):
        undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
        sev = 100.0 * np.mean([(specular_mask(cv2.imread(f), **DET) > 0).mean()
                               for f in sorted(glob.glob(f"{undist}/image01/*.jpg"))[::5]])
        print(f"[glare-masked VO] {seq} (glare {sev:.2f}%)", flush=True)
        glaremask_vo(undist, K, f"results/c3vd/{seq}_glaremask.txt")
        b = ate(seq, f"results/c3vd/{seq}_raw.txt"); g = ate(seq, f"results/c3vd/{seq}_glaremask.txt")
        impr = 100 * (b - g) / b
        rows.append((seq, f"{sev:.2f}", f"{b:.4f}", f"{g:.4f}", f"{impr:.2f}"))
        print(f"=== {seq}: glare {sev:.2f}%  baseline {b:.4f}mm  glare-masked {g:.4f}mm  improvement {impr:+.1f}% ===", flush=True)
    with open("results/c3vd_glaremask.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved results/c3vd_glaremask.csv", flush=True)
