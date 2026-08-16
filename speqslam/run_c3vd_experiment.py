"""
C3VD glare-vs-clean ATE experiment, per sequence. Undistort -> measure glare severity (compact
specular only) -> temporal glare removal -> direct VO on raw and removed -> ATE vs ground truth.
Appends (seq, glare%, raw ATE, removed ATE, improvement) to results/c3vd_severity.csv so several
sequences spanning glare levels build a GROUND-TRUTHED severity curve.

Usage: python run_c3vd_experiment.py cecum_t1_a trans_t1_a ...
"""
import sys, os, subprocess, glob, csv, cv2, numpy as np
sys.path.insert(0, "speqslam")
from c3vd_pipeline import main as undistort_seq
from glare_removal import specular_mask
from temporal_inpaint import process as temporal_process
from direct_vo import run as vo_run

EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
DS = 100.0 / 65535.0 / 1000.0
# compact-specular detector for the bright C3VD phantom: saturated blobs only, not bright tissue
DET = dict(method="fixed", fixed_thr=248, dilate=2, max_area=1500)


def glare_pct(undist):
    fs = sorted(glob.glob(f"{undist}/image01/*.jpg"))[::5]
    return 100.0 * np.mean([(specular_mask(cv2.imread(f), **DET) > 0).mean() for f in fs])


def ate_mm(seq, est):
    out = subprocess.run([EVO, "tum", f"results/c3vd/{seq}_gt.txt", est, "-as", "--no_warnings"],
                         capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000.0
    return float("nan")


def experiment(seq):
    undist = f"data/c3vd/{seq}_undist"
    if not os.path.exists(f"{undist}/rgb.txt"):
        print(f"[undistort] {seq}", flush=True); undistort_seq(seq)
    sev = glare_pct(undist)
    print(f"[glare removal, compact-specular] {seq}  (glare {sev:.2f}%)", flush=True)
    temporal_process(undist, f"{undist}_temporal", window=3, **DET)
    K = np.loadtxt(f"{undist}/K.txt")
    print("[VO raw]", flush=True)
    vo_run(undist, undist, f"results/c3vd/{seq}_raw.txt", K, robust=False, depth_scale=DS)
    print("[VO removed]", flush=True)
    vo_run(f"{undist}_temporal", undist, f"results/c3vd/{seq}_temporal.txt", K, robust=False, depth_scale=DS)
    raw, rem = ate_mm(seq, f"results/c3vd/{seq}_raw.txt"), ate_mm(seq, f"results/c3vd/{seq}_temporal.txt")
    impr = 100 * (raw - rem) / raw
    csvp = "results/c3vd_severity.csv"
    new = not os.path.exists(csvp)
    with open(csvp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seq", "glare_pct", "raw_ate_mm", "removed_ate_mm", "impr_mm", "impr_pct"])
        w.writerow([seq, f"{sev:.3f}", f"{raw:.4f}", f"{rem:.4f}", f"{raw-rem:.4f}", f"{impr:.2f}"])
    print(f"=== {seq}: glare {sev:.2f}%  raw {raw:.4f}mm  removed {rem:.4f}mm  improvement {impr:+.1f}% ===", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["trans_t1_a"]):
        experiment(s)
