"""
C3VD glare-vs-clean ATE experiment. For a sequence: undistort (if needed) -> temporal glare
removal (C3VD threshold: saturated pixels, forced via lo=hi=255) -> run the direct VO on raw and
glare-removed -> ATE vs ground truth with evo. Reports the ATE improvement from removing glare.
"""
import sys, os, subprocess, numpy as np
sys.path.insert(0, "speqslam")
from c3vd_pipeline import main as undistort_seq
from temporal_inpaint import process as temporal_process
from direct_vo import run as vo_run

EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
DS = 100.0 / 65535.0 / 1000.0   # C3VD uint16 depth -> meters


def ate_mm(seq, est):
    out = subprocess.run([EVO, "tum", f"results/c3vd/{seq}_gt.txt", est, "-as", "--no_warnings"],
                         capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000.0
    return float("nan")


def experiment(seq):
    if not os.path.exists(f"data/c3vd/{seq}_undist/rgb.txt"):
        print(f"[undistort] {seq}", flush=True)
        undistort_seq(seq)
    print(f"[glare removal: temporal, saturated-pixel threshold] {seq}", flush=True)
    temporal_process(f"data/c3vd/{seq}_undist", f"data/c3vd/{seq}_undist_temporal",
                     window=3, k=2.0, lo=255, hi=255, dilate=1)
    K = np.loadtxt(f"data/c3vd/{seq}_undist/K.txt")
    print("[VO: raw]", flush=True)
    vo_run(f"data/c3vd/{seq}_undist", f"data/c3vd/{seq}_undist",
           f"results/c3vd/{seq}_raw.txt", K, robust=False, depth_scale=DS)
    print("[VO: glare-removed]", flush=True)
    vo_run(f"data/c3vd/{seq}_undist_temporal", f"data/c3vd/{seq}_undist",
           f"results/c3vd/{seq}_temporal.txt", K, robust=False, depth_scale=DS)
    raw, rem = ate_mm(seq, f"results/c3vd/{seq}_raw.txt"), ate_mm(seq, f"results/c3vd/{seq}_temporal.txt")
    print(f"\n=== {seq}: C3VD ATE vs ground truth (mm) ===", flush=True)
    print(f"RAW           : {raw:.4f}", flush=True)
    print(f"GLARE-REMOVED : {rem:.4f}", flush=True)
    print(f"improvement   : {raw - rem:+.4f} mm  ({100*(raw-rem)/raw:+.1f}%)", flush=True)


if __name__ == "__main__":
    experiment(sys.argv[1] if len(sys.argv) > 1 else "trans_t1_a")
