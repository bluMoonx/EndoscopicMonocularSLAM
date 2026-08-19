"""Retest the flicker weakness WITH affine illumination compensation (DSO-style) in the tracker.
Same seed => identical corrupted sequences as robustness.py. Also runs the clean baseline with
compensation on, to confirm the fix costs nothing when lighting is stable."""
import sys, os, glob, csv, subprocess, shutil
import numpy as np
sys.path.insert(0, "speqslam")
from direct_vo import run as vo_run
from robustness import corrupt_seq, ate_mm, SEQ, DS

def main():
    undist = f"data/c3vd/{SEQ}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    rng = np.random.default_rng(0)                      # same seed/order as robustness.py
    rows = [("condition", "level", "ate_mm")]
    out = f"results/c3vd/{SEQ}_robI_baseline.txt"
    vo_run(undist, undist, out, K, robust=False, depth_scale=DS, depth_sub="depth01_ftc", illum=True)
    a = ate_mm(out); rows.append(("baseline+illum", "-", f"{a:.3f}"))
    print(f"baseline+illum   ATE {a:.3f} mm", flush=True)
    for lv in (0.10, 0.20, 0.35):
        cdir = corrupt_seq("flicker", lv, rng)
        out = f"results/c3vd/{SEQ}_robI_flicker_{lv}.txt"
        vo_run(cdir, undist, out, K, robust=False, depth_scale=DS, depth_sub="depth01_ftc", illum=True)
        shutil.rmtree(cdir, ignore_errors=True)
        a = ate_mm(out); rows.append((f"flicker+illum", lv, f"{a:.3f}"))
        print(f"flicker {lv}+illum  ATE {a:.3f} mm", flush=True)
    with open("results/robustness_illum.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("saved results/robustness_illum.csv", flush=True)

if __name__ == "__main__":
    main()
