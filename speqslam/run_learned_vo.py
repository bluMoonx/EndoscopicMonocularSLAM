"""Run direct VO on LEARNED depth for each sequence -> results/c3vd/{seq}_learned.txt (Stage D)."""
import sys, numpy as np
sys.path.insert(0, "speqslam")
from direct_vo import run as vo_run
DS = 100.0/65535.0/1000.0
for seq in (sys.argv[1:] or ["desc_t4_a","trans_t1_a","trans_t2_a","trans_t2_b"]):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    print(f"[learned-VO] {seq}", flush=True)
    vo_run(undist, undist, f"results/c3vd/{seq}_learned.txt", K, robust=False,
           depth_scale=DS, depth_sub="depth01_learned")
print("DONE", flush=True)
