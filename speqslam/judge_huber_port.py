"""Judge for the pre-registered Huber app-port trial (results/huber_port_gate.md).
Reads the per-arm trajectories the JVM test wrote (arguslite/app/build/trackerab/)
and the GT positions the exporter wrote, computes the project-standard Sim(3)
Umeyama ATE per arm, and applies Gate A exactly as pre-stated: adopt requires a
>= 5% ATE improvement under MEDIAN_SCALED on BOTH windows. Writes
results/huber_port_ab.json. Gate S (on-device speed) is judged separately from
the benchmark card measurement."""
import json, os, sys
import numpy as np

# umeyama_ate copied verbatim from c3vdv2_deform.py:51-63 rather than imported:
# that module runs work at import time (the same trap that once contaminated a
# frozen deform CSV), and a judge must have zero side effects.
def umeyama_ate(gt, est):
    mg, me = gt.mean(0), est.mean(0)
    G, E = gt - mg, est - me
    Sig = G.T @ E / len(gt)
    U, S, Vt = np.linalg.svd(Sig)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    var = (E ** 2).sum() / len(gt)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    aligned = s * (R @ E.T).T + mg
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "arguslite", "app", "build", "trackerab")
FIX = os.path.join(ROOT, "data", "trackerab_fixtures")
SEQS = ["c1_ascending_t4_v1_s1", "c1_sigmoid1_t1_v1_s1"]

def load_csv(path):
    return np.array([[float(x) for x in l.split(",")] for l in open(path) if l.strip()])

out = {"gate": "A", "rule": ">=5% ATE improvement on BOTH windows", "windows": []}
both = True
for seq in SEQS:
    gt = load_csv(os.path.join(FIX, seq, "gt.csv"))
    fixed = load_csv(os.path.join(TRAJ, f"{seq}_FIXED_K.csv"))
    med = load_csv(os.path.join(TRAJ, f"{seq}_MEDIAN_SCALED.csv"))
    assert len(gt) == len(fixed) == len(med), f"{seq}: length mismatch"
    a_fixed = umeyama_ate(gt, fixed)
    a_med = umeyama_ate(gt, med)
    # constant-point baseline, the project's mandatory companion number
    a_const = umeyama_ate(gt, np.zeros_like(gt))
    gain = 100.0 * (a_fixed - a_med) / a_fixed
    ok = gain >= 5.0
    both &= ok
    out["windows"].append({"seq": seq, "ate_fixed_mm": a_fixed, "ate_median_mm": a_med,
                           "ate_constant_mm": a_const, "improvement_pct": gain,
                           "passes": bool(ok)})
    print(f"{seq}: fixed {a_fixed:.3f} mm | median-scaled {a_med:.3f} mm | "
          f"constant {a_const:.3f} mm | gain {gain:+.1f}% -> {'PASS' if ok else 'FAIL'}")

out["gate_A_pass"] = bool(both)
print("GATE A:", "PASS - proceed to on-device Gate S" if both
      else "FAIL - fixed-k stays the shipped default (pre-stated outcome)")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "huber_port_ab.json"), "w"), indent=1)
