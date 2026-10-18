"""Pre-registered baseline head-to-head: out-of-the-box ORB-SLAM3 (monocular) vs the adopted
pipeline-v2 tracker on the five C3VD headline sequences.

Spec: results/orbslam3_baseline_gate.md (committed with this script, before any run).
Phases (idempotent; default runs all three in order):
  --prep     generate the two frozen ORB-SLAM3 configs per sequence; verify the Argus v2
             trajectories (data/redteam/<seq>_traj.npy) reproduce the published red-team ATEs
             to within 0.01 mm, else HALT.
  --run      execute 2 configs x 5 sequences x 5 runs (+1 discarded warm-up) in the
             argus_slam container; raw trajectories kept under results/orbslam3_baseline/.
  --analyze  metrics (completion, Sim(3)/Umeyama ATE beside the constant-point baseline on
             the same timestamp subset, position-based translational RPE at D=1 and D=30
             frames), bootstrap CIs, CSVs, figures.
"""
import argparse, csv, glob, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "speqslam")
from redteam_metric import horn_ate, tum_xyz

SEQS = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
CFGS = {
    # name -> (nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST)
    "stock": (1000, 1.2, 8, 20, 7),   # stock TUM monocular example settings
    "boost": (4000, 1.2, 10, 5, 2),   # the Act I endoscopy-boosted extractor (Hamlyn)
}
N_RUNS = 5
FPS = 30.0
TOL = 1.0 / 60.0          # GT association tolerance (s)
CONTAINER = "argus_slam"
OUT = "results/orbslam3_baseline"
FIGDIR = "results/figures"
PUBLISHED = "results/redteam_retrack.csv"   # frozen, red-team-verified v2 ATEs


# ---------- geometry ----------

def umeyama(src, dst):
    """Sim(3) aligning src -> dst (Umeyama 1991). Returns s, R, t."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sgn[2, 2] = -1.0
    R = U @ sgn @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = (d * np.diag(sgn)).sum() / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse_mm(gt, est):
    s, R, t = umeyama(est, gt)
    aligned = (s * (R @ est.T)).T + t
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean())) * 1000.0, aligned


def const_point_mm(gt):
    """The strongest trivial predictor: the GT centroid (standing policy, retraction 2)."""
    c = gt.mean(0)
    return float(np.sqrt(((gt - c) ** 2).sum(1).mean())) * 1000.0


def rpe_mm(gt, aligned, ts, delta_frames):
    """Position RPE per gate: ||(p^(t+D)-p^(t)) - (gt(t+D)-gt(t))|| over tracked pairs."""
    dt = delta_frames / FPS
    idx = {round(t * FPS): i for i, t in enumerate(ts)}
    vals = []
    for i, t in enumerate(ts):
        j = idx.get(round((t + dt) * FPS))
        if j is not None:
            vals.append(np.linalg.norm((aligned[j] - aligned[i]) - (gt[j] - gt[i])) * 1000.0)
    return np.array(vals)


def boot_ci(vals, n=10000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------- data ----------

def load_tum_full(path):
    ts, xyz = [], []
    for l in open(path):
        v = l.split()
        if l.startswith("#") or len(v) < 8:
            continue
        ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(ts), np.array(xyz)


def seq_info(seq):
    base = f"data/c3vd/{seq}_undist"
    n = len(glob.glob(f"{base}/image01/*.jpg"))
    gt_ts, gt_xyz = load_tum_full(f"results/c3vd/{seq}_gt.txt")
    return base, n, gt_ts, gt_xyz


def match_gt(ts, gt_ts, gt_xyz):
    """Nearest-GT-timestamp association within TOL; returns kept est indices + GT rows."""
    order = np.searchsorted(gt_ts, ts)
    keep, rows = [], []
    for i, t in enumerate(ts):
        cands = [c for c in (order[i] - 1, order[i]) if 0 <= c < len(gt_ts)]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(gt_ts[c] - t))
        if abs(gt_ts[c] - t) <= TOL:
            keep.append(i); rows.append(c)
    return np.array(keep, int), np.array(rows, int)


# ---------- prep ----------

def make_yaml(seq, cfg_name):
    base = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{base}/K.txt")
    import cv2
    img = cv2.imread(sorted(glob.glob(f"{base}/image01/*.jpg"))[0])
    h, w = img.shape[:2]
    nf, sf, nl, ini, mn = CFGS[cfg_name]
    y = f"""%YAML:1.0

# Pre-registered baseline config ({cfg_name}) - results/orbslam3_baseline_gate.md.
# Camera block = the sequence's own undistortion-step pinhole intrinsics (K.txt).
File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {K[0,0]}
Camera1.fy: {K[1,1]}
Camera1.cx: {K[0,2]}
Camera1.cy: {K[1,2]}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0

Camera.fps: 30
Camera.RGB: 1

Camera.width: {w}
Camera.height: {h}

ORBextractor.nFeatures: {nf}
ORBextractor.scaleFactor: {sf}
ORBextractor.nLevels: {nl}
ORBextractor.iniThFAST: {ini}
ORBextractor.minThFAST: {mn}

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    path = f"configs/orbslam3_c3vd/{seq}_{cfg_name}.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", newline="\n").write(y)
    return path


def argus_traj(seq):
    """The red-team-verified v2 trajectory (positions, one per frame)."""
    p = f"data/redteam/{seq}_traj.npy"
    if not os.path.exists(p):
        raise SystemExit(f"HALT: {p} missing - re-run speqslam/redteam_retrack.py first")
    return np.load(p)


def prep():
    for seq in SEQS:
        for cfg in CFGS:
            make_yaml(seq, cfg)
    pub = {r["seq"]: float(r["redteam_v2_mm"]) for r in csv.DictReader(open(PUBLISHED))}
    print("Argus v2 trajectory verification (gate: match published to within 0.01 mm):")
    for seq in SEQS:
        traj = argus_traj(seq)
        gt = tum_xyz(f"results/c3vd/{seq}_gt.txt")
        n = min(len(gt), len(traj))
        a = horn_ate(gt[:n], traj[:n]) * 1000.0
        ok = abs(a - pub[seq]) <= 0.01
        print(f"  {seq}: stored {a:.3f} mm vs published {pub[seq]:.3f} mm -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("HALT per gate: stored trajectory does not reproduce the published ATE")
    print("prep complete: configs written, Argus artifacts verified")


# ---------- run ----------

def dock(cmd):
    return subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", cmd],
                          capture_output=True, text=True)


def run_one(seq, cfg, tag):
    out = f"{OUT}/{seq}_{cfg}_{tag}"
    if os.path.exists(f"{out}/KeyFrameTrajectory.txt"):
        return "cached"
    os.makedirs(out, exist_ok=True)
    yaml = f"configs/orbslam3_c3vd/{seq}_{cfg}.yaml"
    r = dock(f"mkdir -p /workspace/{out} && cd /workspace/{out} && "
             f"DISPLAY=:1 /ORB_SLAM3/Examples/Monocular/mono_tum "
             f"/ORB_SLAM3/Vocabulary/ORBvoc.txt /workspace/{yaml} "
             f"/workspace/data/c3vd/{seq}_undist > run.log 2>&1; true")
    kf = f"{out}/KeyFrameTrajectory.txt"
    return "ok" if os.path.exists(kf) else f"no trajectory (see {out}/run.log)"


def run():
    chk = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        raise SystemExit(f"container '{CONTAINER}' not found - see gate's rebuild provision")
    if chk.stdout.strip() != "true":
        subprocess.run(["docker", "start", CONTAINER], check=True)
    print("warm-up (discarded, not scored):", run_one("cecum_t1_a", "stock", "warmup"))
    for cfg in CFGS:                      # frozen order: all stock, then all boost
        for seq in SEQS:
            for i in range(1, N_RUNS + 1):
                print(f"{seq} {cfg} run{i}:", run_one(seq, cfg, f"run{i}"), flush=True)


# ---------- analyze ----------

def eval_traj(ts, xyz, gt_ts, gt_xyz, n_frames):
    keep, rows = match_gt(ts, gt_ts, gt_xyz)
    if len(keep) < 3:
        return None
    est, gt = xyz[keep], gt_xyz[rows]
    ate, aligned = ate_rmse_mm(gt, est)
    res = {
        "n_poses": len(keep),
        "completion": (ts[keep][-1] - ts[keep][0]) / ((n_frames - 1) / FPS),
        "ate_mm": ate,
        "const_mm": const_point_mm(gt),
        "rpe1": rpe_mm(gt, aligned, ts[keep], 1),
        "rpe30": rpe_mm(gt, aligned, ts[keep], 30),
    }
    return res


def analyze():
    rows = []
    per_seq = {}
    for seq in SEQS:
        base, n_frames, gt_ts, gt_xyz = seq_info(seq)
        # Argus v2 (full-frame trajectory; completion by construction, guard rate = 0 on
        # the headline sequences per the frozen huber-study artifacts)
        traj = argus_traj(seq)
        ts = np.arange(len(traj)) / FPS
        a = eval_traj(ts, traj, gt_ts, gt_xyz, n_frames)
        a["completion"] = 1.0
        rows.append([seq, "argus_v2", "-", a["n_poses"], 1.0, a["ate_mm"], a["const_mm"],
                     np.median(a["rpe1"]) if len(a["rpe1"]) else "",
                     np.median(a["rpe30"]) if len(a["rpe30"]) else ""])
        per_seq.setdefault(seq, {})["argus"] = a
        for cfg in CFGS:
            runs = []
            for i in range(1, N_RUNS + 1):
                kf = f"{OUT}/{seq}_{cfg}_run{i}/KeyFrameTrajectory.txt"
                cam = f"{OUT}/{seq}_{cfg}_run{i}/CameraTrajectory.txt"
                src = cam if os.path.exists(cam) else kf
                if not os.path.exists(src) or os.path.getsize(src) == 0:
                    rows.append([seq, f"orbslam3_{cfg}", i, 0, 0.0, "", "", "", ""])
                    runs.append(None)
                    continue
                ts_o, xyz_o = load_tum_full(src)
                r = eval_traj(ts_o, xyz_o, gt_ts, gt_xyz, n_frames) if len(ts_o) >= 3 else None
                if r is None:
                    rows.append([seq, f"orbslam3_{cfg}", i, len(ts_o), 0.0, "", "", "", ""])
                else:
                    rows.append([seq, f"orbslam3_{cfg}", i, r["n_poses"], r["completion"],
                                 r["ate_mm"], r["const_mm"],
                                 np.median(r["rpe1"]) if len(r["rpe1"]) else "",
                                 np.median(r["rpe30"]) if len(r["rpe30"]) else ""])
                runs.append(r)
            per_seq[seq][cfg] = runs
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/../orbslam3_baseline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "system", "run", "n_poses", "completion", "ate_mm",
                    "const_point_mm", "rpe1_med_mm", "rpe30_med_mm"])
        w.writerows(rows)

    # summary: per sequence, baseline = its better configuration (per gate)
    summary = []
    for seq in SEQS:
        a = per_seq[seq]["argus"]
        best = None
        for cfg in CFGS:
            ok = [r for r in per_seq[seq][cfg] if r]
            if not ok:
                continue
            med_c = float(np.median([r["completion"] for r in ok]))
            ates = [r["ate_mm"] for r in ok]
            cand = {"cfg": cfg, "completion": med_c, "n_ok": len(ok),
                    "ate_med": float(np.median(ates)),
                    "ate_min": float(min(ates)), "ate_max": float(max(ates)),
                    "const": ok[0]["const_mm"]}
            if best is None or (cand["completion"], -cand["ate_med"]) > (best["completion"], -best["ate_med"]):
                best = cand
        summary.append({"seq": seq, "argus_ate": a["ate_mm"], "argus_const": a["const_mm"],
                        "argus_rpe1": float(np.median(a["rpe1"])),
                        "argus_rpe30": float(np.median(a["rpe30"])) if len(a["rpe30"]) else None,
                        "baseline": best})
    with open(f"{OUT}/../orbslam3_baseline_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "argus_ate_mm", "baseline_best_cfg", "baseline_completion_med",
                    "baseline_runs_with_traj", "baseline_ate_med_mm", "baseline_ate_range",
                    "const_point_mm"])
        for s in summary:
            b = s["baseline"]
            w.writerow([s["seq"], f"{s['argus_ate']:.3f}",
                        b["cfg"] if b else "DNF", f"{b['completion']:.3f}" if b else 0,
                        b["n_ok"] if b else 0,
                        f"{b['ate_med']:.2f}" if b else "",
                        f"{b['ate_min']:.2f}-{b['ate_max']:.2f}" if b else "",
                        f"{s['argus_const']:.2f}"])

    # aggregate + bootstrap over sequences
    ag = [s["argus_ate"] for s in summary]
    bl = [s["baseline"]["ate_med"] for s in summary if s["baseline"]]
    print(f"\nArgus v2 mean ATE {np.mean(ag):.3f} mm, 95% CI {boot_ci(ag)}")
    if bl:
        print(f"ORB-SLAM3 (best cfg, completed seqs only, n={len(bl)}) mean ATE "
              f"{np.mean(bl):.2f} mm, 95% CI {boot_ci(bl)}")
    comp = [s["baseline"]["completion"] if s["baseline"] else 0.0 for s in summary]
    print(f"ORB-SLAM3 median completion per seq: " +
          ", ".join(f"{s['seq']}={c:.2f}" for s, c in zip(summary, comp)))
    figures(summary, per_seq)
    print("analyze complete ->", f"{OUT}/../orbslam3_baseline_summary.csv")


def figures(summary, per_seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    x = np.arange(len(SEQS))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    argus = [s["argus_ate"] for s in summary]
    base = [s["baseline"]["ate_med"] if s["baseline"] else np.nan for s in summary]
    const = [s["argus_const"] for s in summary]
    ax.bar(x - 0.2, argus, 0.38, label="Argus pipeline v2", color="#2b6cb0")
    ax.bar(x + 0.2, base, 0.38, label="ORB-SLAM3 (better config)", color="#c05621")
    for i, s in enumerate(summary):
        b = s["baseline"]
        if b:
            ax.plot([x[i] + 0.01, x[i] + 0.39], [b["ate_min"]] * 2, color="k", lw=0.8)
            ax.plot([x[i] + 0.01, x[i] + 0.39], [b["ate_max"]] * 2, color="k", lw=0.8)
            ax.text(x[i] + 0.2, max(b["ate_max"], 0.1), f" n={b['n_ok']}/5\n cmp {b['completion']:.0%}",
                    ha="center", va="bottom", fontsize=7)
        else:
            ax.text(x[i] + 0.2, 0.05, "DNF", ha="center", va="bottom", fontsize=9, color="#c05621")
    ax.scatter(x, const, marker="_", s=400, color="#444", label="constant-point baseline", zorder=5)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(SEQS, rotation=15)
    ax.set_ylabel("ATE RMSE (mm, log)"); ax.legend(fontsize=8)
    ax.set_title("Pre-registered head-to-head on C3VD ground truth (Sim(3) ATE; median of 5 runs)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/orbslam3_baseline_ate.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for k, cfg in enumerate(CFGS):
        meds = []
        for seq in SEQS:
            ok = [r for r in per_seq[seq][cfg] if r]
            meds.append(np.median([r["completion"] for r in ok]) if ok else 0.0)
        ax.bar(x + (k - 0.5) * 0.4, meds, 0.38, label=f"ORB-SLAM3 {cfg}")
    ax.axhline(1.0, color="#2b6cb0", lw=1.2, label="Argus v2 (tracks every frame)")
    ax.set_xticks(x); ax.set_xticklabels(SEQS, rotation=15); ax.set_ylim(0, 1.1)
    ax.set_ylabel("tracking completion"); ax.legend(fontsize=8)
    ax.set_title("Completion: fraction of sequence the system tracked (median of 5 runs)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/orbslam3_baseline_completion.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    data, labels = [], []
    for seq in SEQS:
        a = per_seq[seq]["argus"]
        data.append(a["rpe1"]); labels.append(f"{seq}\nArgus")
        pooled = []
        b = next((s["baseline"] for s in summary if s["seq"] == seq), None)
        if b:
            for r in per_seq[seq][b["cfg"]]:
                if r is not None and len(r["rpe1"]):
                    pooled.extend(r["rpe1"])
        data.append(np.array(pooled) if pooled else np.array([np.nan]))
        labels.append(f"{seq}\nORB3")
    ax.boxplot([d[~np.isnan(d)] if len(d) else [0] for d in data], tick_labels=labels, showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel("RPE @ 1 frame (mm, log)"); ax.tick_params(labelsize=7)
    ax.set_title("Per-step translational drift on Sim(3)-aligned positions (tracked pairs only)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/orbslam3_baseline_rpe.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not (a.prep or a.run or a.analyze):
        a.prep = a.run = a.analyze = True
    if a.prep:
        prep()
    if a.run:
        run()
    if a.analyze:
        analyze()
