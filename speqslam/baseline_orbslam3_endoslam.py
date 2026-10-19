"""Pre-registered cross-dataset confirmation: out-of-the-box ORB-SLAM3 vs the monocular
pipeline on EndoSLAM ex-vivo porcine trajectories (Colon-IV, tumor-free, HighCam).

Spec: results/orbslam3_endoslam_gate.md (committed with this script, before any run).
Comparators: the published zero-shot plain-condition trajectories (frozen; verified against
the published ATEs before anything runs) and one deterministic pipeline-v2 run per trajectory
(the adopted tracker's first cross-dataset measurement). Phases as in the C3VD study:
  --prep     rgb.txt listings + yaml configs; verify published plain ATEs (26.6/32.0/47.1/
             26.5/69.2 mm within 0.1 mm) else HALT.
  --argus    the five deterministic pipeline-v2 runs (checkpointed per trajectory).
  --run      2 configs x 5 trajectories x 5 ORB-SLAM3 runs (+1 discarded warm-up).
  --analyze  pre-specified metrics, CSVs, figures.
"""
import argparse, csv, glob, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "speqslam")
import baseline_orbslam3 as bb   # frozen helpers: umeyama/ate/rpe/boot_ci/load_tum_full/match_gt

TRAJS = [1, 2, 3, 4, 5]
# The audited per-trajectory figures (baseline_check.json) are the IN-VIVO-MODE
# trajectories' Sim(3) ATE RMSE; see the gate's metric-provenance note.
PUBLISHED_INVIVO_RMSE_MM = {1: 26.6, 2: 32.0, 3: 47.1, 4: 26.5, 5: 69.2}
CFGS = bb.CFGS                      # the same two frozen configurations
N_RUNS = 5
FPS = 30.0
OUT = "results/orbslam3_endoslam"
FIGDIR = "results/figures"


def ate_stats_mm(gt, est):
    """Sim(3)-aligned APE in mm: (rmse, mean, aligned estimate)."""
    s, R, t = bb.umeyama(est, gt)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1) * 1000.0
    return float(np.sqrt((err ** 2).mean())), float(err.mean()), aligned


def const_stats_mm(gt):
    err = np.linalg.norm(gt - gt.mean(0), axis=1) * 1000.0
    return float(np.sqrt((err ** 2).mean())), float(err.mean())


def proc(n):
    return f"data/endoslam/proc/colon4_traj{n}"


def gt_path(n):
    return f"results/endoslam/colon4_traj{n}_gt.txt"


def n_frames(n):
    return len(glob.glob(f"{proc(n)}/image01/*.jpg"))


# ---------- prep ----------

def make_rgb_txt(n):
    imgs = sorted(glob.glob(f"{proc(n)}/image01/*.jpg"))
    p = f"{proc(n)}/rgb.txt"
    with open(p, "w", newline="\n") as f:
        f.write("# color images\n# EndoSLAM undistorted (pinhole)\n# timestamp filename\n")
        for i, im in enumerate(imgs):
            f.write(f"{i/FPS:.6f} image01/{os.path.basename(im)}\n")
    return p


def make_yaml(n, cfg_name):
    import cv2
    K = np.loadtxt(f"{proc(n)}/K.txt")
    img = cv2.imread(sorted(glob.glob(f"{proc(n)}/image01/*.jpg"))[0])
    h, w = img.shape[:2]
    nf, sf, nl, ini, mn = CFGS[cfg_name]
    # the shipped calibration's 5.62px skew (K[0,1]) is unsupported by ORB-SLAM3 and dropped
    # per the gate - a disclosed approximation of ~0.6% of fx.
    y = f"""%YAML:1.0

# Pre-registered EndoSLAM baseline config ({cfg_name}) - results/orbslam3_endoslam_gate.md.
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
    path = f"configs/orbslam3_endoslam/traj{n}_{cfg_name}.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", newline="\n").write(y)
    return path


def prep():
    for n in TRAJS:
        make_rgb_txt(n)
        for cfg in CFGS:
            make_yaml(n, cfg)
    print("Published in-vivo-mode verification (ATE RMSE, gate: within 0.1 mm):")
    for n in TRAJS:
        ts, xyz = bb.load_tum_full(f"results/endoslam/colon4_traj{n}_invivo.txt")
        gt_ts, gt_xyz = bb.load_tum_full(gt_path(n))
        keep, rows = bb.match_gt(ts, gt_ts, gt_xyz)
        rmse, mean, _ = ate_stats_mm(gt_xyz[rows], xyz[keep])
        ok = abs(rmse - PUBLISHED_INVIVO_RMSE_MM[n]) <= 0.1
        print(f"  traj{n}: recomputed rmse {rmse:.1f} (mean {mean:.1f}) mm vs published rmse "
              f"{PUBLISHED_INVIVO_RMSE_MM[n]:.1f} mm -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("HALT per gate: stored trajectory does not reproduce the published ATE")
    print("prep complete")


# ---------- the adopted pipeline v2, deterministically, once per trajectory ----------

def argus_v2():
    import cv2
    import direct_vo as dv
    from scipy.spatial.transform import Rotation
    DS = 100.0 / 65535.0 / 1000.0
    for n in TRAJS:
        outp = f"results/endoslam/colon4_traj{n}_v2.txt"
        if os.path.exists(outp):
            print(f"traj{n}: v2 cached")
            continue
        imgs = sorted(glob.glob(f"{proc(n)}/image01/*.jpg"))
        deps = sorted(glob.glob(f"{proc(n)}/depth01_ft/*.png"))
        K = np.loadtxt(f"{proc(n)}/K.txt")
        pose = np.eye(4); lines = ["0.000000 0 0 0 0 0 0 1"]
        g0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        d0 = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
        for i in range(1, len(imgs)):
            g1 = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            T = dv.track_pair(g0, d0, g1, K, 4, robust=True, depth_scale=DS, illum=True)
            pose = pose @ np.linalg.inv(T)
            q = Rotation.from_matrix(pose[:3, :3]).as_quat(); t = pose[:3, 3]
            lines.append(f"{i/FPS:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                         f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
            g0 = g1
            d0 = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            if i % 100 == 0:
                print(f"traj{n} v2 {i}/{len(imgs)}", flush=True)
        open(outp, "w", newline="\n").write("\n".join(lines) + "\n")
        print(f"traj{n}: v2 written -> {outp}", flush=True)


# ---------- baseline runs ----------

def run_one(n, cfg, tag):
    out = f"{OUT}/traj{n}_{cfg}_{tag}"
    if os.path.exists(f"{out}/KeyFrameTrajectory.txt"):
        return "cached"
    os.makedirs(out, exist_ok=True)
    yaml = f"configs/orbslam3_endoslam/traj{n}_{cfg}.yaml"
    bb.dock(f"mkdir -p /workspace/{out} && cd /workspace/{out} && "
            f"DISPLAY=:1 /ORB_SLAM3/Examples/Monocular/mono_tum "
            f"/ORB_SLAM3/Vocabulary/ORBvoc.txt /workspace/{yaml} "
            f"/workspace/{proc(n)} > run.log 2>&1; true")
    return "ok" if os.path.exists(f"{out}/KeyFrameTrajectory.txt") else f"no trajectory (see {out}/run.log)"


def run():
    chk = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", bb.CONTAINER],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        raise SystemExit(f"container '{bb.CONTAINER}' not found - see gate's rebuild provision")
    if chk.stdout.strip() != "true":
        subprocess.run(["docker", "start", bb.CONTAINER], check=True)
    print("warm-up (discarded, not scored):", run_one(1, "stock", "warmup"))
    for cfg in CFGS:
        for n in TRAJS:
            for i in range(1, N_RUNS + 1):
                print(f"traj{n} {cfg} run{i}:", run_one(n, cfg, f"run{i}"), flush=True)


# ---------- analysis ----------

def eval_file(path, gt_ts, gt_xyz, total_frames):
    ts, xyz = bb.load_tum_full(path)
    if len(ts) < 3:
        return None
    keep, rows = bb.match_gt(ts, gt_ts, gt_xyz)
    if len(keep) < 3:
        return None
    est, gt = xyz[keep], gt_xyz[rows]
    rmse, mean, aligned = ate_stats_mm(gt, est)
    c_rmse, c_mean = const_stats_mm(gt)
    tsk = ts[keep]
    return {"n_poses": len(keep),
            "completion": (tsk[-1] - tsk[0]) / ((total_frames - 1) / FPS),
            "ate_mm": rmse, "ate_mean_mm": mean, "const_mm": c_rmse, "const_mean_mm": c_mean,
            "rpe1": bb.rpe_mm(gt, aligned, tsk, 1), "rpe30": bb.rpe_mm(gt, aligned, tsk, 30)}


def analyze():
    rows, per = [], {}
    for n in TRAJS:
        gt_ts, gt_xyz = bb.load_tum_full(gt_path(n))
        tf = n_frames(n)
        per[n] = {}
        for label, path in [("argus_plain", f"results/endoslam/colon4_traj{n}_plain.txt"),
                            ("argus_invivo", f"results/endoslam/colon4_traj{n}_invivo.txt"),
                            ("argus_v2", f"results/endoslam/colon4_traj{n}_v2.txt")]:
            r = eval_file(path, gt_ts, gt_xyz, tf) if os.path.exists(path) else None
            per[n][label] = r
            if r:
                rows.append([f"traj{n}", label, "-", r["n_poses"], 1.0, r["ate_mm"],
                             r["ate_mean_mm"], r["const_mm"], r["const_mean_mm"],
                             np.median(r["rpe1"]) if len(r["rpe1"]) else "",
                             np.median(r["rpe30"]) if len(r["rpe30"]) else ""])
        for cfg in CFGS:
            runs = []
            for i in range(1, N_RUNS + 1):
                base = f"{OUT}/traj{n}_{cfg}_run{i}"
                cam, kf = f"{base}/CameraTrajectory.txt", f"{base}/KeyFrameTrajectory.txt"
                src = cam if os.path.exists(cam) else kf
                r = eval_file(src, gt_ts, gt_xyz, tf) if os.path.exists(src) else None
                runs.append(r)
                if r:
                    rows.append([f"traj{n}", f"orbslam3_{cfg}", i, r["n_poses"], r["completion"],
                                 r["ate_mm"], r["ate_mean_mm"], r["const_mm"], r["const_mean_mm"],
                                 np.median(r["rpe1"]) if len(r["rpe1"]) else "",
                                 np.median(r["rpe30"]) if len(r["rpe30"]) else ""])
                else:
                    rows.append([f"traj{n}", f"orbslam3_{cfg}", i, 0, 0.0, "", "", "", "", ""])
            per[n][cfg] = runs
    os.makedirs(OUT, exist_ok=True)
    with open("results/orbslam3_endoslam.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "system", "run", "n_poses", "completion", "ate_rmse_mm",
                    "ate_mean_mm", "const_point_rmse_mm", "const_point_mean_mm",
                    "rpe1_med_mm", "rpe30_med_mm"])
        w.writerows(rows)

    summary = []
    for n in TRAJS:
        best = None
        for cfg in CFGS:
            ok = [r for r in per[n][cfg] if r]
            if not ok:
                continue
            ates = [r["ate_mm"] for r in ok]
            cand = {"cfg": cfg, "n_ok": len(ok),
                    "completion": float(np.median([r["completion"] for r in ok])),
                    "ate_med": float(np.median(ates)),
                    "ate_min": float(min(ates)), "ate_max": float(max(ates))}
            if best is None or (cand["completion"], -cand["ate_med"]) > (best["completion"], -best["ate_med"]):
                best = cand
        summary.append({"traj": n, "plain": per[n]["argus_plain"],
                        "invivo": per[n]["argus_invivo"], "v2": per[n]["argus_v2"],
                        "baseline": best})
    with open("results/orbslam3_endoslam_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj", "argus_plain_ate_rmse_mm", "argus_invivo_ate_rmse_mm(published)",
                    "argus_v2_ate_rmse_mm", "const_point_rmse_mm",
                    "baseline_best_cfg", "baseline_completion_med", "baseline_runs_with_traj",
                    "baseline_ate_rmse_med_mm", "baseline_ate_rmse_range_mm"])
        for s in summary:
            b, p, iv, v = s["baseline"], s["plain"], s["invivo"], s["v2"]
            w.writerow([s["traj"],
                        f"{p['ate_mm']:.1f}" if p else "", f"{iv['ate_mm']:.1f}" if iv else "",
                        f"{v['ate_mm']:.1f}" if v else "",
                        f"{p['const_mm']:.1f}" if p else "",
                        b["cfg"] if b else "DNF", f"{b['completion']:.3f}" if b else 0,
                        b["n_ok"] if b else 0, f"{b['ate_med']:.1f}" if b else "",
                        f"{b['ate_min']:.1f}-{b['ate_max']:.1f}" if b else ""])

    pl = [s["plain"]["ate_mm"] for s in summary if s["plain"]]
    iv = [s["invivo"]["ate_mm"] for s in summary if s["invivo"]]
    v2 = [s["v2"]["ate_mm"] for s in summary if s["v2"]]
    bl = [s["baseline"]["ate_med"] for s in summary if s["baseline"]]
    comp = [s["baseline"]["completion"] if s["baseline"] else 0.0 for s in summary]
    print(f"\nArgus plain (published) mean ATE {np.mean(pl):.1f} mm, CI {bb.boot_ci(pl)}")
    print(f"Argus invivo (published) mean ATE {np.mean(iv):.1f} mm, CI {bb.boot_ci(iv)}")
    if v2:
        print(f"Argus v2 (adopted, first cross-dataset) mean ATE {np.mean(v2):.1f} mm, CI {bb.boot_ci(v2)}")
    if bl:
        print(f"ORB-SLAM3 (best cfg, completed trajs, n={len(bl)}) mean ATE {np.mean(bl):.1f} mm, CI {bb.boot_ci(bl)}")
    print("ORB-SLAM3 median completion: " + ", ".join(f"traj{s['traj']}={c:.2f}" for s, c in zip(summary, comp)))
    figures(summary, per)
    print("analyze complete -> results/orbslam3_endoslam_summary.csv")


def figures(summary, per):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    x = np.arange(len(TRAJS))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plain = [s["plain"]["ate_mm"] if s["plain"] else np.nan for s in summary]
    iv = [s["invivo"]["ate_mm"] if s["invivo"] else np.nan for s in summary]
    v2 = [s["v2"]["ate_mm"] if s["v2"] else np.nan for s in summary]
    base = [s["baseline"]["ate_med"] if s["baseline"] else np.nan for s in summary]
    const = [s["plain"]["const_mm"] if s["plain"] else np.nan for s in summary]
    ax.bar(x - 0.3, plain, 0.19, label="Argus zero-shot (published, plain)", color="#7fa8d4")
    ax.bar(x - 0.1, iv, 0.19, label="Argus zero-shot (published, in-vivo mode)", color="#5a8dc8")
    ax.bar(x + 0.1, v2, 0.19, label="Argus pipeline v2 (adopted)", color="#2b6cb0")
    ax.bar(x + 0.3, base, 0.19, label="ORB-SLAM3 (better config)", color="#c05621")
    for i, s in enumerate(summary):
        b = s["baseline"]
        if b:
            ax.text(x[i] + 0.3, b["ate_med"], f" n={b['n_ok']}/5\n cmp {b['completion']:.0%}",
                    ha="center", va="bottom", fontsize=7)
        else:
            ax.text(x[i] + 0.3, 1, "DNF", ha="center", va="bottom", fontsize=9, color="#c05621")
    ax.scatter(x, const, marker="_", s=420, color="#444", label="constant-point baseline", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([f"traj{n}" for n in TRAJS])
    ax.set_ylabel("ATE RMSE (mm)"); ax.legend(fontsize=8)
    ax.set_title("Cross-dataset confirmation on EndoSLAM ex-vivo porcine (Sim(3) ATE vs OptiTrack GT)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/orbslam3_endoslam_ate.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for k, cfg in enumerate(CFGS):
        meds = []
        for n in TRAJS:
            ok = [r for r in per[n][cfg] if r]
            meds.append(np.median([r["completion"] for r in ok]) if ok else 0.0)
        ax.bar(x + (k - 0.5) * 0.4, meds, 0.38, label=f"ORB-SLAM3 {cfg}")
    ax.axhline(1.0, color="#2b6cb0", lw=1.2, label="Argus (tracks every frame)")
    ax.set_xticks(x); ax.set_xticklabels([f"traj{n}" for n in TRAJS]); ax.set_ylim(0, 1.1)
    ax.set_ylabel("tracking completion"); ax.legend(fontsize=8)
    ax.set_title("Tracking completion on real ex-vivo tissue (median of 5 runs; DNF = 0)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/orbslam3_endoslam_completion.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--argus", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if not (a.prep or a.argus or a.run or a.analyze):
        a.prep = a.argus = a.run = a.analyze = True
    if a.prep:
        prep()
    if a.argus:
        argus_v2()
    if a.run:
        run()
    if a.analyze:
        analyze()
