"""PIPELINE V2 TRIAL: adopt in-loop Huber weighting (robust=True) across the
deployed tracker configuration - IF it survives staged re-validation of every
tracking benchmark the project publishes. Stop at the first failed gate.

Motivation: the Huber characterization (results/huber_study.csv) showed the
headline C3VD mean would drop 0.90 -> 0.42 mm. robust=False was chosen in the
glare era for experimental reasons and never revisited. Adoption cannot rest
on one benchmark: every published tracking number must be re-earned.

STAGES AND GATES (cheapest first; stop at first failure; stated before any
stage beyond A has run - stage A's data predates this file, so its gate is
deliberately coarse and disclosed as written after that data existed):
  A  C3VD five GT sequences (from huber_study.csv): every Huber ATE must stay
     within the published per-sequence band (<= 1.6 mm) AND the mean must
     improve >= 25%.
  D  deformation controls (from deform_aware.csv, huber column): no rigid
     control worse than +10%.
  C  EndoSLAM zero-shot, all five trajectories, in-vivo mode re-run with
     robust=True: each trajectory's ATE/constant-point-baseline ratio must
     not exceed its recorded value (baseline_check.json) by more than +0.05.
  B  the five stride-3 withdrawals, guarded tracking re-run with robust=True:
     per-sequence ATE within +5% of the recorded frozen-pipeline value,
     median change <= 0 (no net regression at scale), guard rate within +1
     point. Improvement is NOT required here - the v2 case rests on stage A;
     scale must simply not regress.
ADOPTION on all-pass: runners' deployed default flips to robust=True, every
v2 number is published ALONGSIDE the v1 number (v1 stays the record of what
was), the app's Kotlin tracker port is a separate staged change gated on an
on-device measurement, and the native-rate endpoints are re-run post-adoption.

Usage: python speqslam/huber_v2.py [minutes]   (budget-resumable)
"""
import os, sys, csv, glob, json, time
import numpy as np, cv2
BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
from c3vdv2_deform import umeyama_ate, load_poses

STATE = "results/huber_v2_state.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {"stage": "A"}
t0 = time.time()

def save():
    json.dump(state, open(STATE, "w"), indent=1)

def budget():
    if time.time() - t0 > BUDGET_S:
        save()
        print("budget reached; resume me", flush=True)
        sys.exit(0)

def fail(stage, msg):
    state["verdict"] = f"FAIL at stage {stage}: {msg}"
    save()
    print(f"\nPIPELINE V2 VERDICT: {state['verdict']}", flush=True)
    sys.exit(0)

# ---------- stage A ----------
if state["stage"] == "A":
    rows = list(csv.DictReader(open("results/huber_study.csv")))
    hub = [float(r["ate_huber_mm"]) for r in rows]
    base = [float(r["ate_base_mm"]) for r in rows]
    if max(hub) > 1.6:
        fail("A", f"max Huber ATE {max(hub):.2f} mm exceeds the published band")
    imp = 1 - np.mean(hub) / np.mean(base)
    if imp < 0.25:
        fail("A", f"mean improvement only {100*imp:.0f}%")
    state["A"] = {"mean_base": float(np.mean(base)), "mean_hub": float(np.mean(hub))}
    state["stage"] = "D"
    save()
    print(f"stage A PASS: mean {np.mean(base):.3f} -> {np.mean(hub):.3f} mm "
          f"({100*imp:.0f}% better), max {max(hub):.3f}", flush=True)

# ---------- stage D ----------
if state["stage"] == "D":
    rows = list(csv.DictReader(open("results/deform_aware.csv")))
    worst = max(float(r["ate_huber_mm"]) / float(r["ate_base_mm"]) - 1
                for r in rows if r["kind"] == "control")
    if worst > 0.10:
        fail("D", f"a rigid control worsens {100*worst:.0f}%")
    state["D"] = {"worst_control": float(worst)}
    state["stage"] = "C"
    save()
    print(f"stage D PASS: worst deformation-study control {100*worst:+.1f}%",
          flush=True)

# ---------- stage C ----------
if state["stage"] == "C":
    import demo_app, direct_vo as dv
    from glare_removal import specular_mask
    from scipy.spatial.transform import Rotation
    DS = 100.0 / 65535.0 / 1000.0
    rec = json.load(open("results/endoslam/baseline_check.json"))
    if "C" not in state:
        state["C"] = {}
    for traj in range(1, 6):
        key = f"traj{traj}"
        if key in state["C"]:
            continue
        budget()
        PROC = f"data/endoslam/proc/colon4_traj{traj}"
        if not os.path.isdir(f"{PROC}/image01") or \
                len(glob.glob(f"{PROC}/image01/*.jpg")) == 0:
            print(f"[C] re-prepping trajectory {traj}", flush=True)
            env = dict(os.environ, ENDOSLAM_TRAJ=str(traj))
            import subprocess
            subprocess.run([sys.executable, "speqslam/endoslam_pipeline.py",
                            "prep"], env=env, check=True)
        Kl = np.loadtxt(f"{PROC}/K.txt")
        imgs = sorted(glob.glob(f"{PROC}/image01/*.jpg"))
        deps = sorted(glob.glob(f"{PROC}/depth01_ft/*.png"))
        pose = np.eye(4); traj_est = [pose[:3, 3].copy()]
        g0 = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        d0 = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
        m0 = specular_mask(cv2.imread(imgs[0]), method="adaptive", k=2.0,
                           lo=225, hi=245, dilate=1)
        for i in range(1, len(imgs)):
            cur = cv2.imread(imgs[i])
            g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
            T = dv.track_pair(g0, d0, g1, Kl, 4, robust=True, depth_scale=DS,
                              glare=m0, illum=True)
            pose = pose @ np.linalg.inv(T)
            traj_est.append(pose[:3, 3].copy())
            g0 = g1
            d0 = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            m0 = specular_mask(cur, method="adaptive", k=2.0, lo=225, hi=245,
                               dilate=1)
        gt = []
        for l in open(f"results/endoslam/colon4_traj{traj}_gt.txt"):
            v = l.split()
            if len(v) >= 8:
                gt.append([float(v[1]), float(v[2]), float(v[3])])
        gt = np.array(gt)
        n = min(len(gt), len(traj_est))
        ate = umeyama_ate(gt[:n], np.array(traj_est)[:n]) * 1000
        ratio = ate / rec[key]["baseline_mm"]
        state["C"][key] = {"ate_mm": round(ate, 1), "ratio": round(ratio, 3),
                           "recorded_ratio": rec[key]["ratio"]}
        save()
        print(f"[C] {key}: Huber ATE {ate:.1f} mm, ratio {ratio:.2f} "
              f"(recorded {rec[key]['ratio']})", flush=True)
    bad = [k for k, v in state["C"].items()
           if v["ratio"] > v["recorded_ratio"] + 0.05]
    if bad:
        fail("C", f"ratio regressed on {bad}")
    state["stage"] = "B"
    save()
    print("stage C PASS: no EndoSLAM trajectory regressed", flush=True)

# ---------- stage B ----------
WD = {"c1_full_t4_v1_s3": (101.4, 3.0), "c2_full_t1_v1_s3": (92.1, 1.5),
      "c1_full_t1_v1_s3": (114.5, 3.1), "c1_full_t2_v1_s3": (116.5, 4.2),
      "c2_full_t2_v1_s3": (107.6, 2.7)}
if state["stage"] == "B":
    import demo_app, direct_vo as dv
    from glare_removal import specular_mask
    import scaramuzza_undistort as su
    from c3vdv2_deform import V2CAL
    mapu, mapv, K = su.build_map(V2CAL)
    net = demo_app.DepthNet()
    MAX_STEP_M, MAX_ROT_DEG = 0.060, 35.0
    if "B" not in state:
        state["B"] = {}
    for seq, (b_ate, b_g) in WD.items():
        if seq in state["B"]:
            continue
        seqdir = f"data/c3vdv2/{seq}"
        und = sorted(glob.glob(f"{seqdir}/undist/*.jpg"))
        n = len(und)
        ck = f"{seqdir}/poses_hub.npz"
        if os.path.exists(ck):
            z = np.load(ck)
            poses = list(z["poses"]); rejected = list(z["rejected"])
            prev_delta = z["prev_delta"]
        else:
            poses = [np.eye(4)]; rejected = []; prev_delta = np.eye(4)
        start = len(poses)
        if start < n:
            prev_img = cv2.imread(und[start - 1])
            prev_g = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
            prev_d = net.predict(prev_img)
            for i in range(start, n):
                img = cv2.imread(und[i])
                g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                glare = specular_mask(prev_img, method="adaptive", k=2.0,
                                      lo=225, hi=245, dilate=1)
                T = dv.track_pair(prev_g, prev_d, g1, K, 4, robust=True,
                                  depth_scale=1.0, glare=glare, illum=True)
                step = float(np.linalg.norm(T[:3, 3]))
                ang = float(np.degrees(np.arccos(
                    np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
                if step > MAX_STEP_M or ang > MAX_ROT_DEG:
                    T = prev_delta.copy()
                    rejected.append(i)
                else:
                    prev_delta = T.copy()
                poses.append(poses[-1] @ np.linalg.inv(T))
                prev_img, prev_g, prev_d = img, g1, net.predict(img)
                if i % 100 == 0 or time.time() - t0 > BUDGET_S:
                    np.savez(ck, poses=np.array(poses),
                             rejected=np.array(rejected), prev_delta=prev_delta)
                    if time.time() - t0 > BUDGET_S:
                        print(f"[B] {seq}: budget at {i}/{n}; resume me",
                              flush=True)
                        sys.exit(0)
            np.savez(ck, poses=np.array(poses), rejected=np.array(rejected),
                     prev_delta=prev_delta)
        z = np.load(ck)
        poses = z["poses"]; rejected = z["rejected"]
        nums = json.load(open(f"{seqdir}/nums.json"))
        gt_all = load_poses(f"{seqdir}/pose.txt")
        npz = len(poses)
        off = 1 if max(nums[:npz]) >= len(gt_all) else 0
        gt = np.array([gt_all[k - off][1] for k in nums[:npz]])
        gt = gt - gt[0]
        est = np.array([p[:3, 3] for p in poses])
        ate = umeyama_ate(gt, est) * 1000
        guard = 100 * len(rejected) / max(1, npz - 1)
        state["B"][seq] = {"ate_mm": round(float(ate), 1),
                           "guard_pct": round(float(guard), 1),
                           "base_ate": b_ate, "base_guard": b_g}
        save()
        print(f"[B] {seq}: Huber ATE {ate:.1f} (base {b_ate}) | guard "
              f"{guard:.1f}% (base {b_g}%)", flush=True)
    changes = [state["B"][s]["ate_mm"] / state["B"][s]["base_ate"] - 1
               for s in WD]
    gup = [state["B"][s]["guard_pct"] - state["B"][s]["base_guard"] for s in WD]
    if max(changes) > 0.05:
        fail("B", f"a withdrawal ATE worsens {100*max(changes):.0f}%")
    if np.median(changes) > 0:
        fail("B", "median withdrawal ATE regressed")
    if max(gup) > 1.0:
        fail("B", f"guard rate rose {max(gup):.1f} points")
    state["verdict"] = "PASS: all four stages - pipeline v2 (Huber) ADOPTABLE"
    save()
    print(f"\nPIPELINE V2 VERDICT: {state['verdict']}", flush=True)
    print(json.dumps(state["B"], indent=1), flush=True)
