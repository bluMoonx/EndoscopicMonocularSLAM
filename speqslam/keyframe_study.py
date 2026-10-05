"""Keyframe tracking study: the drift attack that loop closure could not be.

The oracle (manuscript 4.16) proved screening-withdrawal drift cannot be
removed by loop closure (near-linear paths offer no revisits where drift
lives). The remaining lever is accumulating LESS error per meter: track each
frame against a stable KEYFRAME instead of the previous frame, so error
compounds once per keyframe instead of once per frame. direct_vo.run_keyframe
has existed, dormant and unvalidated, since the glare era.

PROTOCOL (stated before any result):
  Phase A (characterization, no tuning): the five C3VD GT sequences, keyframe
     arm with the dormant function's own defaults (new keyframe at >4 mm
     translation or >6 deg rotation from the current keyframe, plus a 30-frame
     cap added for safety), same settings as the deployed tracker elsewhere
     (fine-tuned depth, illum=True, robust=False). Compared against the fresh
     frame-to-frame baseline measured by huber_study.py on identical inputs.
  Phase B (the real target, develop-freeze-test): guarded keyframe tracking
     on the stride-3 withdrawals. DEV = c1_full_t4_v1_s3 only: sweep the
     keyframe translation threshold over {2, 4, 8} mm (rotation 6 deg and the
     30-frame cap fixed). FREEZE the winner. TEST = the four other
     withdrawals (c2_t1, c1_t1, c1_t2, c2_t2 stride-3), untouched.
     Guard semantics unchanged: the per-frame DELTA from the previous pose is
     subject to the same 60 mm / 35 deg physical limits; a rejected frame
     coasts on constant velocity and forces a new keyframe.
  CRITERION: PASS iff median test-withdrawal ATE improvement >= 15% vs the
     recorded frozen-pipeline baselines, AND no test withdrawal worsens by
     more than 5%, AND no test guard rate rises by more than 1 point.
     Phase A is reported either way; a C3VD sequence worsening >10% is a
     disqualifying red flag regardless of Phase B.

Usage: python speqslam/keyframe_study.py c3vd [minutes]
       python speqslam/keyframe_study.py dev  [minutes]
       python speqslam/keyframe_study.py test [minutes]
"""
import os, sys, csv, glob, json, time
import numpy as np, cv2
MODE = sys.argv[1] if len(sys.argv) > 1 else "c3vd"
BUDGET_S = (float(sys.argv[2]) if len(sys.argv) > 2 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import direct_vo as dv
from glare_removal import specular_mask
from c3vdv2_deform import umeyama_ate

KF_ROT_DEG = 6.0
KF_MAX_GAP = 30
KF_TRANS_DEV = (0.002, 0.004, 0.008)   # swept on dev only
# FROZEN after dev (results/keyframe_dev.txt): 2mm -11.2% / 4mm +12% / 8mm +8%
# on c1_t4 - small gaps win; wider baselines cost more per track than the
# drift they save. Dev improvement is below the 15% test bar; the frozen
# test runs regardless and the verdict is whatever the rule says.
KF_TRANS = 0.002
MAX_STEP_M = 0.060
MAX_ROT_DEG = 35.0
DS = 100.0 / 65535.0 / 1000.0
C3VD_SEQS = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]
# recorded frozen-pipeline stride-3 baselines (results/c3vdv2_screening.json)
WD_BASE = {"c1_full_t4_v1_s3": (101.4, 3.0), "c2_full_t1_v1_s3": (92.1, 1.5),
           "c1_full_t1_v1_s3": (114.5, 3.1), "c1_full_t2_v1_s3": (116.5, 4.2),
           "c2_full_t2_v1_s3": (107.6, 2.7)}
TESTS = ["c2_full_t1_v1_s3", "c1_full_t1_v1_s3", "c1_full_t2_v1_s3",
         "c2_full_t2_v1_s3"]

def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def kf_track_c3vd(seq, kf_trans):
    base = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{base}/K.txt")
    imgs = sorted(glob.glob(f"{base}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{base}/depth01_ftc/*.png"))
    g = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    d = lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
    kf_g, kf_d, kf_pose, kf_i = g(imgs[0]), d(deps[0]), np.eye(4), 0
    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    for i in range(1, len(imgs)):
        cur = g(imgs[i])
        T = dv.track_pair(kf_g, kf_d, cur, K, 4, robust=False, depth_scale=DS,
                          illum=True)
        pose = kf_pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
        tr = np.linalg.norm(T[:3, 3])          # metric (depth_scale applied inside)
        if tr > kf_trans or rot_deg(T[:3, :3]) > KF_ROT_DEG or i - kf_i >= KF_MAX_GAP:
            kf_g, kf_d, kf_pose, kf_i = cur, d(deps[i]), pose, i
    return np.array(traj)

def kf_track_withdrawal(seqdir, kf_trans, net, mapu, mapv, K, budget_end):
    """Guarded keyframe tracking over undist/*.jpg; checkpointed; returns
    (positions, rejected, done_flag)."""
    und = sorted(glob.glob(f"{seqdir}/undist/*.jpg"))
    n = len(und)
    ck = f"{seqdir}/kf_ck_{kf_trans:g}.npz"
    if os.path.exists(ck):
        z = np.load(ck)
        poses = list(z["poses"]); rejected = list(z["rejected"])
        kf_i = int(z["kf_i"]); kf_pose = z["kf_pose"]
        prev_delta = z["prev_delta"]
    else:
        poses = [np.eye(4)]; rejected = []
        kf_i, kf_pose, prev_delta = 0, np.eye(4), np.eye(4)
    kf_img = cv2.imread(und[kf_i])
    kf_g = cv2.cvtColor(kf_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    kf_d = net.predict(kf_img)
    kf_glare = specular_mask(kf_img, method="adaptive", k=2.0, lo=225, hi=245,
                             dilate=1)
    start = len(poses)
    for i in range(start, n):
        img = cv2.imread(und[i])
        cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        T = dv.track_pair(kf_g, kf_d, cur, K, 4, robust=False, depth_scale=1.0,
                          glare=kf_glare, illum=True)
        cand = kf_pose @ np.linalg.inv(T)
        delta = np.linalg.inv(poses[-1]) @ cand
        step = float(np.linalg.norm(delta[:3, 3]))
        ang = rot_deg(delta[:3, :3])
        if step > MAX_STEP_M or ang > MAX_ROT_DEG:
            rejected.append(i)
            cand = poses[-1] @ prev_delta
            new_kf = True                       # tracking broke: re-anchor
        else:
            prev_delta = delta.copy()
            new_kf = (np.linalg.norm(T[:3, 3]) > kf_trans
                      or rot_deg(T[:3, :3]) > KF_ROT_DEG or i - kf_i >= KF_MAX_GAP)
        poses.append(cand)
        if new_kf:
            kf_i, kf_pose = i, cand
            kf_g = cur
            kf_d = net.predict(img)
            kf_glare = specular_mask(img, method="adaptive", k=2.0, lo=225,
                                     hi=245, dilate=1)
        if i % 100 == 0 or time.time() > budget_end:
            np.savez(ck, poses=np.array(poses), rejected=np.array(rejected),
                     kf_i=kf_i, kf_pose=kf_pose, prev_delta=prev_delta)
            if time.time() > budget_end:
                print(f"  budget at {i}/{n}; resume me", flush=True)
                return None, None, False
    np.savez(ck, poses=np.array(poses), rejected=np.array(rejected),
             kf_i=kf_i, kf_pose=kf_pose, prev_delta=prev_delta)
    return np.array([p[:3, 3] for p in poses]), rejected, True

def wd_eval(seqdir, est, rejected):
    from c3vdv2_deform import load_poses
    nums = json.load(open(f"{seqdir}/nums.json"))
    gt_all = load_poses(f"{seqdir}/pose.txt")
    n = len(est)
    off = 1 if max(nums[:n]) >= len(gt_all) else 0
    gt = np.array([gt_all[k - off][1] for k in nums[:n]])
    gt = gt - gt[0]
    ate = umeyama_ate(gt, est) * 1000
    guard = 100 * len(rejected) / max(1, n - 1)
    return ate, guard

def tum_positions(path):
    out = []
    for l in open(path):
        v = l.split()
        if len(v) >= 8:
            out.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(out)

if __name__ != "__main__":
    MODE = "imported"

if MODE == "c3vd":
    res = "results/keyframe_c3vd.csv"
    done = set()
    if os.path.exists(res):
        done = {r["seq"] for r in csv.DictReader(open(res))}
    else:
        with open(res, "w", newline="") as f:
            csv.writer(f).writerow(["seq", "ate_base_mm", "ate_kf_mm"])
    hub = {r["seq"]: float(r["ate_base_mm"])
           for r in csv.DictReader(open("results/huber_study.csv"))} \
        if os.path.exists("results/huber_study.csv") else {}
    t0 = time.time()
    for seq in C3VD_SEQS:
        if seq in done:
            continue
        if time.time() - t0 > BUDGET_S:
            print("budget reached; resume me", flush=True)
            break
        gt = tum_positions(f"results/c3vd/{seq}_gt.txt")
        est = kf_track_c3vd(seq, KF_TRANS)
        n = min(len(gt), len(est))
        a_kf = umeyama_ate(gt[:n], est[:n]) * 1000
        a_b = hub.get(seq, float("nan"))
        with open(res, "a", newline="") as f:
            csv.writer(f).writerow([seq, f"{a_b:.3f}", f"{a_kf:.3f}"])
        print(f"{seq}: f2f base {a_b:.3f} | keyframe {a_kf:.3f} mm", flush=True)

elif MODE in ("dev", "test"):
    import demo_app
    import scaramuzza_undistort as su
    from c3vdv2_deform import V2CAL
    mapu, mapv, K = su.build_map(V2CAL)
    net = demo_app.DepthNet()
    budget_end = time.time() + BUDGET_S
    if MODE == "dev":
        seqdir = "data/c3vdv2/c1_full_t4_v1_s3"
        for kt in KF_TRANS_DEV:
            est, rej, ok = kf_track_withdrawal(seqdir, kt, net, mapu, mapv, K,
                                               budget_end)
            if not ok:
                sys.exit(0)
            ate, guard = wd_eval(seqdir, est, rej)
            line = (f"dev kf_trans={1000*kt:g}mm: ATE {ate:.1f} mm "
                    f"(base {WD_BASE['c1_full_t4_v1_s3'][0]}) | guard {guard:.1f}% "
                    f"(base {WD_BASE['c1_full_t4_v1_s3'][1]}%)")
            print(line, flush=True)
            with open("results/keyframe_dev.txt", "a") as f:
                f.write(line + "\n")
    else:
        res = "results/keyframe_test.csv"
        done = set()
        if os.path.exists(res):
            done = {r["seq"] for r in csv.DictReader(open(res))}
        else:
            with open(res, "w", newline="") as f:
                csv.writer(f).writerow(["seq", "ate_base_mm", "ate_kf_mm",
                                        "guard_base_pct", "guard_kf_pct"])
        for seq in TESTS:
            if seq in done:
                continue
            seqdir = f"data/c3vdv2/{seq}"
            est, rej, ok = kf_track_withdrawal(seqdir, KF_TRANS, net, mapu,
                                               mapv, K, budget_end)
            if not ok:
                sys.exit(0)
            ate, guard = wd_eval(seqdir, est, rej)
            b_ate, b_g = WD_BASE[seq]
            with open(res, "a", newline="") as f:
                csv.writer(f).writerow([seq, b_ate, f"{ate:.1f}", b_g,
                                        f"{guard:.1f}"])
            print(f"{seq}: base {b_ate} -> kf {ate:.1f} mm | guard {b_g} -> "
                  f"{guard:.1f}%", flush=True)
        rows = list(csv.DictReader(open(res)))
        if len(rows) == len(TESTS):
            imp = [1 - float(r["ate_kf_mm"]) / float(r["ate_base_mm"])
                   for r in rows]
            gup = [float(r["guard_kf_pct"]) - float(r["guard_base_pct"])
                   for r in rows]
            ok2 = (np.median(imp) >= 0.15 and min(imp) >= -0.05
                   and max(gup) <= 1.0)
            print(f"\nVERDICT (pre-registered): median ATE improvement "
                  f"{100*np.median(imp):+.1f}% | worst {100*min(imp):+.1f}% | "
                  f"max guard rise {max(gup):+.1f}pt -> "
                  f"{'PASS' if ok2 else 'FAIL'}", flush=True)
