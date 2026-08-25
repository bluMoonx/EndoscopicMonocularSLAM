"""
EndoSLAM cross-dataset experiment: the fully-monocular pipeline on a dataset it has NEVER seen --
different camera (HighCam), different specimen (real ex-vivo porcine colon), different lab.
Attacks the phantom-only limitation head-on.

Per trajectory: undistort with the dataset's own calibration -> fine-tuned depth (ZERO-SHOT: the
model was tuned on the C3VD phantom, never on this data -> honest domain-gap test) -> direct VO ->
Sim(3) ATE against the OptiTrack 6-DOF ground truth. Two tracker conditions: plain (identical
settings to the C3VD runs, for comparability) and in-vivo mode (glare-aware + illumination
compensation, as the Act I evidence recommends for real tissue).

Usage: python speqslam/endoslam_pipeline.py [prep|vo|eval]   (default: all)
"""
import sys, os, glob, subprocess
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

DS = 100.0 / 65535.0 / 1000.0
EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"
SRC = "data/endoslam/Cameras/HighCam/Colon-IV/TumorfreeTrajectory_1"
POSE_XLSX = f"{SRC}/Poses/low_high_pose_colon_test_1_high_images.xlsx"
PROC = "data/endoslam/proc/colon4_traj1"
OUT = "results/endoslam"

# HighCam calibration as shipped (Calibration/cam.txt.txt)
FX, FY, CX, CY, SKEW = 957.411, 959.386, 282.192, 170.731, 5.6242
K = np.array([[FX, SKEW, CX], [0, FY, CY], [0, 0, 1]])
DIST = np.array([0.2533, -0.2085, 0.0, 0.0, 0.0])


def prep():
    os.makedirs(f"{PROC}/image01", exist_ok=True)
    os.makedirs(f"{PROC}/depth01_ft", exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_excel(POSE_XLSX)
    frames = sorted(glob.glob(f"{SRC}/Frames/*.jpg"))
    by_num = {int(os.path.basename(f).split("_")[1].split(".")[0]): f for f in frames}
    rows = [r for _, r in df.iterrows() if int(r["ImageFrame"]) in by_num]
    print(f"[prep] {len(rows)} pose-matched frames", flush=True)

    # ground truth -> TUM (timestamps = sequential index / 30, same convention as the VO)
    with open(f"{OUT}/colon4_traj1_gt.txt", "w") as f:
        for i, r in enumerate(rows):
            f.write(f"{i/30.0:.6f} {r['trans_x']:.6f} {r['trans_y']:.6f} {r['trans_z']:.6f} "
                    f"{r['quot_x']:.6f} {r['quot_y']:.6f} {r['quot_z']:.6f} {r['quot_w']:.6f}\n")

    # undistort frames + zero-shot fine-tuned depth
    import demo_app
    net = demo_app.DepthNet()
    print(f"[prep] depth model: {net.name}", flush=True)
    valid_fracs = []
    for i, r in enumerate(rows):
        bgr = cv2.imread(by_num[int(r["ImageFrame"])])
        und = cv2.undistort(bgr, K, DIST)
        cv2.imwrite(f"{PROC}/image01/{i:06d}.jpg", und)
        d = net.predict(und)                      # metric decode, zero-shot
        valid_fracs.append((d > 0).mean())
        cv2.imwrite(f"{PROC}/depth01_ft/{i:06d}.png", np.clip(d / DS, 0, 65535).astype(np.uint16))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    np.savetxt(f"{PROC}/K.txt", K)
    print(f"[prep] done. depth valid fraction: mean {np.mean(valid_fracs):.2%} "
          f"min {np.min(valid_fracs):.2%}", flush=True)


def vo():
    from direct_vo import run as vo_run
    Kl = np.loadtxt(f"{PROC}/K.txt")
    print("[vo] plain (C3VD-identical settings)", flush=True)
    vo_run(PROC, PROC, f"{OUT}/colon4_traj1_plain.txt", Kl, robust=False, depth_scale=DS,
           depth_sub="depth01_ft")
    print("[vo] in-vivo mode (glare-aware + illumination compensation)", flush=True)
    from glare_removal import specular_mask
    # generate per-frame glare masks once, reuse inside a wrapped run
    import direct_vo as dv
    imgs = sorted(glob.glob(f"{PROC}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{PROC}/depth01_ft/*.png"))
    pose = np.eye(4); lines = ["0.0 0 0 0 0 0 0 1"]
    prev = cv2.imread(imgs[0]); prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    prev_d = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).astype(np.float32)
    prev_m = specular_mask(prev, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
    from scipy.spatial.transform import Rotation
    for i in range(1, len(imgs)):
        cur = cv2.imread(imgs[i]); cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        T = dv.track_pair(prev_g, prev_d, cur_g, Kl, 4, robust=False, depth_scale=DS,
                          glare=prev_m, illum=True)
        pose = pose @ np.linalg.inv(T)
        q = Rotation.from_matrix(pose[:3, :3]).as_quat(); t = pose[:3, 3]
        lines.append(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        prev_g, prev_d = cur_g, cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        prev_m = specular_mask(cur, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        if i % 100 == 0:
            print(f"  {i}/{len(imgs)}", flush=True)
    open(f"{OUT}/colon4_traj1_invivo.txt", "w").write("\n".join(lines) + "\n")
    print("[vo] done", flush=True)


def evaluate():
    for cond in ["plain", "invivo"]:
        est = f"{OUT}/colon4_traj1_{cond}.txt"
        if not os.path.exists(est):
            continue
        o = subprocess.run([EVO, "tum", f"{OUT}/colon4_traj1_gt.txt", est, "-as", "--no_warnings"],
                           capture_output=True, text=True).stdout
        for l in o.splitlines():
            if l.strip().lower().startswith(("rmse", "mean", "median")):
                print(f"  {cond:<7} {l.strip()}", flush=True)
    # context: trajectory extent
    G = np.loadtxt(f"{OUT}/colon4_traj1_gt.txt")[:, 1:4]
    span = np.linalg.norm(G.max(0) - G.min(0)) * 1000
    path = np.linalg.norm(np.diff(G, axis=0), axis=1).sum() * 1000
    print(f"  GT path length {path:.1f}mm, spatial span {span:.1f}mm", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("prep", "all"):
        prep()
    if step in ("vo", "all"):
        vo()
    if step in ("eval", "all"):
        evaluate()
