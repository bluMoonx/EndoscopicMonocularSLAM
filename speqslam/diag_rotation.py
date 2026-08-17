"""
Diagnose WHY some sequences over-flag blind spots despite sub-mm position ATE.
Hypothesis: it's rotational drift, not translational. Position-only ATE (Umeyama) is blind to
orientation error. Here we align GT<->VO with the SAME Umeyama rotation, then report BOTH the
position error and the per-frame ORIENTATION error (degrees). If the over-flagging sequences show
large rotation error, the tracker's rotation is the culprit -> fix the tracker, not the detector.
"""
import glob, numpy as np
from scipy.spatial.transform import Rotation as Rot
import sys
sys.path.insert(0, "speqslam")
from validate_coverage import load_poses, umeyama

SEQS = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]

print(f"{'seq':<12}{'pos_ATE_mm':>11}{'rot_med_deg':>12}{'rot_max_deg':>12}{'scale':>8}"
      f"{'true_rot_disp':>14}", flush=True)
for seq in SEQS:
    gt = load_poses(f"results/c3vd/{seq}_gt.txt")
    vo = load_poses(f"results/c3vd/{seq}_raw.txt")
    common = sorted(set(gt) & set(vo))
    Pv = np.array([vo[t][:3, 3] for t in common])
    Pg = np.array([gt[t][:3, 3] for t in common])
    A, s = umeyama(Pv, Pg)
    Ralign = A[:3, :3] / s                      # global rotation part of the Sim(3)
    # position error after alignment
    Pv_al = (A[:3, :3] @ Pv.T).T + A[:3, 3]
    pos_mm = np.sqrt(((Pv_al - Pg) ** 2).sum(1)).mean() * 1000
    # orientation error: aligned VO orientation vs GT orientation, per frame
    ang = []
    for t in common:
        Rv = Ralign @ vo[t][:3, :3]
        Rg = gt[t][:3, :3]
        dR = Rg.T @ Rv
        ang.append(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec())))
    ang = np.array(ang)
    # TRUE tracker rotation error = dispersion of the per-frame global offset R_gt @ R_vo^T.
    # If the offsets are all equal, only ONE global rotation separates VO from GT => tracker is
    # good and the 'error' above is just an unrecoverable alignment DOF on straight trajectories.
    offs = [Rot.from_matrix(gt[t][:3, :3] @ vo[t][:3, :3].T) for t in common]
    mean_off = Rot.from_quat(np.mean([o.as_quat() for o in offs], 0)).inv()
    disp = np.array([np.degrees(np.linalg.norm((mean_off * o).as_rotvec()) * 2) for o in offs])
    print(f"{seq:<12}{pos_mm:>11.3f}{np.median(ang):>12.2f}{ang.max():>12.2f}{s:>8.3f}"
          f"{np.median(disp):>14.2f}", flush=True)

print("\nLast column = dispersion of the global orientation offset (deg).", flush=True)
print("Small => tracker rotation is actually good; the big 'rot error' is an alignment DOF.", flush=True)
