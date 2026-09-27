"""Coverage validation: how much of the under-seen statistic is real coverage
and how much is tracking error?

For each full withdrawal, the same fused surface gets its per-vertex view count
computed twice under the identical protocol (700-frame stride-12 temporal
budget, 4 mm occlusion tolerance): once with the pipeline's Sim(3)-aligned
estimated poses, once with the recorded ground-truth poses. The difference in
the under-seen fraction is the coverage error attributable to tracking - the
caveat the app and documents have carried as words, here turned into a number,
at n = 3.

GT rotation convention is auto-checked per sequence: both plausible conventions
are projected on one frame and the one agreeing with the depth map's working
range is used (printed for the record).

Usage: python speqslam/coverage_validation.py <base_s3_dir>
Writes/updates results/coverage_validation.json.
"""
import os, sys, glob, json, time
import numpy as np, cv2

BASE = sys.argv[1].rstrip("/\\")
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

SEQ = os.path.basename(BASE).replace("_s3", "")
RES = "results/coverage_validation.json"
TOL = 0.004
BUDGET_STRIDE = 4          # every 4th stride-3 frame = the stride-12 budget

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
raw = load_poses(f"{BASE}/pose.txt")
t0g = raw[nums[0]][1]
gt_pos = np.array([raw[k][1] for k in nums]) - t0g
gt_R = [raw[k][0] for k in nums]

poses = list(np.load(f"{BASE}/poses_est.npy"))
est = np.array([p[:3, 3] for p in poses])

# Sim(3) est -> GT (the standard alignment)
mg, me = gt_pos.mean(0), est.mean(0)
G, E = gt_pos - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt_pos))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / ((E ** 2).sum() / len(gt_pos)))
apose = []
for p in poses:
    M = np.eye(4)
    M[:3, :3] = R @ p[:3, :3]
    M[:3, 3] = s * (R @ (p[:3, 3] - me)) + mg
    apose.append(M)

# Two SELF-CONSISTENT instruments are compared (per the ablation's logic):
# each pose set fuses its own surface with the same depth maps, and each
# surface's coverage is counted with its own poses. Counting one surface
# from the other's viewpoints conflates surface displacement with coverage
# (the first version of this script measured exactly that artifact).
import open3d as o3d
intr = o3d.camera.PinholeCameraIntrinsic(640, 512, 250.0, 250.0, 320.0, 256.0)

def gt_pose_list(transpose):
    out = []
    for i in range(n):
        M = np.eye(4)
        M[:3, :3] = gt_R[i].T if transpose else gt_R[i]
        M[:3, 3] = gt_pos[i]
        out.append(M)
    return out

# pick the GT rotation convention with a cheap single-frame projection test
net = demo_app.DepthNet()
probe_img = cv2.imread(und[n // 2])
probe_dep = net.predict(probe_img) * s
scores = {}
for tr in (False, True):
    pl = gt_pose_list(tr)
    # forward-project the probe depth into the next GT camera; count in-range
    ys, xs = np.mgrid[0:512:8, 0:640:8]
    z = probe_dep[ys, xs]
    ok0 = z > 0
    pts = np.stack([(xs - 320.0) / 250.0 * z, (ys - 256.0) / 250.0 * z, z], -1)[ok0]
    Pw = (pl[n // 2][:3, :3] @ pts.T).T + pl[n // 2][:3, 3]
    Tcw = np.linalg.inv(pl[n // 2 + BUDGET_STRIDE])
    Pc = (Tcw[:3, :3] @ Pw.T).T + Tcw[:3, 3]
    scores[tr] = int(((Pc[:, 2] > 0.003) & (Pc[:, 2] < 0.20)).sum())
use_tr = max(scores, key=scores.get)
print(f"GT rotation convention: transpose={use_tr} "
      f"({scores[use_tr]} vs {scores[not use_tr]})", flush=True)
gposes = gt_pose_list(use_tr)

vol_p = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0015, sdf_trunc=0.006,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
vol_g = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0015, sdf_trunc=0.006,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
deps = {}
t0 = time.time()
for i in range(n):
    img = cv2.imread(und[i])
    dep = net.predict(img) * s
    if i % BUDGET_STRIDE == 0:
        deps[i] = dep.astype(np.float16)
    col = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, o3d.geometry.Image(dep.astype(np.float32)), depth_scale=1.0,
        depth_trunc=0.20, convert_rgb_to_intensity=False)
    vol_p.integrate(rgbd, intr, np.linalg.inv(apose[i]))
    vol_g.integrate(rgbd, intr, np.linalg.inv(gposes[i]))
    if i % 600 == 0:
        print(f"fuse {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

def surface_of(vol):
    mesh = vol.extract_triangle_mesh()
    mesh = mesh.simplify_quadric_decimation(60000)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices)

def count_views(V, pose_list):
    views = np.zeros(len(V), np.int32)
    for i in sorted(deps.keys()):
        Tcw = np.linalg.inv(pose_list[i])
        P = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
        z = P[:, 2]
        ok = z > 1e-4
        u = np.where(ok, 250.0 * P[:, 0] / np.maximum(z, 1e-6) + 320.0, -1)
        v2 = np.where(ok, 250.0 * P[:, 1] / np.maximum(z, 1e-6) + 256.0, -1)
        ok &= (u >= 0) & (u < 639) & (v2 >= 0) & (v2 < 511)
        ui = u.astype(np.int32).clip(0, 639)
        vi = v2.astype(np.int32).clip(0, 511)
        dm = deps[i].astype(np.float32)[vi, ui]
        ok &= (dm > 0) & (z <= dm + TOL)
        views[ok] += 1
    return views

Vp = surface_of(vol_p)
Vg = surface_of(vol_g)
print(f"surfaces: pipeline {len(Vp)} verts, GT-pose {len(Vg)} verts", flush=True)
v_pipe = count_views(Vp, apose)
v_gt = count_views(Vg, gposes)
u_pipe = 100 * (v_pipe < 3).mean()
u_gt = 100 * (v_gt < 3).mean()
print(f"{SEQ}: under-seen {u_pipe:.1f}% (pipeline poses) vs {u_gt:.1f}% (GT poses) "
      f"| tracking-attributable gap {u_pipe - u_gt:+.1f} points", flush=True)

res = json.load(open(RES)) if os.path.exists(RES) else {}
res[SEQ] = dict(verts_pipeline=len(Vp), verts_gtpose=len(Vg),
                budget_frames=len(deps),
                underseen_pipeline_pct=round(float(u_pipe), 1),
                underseen_gtpose_pct=round(float(u_gt), 1),
                tracking_gap_points=round(float(u_pipe - u_gt), 1),
                gt_rotation_transposed=bool(use_tr))
res["note"] = ("two self-consistent instruments: each pose set fuses its own "
               "surface from the same depth maps and counts its own coverage "
               "(700-frame stride-12 budget, 4 mm tolerance). The GT-pose "
               "instrument isolates what the statistic reads with perfect "
               "tracking; the gap is the tracking-attributable coverage error, "
               "and the GT-pose residual reflects genuine non-inspection plus "
               "depth error.")
json.dump(res, open(RES, "w"), indent=1)
print(f"recorded in {RES}", flush=True)
