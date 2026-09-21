"""Fuse the full-withdrawal display mesh from the stride-3 guarded trajectory
(the frame-rate experiment's winner: ATE 101 mm vs 168 mm baseline, 3.0% guarded).

Fusion uses all 2800 stride-3 frames for the best surface. The under-seen
statistic keeps its original definition - the 700-frame stride-12 temporal
budget (every 4th of these frames), views < 3, 4 mm occlusion tolerance - so the
number stays comparable across trajectory qualities instead of being inflated by
simply analyzing more frames. Sim(3)-aligned metric frame, disclosed as always.
Outputs: mesh_full.bin, traj_full.bin, results/c3vdv2_screening.json update.
"""
import struct, glob, os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1_s3"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
TRAJ = "arguslite/app/src/main/assets/traj_full.bin"
RESULT = "results/c3vdv2_screening.json"
TARGET_TRIS = 60000

poses = list(np.load(f"{BASE}/poses_est.npy"))
est = np.array([p[:3, 3] for p in poses])
nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]

mg, me = gt.mean(0), est.mean(0)
G, E = gt - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / ((E ** 2).sum() / len(gt)))
ate = np.sqrt((((s * (R @ E.T).T + mg) - gt) ** 2).sum(1).mean()) * 1000
print(f"Sim3: scale {s:.4f} | ATE check {ate:.1f} mm", flush=True)

apose = []
for p in poses:
    M = np.eye(4)
    M[:3, :3] = R @ p[:3, :3]
    M[:3, 3] = s * (R @ (p[:3, 3] - me)) + mg
    apose.append(M)

import open3d as o3d
intr = o3d.camera.PinholeCameraIntrinsic(640, 512, 250.0, 250.0, 320.0, 256.0)
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0015, sdf_trunc=0.006,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
net = demo_app.DepthNet()
budget_deps = {}
t0 = time.time()
for i in range(n):
    img = cv2.imread(und[i])
    dep = net.predict(img) * s
    if i % 4 == 0:
        budget_deps[i] = dep.astype(np.float16)   # the 700-frame stat budget
    col = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, o3d.geometry.Image(dep.astype(np.float32)), depth_scale=1.0,
        depth_trunc=0.20, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(apose[i]))
    if i == 0:
        print(f"depth median {np.median(dep)*1000:.1f} mm", flush=True)
    if i % 400 == 0:
        print(f"fused {i}/{n} ({time.time()-t0:.0f}s)", flush=True)

mesh = vol.extract_triangle_mesh()
print(f"raw mesh {len(mesh.vertices)} verts", flush=True)
mesh = mesh.simplify_quadric_decimation(TARGET_TRIS)
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices)
N = np.asarray(mesh.vertex_normals)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles).astype(np.int32)
e = np.linalg.norm(V[T[:, 0]] - V[T[:, 1]], axis=1) * 1000
print(f"decimated {len(V)} verts / {len(T)} tris | median edge {np.median(e):.2f} mm",
      flush=True)

views = np.zeros(len(V), np.int32)
for i, dep16 in budget_deps.items():
    Tcw = np.linalg.inv(apose[i])
    P = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
    z = P[:, 2]
    ok = z > 1e-4
    u = np.where(ok, 250.0 * P[:, 0] / np.maximum(z, 1e-6) + 320.0, -1)
    v2 = np.where(ok, 250.0 * P[:, 1] / np.maximum(z, 1e-6) + 256.0, -1)
    ok &= (u >= 0) & (u < 639) & (v2 >= 0) & (v2 < 511)
    ui = u.astype(np.int32).clip(0, 639); vi = v2.astype(np.int32).clip(0, 511)
    dm = dep16.astype(np.float32)[vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.004)
    views[ok] += 1
blind_pct = float((views < 3).mean() * 100)
print(f"under-seen {blind_pct:.1f}% (700-frame budget) | views median "
      f"{np.median(views):.0f}", flush=True)

Vmm = (V * 1000.0).astype(np.float32)
with open(MESH, "wb") as f:
    f.write(struct.pack("<ii", len(Vmm), len(T)))
    f.write(Vmm.tobytes())
    f.write(N.astype(np.float32).tobytes())
    f.write(C.tobytes())
    f.write(views.clip(0, 255).astype(np.uint8).tobytes())
    f.write(T.tobytes())

center = (Vmm.min(0) + Vmm.max(0)) / 2
spanmm = (Vmm.max(0) - Vmm.min(0)).max() / 2
traj_mm = np.array([m[:3, 3] for m in apose]) * 1000.0
tn = ((traj_mm - center) / spanmm).astype(np.float32)
with open(TRAJ, "wb") as f:
    f.write(struct.pack("<i", len(tn)))
    f.write(tn.tobytes())

path = float((np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000).sum())
base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
res = json.load(open(RESULT))
res["stride3"] = dict(
    frames=n, ate_mm=round(ate, 1), path_mm=round(path),
    ate_pct=round(100 * ate / path, 2),
    constant_point_baseline_mm=round(base, 1),
    guarded_pairs="83 of 2799 (3.0%)",
    blind_pct_700frame_budget=round(blind_pct, 1),
    verdict="tracking informative at ~9 fps effective; frame rate is the lever")
res["mesh_source"] = "stride-3 guarded trajectory, all 2800 frames fused; "\
    "under-seen stat kept on the 700-frame stride-12 budget for comparability"
res["blind_pct"] = round(blind_pct, 1)
res["mesh_verts"] = len(Vmm)
res["mesh_tris"] = len(T)
json.dump(res, open(RESULT, "w"), indent=1)
print(f"wrote {MESH} ({os.path.getsize(MESH)/1e6:.1f} MB), {TRAJ} "
      f"({os.path.getsize(TRAJ)/1e3:.0f} KB), {RESULT} | span {spanmm:.0f} mm",
      flush=True)
