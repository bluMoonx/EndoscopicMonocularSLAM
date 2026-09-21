"""Fuse the full-withdrawal display mesh from the divergence-guarded trajectory.

First-pass forensics: 24 unguarded pairs made the tracker diverge (steps up to
457 m at 22 mm scene depth), the resulting mesh was micron-scale dust, and the
Umeyama-collapsed "3.5% ATE" was no better than a constant-point predictor. The
trajectory is re-estimated by c3vdv2_screening_track2.py with a physical step
guard; this script fuses depth + guarded poses in the Sim(3)-aligned GT frame
(one global scale/rotation, the standard monocular convention, disclosed) and
recomputes per-vertex view counts at the intended 4 mm occlusion tolerance.
Outputs: mesh_full.bin (60k tris), traj_full.bin, results JSON update.
"""
import struct, glob, os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
TRAJ = "arguslite/app/src/main/assets/traj_full.bin"
RESULT = "results/c3vdv2_screening.json"
TARGET_TRIS = 60000

poses = list(np.load(f"{BASE}/poses_est2.npy"))
est = np.array([p[:3, 3] for p in poses])
nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]

# global Sim(3), est -> GT (Umeyama; the exact construction umeyama_ate uses)
mg, me = gt.mean(0), est.mean(0)
G, E = gt - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / ((E ** 2).sum() / len(gt)))
ate = np.sqrt((((s * (R @ E.T).T + mg) - gt) ** 2).sum(1).mean()) * 1000
print(f"Sim3: scale {s:.6f} | ATE check {ate:.1f} mm", flush=True)

# aligned camera-to-world poses: rotation R.R_i, translation s.R.(t_i - me) + mg
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
deps = []
t0 = time.time()
for i in range(n):
    img = cv2.imread(und[i])
    dep = net.predict(img) * s          # metric meters in the aligned frame
    deps.append(dep.astype(np.float16))
    col = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, o3d.geometry.Image(dep.astype(np.float32)), depth_scale=1.0,
        depth_trunc=0.20, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(apose[i]))
    if i == 0:
        print(f"depth median {np.median(dep)*1000:.1f} mm (sanity: tens of mm)",
              flush=True)
    if i % 100 == 0:
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

# per-vertex views at the intended tolerance (4 mm), in the same metric frame
views = np.zeros(len(V), np.int32)
for i in range(n):
    Tcw = np.linalg.inv(apose[i])
    P = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
    z = P[:, 2]
    ok = z > 1e-4
    u = np.where(ok, 250.0 * P[:, 0] / np.maximum(z, 1e-6) + 320.0, -1)
    v2 = np.where(ok, 250.0 * P[:, 1] / np.maximum(z, 1e-6) + 256.0, -1)
    ok &= (u >= 0) & (u < 639) & (v2 >= 0) & (v2 < 511)
    ui = u.astype(np.int32).clip(0, 639); vi = v2.astype(np.int32).clip(0, 511)
    dm = deps[i].astype(np.float32)[vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.004)
    views[ok] += 1
blind_pct = float((views < 3).mean() * 100)
print(f"under-seen {blind_pct:.1f}% | views median {np.median(views):.0f}", flush=True)

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
traj_mm = (s * (R @ (est - me).T).T + mg) * 1000.0
tn = ((traj_mm - center) / spanmm).astype(np.float32)
with open(TRAJ, "wb") as f:
    f.write(struct.pack("<i", len(tn)))
    f.write(tn.tobytes())

path = float((np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000).sum())
base = float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())) * 1000
res = json.load(open(RESULT))
res["ate_mm"] = round(ate, 1)
res["ate_pct"] = round(100 * ate / path, 2)
res["constant_point_baseline_mm"] = round(base, 1)
res["tracking"] = "divergence guard active (see c3vdv2_screening_track2.py); first unguarded pass diverged and its 3.5% figure is retracted"
res["blind_pct"] = round(blind_pct, 1)
res["mesh_verts"] = len(Vmm)
res["mesh_tris"] = len(T)
res["mesh_frame"] = "sim3-aligned metric (one global Umeyama scale/rotation, disclosed)"
json.dump(res, open(RESULT, "w"), indent=1)
print(f"wrote {MESH} ({os.path.getsize(MESH)/1e6:.1f} MB), {TRAJ}, {RESULT} | "
      f"span {spanmm:.0f} mm", flush=True)
