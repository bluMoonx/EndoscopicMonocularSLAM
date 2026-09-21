"""Repair the full-withdrawal app mesh. The screening runner fused the TSDF in the
tracker's own coordinate frame, which for monocular tracking is not metric — scale
drift left the mesh spanning hundreds of meters, so the app's center/span framing
collapsed the actual colon to a sub-pixel dot (blank model view). Fix: apply the
same Umeyama Sim(3) alignment the ATE number already uses, so the mesh lives in
the GT frame in honest millimeters; decimate to the triangle budget the cecum mesh
already proved on-device; and export the aligned camera path (traj_full.bin) so the
"down the tube" view and trajectory ribbon work on the full-withdrawal chip.
"""
import struct, glob, os, sys
import numpy as np
sys.path.insert(0, "speqslam")
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
TRAJ = "arguslite/app/src/main/assets/traj_full.bin"
TARGET_TRIS = 60000   # the run-0 cecum mesh size, known to render on the phone

b = open(MESH, "rb").read()
nv, nt = struct.unpack_from("<ii", b, 0)
off = 8
V = np.frombuffer(b, np.float32, 3 * nv, off).reshape(-1, 3).copy(); off += 12 * nv
N = np.frombuffer(b, np.float32, 3 * nv, off).reshape(-1, 3).copy(); off += 12 * nv
C = np.frombuffer(b, np.uint8, 3 * nv, off).reshape(-1, 3).copy(); off += 3 * nv
W = np.frombuffer(b, np.uint8, nv, off).copy(); off += nv
T = np.frombuffer(b, np.int32, 3 * nt, off).reshape(-1, 3).copy()

span = (V.max(0) - V.min(0)).max() / 2
if span < 2000:
    print(f"mesh span {span:.0f} mm — already metric; nothing to do")
    sys.exit(0)
print(f"before: {nv} verts / {nt} tris, span {span:.0f} mm (non-metric)", flush=True)

poses = list(np.load(f"{BASE}/poses_est.npy"))
est = np.array([p[:3, 3] for p in poses])
nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]

# Umeyama similarity est -> gt (same construction as umeyama_ate, keeping s,R,t)
mg, me = gt.mean(0), est.mean(0)
G, E = gt - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
var = (E ** 2).sum() / len(gt)
s = float(np.trace(np.diag(S) @ D) / var)
ate = np.sqrt((((s * (R @ E.T).T + mg) - gt) ** 2).sum(1).mean()) * 1000
print(f"similarity: scale {s:.6f} | ATE check {ate:.1f} mm", flush=True)

Vm = s * (R @ (V / 1000.0 - me).T).T + mg          # meters, GT frame
traj_mm = (s * (R @ (est - me).T).T + mg) * 1000.0

import open3d as o3d
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vm),
                              o3d.utility.Vector3iVector(T))
m.vertex_colors = o3d.utility.Vector3dVector(C / 255.0)
m = m.simplify_quadric_decimation(TARGET_TRIS)
m.remove_degenerate_triangles()
m.remove_unreferenced_vertices()
m.compute_vertex_normals()
V2 = np.asarray(m.vertices)
N2 = np.asarray(m.vertex_normals)
C2 = (np.asarray(m.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T2 = np.asarray(m.triangles).astype(np.int32)

# view counts don't survive decimation; carry each from its nearest original vertex
kd = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vm)))
W2 = np.array([W[kd.search_knn_vector_3d(v, 1)[1][0]] for v in V2], np.uint8)

Vmm = (V2 * 1000.0).astype(np.float32)
with open(MESH, "wb") as f:
    f.write(struct.pack("<ii", len(Vmm), len(T2)))
    f.write(Vmm.tobytes())
    f.write(N2.astype(np.float32).tobytes())
    f.write(C2.tobytes())
    f.write(W2.tobytes())
    f.write(T2.tobytes())

center = (Vmm.min(0) + Vmm.max(0)) / 2
spanmm = (Vmm.max(0) - Vmm.min(0)).max() / 2
tn = ((traj_mm - center) / spanmm).astype(np.float32)
with open(TRAJ, "wb") as f:
    f.write(struct.pack("<i", len(tn)))
    f.write(tn.tobytes())

print(f"after: {len(Vmm)} verts / {len(T2)} tris, span {spanmm:.0f} mm | "
      f"under-seen {(W2 < 3).mean() * 100:.1f}% | "
      f"{MESH} {os.path.getsize(MESH) / 1e6:.1f} MB, {TRAJ} {os.path.getsize(TRAJ) / 1e3:.0f} KB",
      flush=True)
