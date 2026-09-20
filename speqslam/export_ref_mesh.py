"""Export a workstation-quality reference mesh for the app: the cecum phantom fused
from GT depth + GT poses with Open3D's TSDF at 0.8mm voxels, per-vertex coverage
counts, decimated, written as assets/mesh_cecum.bin:
  int32 nv, int32 nt, f32 verts[3nv] (mm), f32 norms[3nv],
  u8 rgb[3nv], u8 views[nv] (capped 255), int32 tris[3nt]
"""
import os, sys, glob, struct
import numpy as np, cv2, open3d as o3d
from scipy.spatial.transform import Rotation

SEQ = "data/c3vd/cecum_t1_a_undist"
GT = "results/c3vd/cecum_t1_a_gt.txt"
OUT = "arguslite/app/src/main/assets/mesh_cecum.bin"
DS = 100.0 / 65535.0 / 1000.0          # depth units -> meters
FX = 250.0; CX = 320.0; CY = 256.0
W, H = 640, 512

imgs = sorted(glob.glob(f"{SEQ}/image01/*"))
deps = sorted(glob.glob(f"{SEQ}/depth01/*"))
poses = []
for line in open(GT):
    v = line.split()
    if len(v) != 8:
        continue
    t = np.array([float(v[1]), float(v[2]), float(v[3])])
    q = [float(v[4]), float(v[5]), float(v[6]), float(v[7])]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    poses.append(T)
n = min(len(imgs), len(deps), len(poses))
print(f"{n} frames", flush=True)

intr = o3d.camera.PinholeCameraIntrinsic(W, H, FX, FX, CX, CY)
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0008, sdf_trunc=0.0035,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i in range(n):
    col = o3d.io.read_image(imgs[i])
    d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED)
    dep = o3d.geometry.Image(d.astype(np.uint16))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, dep, depth_scale=1.0 / DS, depth_trunc=0.12,
        convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(poses[i]))
mesh = vol.extract_triangle_mesh()
print(f"raw mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris", flush=True)
mesh = mesh.simplify_quadric_decimation(60000)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices)          # meters
Nrm = np.asarray(mesh.vertex_normals)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles).astype(np.int32)
print(f"decimated: {len(V)} verts, {len(T)} tris", flush=True)

# per-vertex coverage: project into every pose, in-bounds + not occluded
views = np.zeros(len(V), np.int32)
for i in range(n):
    d = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
    Tcw = np.linalg.inv(poses[i])
    P = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
    z = P[:, 2]
    ok = z > 1e-4
    u = np.where(ok, FX * P[:, 0] / np.maximum(z, 1e-6) + CX, -1)
    v2 = np.where(ok, FX * P[:, 1] / np.maximum(z, 1e-6) + CY, -1)
    ok &= (u >= 0) & (u < W - 1) & (v2 >= 0) & (v2 < H - 1)
    ui = u.astype(np.int32).clip(0, W - 1); vi = v2.astype(np.int32).clip(0, H - 1)
    dm = d[vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.003)
    views[ok] += 1
print(f"views: median {np.median(views):.0f}, blind(<3) {(views<3).mean()*100:.1f}%",
      flush=True)

with open(OUT, "wb") as f:
    f.write(struct.pack("<ii", len(V), len(T)))
    f.write((V * 1000.0).astype(np.float32).tobytes())   # mm
    f.write(Nrm.astype(np.float32).tobytes())
    f.write(C.tobytes())
    f.write(views.clip(0, 255).astype(np.uint8).tobytes())
    f.write(T.tobytes())
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
