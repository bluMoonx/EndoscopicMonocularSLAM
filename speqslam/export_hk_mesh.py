"""Export the real-footage (Hyper-Kvasir clip1) reconstruction as a mesh for the app:
same pipeline as the validated demo run (static-overlay mask, zero-shot depth, in-vivo
tracker), fused with Open3D TSDF, decimated, per-vertex coverage counted.
Writes arguslite/app/src/main/assets/mesh_hk.bin (same format as mesh_cecum.bin)."""
import os, sys, glob, struct
import numpy as np, cv2, open3d as o3d
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask

P = "data/hyperkvasir/proc/clip1"
OUT = "arguslite/app/src/main/assets/mesh_hk.bin"

imgs = sorted(glob.glob(f"{P}/image01/*.jpg"))
frames = [cv2.imread(f) for f in imgs]
h, w = frames[0].shape[:2]
K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], float)
print(f"{len(frames)} frames {w}x{h}", flush=True)

stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
static = (stack.std(0) < 4.0).astype(np.uint8) * 255
static = cv2.dilate(static, np.ones((7, 7), np.uint8))

net = demo_app.DepthNet()
deps = []
for f in frames:
    d = net.predict(f)
    d[static > 0] = 0
    deps.append(d)

pose = np.eye(4); poses = [pose.copy()]
for i in range(1, len(frames)):
    g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
    bad = np.maximum(glare, static)
    T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0,
                      glare=bad, illum=True)
    pose = pose @ np.linalg.inv(T)
    poses.append(pose.copy())
print("tracked", flush=True)

intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0008, sdf_trunc=0.0035,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
for i in range(len(frames)):
    col = o3d.geometry.Image(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
    dep = o3d.geometry.Image(deps[i].astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, dep, depth_scale=1.0, depth_trunc=0.15, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(poses[i]))
mesh = vol.extract_triangle_mesh()
print(f"raw: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris", flush=True)
mesh = mesh.simplify_quadric_decimation(45000)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices); Nrm = np.asarray(mesh.vertex_normals)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles).astype(np.int32)
print(f"decimated: {len(V)} verts, {len(T)} tris", flush=True)

views = np.zeros(len(V), np.int32)
for i in range(len(frames)):
    Tcw = np.linalg.inv(poses[i])
    Pc = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
    z = Pc[:, 2]
    ok = z > 1e-4
    u = np.where(ok, K[0,0] * Pc[:, 0] / np.maximum(z, 1e-6) + K[0,2], -1)
    v2 = np.where(ok, K[1,1] * Pc[:, 1] / np.maximum(z, 1e-6) + K[1,2], -1)
    ok &= (u >= 0) & (u < w - 1) & (v2 >= 0) & (v2 < h - 1)
    ui = u.astype(np.int32).clip(0, w - 1); vi = v2.astype(np.int32).clip(0, h - 1)
    dm = deps[i][vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.004)
    views[ok] += 1
print(f"views median {np.median(views):.0f}, <3: {(views<3).mean()*100:.1f}%", flush=True)

with open(OUT, "wb") as f:
    f.write(struct.pack("<ii", len(V), len(T)))
    f.write((V * 1000.0).astype(np.float32).tobytes())
    f.write(Nrm.astype(np.float32).tobytes())
    f.write(C.tobytes())
    f.write(views.clip(0, 255).astype(np.uint8).tobytes())
    f.write(T.tobytes())
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)
