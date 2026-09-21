"""The full-procedure study: a real gastroenterologist's complete withdrawal
(C3VDv2 simulated screening video c1_full_t4_v1), sampled every 12th frame, run through
the deployed pipeline. Outputs: tracking error vs the GT camera path over the whole
colon, whole-organ coverage statistics, and a full-colon mesh for the app.

Resumable: undistorted frames and estimated poses are checkpointed.
Usage: python speqslam/c3vdv2_screening_run.py [minutes]
"""
import sys, os, glob, time, json
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL, load_poses, umeyama_ate

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
BASE = "data/c3vdv2/c1_full_t4_v1"
UND = f"{BASE}/undist"
POSE_CK = f"{BASE}/poses_est.npy"
RESULT = "results/c3vdv2_screening.json"
MESH_OUT = "arguslite/app/src/main/assets/mesh_full.bin"

t0 = time.time()
frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
gt_all = load_poses(f"{BASE}/pose.txt")
print(f"{len(frames_raw)} sampled frames, {len(gt_all)} GT poses", flush=True)

# stage 1: undistort once
os.makedirs(UND, exist_ok=True)
mapu, mapv, K = su.build_map(V2CAL)
done_und = len(glob.glob(f"{UND}/*.jpg"))
if done_und < len(frames_raw):
    for i, p in enumerate(frames_raw):
        out = f"{UND}/{i:04d}.jpg"
        if os.path.exists(out):
            continue
        cv2.imwrite(out, su.undistort(cv2.imread(p), mapu, mapv),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if time.time() - t0 > BUDGET_S:
            print("budget during undistort; resume me", flush=True)
            sys.exit(0)
    print("undistort done", flush=True)

und = sorted(glob.glob(f"{UND}/*.jpg"))
n = len(und)
net = demo_app.DepthNet()

# stage 2: tracking with checkpoints
if os.path.exists(POSE_CK):
    poses = list(np.load(POSE_CK))
else:
    poses = [np.eye(4)]
start = len(poses)
if start < n:
    prev_img = cv2.imread(und[start - 1])
    prev_dep = net.predict(prev_img)
    for i in range(start, n):
        img = cv2.imread(und[i])
        dep = net.predict(img)
        g0 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        glare = specular_mask(prev_img, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
        T = dv.track_pair(g0, prev_dep, g1, K, 4, robust=False, depth_scale=1.0,
                          glare=glare, illum=True)
        poses.append(poses[-1] @ np.linalg.inv(T))
        prev_img, prev_dep = img, dep
        if i % 25 == 0:
            np.save(POSE_CK, np.array(poses))
            print(f"tracked {i}/{n}", flush=True)
        if time.time() - t0 > BUDGET_S:
            np.save(POSE_CK, np.array(poses))
            print(f"budget at pair {i}; resume me", flush=True)
            sys.exit(0)
    np.save(POSE_CK, np.array(poses))
    print("tracking done", flush=True)

poses = list(np.load(POSE_CK))
est = np.array([p[:3, 3] for p in poses])
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]
steps = np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000
ate = umeyama_ate(gt, est) * 1000
path = float(steps.sum())
print(f"ATE {ate:.1f} mm over {path:.0f} mm GT path ({100*ate/max(1,path):.1f}%)",
      flush=True)

# stage 3: coverage + full-colon mesh (Open3D TSDF, decimated to app size)
import open3d as o3d
import struct
intr = o3d.camera.PinholeCameraIntrinsic(640, 512, 250.0, 250.0, 320.0, 256.0)
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.0012, sdf_trunc=0.005,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
deps_cache = []
for i in range(n):
    img = cv2.imread(und[i])
    dep = net.predict(img)
    deps_cache.append(dep.astype(np.float16))
    col = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dm = o3d.geometry.Image(dep.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, dm, depth_scale=1.0, depth_trunc=0.15, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(poses[i]))
    if time.time() - t0 > BUDGET_S * 1.4:
        print("budget during fusion; rerun to redo stage 3", flush=True)
        sys.exit(0)
mesh = vol.extract_triangle_mesh()
print(f"raw mesh: {len(mesh.vertices)} verts", flush=True)
mesh = mesh.simplify_quadric_decimation(80000)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices); Nrm = np.asarray(mesh.vertex_normals)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles).astype(np.int32)

views = np.zeros(len(V), np.int32)
for i in range(n):
    Tcw = np.linalg.inv(poses[i])
    P = (Tcw[:3, :3] @ V.T).T + Tcw[:3, 3]
    z = P[:, 2]
    ok = z > 1e-4
    u = np.where(ok, 250.0 * P[:, 0] / np.maximum(z, 1e-6) + 320.0, -1)
    v2 = np.where(ok, 250.0 * P[:, 1] / np.maximum(z, 1e-6) + 256.0, -1)
    ok &= (u >= 0) & (u < 639) & (v2 >= 0) & (v2 < 511)
    ui = u.astype(np.int32).clip(0, 639); vi = v2.astype(np.int32).clip(0, 511)
    dm = deps_cache[i].astype(np.float32)[vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.004)
    views[ok] += 1
blind_pct = float((views < 3).mean() * 100)
print(f"mesh {len(V)} verts / {len(T)} tris | views median {np.median(views):.0f} | "
      f"under-seen {blind_pct:.1f}%", flush=True)

with open(MESH_OUT, "wb") as f:
    f.write(struct.pack("<ii", len(V), len(T)))
    f.write((V * 1000.0).astype(np.float32).tobytes())
    f.write(Nrm.astype(np.float32).tobytes())
    f.write(C.tobytes())
    f.write(views.clip(0, 255).astype(np.uint8).tobytes())
    f.write(T.tobytes())
json.dump(dict(seq="c1_full_t4_v1", frames=n, sample_stride=12,
               ate_mm=round(ate, 1), path_mm=round(path), ate_pct=round(100*ate/path, 2),
               mesh_verts=len(V), mesh_tris=len(T), blind_pct=round(blind_pct, 1)),
          open(RESULT, "w"), indent=1)
print(f"wrote {MESH_OUT} ({os.path.getsize(MESH_OUT)/1e6:.1f} MB) and {RESULT}",
      flush=True)
