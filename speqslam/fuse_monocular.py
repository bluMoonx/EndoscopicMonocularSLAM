"""
Dense monocular colon reconstruction via TSDF volumetric fusion (KinectFusion-style).

Integrates RGB + depth over the estimated trajectory into a truncated-signed-distance volume, which
AVERAGES per-frame depth noise into a clean surface, then extracts a mesh. Runs in two modes:
  mono : learned depth + learned-VO poses  -> the fully-monocular reconstruction (no depth sensor)
  gt   : GT depth + GT poses               -> the ground-truth reconstruction (for comparison)
Saves a mesh (.ply) and a colored point sample (.npz) for the viewer. If both exist, reports the
Chamfer distance (mm) between the monocular and GT surfaces = reconstruction accuracy.
"""
import sys, glob, os, numpy as np, cv2, open3d as o3d
from scipy.spatial.transform import Rotation as Rot

DS = 100.0 / 65535.0 / 1000.0
DEPTH_SCALE = 1.0 / DS          # o3d: meters = raw / depth_scale


def load_poses(path):
    P = {}
    for l in open(path):
        v = l.split()
        if len(v) < 8:
            continue
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat(np.array(v[4:8], float)).as_matrix()
        T[:3, 3] = np.array(v[1:4], float)
        P[round(float(v[0]), 6)] = T
    return P


MODES = {  # mode -> (depth subfolder, trajectory suffix)
    "gt":     ("depth01", "gt"),
    "mono":   ("depth01_learned", "learned"),   # off-the-shelf depth + VO on it
    "ftmono": ("depth01_ftc", "ft"),            # fine-tuned depth + VO on it
}


def fuse(seq, mode, voxel=0.0006):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    dsub, tsuf = MODES[mode]
    deps = sorted(glob.glob(f"{undist}/{dsub}/*.png"))
    poses = load_poses(f"results/c3vd/{seq}_{tsuf}.txt")
    h, w = cv2.imread(deps[0], cv2.IMREAD_UNCHANGED).shape[:2]
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    used = 0
    for i in range(len(imgs)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        color = o3d.geometry.Image(cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2RGB))
        depth = o3d.geometry.Image(cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=DEPTH_SCALE, depth_trunc=0.12, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(poses[ts]))   # extrinsic = world->cam
        used += 1
    mesh = vol.extract_triangle_mesh()
    # remove floating noise islands: keep only sizeable connected components
    tc, n_tri, _ = mesh.cluster_connected_triangles()
    tc, n_tri = np.asarray(tc), np.asarray(n_tri)
    mesh.remove_triangles_by_mask(n_tri[tc] < 0.02 * n_tri.max())
    mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
    os.makedirs("results/mesh", exist_ok=True)
    ply = f"results/mesh/{seq}_{mode}.ply"
    o3d.io.write_triangle_mesh(ply, mesh)
    print(f"{seq} [{mode}]: fused {used} frames -> {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris -> {ply}", flush=True)
    return mesh


def chamfer_mm(mesh_a, mesh_b, transform=None, n=80000, icp=True):
    a = mesh_a.sample_points_uniformly(n); b = mesh_b.sample_points_uniformly(n)
    if transform is not None:
        a.transform(transform)
    if icp:                                 # refine the coarse Sim(3) with rigid ICP (fair shape eval)
        a.estimate_normals(); b.estimate_normals()
        reg = o3d.pipelines.registration.registration_icp(
            a, b, 0.01, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        a.transform(reg.transformation)
    da = np.asarray(b.compute_point_cloud_distance(a))   # b->a
    db = np.asarray(a.compute_point_cloud_distance(b))   # a->b
    return 0.5 * (da.mean() + db.mean()) * 1000.0, np.median(np.concatenate([da, db])) * 1000.0


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a"
    modes = sys.argv[2:] or ["mono", "gt"]
    meshes = {m: fuse(seq, m) for m in modes}
    est = next((m for m in modes if m != "gt"), None)
    if est and "gt" in meshes:
        # estimated mesh is in its VO frame; map it into GT frame with the SAME rotation-aware
        # Sim(3) used everywhere: p_gt = s*Rg*p + tg  (one global 7-DOF transform, standard for
        # monocular reconstruction which is scale/frame-ambiguous).
        sys.path.insert(0, "speqslam")
        from validate_coverage import load_poses as lp, umeyama
        gt, ln = lp(f"results/c3vd/{seq}_gt.txt"), lp(f"results/c3vd/{seq}_{MODES[est][1]}.txt")
        common = sorted(set(gt) & set(ln))
        Pv = np.array([ln[t][:3, 3] for t in common]); Pg = np.array([gt[t][:3, 3] for t in common])
        Rg = Rot.from_matrix(np.array([gt[t][:3, :3] @ ln[t][:3, :3].T for t in common])).mean().as_matrix()
        _, s = umeyama(Pv, Pg)
        T = np.eye(4); T[:3, :3] = s * Rg; T[:3, 3] = Pg.mean(0) - s * (Rg @ Pv.mean(0))
        cd_mean, cd_med = chamfer_mm(meshes[est], meshes["gt"], transform=T)
        print(f"\n{seq}: {est}-vs-GT surface Chamfer = {cd_mean:.2f} mm mean, "
              f"{cd_med:.2f} mm median  (scale {s:.3f})", flush=True)
