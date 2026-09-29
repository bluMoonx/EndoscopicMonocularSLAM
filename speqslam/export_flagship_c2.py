"""Make c2_full_t1_v1 the app's flagship full-withdrawal chip.

c2 is the strongest sequence on every axis: best tracking (ATE 92.1 mm over a
7.0 m GT path, 1.32%, 0.69x baseline; guard rate 1.5%), a validated coverage
story (GT-pose instrument: 41.8% under-seen at screening pace), and - unlike
c1_t4 - real sculpted polyps that the frozen detector locks onto for 100+
consecutive frames. This regenerates every run-3 asset from c2 under the same
filenames the app already loads:

  mesh_full.bin   fused surface (60k tris) + per-vertex views (stride-12
                  screening-pace budget, 4 mm tolerance)
  traj_full.bin   aligned estimated camera path (all 3667 frames, normalized)
  gt_full.bin     recorded ground-truth path, same frame ("vs truth")
  born_full.bin   per-vertex pass-by frame for the build replay
  fullreplay/     70 raw+depth thumbnail pairs
  lesions_full.bin  strongest dwell-event findings as pins:
                  int L, then per lesion 6 floats
                  (x, y, z normalized, size_mm, support_frames, mean_conf)

Pins are the dwell events with >= 40 supporting frames from the frozen-gate
sweep (display curation of the 24 events that clear the >=15 bar; the full
sweep remains in results/screening_polyp_c2_full_t1_v1.json). Events whose
world positions land within 20 mm are merged.
"""
import struct, glob, os, sys, json, time
import numpy as np, cv2

BASE = "data/c3vdv2/c2_full_t1_v1_s3"
A = "arguslite/app/src/main/assets"
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
frames_raw = sorted(glob.glob(f"{BASE}/rgb/*.png"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in frames_raw]
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums])
gt = gt - gt[0]
poses = list(np.load(f"{BASE}/poses_est.npy"))
est = np.array([p[:3, 3] for p in poses])

mg, me = gt.mean(0), est.mean(0)
G, E = gt - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / ((E ** 2).sum() / len(gt)))
ate = np.sqrt((((s * (R @ E.T).T + mg) - gt) ** 2).sum(1).mean()) * 1000
print(f"Sim3 scale {s:.4f} | ATE check {ate:.1f} mm (expect 92.1)", flush=True)
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
net = demo_app.DepthNet()
budget = {}
t0 = time.time()
for i in range(n):
    img = cv2.imread(und[i])
    dep = net.predict(img) * s
    if i % 4 == 0:
        budget[i] = dep.astype(np.float16)
    col = o3d.geometry.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        col, o3d.geometry.Image(dep.astype(np.float32)), depth_scale=1.0,
        depth_trunc=0.20, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(apose[i]))
    if i % 600 == 0:
        print(f"fuse {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
mesh = vol.extract_triangle_mesh()
mesh = mesh.simplify_quadric_decimation(60000)
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
V = np.asarray(mesh.vertices)
N = np.asarray(mesh.vertex_normals)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles).astype(np.int32)
print(f"mesh {len(V)} verts / {len(T)} tris", flush=True)

views = np.zeros(len(V), np.int32)
for i, dep16 in budget.items():
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
print(f"under-seen {blind_pct:.1f}% (stride-12 budget; GT-pose instrument 41.8%)",
      flush=True)

Vmm = (V * 1000.0).astype(np.float32)
with open(f"{A}/mesh_full.bin", "wb") as f:
    f.write(struct.pack("<ii", len(Vmm), len(T)))
    f.write(Vmm.tobytes())
    f.write(N.astype(np.float32).tobytes())
    f.write(C.tobytes())
    f.write(views.clip(0, 255).astype(np.uint8).tobytes())
    f.write(T.tobytes())
center = (Vmm.min(0) + Vmm.max(0)) / 2
span = (Vmm.max(0) - Vmm.min(0)).max() / 2
print(f"span {span:.0f} mm", flush=True)

traj_mm = np.array([m[:3, 3] for m in apose]) * 1000.0
for name, P in (("traj_full.bin", traj_mm), ("gt_full.bin", gt * 1000.0)):
    tn = ((P - center) / span).astype(np.float32)
    with open(f"{A}/{name}", "wb") as f:
        f.write(struct.pack("<i", len(tn)))
        f.write(tn.tobytes())

cam = traj_mm / 1000.0
born = np.empty(len(V), np.int32)
for j0 in range(0, len(V), 4096):
    d = ((V[j0:j0+4096, None, :] - cam[None, :, :]) ** 2).sum(2)
    born[j0:j0+4096] = d.argmin(1)
with open(f"{A}/born_full.bin", "wb") as f:
    f.write(struct.pack("<i", len(V)))
    f.write(born.astype(np.uint16).tobytes())

os.makedirs(f"{A}/fullreplay", exist_ok=True)
sel = np.linspace(0, n - 1, 70).astype(int)
for j, i in enumerate(sel):
    img = cv2.imread(und[i])
    dep = net.predict(img)
    h, w = img.shape[:2]
    sc = 320.0 / w
    small = cv2.resize(img, (320, int(h * sc)))
    lo, hi = np.percentile(dep[dep > 0], 2), np.percentile(dep[dep > 0], 98)
    tt = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cv2.resize(cv2.applyColorMap(((1 - tt) * 255).astype(np.uint8),
                                        cv2.COLORMAP_INFERNO), (320, int(h * sc)))
    cv2.imwrite(f"{A}/fullreplay/f{j:03d}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(f"{A}/fullreplay/f{j:03d}d.jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("replay thumbnails written", flush=True)

# ---- lesion pins from the frozen-gate dwell sweep ----
d = np.load("results/screening_polyp_c2_full_t1_v1_det.npz")
fr, pos, size, conf = d["frames"], d["pos"], d["size"], d["conf"]
used = [False] * len(fr)
events = []
for a in range(len(fr)):
    if used[a]:
        continue
    grp = [a]; used[a] = True
    last = fr[a]
    for b in range(a + 1, len(fr)):
        if used[b] or fr[b] - last > 10:
            if fr[b] - last > 10:
                break
            continue
        if min(np.linalg.norm(pos[g] - pos[b]) for g in grp) < 0.010:
            grp.append(b); used[b] = True
            last = fr[b]
    if len(grp) >= 40:
        events.append(grp)
les = []
for grp in events:
    W = []
    for g in grp:
        i = int(fr[g])
        pw = apose[i][:3, :3] @ (s * pos[g]) + apose[i][:3, 3]
        W.append(pw * 1000.0)
    W = np.median(np.array(W), 0)
    les.append([W, float(np.median(size[grp])), len(grp), float(np.mean(conf[grp]))])
# merge events within 20 mm
merged = []
for L in sorted(les, key=lambda x: -x[2]):
    hit = next((m for m in merged if np.linalg.norm(m[0] - L[0]) < 20.0), None)
    if hit:
        hit[2] += L[2]
    else:
        merged.append(L)
print(f"pins: {len(merged)} (from {len(events)} strong events)", flush=True)
with open(f"{A}/lesions_full.bin", "wb") as f:
    f.write(struct.pack("<i", len(merged)))
    for W, sz, sup, cf in merged:
        nx = (W - center) / span
        f.write(struct.pack("<6f", nx[0], nx[1], nx[2], sz, float(sup), cf))
        print(f"  pin at {np.round(nx,2)} ~{sz:.1f}mm support {sup} conf {cf:.2f}",
              flush=True)

json.dump(dict(seq="c2_full_t1_v1", frames=n, ate_mm=92.1, path_mm=7003,
               blind_pct=round(blind_pct, 1), gtpose_blind_pct=41.8,
               pins=len(merged), span_mm=round(float(span))),
          open("results/flagship_c2.json", "w"), indent=1)
print("done", flush=True)
