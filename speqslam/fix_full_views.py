"""Recompute per-vertex view counts for the shipped full-withdrawal mesh.

fix_full_mesh.py decimated the mesh to the on-device triangle budget and carried
view counts by nearest-vertex transfer — an approximation that moved the under-seen
figure by ~1 point. This recomputes the counts exactly at the shipped vertices with
the same visibility test the study used (project into every frame, occlusion-check
against that frame's depth), then rewrites the views channel in mesh_full.bin and
the blind_pct in results/c3vdv2_screening.json so every surface shows one measured
number.
"""
import struct, glob, os, sys, json, time
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
RESULT = "results/c3vdv2_screening.json"

b = bytearray(open(MESH, "rb").read())
nv, nt = struct.unpack_from("<ii", b, 0)
V = np.frombuffer(bytes(b), np.float32, 3 * nv, 8).reshape(-1, 3)  # GT frame, mm

poses = list(np.load(f"{BASE}/poses_est.npy"))
est = np.array([p[:3, 3] for p in poses])
nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums[:len(poses)]])
gt = gt - gt[0]

# the same similarity fix_full_mesh.py applied; invert it to get back to the
# tracker frame the depth maps live in
mg, me = gt.mean(0), est.mean(0)
G, E = gt - mg, est - me
U, S, Vt = np.linalg.svd(G.T @ E / len(gt))
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
R = U @ D @ Vt
s = float(np.trace(np.diag(S) @ D) / ((E ** 2).sum() / len(gt)))
Vest = (R.T @ ((V / 1000.0 - mg) / s).T).T + me   # tracker frame, meters

und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
net = demo_app.DepthNet()
views = np.zeros(len(Vest), np.int32)
t0 = time.time()
for i in range(n):
    dep = net.predict(cv2.imread(und[i]))
    Tcw = np.linalg.inv(poses[i])
    P = (Tcw[:3, :3] @ Vest.T).T + Tcw[:3, 3]
    z = P[:, 2]
    ok = z > 1e-4
    u = np.where(ok, 250.0 * P[:, 0] / np.maximum(z, 1e-6) + 320.0, -1)
    v2 = np.where(ok, 250.0 * P[:, 1] / np.maximum(z, 1e-6) + 256.0, -1)
    ok &= (u >= 0) & (u < 639) & (v2 >= 0) & (v2 < 511)
    ui = u.astype(np.int32).clip(0, 639); vi = v2.astype(np.int32).clip(0, 511)
    dm = dep[vi, ui]
    ok &= (dm > 0) & (z <= dm + 0.004)
    views[ok] += 1
    if i % 50 == 0:
        print(f"{i}/{n} ({time.time() - t0:.0f}s)", flush=True)

blind_pct = float((views < 3).mean() * 100)
print(f"under-seen {blind_pct:.1f}% | views median {np.median(views):.0f}", flush=True)

off = 8 + 12 * nv + 12 * nv + 3 * nv
b[off:off + nv] = views.clip(0, 255).astype(np.uint8).tobytes()
open(MESH, "wb").write(bytes(b))

res = json.load(open(RESULT))
res["blind_pct"] = round(blind_pct, 1)
res["mesh_verts"] = nv
res["mesh_tris"] = nt
json.dump(res, open(RESULT, "w"), indent=1)
print(f"rewrote {MESH} views + {RESULT}", flush=True)
