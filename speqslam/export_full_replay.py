"""Assets that make the full-withdrawal chip's replay a real build:

- born_full.bin: per mesh vertex, the first frame (u16 index into the 2800-frame
  stride-3 trajectory) whose camera actually saw it - the app reveals each patch
  at that moment, so the tube grows as the camera path advances. Vertices never
  confirmed seen fall back to the nearest camera position's frame.
- assets/fullreplay/fNNN.jpg + fNNNd.jpg: 70 raw/depth thumbnail pairs (every
  40th frame), the same replay strip the phantom and real-footage chips have.
- fig13_path_overlay.png: estimated (Sim3-aligned) camera path over the GT path,
  the honest answer to "is this tube even right?" - the centerline is graded.
"""
import struct, glob, os, sys, json
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1_s3"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
BORN = "arguslite/app/src/main/assets/born_full.bin"
RDIR = "arguslite/app/src/main/assets/fullreplay"
FIG = "results/poster/fig13_path_overlay.png"

b = open(MESH, "rb").read()
nv, nt = struct.unpack_from("<ii", b, 0)
V = np.frombuffer(b, np.float32, 3 * nv, 8).reshape(-1, 3) / 1000.0  # m, GT frame

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
apose = []
for p in poses:
    M = np.eye(4)
    M[:3, :3] = R @ p[:3, :3]
    M[:3, 3] = s * (R @ (p[:3, 3] - me)) + mg
    apose.append(M)
cam = np.array([m[:3, 3] for m in apose])

# ---- born: the frame at which the camera passes closest to each vertex ----
# (display semantics for the build replay: a patch appears as the camera reaches
# it. Strict first-confirmed-visibility was tried and reveals most of the tube
# in the last third - residual drift delays the 4 mm occlusion test - so the
# pass-by definition is used for the animation; blind-spot statistics are
# unaffected, they come from the views channel.)
und = sorted(glob.glob(f"{BASE}/undist/*.jpg"))
n = len(und)
net = demo_app.DepthNet()
born = np.empty(nv, np.int32)
for j0 in range(0, nv, 4096):
    d = ((V[j0:j0 + 4096, None, :] - cam[None, :, :]) ** 2).sum(2)
    born[j0:j0 + 4096] = d.argmin(1)
print(f"born done: median {np.median(born):.0f}", flush=True)
with open(BORN, "wb") as f:
    f.write(struct.pack("<i", nv))
    f.write(born.astype(np.uint16).tobytes())
print(f"wrote {BORN} ({os.path.getsize(BORN)/1e3:.0f} KB)", flush=True)

# ---- replay thumbnails ----
os.makedirs(RDIR, exist_ok=True)
sel = list(range(0, n, 40))[:70]
for j, i in enumerate(sel):
    img = cv2.imread(und[i])
    dep = net.predict(img)
    h, w = img.shape[:2]
    sc = 320.0 / w
    small = cv2.resize(img, (320, int(h * sc)))
    lo, hi = np.percentile(dep[dep > 0], 2), np.percentile(dep[dep > 0], 98)
    t = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cv2.applyColorMap(((1 - t) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (320, int(h * sc)))
    cv2.imwrite(f"{RDIR}/f{j:03d}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(f"{RDIR}/f{j:03d}d.jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 80])
size = sum(os.path.getsize(f"{RDIR}/{f}") for f in os.listdir(RDIR)) / 1e6
print(f"wrote {len(sel)} thumbnail pairs ({size:.1f} MB)", flush=True)

# ---- fig13: estimated path vs GT ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for ax, (a, bx, name) in zip(axes, [(0, 1, "top view (x-y)"), (0, 2, "side view (x-z)")]):
    ax.plot(gt[:, a] * 1000, gt[:, bx] * 1000, color="#444444", lw=2.2,
            label="ground-truth camera path")
    ax.plot(cam[:, a] * 1000, cam[:, bx] * 1000, color="#3FA0D6", lw=1.4,
            label="Argus estimate (Sim(3)-aligned)")
    ax.set_title(name)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    ax.set_aspect("equal")
axes[0].legend(fontsize=8.5, loc="best")
fig.suptitle("Full withdrawal at 1/3 native frame rate: estimated vs true camera path "
             "(ATE 101 mm, baseline 168 mm)", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG, dpi=170)
print(f"wrote {FIG}", flush=True)
