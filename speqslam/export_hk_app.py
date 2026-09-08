"""Package the real-footage closed-loop run for the app: replay frames (raw+depth views),
3D cloud with coverage/born-frames, camera path, and the found lesion — hk_map.bin uses the
colon_map2.bin layout plus a trailing lesion block (int L, then L x [xyz float, size_mm float])."""
import sys, os, glob, json, struct
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask

DS = 100.0 / 65535.0 / 1000.0
P = "data/hyperkvasir/proc/clip1"
A = "arguslite/app/src/main/assets"

imgs = sorted(glob.glob(f"{P}/image01/*.jpg"))
frames = [cv2.imread(f) for f in imgs]
h, w = frames[0].shape[:2]
K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], float)
stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
static = cv2.dilate(((stack.std(0) < 4.0) * 255).astype(np.uint8), np.ones((7, 7), np.uint8))

net = demo_app.DepthNet()
deps = []
os.makedirs(f"{A}/hkreplay", exist_ok=True)
for j, f in enumerate(frames):
    d = net.predict(f); d[static > 0] = 0
    deps.append(d)
    cv2.imwrite(f"{A}/hkreplay/f{j:03d}.jpg", cv2.resize(f, (320, 256)),
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    v = d[d > 0]; lo, hi = np.percentile(v, 3), np.percentile(v, 97)
    t = 1.0 - np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1); t[d <= 0] = 1.0
    cv2.imwrite(f"{A}/hkreplay/f{j:03d}d.jpg",
                cv2.resize(cv2.applyColorMap((255 * t).astype(np.uint8), cv2.COLORMAP_MAGMA),
                           (320, 256)), [cv2.IMWRITE_JPEG_QUALITY, 70])

pose = np.eye(4); poses = [pose.copy()]
for i in range(1, len(frames)):
    g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    bad = np.maximum(specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245, dilate=1), static)
    T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0, glare=bad, illum=True)
    pose = pose @ np.linalg.inv(T); poses.append(pose.copy())
print("[vo] done", flush=True)

VOX = 0.0015
acc = {}
for i, f in enumerate(frames):
    d = deps[i]
    ys, xs = np.nonzero(d > 0)
    ys, xs = ys[::9], xs[::9]
    z = d[ys, xs]
    Pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    T = poses[i]
    Pw = Pc @ T[:3, :3].T + T[:3, 3]
    cols = f[ys, xs][:, ::-1]
    for p, c in zip(Pw, cols):
        k = tuple(np.floor(p / VOX).astype(np.int64))
        if k in acc:
            e = acc[k]; e[0] += p; e[1] += c; e[2] += 1; e[3] = min(e[3], i)
        else:
            acc[k] = [p.copy(), c.astype(np.float64), 1, i]
pts = np.array([e[0]/e[2] for e in acc.values()])
cols = np.array([e[1]/e[2] for e in acc.values()]).clip(0, 255).astype(np.uint8)
cnt = np.array([e[2] for e in acc.values()])
born = np.array([e[3] for e in acc.values()], np.uint16)
blind = (cnt < 3).astype(np.uint8)
if len(pts) > 4500:
    keep = np.random.default_rng(0).choice(len(pts), 4500, replace=False)
    pts, cols, blind, born = pts[keep], cols[keep], blind[keep], born[keep]
traj = np.array([p[:3, 3] for p in poses])
les = json.load(open("results/closed_loop_hk1.json"))["lesions"]
lpos = np.array([L["pos"] for L in les]); lsize = [L["size_mm"] for L in les]
mu = pts.mean(0); sc = np.abs(pts - mu).max()
pn, tn, ln = (pts-mu)/sc, (traj-mu)/sc, (lpos-mu)/sc
with open(f"{A}/hk_map.bin", "wb") as f:
    f.write(struct.pack("<i", len(pn)))
    for i in range(len(pn)):
        f.write(struct.pack("<fff", *pn[i]))
        f.write(struct.pack("<BBBB", cols[i][0], cols[i][1], cols[i][2], blind[i]))
        f.write(struct.pack("<H", born[i]))
    f.write(struct.pack("<i", len(tn)))
    for t in tn:
        f.write(struct.pack("<fff", *t))
    f.write(struct.pack("<i", len(ln)))
    for i in range(len(ln)):
        f.write(struct.pack("<ffff", *ln[i], lsize[i]))
print(f"hk_map.bin: {len(pn)} pts, {100*blind.mean():.1f}% blind, {len(tn)} poses, "
      f"{len(ln)} lesion(s) at size {lsize}", flush=True)
