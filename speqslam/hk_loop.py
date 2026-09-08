"""THE closed-loop test on REAL polyp footage (Hyper-Kvasir clip, human colonoscopy).
Overlays auto-masked by temporal variance (UI never moves, tissue does). Tracker runs in in-vivo
mode with overlay+glare pixels excluded. Detection uses the FROZEN operating point calibrated on
the negative controls: size 2-22mm, confidence>=0.9, support>=15 frames.
No pose GT exists -> this is the qualitative end-to-end demo; sizes use a nominal calibration."""
import sys, os, glob, json
import numpy as np, cv2
sys.path.insert(0, "speqslam")
import demo_app, direct_vo as dv
from glare_removal import specular_mask
from polyp_detect import load_model, segment
from polyp_measure import diameter_mm
from scipy.spatial.transform import Rotation

DS = 100.0 / 65535.0 / 1000.0
P = "data/hyperkvasir/proc/clip1"
CONF, SUPPORT, SMIN, SMAX = 0.7, 15, 2.0, 22.0  # conf had no effect at support>=15 on controls

imgs = sorted(glob.glob(f"{P}/image01/*.jpg"))
frames = [cv2.imread(f) for f in imgs]
h, w = frames[0].shape[:2]
K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], float)   # nominal (uncalibrated scope)

# 1. static-overlay mask: pixels that never change are UI, not tissue
stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
static = (stack.std(0) < 4.0).astype(np.uint8) * 255
static = cv2.dilate(static, np.ones((7, 7), np.uint8))
cv2.imwrite("results/hk_clip1_overlay_mask.png", static)
print(f"[mask] {100*(static>0).mean():.1f}% of frame is static overlay/border", flush=True)

# 2. zero-shot depth
net = demo_app.DepthNet()
deps = []
for f in frames:
    d = net.predict(f)
    d[static > 0] = 0
    deps.append(d)
print("[depth] done", flush=True)

# 3. tracking, in-vivo mode, overlay+glare excluded
pose = np.eye(4); poses = [pose.copy()]
for i in range(1, len(frames)):
    g0 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    glare = specular_mask(frames[i-1], method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
    bad = np.maximum(glare, static)
    T = dv.track_pair(g0, deps[i-1], g1, K, 4, robust=False, depth_scale=1.0, glare=bad, illum=True)
    pose = pose @ np.linalg.inv(T)
    poses.append(pose.copy())
print(f"[vo] tracked {len(poses)} frames, path {1000*sum(np.linalg.norm(poses[i][:3,3]-poses[i-1][:3,3]) for i in range(1,len(poses))):.1f}mm (nominal scale)", flush=True)

# 4. detection with FROZEN filters, pinned to 3D
model = load_model()
hits = []
for i, f in enumerate(frames):
    mask = segment(f, model)
    mask[static > 0] = 0
    binm = mask > 0.5
    if binm.mean() < 0.005:
        continue
    ys, xs = np.nonzero(binm)
    u0, v0 = int(xs.mean()), int(ys.mean())
    vals = deps[i][max(0,v0-2):v0+3, max(0,u0-2):u0+3]; vals = vals[vals > 0]
    if len(vals) == 0:
        continue
    z = float(np.median(vals))
    Pc = np.array([(u0-K[0,2])/K[0,0]*z, (v0-K[1,2])/K[1,1]*z, z])
    T = poses[i]; Pw = T[:3,:3] @ Pc + T[:3,3]
    size = diameter_mm(mask.astype(np.float32), deps[i], K) or 0.0
    conf = float(mask[binm].mean())
    hits.append(dict(i=i, pos=Pw, size=size, conf=conf, u=u0, v=v0))
print(f"[detect] {len(hits)} raw detections over {len(frames)} frames", flush=True)

kept = [x for x in hits if x["conf"] >= CONF and SMIN <= x["size"] <= SMAX]
used = [False]*len(kept); lesions = []
for a in range(len(kept)):
    if used[a]: continue
    grp = [kept[a]]; used[a] = True
    for b in range(a+1, len(kept)):
        if not used[b] and np.linalg.norm(kept[a]["pos"]-kept[b]["pos"]) < 0.010:
            grp.append(kept[b]); used[b] = True
    if len(grp) >= SUPPORT:
        lesions.append(dict(
            frames=len(grp),
            size_mm=round(float(np.median([g["size"] for g in grp])), 1),
            conf=round(float(np.mean([g["conf"] for g in grp])), 2),
            pos=[round(float(v),4) for v in np.mean([g["pos"] for g in grp], 0)]))
print(f"[loop] {len(lesions)} lesion(s) at frozen operating point:", flush=True)
for L in lesions:
    print(f"   {L['size_mm']}mm (nominal scale), confidence {L['conf']}, "
          f"persisted {L['frames']}/{len(frames)} frames at {L['pos']}", flush=True)
json.dump(dict(clip="hyperkvasir_9e3f0999", frames=len(frames), raw=len(hits),
               operating_point=dict(conf=CONF, support=SUPPORT, size=[SMIN, SMAX]),
               lesions=lesions), open("results/closed_loop_hk1.json", "w"), indent=1)

# 5. annotated evidence frame
if lesions and kept:
    mid = kept[len(kept)//2]
    f = frames[mid["i"]].copy()
    m = segment(f, model); m[static > 0] = 0
    binm = (m > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(f, cnts, -1, (60, 255, 60), 3)
    cv2.putText(f, f"lesion: ~{lesions[0]['size_mm']}mm  conf {lesions[0]['conf']}  "
                f"{lesions[0]['frames']} frames", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(f, f"lesion: ~{lesions[0]['size_mm']}mm  conf {lesions[0]['conf']}  "
                f"{lesions[0]['frames']} frames", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90,255,90), 2, cv2.LINE_AA)
    cv2.imwrite("results/hk_clip1_loop_result.jpg", f)
    print("evidence frame -> results/hk_clip1_loop_result.jpg", flush=True)
