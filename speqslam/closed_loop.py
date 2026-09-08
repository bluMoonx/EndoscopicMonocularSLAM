"""
The closed loop: detection wired INTO the mapping pipeline. While a sequence is reconstructed,
every frame also runs polyp segmentation; detections are back-projected through that frame's depth
and camera pose into 3D, clustered across frames (a real lesion persists across many frames --
one-frame flashes are discarded), then sized. Output: lesion sites (location, size, confidence,
frame support) alongside the blind-spot map = the full procedure report.

All available video data is deliberately lesion-free (C3VD phantom, EndoSLAM tumor-free), so on
these sequences the loop acts as a NEGATIVE CONTROL: few/no persistent clusters should survive.

Usage: python speqslam/closed_loop.py cecum_t1_a | endoslam1
"""
import sys, os, glob, json
import numpy as np, cv2
sys.path.insert(0, "speqslam")
from validate_coverage import load_poses
from polyp_detect import load_model, segment
from polyp_measure import diameter_mm

DS = 100.0 / 65535.0 / 1000.0
MIN_AREA = 0.005          # detection must cover >0.5% of frame
MIN_SUPPORT = 5           # lesion must persist in >=5 frames
CLUSTER_R = 0.010         # detections within 10mm belong to one lesion


def frames_for(name):
    if name.startswith("endoslam"):
        p = f"data/endoslam/proc/colon4_traj{name[-1]}"
        return (sorted(glob.glob(f"{p}/image01/*.jpg")), sorted(glob.glob(f"{p}/depth01_ft/*.png")),
                np.loadtxt(f"{p}/K.txt"), load_poses(f"results/endoslam/colon4_traj{name[-1]}_plain.txt"))
    u = f"data/c3vd/{name}_undist"
    return (sorted(glob.glob(f"{u}/image01/*.jpg")), sorted(glob.glob(f"{u}/depth01_ftc/*.png")),
            np.loadtxt(f"{u}/K.txt"), load_poses(f"results/c3vd/{name}_ft.txt"))


def run(name):
    imgs, deps, K, poses = frames_for(name)
    model = load_model()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hits = []          # (world_xyz, size_mm, conf, frame)
    n_det_frames = 0
    for i in range(len(imgs)):
        ts = round(i / 30.0, 6)
        if ts not in poses:
            continue
        bgr = cv2.imread(imgs[i])
        mask = segment(bgr, model)
        binm = mask > 0.5
        if binm.mean() < MIN_AREA:
            continue
        n_det_frames += 1
        dep = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        ys, xs = np.nonzero(binm)
        u0, v0 = int(xs.mean()), int(ys.mean())
        patch = dep[max(0, v0 - 2):v0 + 3, max(0, u0 - 2):u0 + 3]
        vals = patch[patch > 0]
        if len(vals) == 0:
            continue
        z = float(np.median(vals))
        Pc = np.array([(u0 - cx) / fx * z, (v0 - cy) / fy * z, z])
        T = poses[ts]
        Pw = T[:3, :3] @ Pc + T[:3, 3]
        size = diameter_mm(mask.astype(np.float32), dep, K) or 0.0
        hits.append((Pw, size, float(mask[binm].mean()), i))
    # greedy 3D clustering with temporal-support filter
    lesions = []
    used = np.zeros(len(hits), bool)
    for a in range(len(hits)):
        if used[a]:
            continue
        group = [a]
        for b in range(a + 1, len(hits)):
            if not used[b] and np.linalg.norm(hits[a][0] - hits[b][0]) < CLUSTER_R:
                group.append(b); used[b] = True
        used[a] = True
        if len(group) >= MIN_SUPPORT:
            g = [hits[j] for j in group]
            lesions.append(dict(
                pos=[round(float(v), 4) for v in np.mean([h[0] for h in g], 0)],
                size_mm=round(float(np.median([h[1] for h in g])), 1),
                confidence=round(float(np.mean([h[2] for h in g])), 2),
                frames=len(g)))
    out = dict(sequence=name, frames_total=len(imgs), frames_with_detection=n_det_frames,
               raw_detections=len(hits), persistent_lesions=lesions)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/closed_loop_{name}.json", "w"), indent=1)
    print(f"{name}: {len(imgs)} frames | {n_det_frames} frames fired | {len(hits)} raw detections "
          f"| {len(lesions)} persistent lesion(s) after temporal filter", flush=True)
    for L in lesions:
        print(f"   lesion: {L['size_mm']}mm, confidence {L['confidence']}, "
              f"seen in {L['frames']} frames at {L['pos']}", flush=True)
    print(f"-> results/closed_loop_{name}.json", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["cecum_t1_a", "endoslam1"]):
        run(s)
