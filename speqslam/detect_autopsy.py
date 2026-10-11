"""Dense autopsy of the 10 polyp clips the frozen 41-clip batch study missed.

The batch protocol samples 80 frames/clip. This script runs the SAME deployed
detector (polyp_ft/dlv3p_ft_ep8) over up to 400 frames per missed clip and asks,
per clip: does the polyp become visible to the detector in frames the protocol
never sampled (sampling miss), or is the detector near-silent everywhere
(recognition miss)? Fire criterion identical to batch_run.py: mask>0.5 over
>=0.5% of the frame, static-overlay pixels zeroed.

Diagnostic only - no pass/fail claim. Writes results/detect_autopsy/summary.csv
and the 6 highest-area overlay JPEGs per clip for visual inspection.
"""
import csv, os, sys, time
import numpy as np, cv2, torch

sys.path.insert(0, "speqslam")
from polyp_detect import load_model, segment, DEV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "results/detect_autopsy"
os.makedirs(OUT, exist_ok=True)

rows = []
for f in ["results/batch_study/results_ft.csv", "results/batch_study/results_ft_ext.csv"]:
    rows += list(csv.DictReader(open(f)))
missed = [r for r in rows if r["split"] != "calib" and r["label"] == "polyp"
          and int(r["maxC"]) < 14]
print(f"{len(missed)} missed clips")

model = load_model()
model.load_state_dict(torch.load("models/polyp_ft/dlv3p_ft_ep8.pth",
                                 map_location=DEV, weights_only=True))
model.eval()

sumpath = os.path.join(OUT, "summary.csv")
done = set()
if os.path.exists(sumpath):
    done = {r["file"] for r in csv.DictReader(open(sumpath))}
else:
    with open(sumpath, "w", newline="") as f:
        csv.writer(f).writerow(["file", "batch_fired", "total_frames", "dense_n",
                                "dense_fired", "fire_frac", "max_area_pct",
                                "max_conf", "verdict"])

for r in missed:
    if r["file"] in done:
        continue
    t0 = time.time()
    stem = os.path.splitext(os.path.basename(r["file"]))[0]
    cap = cv2.VideoCapture(r["file"])
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, N // 400)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    # static-overlay mask, same rule as the batch (std over a subsample < 4 gray levels)
    sub = frames[:: max(1, len(frames) // 80)]
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sub]).astype(np.float32)
    static = (stack.std(0) < 4.0).astype(np.uint8) * 255
    static = cv2.dilate(static, np.ones((7, 7), np.uint8))

    fired = 0; best = []   # (area, idx, conf, mask)
    for k, f in enumerate(frames):
        mask = segment(f, model)
        mask[static > 0] = 0
        binm = mask > 0.5
        area = float(binm.mean())
        conf = float(mask[binm].mean()) if binm.any() else 0.0
        if area >= 0.005:
            fired += 1
        best.append((area, k, conf))
    best.sort(reverse=True)
    cdir = os.path.join(OUT, stem)
    os.makedirs(cdir, exist_ok=True)
    for rank, (area, k, conf) in enumerate(best[:6]):
        f = frames[k].copy()
        m = segment(f, model)
        m[static > 0] = 0
        f[m > 0.5] = (0.35 * f[m > 0.5] + np.array([0, 160, 0]) * 0.65).astype(np.uint8)
        cv2.putText(f, f"frame {k*step}/{N} area {area*100:.2f}% conf {conf:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(cdir, f"r{rank}_f{k*step}.jpg"), f)
    frac = fired / len(frames)
    mx_a, _, mx_c = best[0]
    verdict = ("sampling_miss" if fired >= 8 and int(r["fired"]) <= 3 else
               "weak_recognition" if mx_a >= 0.005 else "recognition_dead")
    with open(sumpath, "a", newline="") as f:
        csv.writer(f).writerow([r["file"], r["fired"], N, len(frames), fired,
                                f"{frac:.3f}", f"{mx_a*100:.2f}", f"{mx_c:.2f}", verdict])
    print(f"{stem}: batch fired {r['fired']}/80 -> dense {fired}/{len(frames)} "
          f"(max area {mx_a*100:.2f}%, conf {mx_c:.2f}) => {verdict} "
          f"({time.time()-t0:.0f}s)", flush=True)

print("autopsy complete")
