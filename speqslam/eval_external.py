"""Head-to-head zero-shot evaluation on the four external polyp benchmarks
(protocol v2, results/distill_protocol_v2.md - committed before this ran).

Both models see each image exactly as deployed: teacher at 384x288, student
at its trained resolution; predicted probability is bilinearly upsampled to
the ground-truth mask's native resolution and thresholded at 0.5; Dice per
image, mean per set and pooled. No thresholds, no tuning, single pass.

Usage: python speqslam/eval_external.py <student.pth> <stu_w> <stu_h> <tag>
"""
import os, sys, glob, json
import numpy as np, cv2, torch
sys.path.insert(0, "speqslam")
from polyp_detect import load_model, IN_W, IN_H, MEAN, STD, DEV

STUDENT_PTH = sys.argv[1]
STU_W, STU_H = int(sys.argv[2]), int(sys.argv[3])
TAG = sys.argv[4]
SETS = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
ROOT = "data/polyp_external"

teacher = load_model()
teacher.load_state_dict(torch.load("models/polyp_ft/dlv3p_ft_ep8.pth",
                                   map_location=DEV, weights_only=True))
teacher.eval()
import segmentation_models_pytorch as smp
student = smp.DeepLabV3Plus("efficientnet-b0", encoder_weights=None,
                            classes=2, activation="sigmoid").to(DEV)
student.load_state_dict(torch.load(STUDENT_PTH, map_location=DEV, weights_only=True))
student.eval()

def predict(model, bgr, w, h, gt_shape):
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        p = model(torch.from_numpy(x).float().to(DEV))[0, 1].cpu().numpy()
    return cv2.resize(p, (gt_shape[1], gt_shape[0]),
                      interpolation=cv2.INTER_LINEAR) > 0.5

out = {"student": STUDENT_PTH, "student_res": [STU_W, STU_H], "sets": {}}
pooled = {"t": [], "s": []}
for ds in SETS:
    imgs = sorted(glob.glob(f"{ROOT}/{ds}/images/*"))
    td, sd = [], []
    for ip in imgs:
        mp = ip.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
        if not os.path.exists(mp):
            base = os.path.splitext(os.path.basename(ip))[0]
            cand = glob.glob(f"{ROOT}/{ds}/masks/{base}.*")
            if not cand:
                continue
            mp = cand[0]
        bgr = cv2.imread(ip)
        gt = cv2.imread(mp, 0) > 127
        for key, model, w, h in (("t", teacher, IN_W, IN_H),
                                 ("s", student, STU_W, STU_H)):
            pred = predict(model, bgr, w, h, gt.shape)
            d = 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-9)
            (td if key == "t" else sd).append(float(d))
    out["sets"][ds] = {"n": len(td), "teacher_dice": round(float(np.mean(td)), 4),
                       "student_dice": round(float(np.mean(sd)), 4),
                       "gap": round(float(np.mean(td) - np.mean(sd)), 4)}
    pooled["t"] += td; pooled["s"] += sd
    print(f"{ds:20s} n={len(td):3d}  teacher {np.mean(td):.4f}  "
          f"student {np.mean(sd):.4f}  gap {np.mean(td)-np.mean(sd):+.4f}", flush=True)

pt, ps = float(np.mean(pooled["t"])), float(np.mean(pooled["s"]))
out["pooled"] = {"n": len(pooled["t"]), "teacher_dice": round(pt, 4),
                 "student_dice": round(ps, 4), "gap": round(pt - ps, 4)}
maxset = max(abs(v["gap"]) if v["gap"] > 0 else 0 for v in out["sets"].values())
out["gate_G1prime"] = {"pooled_gap_le_0.03": pt - ps <= 0.03,
                       "per_set_gap_le_0.05": maxset <= 0.05,
                       "verdict": "PASS" if (pt - ps <= 0.03 and maxset <= 0.05) else "FAIL"}
print(f"POOLED n={len(pooled['t'])}  teacher {pt:.4f}  student {ps:.4f}  "
      f"gap {pt-ps:+.4f}  G1' {out['gate_G1prime']['verdict']}", flush=True)
with open(f"results/external_eval_{TAG}.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"written results/external_eval_{TAG}.json", flush=True)
