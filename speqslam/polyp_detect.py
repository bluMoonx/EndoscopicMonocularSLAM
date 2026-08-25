"""
Polyp detection: segment polyps in colonoscopy frames and outline them on screen.

Blind-spot mapping tells the clinician WHERE they haven't looked; this layer flags WHAT was seen --
polyps are the lesions whose miss rate (~25% in routine colonoscopy) motivates the whole system.
Uses a PRE-TRAINED DeepLabV3+ (EfficientNet-B4 encoder, ImageNet pre-training) fine-tuned for
polyp segmentation, from HuggingFace ahmeterdempmk/DeepLabV3Plus-With-EfficientNetB4-Encoder-
50-Epoch-Polyp-Segmentation-0.9793-IoU. We did NOT train it. Weights are stored as a plain
state_dict in models/polyp_seg/ (converted from the original full-model pickle).

CAVEAT for honest reporting: the model author does not document the training split. Its accuracy
on Kvasir-SEG is so high (mean Dice ~0.96 over all 1000 images) that the training data almost
certainly included part of Kvasir-SEG, so the eval number is best read as "reproduces expert
masks on in-distribution data", not as unseen-data performance.

Detect:  python speqslam/polyp_detect.py --image data/kvasir_seg/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg
         writes <image>_polyp.png next to results/polyp_examples/ (or --out path).
Eval:    python speqslam/polyp_detect.py --eval
         Dice + IoU on all 1000 Kvasir-SEG image/mask pairs -> results/polyp_eval.csv
"""
import os, glob, csv, time, argparse
import numpy as np, cv2, torch

WEIGHTS = "models/polyp_seg/dlv3p_effb4_polyp.pth"
HF_URL = ("https://huggingface.co/ahmeterdempmk/DeepLabV3Plus-With-EfficientNetB4-Encoder-"
          "50-Epoch-Polyp-Segmentation-0.9793-IoU/resolve/main/best_model.pth")
DATA = "data/kvasir_seg/Kvasir-SEG"
IN_W, IN_H = 384, 288
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    import segmentation_models_pytorch as smp
    m = smp.DeepLabV3Plus(encoder_name="efficientnet-b4", encoder_weights=None,
                          classes=2, activation="sigmoid")
    if not os.path.exists(WEIGHTS):  # first run: fetch original pickle, keep only the state_dict
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        torch.hub.download_url_to_file(HF_URL, WEIGHTS + ".orig")
        torch.save(torch.load(WEIGHTS + ".orig", map_location="cpu", weights_only=False).state_dict(), WEIGHTS)
        os.remove(WEIGHTS + ".orig")
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return m.to(DEV).eval()


def segment(bgr, model):
    """BGR frame -> float polyp-probability mask, same HxW, values in [0,1]."""
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        prob = model(t)[0, 1].cpu().numpy()  # channel 1 = polyp (0 = background)
    return np.clip(cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def overlay(bgr, mask, thr=0.5):
    """Bright-green contour + translucent fill + bounding box around each detected polyp."""
    out = bgr.copy()
    binm = (mask > thr).astype(np.uint8)
    fill = out.copy(); fill[binm > 0] = (90, 255, 90)
    out = cv2.addWeighted(fill, 0.30, out, 0.70, 0)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 0.0005 * binm.size:  # skip specks
            continue
        cv2.drawContours(out, [c], -1, (60, 255, 60), 2, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 255, 60), 1)
        cv2.putText(out, "polyp", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, "polyp", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (90, 255, 90), 1, cv2.LINE_AA)
    return out


def dice_iou(pred, gt):
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-9), inter / ((pred | gt).sum() + 1e-9)


def evaluate(thr=0.5):
    model = load_model()
    imgs = sorted(glob.glob(f"{DATA}/images/*.jpg"))
    rows, dices, ious, times = [("image", "dice", "iou")], [], [], []
    for p in imgs:
        bgr = cv2.imread(p)
        gt = cv2.imread(f"{DATA}/masks/{os.path.basename(p)}", 0) > 127
        t0 = time.perf_counter()
        mask = segment(bgr, model)
        times.append(time.perf_counter() - t0)
        d, i = dice_iou(mask > thr, gt)
        dices.append(d); ious.append(i)
        rows.append((os.path.basename(p), f"{d:.4f}", f"{i:.4f}"))
    rows.append(("MEAN", f"{np.mean(dices):.4f}", f"{np.mean(ious):.4f}"))
    rows.append(("MEDIAN", f"{np.median(dices):.4f}", f"{np.median(ious):.4f}"))
    os.makedirs("results", exist_ok=True)
    with open("results/polyp_eval.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"n={len(dices)}  mean Dice {np.mean(dices):.4f}  mean IoU {np.mean(ious):.4f}  "
          f"median Dice {np.median(dices):.4f}  |  {1000 * np.mean(times[1:]):.1f} ms/frame on {DEV}")
    print("saved results/polyp_eval.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    if a.eval:
        evaluate(a.thr)
    elif a.image:
        model = load_model()
        bgr = cv2.imread(a.image)
        vis = overlay(bgr, segment(bgr, model), a.thr)
        out = a.out or f"results/polyp_examples/{os.path.splitext(os.path.basename(a.image))[0]}_polyp.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, vis)
        print("saved", out)
    else:
        ap.print_help()
