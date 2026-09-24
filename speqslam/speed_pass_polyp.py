"""Speed pass for the on-device polyp detector, accuracy-gated like the depth one.

Builds a MatMul-only dynamic-int8 variant of the fine-tuned polyp ONNX and
promotes it to the app (assets/polyp_q.onnx) only if it passes two gates on the
documented Kvasir-SEG validation split (the 200 images the fine-tune held out):
  1. val Dice within 0.003 of fp32;
  2. per-image mask agreement with fp32 (IoU of thresholded masks) >= 0.98 mean
     - if int8 reproduces fp32's masks almost exactly, downstream behavior
     (fire rates, persistence counts, the frozen operating point) cannot move.
Motivation: detection is now the live study's dominant stage (~1.8 s/frame CPU
after the NNAPI revert); depth's identical treatment gave 1.6-2.3x at zero cost.
"""
import os, sys, glob, time
import numpy as np, cv2
import onnxruntime as ort

SRC = "models/argus_lite_polyp_ft.onnx"
INT8 = "models/argus_lite_polyp_ft_int8.onnx"
APP = "arguslite/app/src/main/assets/polyp_q.onnx"
VAL_DIR = "data/kvasir_seg"
IN_W, IN_H = 384, 288
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

if not os.path.exists(INT8):
    # STATIC quantization (QDQ, per-channel): this network is convolution-heavy,
    # so the MatMul-only dynamic trick that sped depth up did nothing here
    # (verified: same size, no speedup). Static quant needs calibration data.
    from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                          CalibrationDataReader)

    class Reader(CalibrationDataReader):
        def __init__(self, paths):
            self.paths = list(paths); self.i = 0
        def get_next(self):
            if self.i >= len(self.paths):
                return None
            bgr = cv2.imread(self.paths[self.i]); self.i += 1
            x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
            return {"pixels": x.astype(np.float32)}

    calib = sorted(glob.glob(f"{VAL_DIR}/Kvasir-SEG/images/*.jpg"))[:64]
    quantize_static(SRC, INT8, Reader(calib), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)
    print("static int8 written", flush=True)
for p in (SRC, INT8):
    print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB", flush=True)

# the documented split, replicated exactly from train_detector.py:
# random.Random(7).shuffle over the sorted image list, val = [800:]
import random
imgs = sorted(glob.glob(f"{VAL_DIR}/Kvasir-SEG/images/*.jpg"))
assert len(imgs) >= 1000, f"kvasir layout? {len(imgs)}"
rng = random.Random(7)
rng.shuffle(imgs)
val_imgs = imgs[800:1000]
# pair by basename into the masks directory (never string-replace path
# separators: a mixed-slash Windows glob path silently defeats it, and the
# first version of this harness scored predictions against the RGB image)
val_masks = [f"{VAL_DIR}/Kvasir-SEG/masks/{os.path.basename(p)}" for p in val_imgs]
assert all(os.path.exists(m) for m in val_masks), "mask pairing failed"
assert val_masks[0] != val_imgs[0]
print(f"validating on {len(val_imgs)} held-out images (train_detector split)",
      flush=True)

def prep(bgr):
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]

def run_all(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    outs = []
    t0 = time.perf_counter()
    for p in val_imgs:
        x = prep(cv2.imread(p))
        prob = sess.run(None, {iname: x})[0][0, 1]
        outs.append(prob)
    ms = 1000 * (time.perf_counter() - t0) / len(val_imgs)
    return outs, ms

def dice_of(prob, mask_path):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (prob.shape[1], prob.shape[0]),
                    interpolation=cv2.INTER_NEAREST) > 127
    pr = prob > 0.5
    inter = (gt & pr).sum()
    return 2 * inter / max(1, gt.sum() + pr.sum())

o32, ms32 = run_all(SRC)
o8, ms8 = run_all(INT8)
d32 = float(np.mean([dice_of(p, m) for p, m in zip(o32, val_masks)]))
d8 = float(np.mean([dice_of(p, m) for p, m in zip(o8, val_masks)]))
ious = []
for a, b in zip(o32, o8):
    A = a > 0.5; B = b > 0.5
    u = (A | B).sum()
    ious.append(1.0 if u == 0 else (A & B).sum() / u)
agree = float(np.mean(ious))
print(f"fp32: {ms32:.0f} ms/img (workstation proxy) | val Dice {d32:.4f}", flush=True)
print(f"int8: {ms8:.0f} ms/img | val Dice {d8:.4f} | mask agreement IoU {agree:.4f}",
      flush=True)

# third gate: identical FIRE verdicts on the training's two held-out control
# clips (the frozen operating point's negative behavior must not move)
import csv
rows = [r for r in csv.DictReader(open("results/batch_study/manifest.csv"))
        if r["split"] == "calib" and r["label"] == "nopolyp"]
clips = [r["file"] for r in rows][-2:]
frames = []
for c in clips:
    cap = cv2.VideoCapture(c)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(40):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(1, N) / 40))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()

def fires(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = []
    for f in frames:
        prob = sess.run(None, {iname: prep(f)})[0][0, 1]
        binm = prob > 0.5
        out.append(binm.mean() > 0.001 and (prob[binm].mean() if binm.any() else 0) >= 0.7)
    return np.array(out)

f32 = fires(SRC)
f8 = fires(INT8)
match = float((f32 == f8).mean())
print(f"control fire verdicts: fp32 {f32.mean()*100:.1f}% | int8 {f8.mean()*100:.1f}% "
      f"| identical on {match*100:.1f}% of {len(frames)} frames", flush=True)

if d32 - d8 <= 0.003 and agree >= 0.98 and match >= 0.97:
    import shutil
    shutil.copy(INT8, APP)
    print(f"PROMOTED int8 -> {APP} (Dice {d8:.4f} vs {d32:.4f}, agree {agree:.3f}, "
          f"fire-match {match:.3f})", flush=True)
else:
    if os.path.exists(APP):
        os.remove(APP)
        print(f"removed stale {APP}", flush=True)
    print("int8 failed a gate; app keeps fp32", flush=True)
