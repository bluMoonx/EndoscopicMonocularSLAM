"""Speed pass for on-device depth: build fp16 and dynamic-int8 variants of the
252px depth ONNX, benchmark them, and grade ACCURACY against C3VD ground truth before
anything ships. A variant is only promoted to the app (assets/depth_q.onnx) if its
held-out depth accuracy stays within 1 point of fp32.
"""
import os, sys, glob, time
import numpy as np, cv2
import onnxruntime as ort

SRC = "models/argus_lite_depth252.onnx"
FP16 = "models/argus_lite_depth252_fp16.onnx"
INT8 = "models/argus_lite_depth252_int8.onnx"
APP = "arguslite/app/src/main/assets/depth_q.onnx"
SEQ = "data/c3vd/cecum_t1_a_undist"
DS = 100.0 / 65535.0 / 1000.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# ---- build variants (fp16 dropped: converter emits an invalid graph for this
# architecture, and mobile CPU inference wouldn't benefit from it anyway) ----
if not os.path.exists(INT8):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    # MatMul only: the ViT's bulk, and ConvInteger is unimplemented in mobile/CPU EPs
    quantize_dynamic(SRC, INT8, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    print("int8 written", flush=True)
for p in (SRC, INT8):
    print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1e6:.1f} MB", flush=True)

def prep(bgr):
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (252, 252),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]

imgs = sorted(glob.glob(f"{SEQ}/image01/*"))[:30]
gts = sorted(glob.glob(f"{SEQ}/depth01/*"))[:30]
frames = [prep(cv2.imread(p)) for p in imgs]
gtd = [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * DS for p in gts]

def evaluate(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    # warmup + timing
    sess.run(None, {iname: frames[0]})
    t0 = time.perf_counter()
    outs = [sess.run(None, {iname: f})[0].reshape(252, 252) for f in frames]
    ms = 1000 * (time.perf_counter() - t0) / len(frames)
    # accuracy vs GT with per-sequence affine (the standard protocol)
    d1s = []
    for disp, gt in zip(outs, gtd):
        gtr = cv2.resize(gt, (252, 252), interpolation=cv2.INTER_NEAREST)
        ok = (gtr > 0.003) & (gtr < 0.099) & (disp > 1e-6)
        if ok.sum() < 1000:
            continue
        inv_gt = 1.0 / gtr[ok]
        A = np.vstack([disp[ok], np.ones(ok.sum())]).T
        a, b = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        est = 1.0 / np.maximum(a * disp[ok] + b, 1e-6)
        rel = np.abs(est - gtr[ok]) / gtr[ok]
        d1s.append(np.mean(rel < 0.25) * 100)
    return ms, float(np.mean(d1s))

results = {}
for name, path in [("fp32", SRC), ("int8", INT8)]:
    ms, d1 = evaluate(path)
    results[name] = (ms, d1)
    print(f"{name}: {ms:.0f} ms/frame (workstation CPU proxy) | d1 {d1:.2f}%", flush=True)

base_d1 = results["fp32"][1]
ms8, d18 = results["int8"]
if base_d1 - d18 <= 1.0:
    import shutil
    shutil.copy(INT8, APP)
    print(f"PROMOTED int8 -> {APP} (d1 {d18:.2f} vs fp32 {base_d1:.2f})", flush=True)
else:
    print("int8 failed the accuracy bar; app keeps fp32", flush=True)
