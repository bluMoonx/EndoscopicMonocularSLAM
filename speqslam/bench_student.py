"""Gate G3 for the distilled student: export to ONNX, parity-check, then time
teacher vs student on CPU (the workstation proxy for the phone). Prints the
speedup; gate is >= 2.5x. Does NOT touch the app assets - installation only
happens after ALL gates pass and the operating point is re-derived.
Usage: python speqslam/bench_student.py [weights (default student_best.pth)]"""
import sys, os, time
import numpy as np, cv2, torch
sys.path.insert(0, "speqslam")
from polyp_detect import IN_W, IN_H, MEAN, STD

W = sys.argv[1] if len(sys.argv) > 1 else "models/polyp_student/student_best.pth"
OUT = "models/polyp_student/student.onnx"
TEACHER_ONNX = "models/argus_lite_polyp_ft.onnx"

import segmentation_models_pytorch as smp
student = smp.DeepLabV3Plus(encoder_name="efficientnet-b0", encoder_weights=None,
                            classes=2, activation="sigmoid").cpu().eval()
student.load_state_dict(torch.load(W, map_location="cpu", weights_only=True))
torch.onnx.export(student, torch.zeros(1, 3, IN_H, IN_W), OUT,
                  input_names=["pixels"], output_names=["mask"], opset_version=17)
print(f"exported {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, "
      f"teacher {os.path.getsize(TEACHER_ONNX)/1e6:.1f} MB)", flush=True)

import onnxruntime as ort
img = cv2.imread("data/kvasir_seg/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg")
x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
with torch.no_grad():
    ref = student(torch.from_numpy(x))[0, 1].numpy()
s_sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
diff = np.abs(ref - s_sess.run(None, {"pixels": x})[0][0, 1]).max()
print(f"parity: max |pytorch - onnx| = {diff:.2e} {'OK' if diff < 1e-3 else 'FAIL'}",
      flush=True)

# One session alive at a time: two live ORT sessions each hold a spin-waiting
# full-core thread pool and inflate BOTH timings 2-5x (measured).
def med_ms(path, n=50):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for _ in range(5):
        sess.run(None, {"pixels": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"pixels": x})
        ts.append((time.perf_counter() - t0) * 1000)
    del sess
    return float(np.median(ts))
del s_sess
tt, ts_ = med_ms(TEACHER_ONNX), med_ms(OUT)
print(f"teacher {tt:.0f} ms | student {ts_:.0f} ms | speedup {tt/ts_:.2f}x | "
      f"gate >=2.5x: {'PASS' if tt/ts_ >= 2.5 else 'FAIL'}", flush=True)
