"""Export the fine-tuned polyp segmenter to ONNX for Argus Lite, with a parity check.
Writes models/argus_lite_polyp_ft.onnx and, if parity passes, installs it as the app's
assets/polyp.onnx (previous asset backed up alongside)."""
import sys, os, shutil
import numpy as np, cv2, torch
sys.path.insert(0, "speqslam")
from polyp_detect import load_model, IN_W, IN_H, MEAN, STD

FT = "models/polyp_ft/dlv3p_ft_ep8.pth"
OUT = "models/argus_lite_polyp_ft.onnx"
APP = "arguslite/app/src/main/assets/polyp.onnx"

model = load_model().cpu().eval()
model.load_state_dict(torch.load(FT, map_location="cpu", weights_only=True))

dummy = torch.zeros(1, 3, IN_H, IN_W)
torch.onnx.export(model, dummy, OUT, input_names=["pixels"], output_names=["mask"],
                  opset_version=17)
print(f"exported {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

# parity: pytorch vs onnxruntime on a real Kvasir image
import onnxruntime as ort
img = cv2.imread("data/kvasir_seg/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg")
x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
with torch.no_grad():
    ref = model(torch.from_numpy(x))[0, 1].numpy()
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
got = sess.run(None, {"pixels": x})[0][0, 1]
diff = np.abs(ref - got).max()
print(f"parity: max |pytorch - onnx| = {diff:.2e}", flush=True)
if diff < 1e-3:
    if os.path.exists(APP):
        shutil.copy(APP, APP + ".pre_ft.bak")
    shutil.copy(OUT, APP)
    print(f"installed as {APP} (old asset backed up)", flush=True)
else:
    print("PARITY FAILED - app asset NOT replaced", flush=True)
