"""Depth speed push: the phone's bottleneck is now the depth network (113 ms
int8 at 252 px, measured in-app). A ViT's cost falls superlinearly with input
resolution, and the detector campaign proved resolution is the honest speed
lever. Candidates: the SAME deployed model exported at 196 and 168 px
(multiples of the ViT patch size), int8-dynamic MatMul like the shipped
model.

PRE-REGISTERED GATES (per candidate, vs the shipped 252-int8 baseline, all
must pass; stated before any result):
  G1 accuracy    delta<1.25 on C3VD GT within 1.0 point of the 252-int8
                 baseline on EACH of the five GT sequences; every model's
                 output is upsampled to the common 252 grid before scoring,
                 one global scale per sequence (the standard protocol).
  G2 downstream  direct-VO ATE driven by the candidate's ONNX depth within
                 +10% of the 252-int8-driven ATE on both smoke sequences
                 (trans_t1_a, cecum_t1_a).
  G3 speed       >= 1.5x the 252-int8 latency, workstation CPU proxy, one
                 session alive, median of 50.
SHIP RULE: the smallest resolution passing all gates is STAGED as
models/depth_speed/depth_q<res>.onnx; the app asset swap happens only after
the in-app benchmark card confirms the speedup on-device (next phone
session) - the NNAPI lesson: no on-device claim without an on-device
measurement.

Usage: python speqslam/depth_speed.py [minutes]   (budget-resumable)
"""
import os, sys, glob, json, time
import numpy as np, cv2
BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import onnxruntime as ort

OUT = "models/depth_speed"
os.makedirs(OUT, exist_ok=True)
BASE_INT8 = "models/argus_lite_depth252_int8.onnx"
RESES = (196, 168)
SEQS5 = ["trans_t1_a", "trans_t2_a", "trans_t2_b", "desc_t4_a", "cecum_t1_a"]
SMOKE = ["trans_t1_a", "cecum_t1_a"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DS = 100.0 / 65535.0 / 1000.0
STATE = f"{OUT}/state.json"
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
t0 = time.time()

def save():
    json.dump(state, open(STATE, "w"), indent=1)

def budget():
    if time.time() - t0 > BUDGET_S:
        save()
        print("budget reached; resume me", flush=True)
        sys.exit(0)

# ---- 1. export + quantize candidates ----
for res in RESES:
    fp = f"{OUT}/depth{res}.onnx"
    q = f"{OUT}/depth_q{res}.onnx"
    if os.path.exists(q):
        continue
    budget()
    import torch
    from transformers import AutoModelForDepthEstimation
    m = AutoModelForDepthEstimation.from_pretrained("models/depth_c3vd_demo").cpu().eval()
    class Wrap(torch.nn.Module):
        def __init__(self, mm):
            super().__init__(); self.m = mm
        def forward(self, x):
            return self.m(pixel_values=x).predicted_depth
    torch.onnx.export(Wrap(m), torch.zeros(1, 3, res, res), fp,
                      input_names=["pixel_values"], output_names=["depth"],
                      opset_version=17)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp, q, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    print(f"exported {q} ({os.path.getsize(q)/1e6:.1f} MB)", flush=True)

def sess_of(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def predict(sess, bgr, res):
    rgb = cv2.cvtColor(cv2.resize(bgr, (res, res)), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    d = sess.run(None, {sess.get_inputs()[0].name:
                        x.transpose(2, 0, 1)[None]})[0][0, 0]
    return d

# ---- 2. G1 accuracy on the five GT sequences ----
def d1_for(path, res, seq):
    sess = sess_of(path)
    imgs = sorted(glob.glob(f"data/c3vd/{seq}_undist/image01/*.jpg"))[::4]
    gts = sorted(glob.glob(f"data/c3vd/{seq}_undist/depth01/*.png"))[::4]
    scale = None
    accs = []
    for ip, gp in zip(imgs, gts):
        bgr = cv2.imread(ip)
        disp = predict(sess, bgr, res)
        disp = cv2.resize(disp, (252, 252), interpolation=cv2.INTER_LINEAR)
        d = 1.0 / np.clip(disp, 1.0 / 0.1, None)
        gt = cv2.imread(gp, cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        gt = cv2.resize(gt, (252, 252), interpolation=cv2.INTER_NEAREST)
        ok = (gt > 0.003) & (gt < 0.099)
        if ok.sum() < 500:
            continue
        if scale is None:
            scale = float(np.median(gt[ok] / np.maximum(d[ok], 1e-9)))
        e = d * scale
        r = np.maximum(e[ok] / gt[ok], gt[ok] / np.maximum(e[ok], 1e-9))
        accs.append(float((r < 1.25).mean() * 100))
    return float(np.mean(accs))

if "g1" not in state:
    state["g1"] = {}
for tag, path, res in [("252", BASE_INT8, 252)] + \
        [(str(r), f"{OUT}/depth_q{r}.onnx", r) for r in RESES]:
    for seq in SEQS5:
        key = f"{tag}/{seq}"
        if key in state["g1"]:
            continue
        budget()
        state["g1"][key] = d1_for(path, res, seq)
        print(f"G1 {key}: d1 {state['g1'][key]:.2f}%", flush=True)
        save()

# ---- 3. G2 downstream VO smoke ----
import direct_vo as dv
from c3vdv2_deform import umeyama_ate

def tum_positions(path):
    out = []
    for l in open(path):
        v = l.split()
        if len(v) >= 8:
            out.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(out)

def vo_ate(path, res, seq):
    sess = sess_of(path)
    base = f"data/c3vd/{seq}_undist"
    K = np.loadtxt(f"{base}/K.txt")
    imgs = sorted(glob.glob(f"{base}/image01/*.jpg"))
    gt = tum_positions(f"results/c3vd/{seq}_gt.txt")
    def dep_of(bgr):
        disp = predict(sess, bgr, res)
        disp = cv2.resize(disp, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return (1.0 / np.clip(disp, 1.0 / 0.1, None)).astype(np.float32)
    pose = np.eye(4); traj = [pose[:3, 3].copy()]
    b0 = cv2.imread(imgs[0])
    g0 = cv2.cvtColor(b0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d0 = dep_of(b0)
    for i in range(1, len(imgs)):
        b1 = cv2.imread(imgs[i])
        g1 = cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        T = dv.track_pair(g0, d0, g1, K, 4, robust=False, depth_scale=1.0,
                          illum=True)
        pose = pose @ np.linalg.inv(T)
        traj.append(pose[:3, 3].copy())
        g0, d0 = g1, dep_of(b1)
    n = min(len(gt), len(traj))
    return umeyama_ate(gt[:n], np.array(traj)[:n]) * 1000

if "g2" not in state:
    state["g2"] = {}
for tag, path, res in [("252", BASE_INT8, 252)] + \
        [(str(r), f"{OUT}/depth_q{r}.onnx", r) for r in RESES]:
    for seq in SMOKE:
        key = f"{tag}/{seq}"
        if key in state["g2"]:
            continue
        budget()
        state["g2"][key] = vo_ate(path, res, seq)
        print(f"G2 {key}: ATE {state['g2'][key]:.3f} mm", flush=True)
        save()

# ---- 4. G3 speed ----
if "g3" not in state:
    state["g3"] = {}
    for tag, path, res in [("252", BASE_INT8, 252)] + \
            [(str(r), f"{OUT}/depth_q{r}.onnx", r) for r in RESES]:
        sess = sess_of(path)
        x = np.random.randn(1, 3, res, res).astype(np.float32)
        nm = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {nm: x})
        ts = []
        for _ in range(50):
            a = time.perf_counter()
            sess.run(None, {nm: x})
            ts.append((time.perf_counter() - a) * 1000)
        del sess
        state["g3"][tag] = float(np.median(ts))
        print(f"G3 {tag}: {state['g3'][tag]:.1f} ms", flush=True)
    save()

# ---- verdict ----
print("\nVERDICT (pre-registered, vs 252-int8 baseline):", flush=True)
for r in RESES:
    tag = str(r)
    g1ok = all(state["g1"][f"{tag}/{s}"] >= state["g1"][f"252/{s}"] - 1.0
               for s in SEQS5)
    g2ok = all(state["g2"][f"{tag}/{s}"] <= state["g2"][f"252/{s}"] * 1.10
               for s in SMOKE)
    sp = state["g3"]["252"] / state["g3"][tag]
    g3ok = sp >= 1.5
    print(f"  {r}px: G1 {'PASS' if g1ok else 'FAIL'} | G2 "
          f"{'PASS' if g2ok else 'FAIL'} | G3 {sp:.2f}x "
          f"{'PASS' if g3ok else 'FAIL'} -> "
          f"{'CANDIDATE' if (g1ok and g2ok and g3ok) else 'rejected'}",
          flush=True)
