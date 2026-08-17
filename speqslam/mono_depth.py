"""
Stage C prerequisite: MONOCULAR depth estimation with a pre-trained network (no depth sensor).

Uses Depth-Anything-V2 (implemented, not trained -- matches the project's CV-first, pre-trained-model
approach). The network outputs affine-invariant inverse depth (disparity). We evaluate it against
C3VD's ground-truth metric depth using the standard MiDaS/DPT protocol: least-squares align the
predicted disparity to GT disparity (scale + shift), invert to metric depth, then report AbsRel and
delta<1.25. This tells us honestly whether off-the-shelf depth is good enough to carry a colon map.
"""
import sys, os, glob, numpy as np, cv2, torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DS = 100.0 / 65535.0 / 1000.0
MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
_proc = _model = None


def _load():
    global _proc, _model
    if _model is None:
        _proc = AutoImageProcessor.from_pretrained(MODEL)
        _model = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    return _proc, _model


@torch.no_grad()
def predict_disparity(bgr):
    """Return per-pixel affine-invariant inverse depth (disparity), same HxW as input."""
    proc, model = _load()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = proc(images=rgb, return_tensors="pt").to("cuda")
    pred = model(**inp).predicted_depth                       # (1,h,w), larger = nearer
    disp = torch.nn.functional.interpolate(pred[:, None], size=bgr.shape[:2],
                                           mode="bicubic", align_corners=False)[0, 0]
    return disp.float().cpu().numpy()


def align_disp_to_metric(disp, depth_gt):
    """Least-squares fit disp_gt = a*disp + b over valid pixels, return predicted METRIC depth."""
    m = depth_gt > 0
    dg = 1.0 / depth_gt[m]
    A = np.stack([disp[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, dg, rcond=None)[0]
    pred_disp = a * disp + b
    pred_depth = 1.0 / np.clip(pred_disp, 1e-6, None)
    return pred_depth, (a, b)


def evaluate(seq, n=25):
    undist = f"data/c3vd/{seq}_undist"
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))
    deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    absrel, d1, corr = [], [], []
    for i in idx:
        bgr = cv2.imread(imgs[i])
        dgt = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        disp = predict_disparity(bgr)
        pred, _ = align_disp_to_metric(disp, dgt)
        m = (dgt > 0) & np.isfinite(pred) & (pred > 0) & (pred < 0.5)   # drop non-physical tail
        if m.sum() < 100:
            continue
        ratio = np.maximum(pred[m] / dgt[m], dgt[m] / pred[m])
        absrel.append(np.median(np.abs(pred[m] - dgt[m]) / dgt[m]))     # median = robust to tail
        d1.append((ratio < 1.25).mean())
        # alignment-free STRUCTURAL quality: does predicted disparity rank-match GT inverse depth?
        mm = dgt > 0
        corr.append(np.corrcoef(disp[mm], 1.0 / dgt[mm])[0, 1])
    print(f"{seq:<12} frames={len(absrel):>3}  medAbsRel={np.mean(absrel):.3f}  "
          f"delta<1.25={100*np.mean(d1):.1f}%  struct_corr={np.mean(corr):.3f}  "
          f"(depth {np.median(dgt[dgt>0])*1000:.0f}mm)", flush=True)
    return np.mean(absrel), np.mean(d1), np.mean(corr)


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
    print(f"Monocular depth ({MODEL}) vs C3VD ground truth:\n", flush=True)
    for s in seqs:
        evaluate(s)
