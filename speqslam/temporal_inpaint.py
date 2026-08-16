"""
Argus - TEMPORAL inpainting condition. Instead of hallucinating what's under the glare
(single-frame AI), borrow the REAL tissue from nearby frames where that spot wasn't glared,
and warp it into place with dense optical flow. This yields temporally-consistent (real)
features - the thing single-frame inpainting can't provide. Any pixel still glared in every
neighbor falls back to Telea inpainting.
"""
import cv2, numpy as np, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glare_removal import specular_mask, load_rgb_txt


def _flow_warp(gray_t, frame_tp, mask_tp, xx, yy):
    """Warp neighbor frame+mask into the current frame's coordinates via Farneback flow."""
    gray_tp = cv2.cvtColor(frame_tp, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_t, gray_tp, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    mapx = (xx + flow[..., 0]).astype(np.float32)
    mapy = (yy + flow[..., 1]).astype(np.float32)
    w_frame = cv2.remap(frame_tp, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # warp the neighbor's glare mask too; off-image samples count as "glared" (255) so we don't trust them
    w_mask = cv2.remap(mask_tp, mapx, mapy, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return w_frame, w_mask


def temporal_fill(get, t, n, window=3):
    cur, M = get(t)
    filled = cur.copy()
    remaining = M > 0
    gray_t = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    h, w = M.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    offsets = []
    for d in range(1, window + 1):
        offsets += [d, -d]  # nearest neighbors first
    for off in offsets:
        if not remaining.any():
            break
        tp = t + off
        if tp < 0 or tp >= n:
            continue
        f_tp, m_tp = get(tp)
        w_frame, w_mask = _flow_warp(gray_t, f_tp, m_tp, xx, yy)
        fillable = remaining & (w_mask == 0)  # glared here, but this neighbor saw clean tissue
        filled[fillable] = w_frame[fillable]
        remaining &= ~fillable
    if remaining.any():  # glared in every neighbor -> classical fallback
        filled = cv2.inpaint(filled, (remaining * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return filled


def process(in_seq, out_seq, window=3, k=2.0, lo=225, hi=245, dilate=1):
    header, entries = load_rgb_txt(in_seq)
    n = len(entries)
    cache = {}

    def get(i):
        if i not in cache:
            bgr = cv2.imread(os.path.join(in_seq, entries[i][1]))
            cache[i] = (bgr, specular_mask(bgr, method="adaptive", k=k, lo=lo, hi=hi, dilate=dilate))
        return cache[i]

    os.makedirs(out_seq, exist_ok=True)
    for t in range(n):
        cur, M = get(t)
        out = cur if M.max() == 0 else temporal_fill(get, t, n, window)
        op = os.path.join(out_seq, entries[t][1])
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, out)
        for key in [kk for kk in cache if kk < t - window]:
            del cache[key]
        if t % 200 == 0:
            print(f"  {t}/{n}", flush=True)
    with open(os.path.join(out_seq, "rgb.txt"), "w") as f:
        f.write("\n".join(header) + "\n")
    print(f"temporally-inpainted sequence -> {out_seq}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True)
    ap.add_argument("--out_seq", required=True)
    ap.add_argument("--window", type=int, default=3)
    a = ap.parse_args()
    process(a.in_seq, a.out_seq, a.window)
