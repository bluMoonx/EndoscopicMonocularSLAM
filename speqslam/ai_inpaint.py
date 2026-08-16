"""
Argus - AI inpainting condition. Same glare detector as Stage 1, but fills the glare with a
pretrained deep inpainter (LaMa) that produces realistic TEXTURE instead of Telea's smooth mush,
so ORB-SLAM3 gets features where the glare was. Writes a cleaned sequence (drop-in for SLAM).
"""
import cv2, numpy as np, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glare_removal import specular_mask, load_rgb_txt
from simple_lama_inpainting import SimpleLama
from PIL import Image


def process(in_seq, out_seq, k=2.0, lo=225, hi=245, dilate=1):
    os.makedirs(out_seq, exist_ok=True)
    header, entries = load_rgb_txt(in_seq)
    lama = SimpleLama()
    for i, (ts, rel) in enumerate(entries):
        bgr = cv2.imread(os.path.join(in_seq, rel))
        mask = specular_mask(bgr, method="adaptive", k=k, lo=lo, hi=hi, dilate=dilate)
        if mask.max() == 0:
            out = bgr  # nothing to inpaint
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = np.array(lama(Image.fromarray(rgb), Image.fromarray(mask).convert("L")))
            out = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            if out.shape[:2] != bgr.shape[:2]:
                out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]))
        op = os.path.join(out_seq, rel)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, out)
        if i % 200 == 0:
            print(f"  {i}/{len(entries)} frames", flush=True)
    with open(os.path.join(out_seq, "rgb.txt"), "w") as f:
        f.write("\n".join(header) + "\n")
    print(f"AI-inpainted sequence -> {out_seq}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True)
    ap.add_argument("--out_seq", required=True)
    a = ap.parse_args()
    process(a.in_seq, a.out_seq)
