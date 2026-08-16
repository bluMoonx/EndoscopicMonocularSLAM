"""
Argus - Stage 1 glare-removal preprocessing layer.

Sits BEFORE ORB-SLAM3. For each frame it:
  1. finds the specular (glare) pixels with a per-frame ADAPTIVE brightness threshold,
  2. removes them by inpainting (filling the hole with surrounding tissue) so the SLAM
     feature extractor never locks onto the bright, lying blobs.

It reads a sequence folder that contains an ORB-SLAM3-style rgb.txt (3 header lines then
"timestamp relative/path.jpg" per line), writes a cleaned copy to a new sequence folder with
the SAME rgb.txt, and records a per-frame "specularity severity" score (the fraction of pixels
flagged as glare) -> this is the raw material for the severity curve later.

Methods:
  adaptive : threshold = max(abs_min, mean + k*std) computed PER FRAME  (Argus condition 3)
  fixed    : a single global brightness threshold                       (prior-work condition 2)
"""
import cv2, numpy as np, glob, os, argparse, csv


def specular_mask(img, method="adaptive", k=1.5, lo=200, hi=235, fixed_thr=220, dilate=2, max_area=None):
    """Return a uint8 mask (255 = glare) for a BGR image.

    Specular highlights are bright in every channel, so the per-pixel channel MAX is a clean
    "how white is this pixel" signal. The adaptive threshold adjusts to each frame's brightness
    but is CLAMPED to [lo, hi] so it never runs away above the near-saturated glare (the bug that
    made it catch nothing before) and never dips into normal tissue.

    max_area (optional): keep only connected bright components with area <= max_area pixels. Real
    specular highlights are compact blobs; large bright regions are just brightly-lit tissue. This
    is needed on evenly-lit footage (e.g. the C3VD phantom) where a plain threshold over-masks.
    """
    bright = img.max(axis=2).astype(np.float32)
    if method == "fixed":
        thr = fixed_thr  # one global threshold for the prior-work condition
    else:  # adaptive
        thr = float(np.clip(bright.mean() + k * bright.std(), lo, hi))
    mask = (bright >= thr).astype(np.uint8)
    if max_area is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] <= max_area:
                keep[labels == i] = 1
        mask = keep
    mask = mask * 255
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, kernel)  # grow the mask to cover the soft halo around each highlight
    return mask


def load_rgb_txt(seq_dir):
    lines = open(os.path.join(seq_dir, "rgb.txt")).read().splitlines()
    entries = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        ts, rel = ln.split()
        entries.append((ts, rel))
    return lines, entries


def process(in_seq, out_seq, method="adaptive", k=1.5, lo=200, hi=235, fixed_thr=220,
            dilate=2, inpaint_radius=3):
    os.makedirs(out_seq, exist_ok=True)
    header_lines, entries = load_rgb_txt(in_seq)
    severities = []
    for ts, rel in entries:
        img = cv2.imread(os.path.join(in_seq, rel))
        mask = specular_mask(img, method=method, k=k, lo=lo, hi=hi, fixed_thr=fixed_thr, dilate=dilate)
        severities.append(float((mask > 0).mean()))
        cleaned = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
        out_path = os.path.join(out_seq, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cleaned)
    # copy rgb.txt verbatim so the cleaned sequence is a drop-in replacement
    with open(os.path.join(out_seq, "rgb.txt"), "w") as f:
        f.write("\n".join(header_lines) + "\n")
    # save per-frame severity for the severity curve
    with open(os.path.join(out_seq, "severity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "specular_fraction"])
        for (ts, rel), s in zip(entries, severities):
            w.writerow([rel, f"{s:.6f}"])
    sev = np.array(severities)
    print(f"method={method}  frames={len(sev)}")
    print(f"per-sequence specularity severity: mean={sev.mean()*100:.2f}%  max={sev.max()*100:.2f}%")
    print(f"cleaned sequence written to: {out_seq}")
    return sev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_seq", required=True)
    ap.add_argument("--out_seq", required=True)
    ap.add_argument("--method", default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--k", type=float, default=1.5)
    ap.add_argument("--lo", type=float, default=200)
    ap.add_argument("--hi", type=float, default=235)
    ap.add_argument("--fixed_thr", type=float, default=220)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--inpaint_radius", type=int, default=3)
    a = ap.parse_args()
    process(a.in_seq, a.out_seq, a.method, a.k, a.lo, a.hi, a.fixed_thr, a.dilate, a.inpaint_radius)
