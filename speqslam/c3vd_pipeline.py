"""
C3VD -> Argus-ready. Undistorts a C3VD sequence (fisheye -> pinhole) into the image01/depth01 +
rgb.txt layout the direct VO reads, and converts C3VD's ground-truth pose.txt into a TUM
trajectory for `evo`. Pose mapping verified against the real file: translation is in the LAST ROW
(mm -> m), and the top-left 3x3 is transposed to column-vector convention.
"""
import sys, os, glob, re, cv2, numpy as np, argparse
sys.path.insert(0, "speqslam")
from scaramuzza_undistort import load_calib, build_map, undistort
from scipy.spatial.transform import Rotation as R

CALIB = "data/c3vd/calib/cfhq190l_omnidirectional_params/cfhq190l_omnidirectional_values.txt"


def main(seq):
    base = f"data/c3vd/{seq}"
    out = f"data/c3vd/{seq}_undist"
    os.makedirs(f"{out}/image01", exist_ok=True)
    os.makedirs(f"{out}/depth01", exist_ok=True)
    cal = load_calib(CALIB)
    mapu, mapv, K = build_map(cal, 640, 512, 250.0)
    np.savetxt(f"{out}/K.txt", K)

    # pair color<->depth by FRAME NUMBER (robust to padded vs unpadded naming), ordered numerically
    def _num(p, tag):
        m = re.search(r"(\d+)_" + tag, os.path.basename(p))
        return int(m.group(1)) if m else -1
    colors = {_num(p, "color"): p for p in glob.glob(f"{base}/*_color.png")}
    depths = {_num(p, "depth"): p for p in glob.glob(f"{base}/*_depth.tiff")}
    nums = sorted(set(colors) & set(depths))
    with open(f"{out}/rgb.txt", "w") as f:
        f.write("# color images\n# C3VD undistorted (pinhole)\n# timestamp filename\n")
        for i, n in enumerate(nums):
            und = undistort(cv2.imread(colors[n]), mapu, mapv)
            cv2.imwrite(f"{out}/image01/{i:04d}.jpg", und)
            du = undistort(cv2.imread(depths[n], cv2.IMREAD_UNCHANGED), mapu, mapv, nearest=True)
            cv2.imwrite(f"{out}/depth01/{i:04d}.png", du.astype(np.uint16))
            f.write(f"{i/30.0:.6f} image01/{i:04d}.jpg\n")
    print(f"undistorted {len(nums)} frames -> {out}")

    # ground-truth poses -> TUM (aligned to the same paired frames)
    os.makedirs("results/c3vd", exist_ok=True)
    lines = [l for l in open(f"{base}/pose.txt") if l.strip()]
    with open(f"results/c3vd/{seq}_gt.txt", "w") as g:
        for i, n in enumerate(nums):
            M = np.fromstring(lines[n].strip(), sep=',').reshape(4, 4)  # row-major
            t = M[3, :3] / 1000.0                                        # translation in LAST ROW, mm->m
            q = R.from_matrix(M[:3, :3].T).as_quat()                     # transpose -> col-vector; xyzw
            g.write(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"wrote GT trajectory -> results/c3vd/{seq}_gt.txt ({len(nums)} poses)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seq", default="trans_t1_a")
    main(ap.parse_args().seq)
