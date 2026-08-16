"""
C3VD -> Argus-ready. Undistorts a C3VD sequence (fisheye -> pinhole) into the image01/depth01 +
rgb.txt layout the direct VO reads, and converts C3VD's ground-truth pose.txt into a TUM
trajectory for `evo`. Pose mapping verified against the real file: translation is in the LAST ROW
(mm -> m), and the top-left 3x3 is transposed to column-vector convention.
"""
import sys, os, glob, cv2, numpy as np, argparse
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

    colors = sorted(glob.glob(f"{base}/*_color.png"))
    with open(f"{out}/rgb.txt", "w") as f:
        f.write("# color images\n# C3VD undistorted (pinhole)\n# timestamp filename\n")
        for i, c in enumerate(colors):
            und = undistort(cv2.imread(c), mapu, mapv)
            cv2.imwrite(f"{out}/image01/{i:04d}.jpg", und)
            d = cv2.imread(c.replace("_color.png", "_depth.tiff"), cv2.IMREAD_UNCHANGED)
            du = undistort(d, mapu, mapv, nearest=True)
            cv2.imwrite(f"{out}/depth01/{i:04d}.png", du.astype(np.uint16))
            f.write(f"{i/30.0:.6f} image01/{i:04d}.jpg\n")
    print(f"undistorted {len(colors)} frames -> {out}")

    # ground-truth poses -> TUM
    os.makedirs("results/c3vd", exist_ok=True)
    lines = [l for l in open(f"{base}/pose.txt") if l.strip()]
    with open(f"results/c3vd/{seq}_gt.txt", "w") as g:
        for i, l in enumerate(lines):
            M = np.fromstring(l.strip(), sep=',').reshape(4, 4)   # row-major
            t = M[3, :3] / 1000.0                                  # translation in LAST ROW, mm->m
            q = R.from_matrix(M[:3, :3].T).as_quat()               # transpose -> col-vector; xyzw
            g.write(f"{i/30.0:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(f"wrote GT trajectory -> results/c3vd/{seq}_gt.txt ({len(lines)} poses)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seq", default="trans_t1_a")
    main(ap.parse_args().seq)
