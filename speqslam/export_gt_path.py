"""Export the full withdrawal's GROUND-TRUTH camera path for the app
(assets/gt_full.bin, same normalized frame as traj_full.bin), so the phone can
draw the truth next to its own estimate - the instrument displaying its own
error is the point.  Format: int32 n, then 3n float32 (mesh-normalized)."""
import struct, glob, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
from c3vdv2_deform import load_poses

BASE = "data/c3vdv2/c1_full_t4_v1_s3"
MESH = "arguslite/app/src/main/assets/mesh_full.bin"
OUT = "arguslite/app/src/main/assets/gt_full.bin"

b = open(MESH, "rb").read()
nv, nt = struct.unpack_from("<ii", b, 0)
V = np.frombuffer(b, np.float32, 3 * nv, 8).reshape(-1, 3)
center = (V.min(0) + V.max(0)) / 2
span = (V.max(0) - V.min(0)).max() / 2

nums = sorted(int(os.path.splitext(os.path.basename(p))[0])
              for p in glob.glob(f"{BASE}/rgb/*.png"))
gt_all = load_poses(f"{BASE}/pose.txt")
gt = np.array([gt_all[k][1] for k in nums])
gt = (gt - gt[0]) * 1000.0                      # mm, the frame the mesh lives in
tn = ((gt - center) / span).astype(np.float32)
with open(OUT, "wb") as f:
    f.write(struct.pack("<i", len(tn)))
    f.write(tn.tobytes())
print(f"wrote {OUT}: {len(tn)} pts, range {tn.min(0).round(2)}..{tn.max(0).round(2)}")
