"""Native-rate extraction without the disk blow-up: stream every frame straight
from the retained zip through undistortion into the compact JPEGs the tracker
reads, never writing the raw PNGs (whose full set exceeds free disk).
Produces data/c3vdv2/c1_full_t4_v1_s1/{undist/*.jpg, pose.txt, nums.json};
deletes the zip when every frame is done. Budget-resumable.
Usage: python speqslam/fetch_native_undist.py [minutes]
"""
import os, sys, json, time, zipfile
import numpy as np, cv2

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL

ZIP = "data/c3vdv2/c1_full_t4_v1.zip"
DEST = "data/c3vdv2/c1_full_t4_v1_s1"
UND = f"{DEST}/undist"
os.makedirs(UND, exist_ok=True)

mapu, mapv, K = su.build_map(V2CAL)
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    def framenum(n):
        stem = os.path.splitext(os.path.basename(n))[0]
        digits = "".join(c for c in stem if c.isdigit())
        return int(digits) if digits else 0
    imgs = sorted((n for n in names
                   if n.lower().endswith((".png", ".jpg", ".jpeg"))
                   and "depth" not in n.lower() and "normal" not in n.lower()
                   and "flow" not in n.lower() and "occlusion" not in n.lower()
                   and "diffuse" not in n.lower()), key=framenum)
    poses = [n for n in names
             if "pose" in n.replace("\\", "/").lower() and n.lower().endswith(".txt")]
    if not os.path.exists(f"{DEST}/pose.txt"):
        with z.open(poses[0]) as srcf, open(f"{DEST}/pose.txt", "wb") as outf:
            outf.write(srcf.read())
    json.dump([framenum(p) for p in imgs], open(f"{DEST}/nums.json", "w"))
    done = 0
    for i, name in enumerate(imgs):
        out = f"{UND}/{i:04d}.jpg"
        if os.path.exists(out):
            done += 1
            continue
        buf = np.frombuffer(z.read(name), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        cv2.imwrite(out, su.undistort(img, mapu, mapv),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        done += 1
        if done % 500 == 0:
            print(f"{done}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
        if time.time() - t0 > BUDGET_S:
            print(f"budget at {done}/{len(imgs)}; resume me", flush=True)
            sys.exit(0)
print(f"all {len(imgs)} frames undistorted; removing zip", flush=True)
os.remove(ZIP)
print("done", flush=True)
