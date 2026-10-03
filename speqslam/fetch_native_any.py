"""Native-rate fetch for any C3VDv2 full withdrawal: resumable zip download
(curl_cffi, Chrome TLS - the JHU archive blocks ordinary clients), then every
frame streamed from the zip through undistortion into compact JPEGs - raw PNGs
are never written (their full set exceeds free disk; learned the hard way on
c1_full_t4). Zip is deleted when every frame is done.

Extends the native-rate frame-point (currently n=1, c1_full_t4) to the other
two replication withdrawals, so the frame-rate curve's endpoint carries the
same n=3 as its stride points.

Usage: python speqslam/fetch_native_any.py <name> <file_id> [minutes]
e.g.:  python speqslam/fetch_native_any.py c2_full_t1_v1 14432 300
ONE session at a time - concurrent fetches corrupted an archive once.
"""
import os, sys, json, time, zipfile
import numpy as np, cv2
from curl_cffi import requests as cr

NAME = sys.argv[1]
FID = int(sys.argv[2])
BUDGET_S = (float(sys.argv[3]) if len(sys.argv) > 3 else 9.0) * 60
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL

OUT = "data/c3vdv2"
ZIP = f"{OUT}/{NAME}.zip"
DEST = f"{OUT}/{NAME}_s1"
UND = f"{DEST}/undist"
mark = f"{DEST}.extracted"
if os.path.exists(mark):
    print("already extracted", flush=True)
    sys.exit(0)
os.makedirs(UND, exist_ok=True)

t0 = time.time()
if not os.path.exists(ZIP):
    part = ZIP + ".part"
    url = f"https://archive.data.jhu.edu/api/access/datafile/{FID}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    print(f"fetching {NAME} ({have/1e9:.2f} GB so far)", flush=True)
    headers = {"Range": f"bytes={have}-"} if have else {}
    with cr.Session(impersonate="chrome") as s:
        r = s.get(url, headers=headers, stream=True, timeout=600)
        if r.status_code == 200 and have:
            have = 0
        if r.status_code not in (200, 206):
            print(f"HTTP {r.status_code}", flush=True)
            sys.exit(1)
        mode = "ab" if (have and r.status_code == 206) else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content():
                f.write(chunk)
                if time.time() - t0 > BUDGET_S:
                    print(f"budget at {os.path.getsize(part)/1e9:.2f} GB; resume me",
                          flush=True)
                    sys.exit(0)
    os.replace(part, ZIP)
print(f"zip complete: {os.path.getsize(ZIP)/1e9:.2f} GB; streaming undistort",
      flush=True)

mapu, mapv, K = su.build_map(V2CAL)
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
    if not imgs or not poses:
        print("layout not recognized - zip retained for inspection", flush=True)
        sys.exit(1)
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
open(mark, "w").write(f"native rate: {len(imgs)} undistorted frames\n")
print("done", flush=True)
