"""Re-fetch the full-withdrawal video and keep every 3rd frame (stride 3) into
data/c3vdv2/c1_full_t4_v1_s3, alongside the existing stride-12 set.

Motivation: at stride 12 (true screening speed sampled at ~2.3 fps effective)
16% of frame pairs exceed the photometric tracker's convergence basin and the
global trajectory collapses to the constant-point baseline. The scientific
question is the frame-rate requirement: does the same frozen pipeline hold the
trajectory at ~9 fps effective? Same archive, same rules, one new stride.

Resumable like the original fetcher. ONE session at a time - concurrent fetches
corrupted this archive once before.
Usage: python speqslam/c3vdv2_screening_fetch3.py [minutes]
"""
import os, sys, time, zipfile
from curl_cffi import requests as cr

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
NAME = "c1_full_t4_v1"
FID = 14429
OUT = "data/c3vdv2"
KEEP_EVERY = 3
DEST = f"{OUT}/{NAME}_s3"

final = f"{OUT}/{NAME}.zip"
mark = f"{DEST}.extracted"
if os.path.exists(mark):
    print("already extracted", flush=True)
    sys.exit(0)

t0 = time.time()
if not os.path.exists(final):
    part = final + ".part"
    url = f"https://archive.data.jhu.edu/api/access/datafile/{FID}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    print(f"fetching ({have/1e9:.2f} GB so far)", flush=True)
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
    os.replace(part, final)
print(f"zip complete: {os.path.getsize(final)/1e9:.2f} GB; selective extract",
      flush=True)

os.makedirs(DEST, exist_ok=True)
kept = 0
with zipfile.ZipFile(final) as z:
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
    print(f"{len(imgs)} image frames, {len(poses)} pose files; "
          f"keeping every {KEEP_EVERY}th", flush=True)
    if not imgs or not poses:
        print("layout not recognized - keeping zip for inspection", flush=True)
        sys.exit(1)
    os.makedirs(f"{DEST}/rgb", exist_ok=True)
    for i in range(0, len(imgs), KEEP_EVERY):
        with z.open(imgs[i]) as src:
            with open(f"{DEST}/rgb/{os.path.basename(imgs[i])}", "wb") as out:
                out.write(src.read())
        kept += 1
    with z.open(poses[0]) as src:
        with open(f"{DEST}/pose.txt", "wb") as out:
            out.write(src.read())
if kept == 0:
    print("kept nothing - zip retained", flush=True)
    sys.exit(1)
os.remove(final)
open(mark, "w").write(f"kept {kept} of {len(imgs)} frames, every {KEEP_EVERY}th\n")
print(f"kept {kept} frames + pose.txt; zip removed", flush=True)
