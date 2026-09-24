"""Fetch any C3VDv2 full-withdrawal video by name+id and keep every Nth frame.
Generalization of c3vdv2_screening_fetch3.py for the replication study: the
speed-limit and coverage findings currently rest on one withdrawal (c1_full_t4);
this fetches others (different colon model / texture) for the same frozen
pipeline. ONE session at a time - concurrent fetches corrupted an archive once.

Usage: python speqslam/c3vdv2_fetch_full.py <name> <file_id> <keep_every> [minutes]
e.g.:  python speqslam/c3vdv2_fetch_full.py c2_full_t1_v1 14432 3 9.5
Extracts to data/c3vdv2/<name>_s<keep_every>/; deletes the zip afterwards.
"""
import os, sys, time, zipfile
from curl_cffi import requests as cr

NAME = sys.argv[1]
FID = int(sys.argv[2])
KEEP_EVERY = int(sys.argv[3])
BUDGET_S = (float(sys.argv[4]) if len(sys.argv) > 4 else 9.0) * 60
OUT = "data/c3vdv2"
DEST = f"{OUT}/{NAME}_s{KEEP_EVERY}"

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
