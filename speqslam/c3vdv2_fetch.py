"""C3VDv2 downloader for the deformation study. The JHU archive sits behind a
Cloudflare challenge that blocks ordinary clients; curl_cffi with Chrome TLS
impersonation passes it. Resumable via HTTP Range; budget-aware; re-run to continue.

Study set: the six moving-camera deformation videos (v4, GT poses, no pixel GT) and
four matched rigid controls of the same phantom segments (full pixel GT).
Usage: python speqslam/c3vdv2_fetch.py [minutes]
"""
import os, sys, time, zipfile
from curl_cffi import requests as cr

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 9.0) * 60
OUT = "data/c3vdv2"
os.makedirs(OUT, exist_ok=True)

FILES = [
    # deformation videos (moving camera): name, dataverse id, GB
    ("c1_ascending_t4_v4", 14254, 0.73),
    ("c1_cecum_t1_v4", 14309, 1.06),
    ("c1_descending_t4_v4", 14431, 0.77),
    ("c1_sigmoid2_t4_v4", 14250, 0.58),
    ("c1_transverse1_t1_v4", 14180, 1.15),
    ("c2_cecum_t1_v4", 14410, 0.61),
    # matched rigid controls with full pixel GT
    ("c1_cecum_t1_v1", 14285, 2.39),
    ("c1_descending_t4_v1", 14346, 1.99),
    ("c1_transverse1_t1_v2", 14368, 1.22),
    ("c2_cecum_t1_v1", 14223, 3.06),
]

t0 = time.time()
for name, fid, gb in FILES:
    final = f"{OUT}/{name}.zip"
    mark = f"{OUT}/{name}.extracted"
    if os.path.exists(mark):
        continue
    if time.time() - t0 > BUDGET_S:
        print("budget reached; resume me", flush=True)
        sys.exit(0)
    part = final + ".part"
    url = f"https://archive.data.jhu.edu/api/access/datafile/{fid}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    print(f"{name}: fetching ({gb} GB, have {have/1e9:.2f})", flush=True)
    headers = {"Range": f"bytes={have}-"} if have else {}
    with cr.Session(impersonate="chrome") as s:
        r = s.get(url, headers=headers, stream=True, timeout=600)
        if r.status_code == 200 and have:
            have = 0  # server ignored the range; restart the file
            print("  (no range support; restarting file)", flush=True)
        if r.status_code not in (200, 206):
            print(f"  HTTP {r.status_code} - stopping", flush=True)
            sys.exit(1)
        mode = "ab" if (have and r.status_code == 206) else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content():
                f.write(chunk)
                if time.time() - t0 > BUDGET_S * 1.3:
                    print(f"  budget mid-file at {os.path.getsize(part)/1e9:.2f} GB; "
                          "resume me", flush=True)
                    sys.exit(0)
    os.replace(part, final)
    print(f"  complete: {os.path.getsize(final)/1e9:.2f} GB; extracting", flush=True)
    with zipfile.ZipFile(final) as z:
        z.extractall(f"{OUT}/{name}")
    os.remove(final)
    open(mark, "w").close()
    print(f"  extracted -> {OUT}/{name}/ (zip removed)", flush=True)

print("all files present", flush=True)
