"""Batch-study population builder: pulls ~30 short Hyper-Kvasir clips (half labeled
polyp, half not) via HTTP-Range from the 30GB archive, transcodes to 512px H.264,
and writes the manifest with the calibration/test split.

Split rules (methodology, not convenience):
- clips this project has EVER touched (clip1/clip2, the two pack polyp clips) are
  tainted by exploration -> forced into the CALIBRATION split, never test;
- everything else alternates by sorted videoID -> deterministic, no cherry-picking.
"""
import os, csv, glob, shutil, subprocess
import requests, urllib3
from remotezip import RemoteZip
import imageio_ffmpeg

urllib3.disable_warnings()
RAW = "data/hyperkvasir/batch/raw"
PROC = "data/hyperkvasir/batch/proc"
os.makedirs(RAW, exist_ok=True)
os.makedirs(PROC, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()

TAINTED = {
    "9e3f0999-0279-44c2-ad4a-e3940df9991c",   # clip1, the original demo
    "16e7e461-a9a7-4b04-8d2c-155de3eb1c90",   # clip2
    "29ce8d04-c184-42fa-ac05-322d3a2d37bd",   # phone cold run 1
    "7d8e1601-a407-426f-aded-8ed1d60dac8e",   # phone cold run 2 + all sim tuning
}
PER_CLASS = 15
MAX_MB = 20

ann = {}
with open("data/hyperkvasir/video-annotations.csv") as f:
    for row in csv.reader(f, delimiter=";"):
        if len(row) == 2 and row[0] != "videoID":
            ann[row[0]] = row[1].strip()

def label_of(vid):
    return "polyp" if "polyp" in ann[vid].lower() else "nopolyp"

URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-videos.zip"
s = requests.Session(); s.verify = False

chosen = []
with RemoteZip(URL, session=s) as z:
    infos = []
    for zi in z.infolist():
        base = os.path.basename(zi.filename)
        vid = os.path.splitext(base)[0]
        if vid in ann and base.lower().endswith(".avi") and zi.file_size <= MAX_MB * 1e6:
            infos.append((zi, vid, zi.file_size))
    pool = {"polyp": [], "nopolyp": []}
    # tainted first (they must be in the study, in calibration), then smallest unseen
    for zi, vid, size in sorted(infos, key=lambda x: x[2]):
        pool[label_of(vid)].append((zi, vid, size))
    for lab in pool:
        tainted_first = [e for e in pool[lab] if e[1] in TAINTED]
        rest = [e for e in pool[lab] if e[1] not in TAINTED]
        pool[lab] = (tainted_first + rest)[:PER_CLASS]
    for lab, entries in pool.items():
        for zi, vid, size in entries:
            dest_raw = f"{RAW}/{lab}_{vid[:8]}.avi"
            dest = f"{PROC}/{lab}_{vid[:8]}.mp4"
            chosen.append((vid, lab))
            if os.path.exists(dest):
                print("have", dest, flush=True)
                continue
            if not os.path.exists(dest_raw):
                print(f"fetch {vid} ({lab}, {size/1e6:.1f} MB)", flush=True)
                with z.open(zi.filename) as src, open(dest_raw, "wb") as out:
                    shutil.copyfileobj(src, out)
            r = subprocess.run([FF, "-y", "-loglevel", "error", "-i", dest_raw,
                                "-vf", "scale=512:-2", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "24", dest])
            if r.returncode == 0 and os.path.exists(dest):
                os.remove(dest_raw)
                print("  ->", dest, flush=True)
            else:
                print("  TRANSCODE FAILED", vid, flush=True)

rows = [("file", "videoID", "label", "tainted", "split")]
for vid, lab in sorted(chosen):
    f = f"{PROC}/{lab}_{vid[:8]}.mp4"
    if not os.path.exists(f):
        continue
    tainted = vid in TAINTED
    idx = len(rows) - 1
    split = "calib" if tainted else ("calib" if idx % 2 == 0 else "test")
    rows.append((f, vid, lab, int(tainted), split))
os.makedirs("results/batch_study", exist_ok=True)
with open("results/batch_study/manifest.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
n = len(rows) - 1
print(f"\nmanifest: {n} clips "
      f"({sum(1 for r in rows[1:] if r[2]=='polyp')} polyp / "
      f"{sum(1 for r in rows[1:] if r[2]=='nopolyp')} nopolyp; "
      f"{sum(1 for r in rows[1:] if r[4]=='calib')} calib / "
      f"{sum(1 for r in rows[1:] if r[4]=='test')} test)", flush=True)
