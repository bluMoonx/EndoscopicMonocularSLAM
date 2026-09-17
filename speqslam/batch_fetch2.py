"""Extended-test fetch: the NEXT 15 polyp + 15 nopolyp smallest unseen clips, all
assigned split 'test2'. No clip here influences any choice — the operating point is
already frozen — so every row is pure test evidence. Writes manifest_ext.csv."""
import os, csv, shutil, subprocess
import requests, urllib3
from remotezip import RemoteZip
import imageio_ffmpeg

urllib3.disable_warnings()
RAW = "data/hyperkvasir/batch/raw"
PROC = "data/hyperkvasir/batch/proc"
os.makedirs(RAW, exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
PER_CLASS = 15
MAX_MB = 45

have = {r["videoID"] for r in csv.DictReader(open("results/batch_study/manifest.csv"))}
ann = {}
with open("data/hyperkvasir/video-annotations.csv") as f:
    for row in csv.reader(f, delimiter=";"):
        if len(row) == 2 and row[0] != "videoID":
            ann[row[0]] = row[1].strip()

URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-videos.zip"
s = requests.Session(); s.verify = False
chosen = []
with RemoteZip(URL, session=s) as z:
    infos = []
    for zi in z.infolist():
        base = os.path.basename(zi.filename)
        vid = os.path.splitext(base)[0]
        if vid in ann and vid not in have and base.lower().endswith(".avi") \
                and zi.file_size <= MAX_MB * 1e6:
            infos.append((zi, vid, zi.file_size))
    pool = {"polyp": [], "nopolyp": []}
    for zi, vid, size in sorted(infos, key=lambda x: x[2]):
        lab = "polyp" if "polyp" in ann[vid].lower() else "nopolyp"
        if len(pool[lab]) < PER_CLASS:
            pool[lab].append((zi, vid, size))
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
    if os.path.exists(f):
        rows.append((f, vid, lab, 0, "test2"))
with open("results/batch_study/manifest_ext.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\nmanifest_ext: {len(rows)-1} clips "
      f"({sum(1 for r in rows[1:] if r[2]=='polyp')}P/"
      f"{sum(1 for r in rows[1:] if r[2]=='nopolyp')}N)", flush=True)
