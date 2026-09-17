"""Domain fine-tune of the polyp segmenter — the depth-fix recipe applied to detection.

The batch study showed the borrowed detector fires persistently on normal/inflamed tissue
it never saw in training. The fix mirrors the depth story: keep the pretrained weights,
continue training briefly on (a) Kvasir-SEG polyp images with expert masks, under a
DOCUMENTED split this time (800 train / 200 val, seeded), and (b) HARD NEGATIVES —
frames sampled from the batch study's CALIBRATION-split no-polyp clips with all-zero
masks, teaching the network what normal colon looks like. Test-split clips are never
touched, so the frozen batch protocol stays valid for re-evaluation.

Resumable: checkpoints each epoch to models/polyp_ft/; run with a minutes budget.
Usage: python speqslam/train_detector.py [minutes]  (default 8)
"""
import sys, os, csv, glob, time, random
import numpy as np, cv2, torch
import torch.nn.functional as Fn
sys.path.insert(0, "speqslam")
from polyp_detect import load_model, IN_W, IN_H, MEAN, STD, DEV

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 8.0) * 60
OUT = "models/polyp_ft"
os.makedirs(OUT, exist_ok=True)
EPOCHS = 8
LR = 2e-5
BATCH = 6
SEED = 7
NEG_PER_CLIP = 60

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- data lists ----
imgs = sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/images/*.jpg"))
rng = random.Random(SEED)
rng.shuffle(imgs)
train_pos, val_pos = imgs[:800], imgs[800:]

calib_neg_clips = [r["file"] for r in csv.DictReader(open("results/batch_study/manifest.csv"))
                   if r["split"] == "calib" and r["label"] == "nopolyp"]
# last two calibration clips are held out to measure false-fire rate, never trained on
train_neg_clips, valneg_clips = calib_neg_clips[:-2], calib_neg_clips[-2:]

def clip_frames(path, count):
    cap = cv2.VideoCapture(path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(1, N) / count))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out

def to_tensor(bgr, flip):
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IN_W, IN_H),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if flip:
        x = x[:, ::-1]
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

def mask_tensor(path, flip):
    m = cv2.imread(path, 0)
    m = cv2.resize(m, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
    if flip:
        m = m[:, ::-1]
    return torch.from_numpy((np.ascontiguousarray(m) > 127).astype(np.float32))

print("caching hard negatives...", flush=True)
neg_frames = []
for c in train_neg_clips:
    neg_frames += clip_frames(c, NEG_PER_CLIP)
print(f"  {len(neg_frames)} negative frames from {len(train_neg_clips)} calib control clips",
      flush=True)
valneg_frames = []
for c in valneg_clips:
    valneg_frames += clip_frames(c, 40)
print(f"  {len(valneg_frames)} held-out negative frames from {len(valneg_clips)} clips",
      flush=True)

model = load_model()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
start_ep = 0
ck_path = f"{OUT}/ckpt.pth"
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_ep = ck["epoch"]
    print(f"resumed at epoch {start_ep}", flush=True)

def loss_fn(out, y):
    # out: Bx2xHxW sigmoided probabilities (channel 1 = polyp); y: BxHxW in {0,1}
    p = out[:, 1].clamp(1e-5, 1 - 1e-5)
    b = out[:, 0].clamp(1e-5, 1 - 1e-5)
    bce = Fn.binary_cross_entropy(p, y) + Fn.binary_cross_entropy(b, 1 - y)
    inter = (p * y).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()

def validate():
    model.eval()
    dices = []
    with torch.no_grad():
        for pth in val_pos:
            x = to_tensor(cv2.imread(pth), False)[None].to(DEV)
            p = model(x)[0, 1].cpu().numpy() > 0.5
            gt = cv2.resize(cv2.imread(
                f"data/kvasir_seg/Kvasir-SEG/masks/{os.path.basename(pth)}", 0),
                (IN_W, IN_H), interpolation=cv2.INTER_NEAREST) > 127
            dices.append(2 * (p & gt).sum() / (p.sum() + gt.sum() + 1e-9))
        fires = 0
        for f in valneg_frames:
            x = to_tensor(f, False)[None].to(DEV)
            p = model(x)[0, 1].cpu().numpy()
            binm = p > 0.5
            if binm.mean() > 0.001 and (p[binm].mean() if binm.any() else 0) >= 0.7:
                fires += 1
    model.train()
    return float(np.mean(dices)), fires / max(1, len(valneg_frames))

t0 = time.time()
samples = [("pos", p) for p in train_pos] + [("neg", i) for i in range(len(neg_frames))]
mcsv = f"{OUT}/metrics.csv"
if not os.path.exists(mcsv):
    with open(mcsv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_dice", "valneg_fire_rate"])
    d0, f0 = validate()
    with open(mcsv, "a", newline="") as f:
        csv.writer(f).writerow([0, f"{d0:.4f}", f"{f0:.3f}"])
    print(f"BEFORE: val Dice {d0:.4f} | held-out control fire rate {100*f0:.1f}%", flush=True)

model.train()
for ep in range(start_ep, EPOCHS):
    if time.time() - t0 > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    rng2 = random.Random(SEED + ep + 1)
    rng2.shuffle(samples)
    losses = []
    for bi in range(0, len(samples), BATCH):
        batch = samples[bi:bi + BATCH]
        xs, ys = [], []
        for kind, ref in batch:
            flip = rng2.random() < 0.5
            if kind == "pos":
                xs.append(to_tensor(cv2.imread(ref), flip))
                ys.append(mask_tensor(
                    f"data/kvasir_seg/Kvasir-SEG/masks/{os.path.basename(ref)}", flip))
            else:
                xs.append(to_tensor(neg_frames[ref], flip))
                ys.append(torch.zeros(IN_H, IN_W))
        x = torch.stack(xs).to(DEV); y = torch.stack(ys).to(DEV)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if time.time() - t0 > BUDGET_S * 1.15:
            break
    d, fr = validate()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1},
               ck_path)
    torch.save(model.state_dict(), f"{OUT}/dlv3p_ft_ep{ep+1}.pth")
    with open(mcsv, "a", newline="") as f:
        csv.writer(f).writerow([ep + 1, f"{d:.4f}", f"{fr:.3f}"])
    print(f"epoch {ep+1}/{EPOCHS}: loss {np.mean(losses):.4f} | val Dice {d:.4f} | "
          f"held-out control fire rate {100*fr:.1f}%", flush=True)

print("session done", flush=True)
