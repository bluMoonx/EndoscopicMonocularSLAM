"""Detector fine-tune, round two — aimed at the miss categories the 41-clip batch study
characterized: dye-stained polyps (hue rotation + synthetic blue-dye tinting of real
polyp images), and lighting variation (brightness/gamma jitter). Continues from round
one's checkpoint; identical holdouts (Kvasir 800/200 split, two held-out calibration
control clips); identical metrics per epoch, so rounds are directly comparable.

Usage: python speqslam/train_detector2.py [minutes]   (resumable)
"""
import sys, os, csv, glob, time, random
import numpy as np, cv2, torch
import torch.nn.functional as Fn
sys.path.insert(0, "speqslam")
from polyp_detect import load_model, IN_W, IN_H, MEAN, STD, DEV

BUDGET_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 8.0) * 60
BASE = "models/polyp_ft/dlv3p_ft_ep8.pth"
OUT = "models/polyp_ft2"
os.makedirs(OUT, exist_ok=True)
EPOCHS = 6
LR = 1.5e-5
BATCH = 6
SEED = 7
NEG_PER_CLIP = 60

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

imgs = sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/images/*.jpg"))
rng = random.Random(SEED)
rng.shuffle(imgs)
train_pos, val_pos = imgs[:800], imgs[800:]

calib_neg_clips = [r["file"] for r in csv.DictReader(open("results/batch_study/manifest.csv"))
                   if r["split"] == "calib" and r["label"] == "nopolyp"]
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

def augment(bgr, r2):
    """Photometric augmentation targeting the batch study's miss categories."""
    img = bgr
    if r2.random() < 0.30:                          # hue rotation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + r2.randint(-15, 45)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if r2.random() < 0.20:                          # synthetic blue dye (indigo carmine)
        a = r2.uniform(0.20, 0.45)
        dye = np.zeros_like(img); dye[:, :] = (140, 60, 20)   # deep blue, BGR
        img = cv2.addWeighted(img, 1 - a, dye, a, 0)
    if r2.random() < 0.35:                          # brightness / gamma
        g = r2.uniform(0.6, 1.5)
        img = np.clip(((img / 255.0) ** g) * 255.0 * r2.uniform(0.8, 1.2),
                      0, 255).astype(np.uint8)
    return img

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
valneg_frames = []
for c in valneg_clips:
    valneg_frames += clip_frames(c, 40)
print(f"  {len(neg_frames)} train negatives, {len(valneg_frames)} held-out negatives",
      flush=True)

model = load_model()
model.load_state_dict(torch.load(BASE, map_location=DEV, weights_only=True))
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
    p = out[:, 1].clamp(1e-5, 1 - 1e-5)
    b = out[:, 0].clamp(1e-5, 1 - 1e-5)
    bce = Fn.binary_cross_entropy(p, y) + Fn.binary_cross_entropy(b, 1 - y)
    inter = (p * y).sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) + 1)
    return bce + dice.mean()

def validate():
    model.eval()
    dices, dyed_dices = [], []
    r3 = random.Random(99)
    with torch.no_grad():
        for pth in val_pos:
            img = cv2.imread(pth)
            x = to_tensor(img, False)[None].to(DEV)
            p = model(x)[0, 1].cpu().numpy() > 0.5
            gt = cv2.resize(cv2.imread(
                f"data/kvasir_seg/Kvasir-SEG/masks/{os.path.basename(pth)}", 0),
                (IN_W, IN_H), interpolation=cv2.INTER_NEAREST) > 127
            dices.append(2 * (p & gt).sum() / (p.sum() + gt.sum() + 1e-9))
            # dyed-polyp proxy: same image under synthetic blue dye
            a = 0.35
            dye = np.zeros_like(img); dye[:, :] = (140, 60, 20)
            dyed = cv2.addWeighted(img, 1 - a, dye, a, 0)
            xd = to_tensor(dyed, False)[None].to(DEV)
            pd = model(xd)[0, 1].cpu().numpy() > 0.5
            dyed_dices.append(2 * (pd & gt).sum() / (pd.sum() + gt.sum() + 1e-9))
        fires = 0
        for f in valneg_frames:
            x = to_tensor(f, False)[None].to(DEV)
            p = model(x)[0, 1].cpu().numpy()
            binm = p > 0.5
            if binm.mean() > 0.001 and (p[binm].mean() if binm.any() else 0) >= 0.7:
                fires += 1
    model.train()
    return (float(np.mean(dices)), float(np.mean(dyed_dices)),
            fires / max(1, len(valneg_frames)))

t0 = time.time()
samples = [("pos", p) for p in train_pos] + [("neg", i) for i in range(len(neg_frames))]
mcsv = f"{OUT}/metrics.csv"
if not os.path.exists(mcsv):
    with open(mcsv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_dice", "val_dice_dyed", "valneg_fire_rate"])
    d0, dd0, f0 = validate()
    with open(mcsv, "a", newline="") as f:
        csv.writer(f).writerow([0, f"{d0:.4f}", f"{dd0:.4f}", f"{f0:.3f}"])
    print(f"ROUND-1 BASELINE: Dice {d0:.4f} | dyed-Dice {dd0:.4f} | fire {100*f0:.1f}%",
          flush=True)

model.train()
for ep in range(start_ep, EPOCHS):
    if time.time() - t0 > BUDGET_S:
        print("budget reached; resume me", flush=True)
        break
    r2 = random.Random(SEED + 100 + ep)
    r2.shuffle(samples)
    losses = []
    for bi in range(0, len(samples), BATCH):
        batch = samples[bi:bi + BATCH]
        xs, ys = [], []
        for kind, ref in batch:
            flip = r2.random() < 0.5
            if kind == "pos":
                img = augment(cv2.imread(ref), r2)
                xs.append(to_tensor(img, flip))
                ys.append(mask_tensor(
                    f"data/kvasir_seg/Kvasir-SEG/masks/{os.path.basename(ref)}", flip))
            else:
                img = augment(neg_frames[ref], r2)
                xs.append(to_tensor(img, flip))
                ys.append(torch.zeros(IN_H, IN_W))
        x = torch.stack(xs).to(DEV); y = torch.stack(ys).to(DEV)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if time.time() - t0 > BUDGET_S * 1.15:
            break
    d, dd, fr = validate()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1},
               ck_path)
    torch.save(model.state_dict(), f"{OUT}/dlv3p_ft2_ep{ep+1}.pth")
    with open(mcsv, "a", newline="") as f:
        csv.writer(f).writerow([ep + 1, f"{d:.4f}", f"{dd:.4f}", f"{fr:.3f}"])
    print(f"epoch {ep+1}/{EPOCHS}: loss {np.mean(losses):.4f} | Dice {d:.4f} | "
          f"dyed-Dice {dd:.4f} | fire {100*fr:.1f}%", flush=True)

print("session done", flush=True)
