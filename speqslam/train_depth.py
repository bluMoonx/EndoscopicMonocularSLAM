"""
Domain fine-tuning of Depth-Anything-V2 on C3VD endoscopy, leave-one-sequence-out.

For a held-out test sequence, train on the OTHER sequences' (RGB -> metric depth) pairs and then
predict depth on the held-out one -> data/c3vd/<seq>_undist/depth01_ft/. The model is fine-tuned to
regress METRIC disparity (1/depth in 1/m), so at test time depth = 1/prediction is metric with NO
ground-truth scale alignment -- fully monocular. Loss = L1 + gradient-matching + scale-invariant.

Usage: python train_depth.py <holdout_seq> [epochs]
"""
import sys, os, glob, random, numpy as np, cv2, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DS = 100.0 / 65535.0 / 1000.0
ALL = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
MODEL = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DEV = "cuda"
RES = 392
DMIN, DMAX = 0.003, 0.11


def pairs(seq):
    u = f"data/c3vd/{seq}_undist"
    im = sorted(glob.glob(f"{u}/image01/*.jpg")); dp = sorted(glob.glob(f"{u}/depth01/*.png"))
    return list(zip(im, dp))


class C3VD(Dataset):
    def __init__(self, seqs, train=True):
        self.items = [p for s in seqs for p in pairs(s)]
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, dpth = self.items[i]
        bgr = cv2.imread(ip); d = cv2.imread(dpth, cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (RES, RES)); d = cv2.resize(d, (RES, RES), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                rgb = rgb[:, ::-1].copy(); d = d[:, ::-1].copy()
            g = random.uniform(0.8, 1.2)                       # brightness/gamma jitter
            rgb = np.clip(((rgb / 255.0) ** (1 / g)) * 255.0, 0, 255).astype(np.uint8)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        disp = np.where(d > 0, 1.0 / np.clip(d, DMIN, DMAX), 0.0).astype(np.float32)  # metric disparity
        return x, torch.from_numpy(disp), torch.from_numpy((d > 0).astype(np.float32))


def grad_loss(pred, gt, mask):
    def gx(t): return t[:, :, 1:] - t[:, :, :-1]
    def gy(t): return t[:, 1:, :] - t[:, :-1, :]
    mx, my = mask[:, :, 1:] * mask[:, :, :-1], mask[:, 1:, :] * mask[:, :-1, :]
    return ((gx(pred) - gx(gt)).abs() * mx).sum() / (mx.sum() + 1) + \
           ((gy(pred) - gy(gt)).abs() * my).sum() / (my.sum() + 1)


def predict_metric(model, proc, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(cv2.resize(rgb, (RES, RES))).permute(2, 0, 1).float() / 255.0
    x = ((x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) /
         torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        disp = model(pixel_values=x).predicted_depth[0]
    disp = F.interpolate(disp[None, None], size=bgr.shape[:2], mode="bilinear")[0, 0]
    depth = 1.0 / torch.clamp(disp, min=1.0 / DMAX)
    return depth.cpu().numpy()


def main(holdout, epochs=40):
    train_seqs = [s for s in ALL if s != holdout]
    print(f"[FT] holdout={holdout}  train={train_seqs}  epochs={epochs}", flush=True)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(DEV)
    dl = DataLoader(C3VD(train_seqs, True), batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x, gt, m in dl:
            x, gt, m = x.to(DEV), gt.to(DEV), m.to(DEV)
            pred = model(pixel_values=x).predicted_depth
            pred = F.interpolate(pred[:, None], size=gt.shape[1:], mode="bilinear")[:, 0]
            pred = torch.clamp(pred, min=1e-3)
            mb = m > 0
            l1 = (((pred - gt).abs()) * m).sum() / (m.sum() + 1)
            dlog = (torch.log(pred[mb]) - torch.log(torch.clamp(gt[mb], min=1e-3)))
            silog = torch.sqrt((dlog ** 2).mean() - 0.85 * dlog.mean() ** 2 + 1e-6)
            loss = l1 + 0.5 * grad_loss(pred, gt, m) + 0.5 * silog
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:>3} loss={tot/len(dl):.4f}", flush=True)
    # generate held-out depth
    model.eval()
    u = f"data/c3vd/{holdout}_undist"; out = f"{u}/depth01_ft"; os.makedirs(out, exist_ok=True)
    for ip, _ in pairs(holdout):
        depth = predict_metric(model, proc, cv2.imread(ip))
        depth[(depth < DMIN) | (depth > DMAX) | ~np.isfinite(depth)] = 0.0
        raw = np.clip(depth / DS, 0, 65535).astype(np.uint16)
        cv2.imwrite(f"{out}/{os.path.basename(ip).replace('.jpg','.png')}", raw)
    print(f"  -> {out}  ({len(pairs(holdout))} maps)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
