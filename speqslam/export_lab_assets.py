"""Package the workstation's remaining demos for the phone: measure frames+depth,
trajectory pairs (est vs GT), and precomputed depth X-ray strips."""
import sys, os, glob, struct
import numpy as np, cv2
sys.path.insert(0, "speqslam")
from validate_coverage import load_poses, align_fullpose

DS = 100.0 / 65535.0 / 1000.0
A = "arguslite/app/src/main/assets"
FRAMES = [("cecum_t1_a", 40), ("cecum_t1_a", 150), ("desc_t4_a", 60), ("trans_t2_a", 90)]

# 1. measure: frame jpg + half-res depth bin + K (halved)
os.makedirs(f"{A}/measure", exist_ok=True)
for j, (seq, fr) in enumerate(FRAMES):
    u = f"data/c3vd/{seq}_undist"
    img = cv2.imread(sorted(glob.glob(f"{u}/image01/*.jpg"))[fr])
    dep = cv2.imread(sorted(glob.glob(f"{u}/depth01_ftc/*.png"))[fr], cv2.IMREAD_UNCHANGED)
    K = np.loadtxt(f"{u}/K.txt")
    h2, w2 = dep.shape[0] // 2, dep.shape[1] // 2
    d2 = cv2.resize(dep, (w2, h2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{A}/measure/m{j}.jpg", cv2.resize(img, (w2, h2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(f"{A}/measure/m{j}.bin", "wb") as f:
        f.write(struct.pack("<iiffff", w2, h2, K[0,0]/2, K[1,1]/2, K[0,2]/2, K[1,2]/2))
        f.write(d2.astype("<u2").tobytes())
print("[measure] 4 presets", flush=True)

# 2. trajectories: gt + aligned ft, normalized together
os.makedirs(f"{A}/traj", exist_ok=True)
for seq in ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]:
    gt = load_poses(f"results/c3vd/{seq}_gt.txt")
    ft = load_poses(f"results/c3vd/{seq}_ft.txt")
    ft_al, _ = align_fullpose(gt, ft, sorted(set(gt) & set(ft)))
    G = np.array([gt[t][:3, 3] for t in sorted(gt)])
    F = np.array([ft_al[t][:3, 3] for t in sorted(ft_al)])
    mu = G.mean(0); sc = np.abs(G - mu).max()
    G, F = (G - mu) / sc, (F - mu) / sc
    with open(f"{A}/traj/{seq}.bin", "wb") as f:
        f.write(struct.pack("<i", len(G)))
        for p in G: f.write(struct.pack("<fff", *p))
        f.write(struct.pack("<i", len(F)))
        for p in F: f.write(struct.pack("<fff", *p))
print("[traj] 5 sequences", flush=True)

# 3. depth x-ray strips (precomputed on workstation, labeled as such in-app)
import torch, demo_app
from transformers import AutoModelForDepthEstimation
net = demo_app.DepthNet()
off = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf").to("cuda" if torch.cuda.is_available() else "cpu").eval()
def offdepth(bgr):
    import numpy as np
    rgb = cv2.cvtColor(cv2.resize(bgr, (392, 392)), cv2.COLOR_BGR2RGB)
    x = (torch.from_numpy(rgb).permute(2,0,1).float()/255.0 -
         torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    with torch.no_grad():
        d = off(pixel_values=x.unsqueeze(0).to(next(off.parameters()).device)).predicted_depth[0]
    return torch.nn.functional.interpolate(d[None,None], size=bgr.shape[:2], mode="bilinear")[0,0].cpu().numpy()
def vis(d, invert=False):
    v = d[np.isfinite(d) & (d != 0)]
    lo, hi = np.percentile(v, 3), np.percentile(v, 97)
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if invert: t = 1 - t
    return cv2.applyColorMap((255 * t).astype(np.uint8), cv2.COLORMAP_MAGMA)
os.makedirs(f"{A}/xray", exist_ok=True)
for j, (seq, fr) in enumerate(FRAMES):
    u = f"data/c3vd/{seq}_undist"
    bgr = cv2.imread(sorted(glob.glob(f"{u}/image01/*.jpg"))[fr])
    o = vis(offdepth(bgr))
    ftd = vis(net.predict(bgr), invert=True)
    hh = 240
    def rs(im): return cv2.resize(im, (int(im.shape[1]*hh/im.shape[0]), hh))
    cv2.imwrite(f"{A}/xray/x{j}.jpg", np.hstack([rs(bgr), rs(o), rs(ftd)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
print("[xray] 4 strips", flush=True)
