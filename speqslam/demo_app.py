"""
Argus live demo — the booth application.

Plays an endoscopy sequence and runs the REAL pipeline on every frame, live, in front of you:
  window 1 (video panels):  raw frame | specular-glare mask | learned depth
  window 2 (3D):            the colon reconstruction growing point-by-point + the camera path
At the end (or press B) the 3D view recolors into the blind-spot map: red = patches the camera
never got a good look at.

Controls:  SPACE pause/resume · B blind-spot recolor · ESC quit
Usage:     python speqslam/demo_app.py --seq cecum_t1_a
           python speqslam/demo_app.py --selftest        (5 frames, no windows, print timings)

Uses the demo depth model (models/depth_c3vd_demo, fine-tuned on C3VD) if present, else the
off-the-shelf Depth-Anything-V2. NOTE: the demo model saw all sequences during training -- fine for
a demo, but reported RESULTS always come from the leave-one-out folds, never from this model.
"""
import sys, os, glob, time, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
from glare_removal import specular_mask
from direct_vo import track_pair

DET = dict(method="fixed", fixed_thr=248, dilate=2, max_area=1500)   # compact-specular (C3VD)
DMIN, DMAX = 0.003, 0.11
RES = 392                    # depth network input
VO_SCALE = 0.5               # VO runs at half resolution for speed
ADD_EVERY = 2                # add 3D points every Nth frame
VOXEL = 0.0012               # 3D accumulation voxel (m)
THR_VIEWS = 3                # blind = seen fewer than this many times
TISSUE = np.array([0.42, 0.36, 0.62])   # fallback point color (RGB 0-1) if frame color unused


class DepthNet:
    def __init__(self):
        from transformers import AutoModelForDepthEstimation
        local = os.path.join("models", "depth_c3vd_demo")
        self.name = "fine-tuned (C3VD demo model)" if os.path.isdir(local) else "off-the-shelf"
        src = local if os.path.isdir(local) else "depth-anything/Depth-Anything-V2-Small-hf"
        self.model = AutoModelForDepthEstimation.from_pretrained(src).to("cuda").eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @torch.no_grad()
    def predict(self, bgr):
        """Metric depth (m), same HxW as input. The fine-tuned demo model outputs metric disparity
        directly; the off-the-shelf fallback outputs RELATIVE disparity, which we map into the
        endoscopic working range per-frame so the demo stays sensible without the local model."""
        rgb = cv2.cvtColor(cv2.resize(bgr, (RES, RES)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = ((x - self.mean) / self.std).unsqueeze(0).to("cuda")
        disp = self.model(pixel_values=x).predicted_depth[0]
        disp = torch.nn.functional.interpolate(disp[None, None], size=bgr.shape[:2],
                                               mode="bilinear")[0, 0].cpu().numpy()
        if self.name == "off-the-shelf":
            lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
            disp = 1.0 / DMAX + (np.clip(disp, lo, hi) - lo) / max(hi - lo, 1e-6) * (1.0 / 0.012 - 1.0 / DMAX)
        d = 1.0 / np.clip(disp, 1.0 / DMAX, None)
        d[(d < DMIN) | (d > DMAX) | ~np.isfinite(d)] = 0.0
        return d


def colormap_depth(d):
    v = np.where(d > 0, d, np.nan)
    lo, hi = np.nanpercentile(v, 3), np.nanpercentile(v, 97)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img[d <= 0] = 30
    return img


def backproject(depth, bgr, pose, K, step=6):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(depth > 0)
    ys, xs = ys[::step], xs[::step]
    z = depth[ys, xs]
    Pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
    Pw = Pc @ pose[:3, :3].T + pose[:3, 3]
    cols = bgr[ys, xs][:, ::-1].astype(np.float64) / 255.0
    return Pw, cols


def hud(canvas, lines):
    y = 22
    for t in lines:
        cv2.putText(canvas, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 235, 225), 1, cv2.LINE_AA)
        y += 24
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="cecum_t1_a")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    undist = f"data/c3vd/{a.seq}_undist"
    imgs = sorted(glob.glob(f"{undist}/image01/*.jpg"))[::a.stride]
    K = np.loadtxt(f"{undist}/K.txt")
    assert imgs, f"no frames in {undist}/image01"
    net = DepthNet()
    print(f"[demo] {a.seq}: {len(imgs)} frames | depth model: {net.name}", flush=True)

    Ks = K.copy(); Ks[:2, :] *= VO_SCALE          # VO intrinsics at reduced res
    pose = np.eye(4)
    prev_g = prev_d = None
    traj = [pose[:3, 3].copy()]

    # 3D state
    vox_count = {}
    pts_all, col_all, keys_all = [], [], []

    vis = pcd = ls = None
    if not a.selftest:
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window("Argus - live 3D reconstruction", width=960, height=720)
        vis.get_render_option().background_color = np.array([0.05, 0.055, 0.075])
        vis.get_render_option().point_size = 2.5
        pcd = o3d.geometry.PointCloud()
        ls = o3d.geometry.LineSet()
        added = False

    t0 = time.time(); n_done = 0; paused = False; blind_mode = False
    for i, path in enumerate(imgs):
        bgr = cv2.imread(path)
        t_a = time.time()
        mask = specular_mask(bgr, **DET)
        t_b = time.time()
        depth = net.predict(bgr)
        t_c = time.time()

        # --- VO (half resolution, live) ---
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g_s = cv2.resize(g, None, fx=VO_SCALE, fy=VO_SCALE)
        d_s = cv2.resize(depth, None, fx=VO_SCALE, fy=VO_SCALE, interpolation=cv2.INTER_NEAREST)
        if prev_g is not None:
            T_rel = track_pair(prev_g, prev_d, g_s, Ks, levels=3, iters=10, robust=False,
                               depth_scale=1.0)
            pose = pose @ np.linalg.inv(T_rel)
            traj.append(pose[:3, 3].copy())
        prev_g, prev_d = g_s, d_s
        t_d = time.time()

        # --- accumulate 3D ---
        Pw, cols = backproject(depth, bgr, pose, K)
        kv = np.floor(Pw / VOXEL).astype(np.int64)
        for k in map(tuple, kv[::4]):
            vox_count[k] = vox_count.get(k, 0) + 1
        if i % ADD_EVERY == 0:
            pts_all.append(Pw[::3]); col_all.append(cols[::3]); keys_all.append(kv[::3])

        n_done += 1
        if a.selftest:
            print(f"  frame {i}: glare {1000*(t_b-t_a):.0f}ms  depth {1000*(t_c-t_b):.0f}ms  "
                  f"VO {1000*(t_d-t_c):.0f}ms", flush=True)
            if n_done >= 5:
                break
            continue

        # --- video panels ---
        show = bgr.copy(); show[mask > 0] = (40, 40, 230)
        panels = [bgr, show, colormap_depth(depth)]
        h = 384
        canvas = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels])
        fps = n_done / max(time.time() - t0, 1e-6)
        canvas = hud(canvas, [
            f"ARGUS  |  raw   |   specular mask   |   learned depth ({net.name})",
            f"frame {i+1}/{len(imgs)}   {fps:.1f} fps   glare {1000*(t_b-t_a):.0f}ms  "
            f"depth {1000*(t_c-t_b):.0f}ms  VO {1000*(t_d-t_c):.0f}ms",
            f"map: {sum(len(p) for p in pts_all):,} points  {len(vox_count):,} voxels   "
            f"[SPACE pause  B blind spots  ESC quit]"])
        cv2.imshow("Argus - live monocular pipeline", canvas)

        # --- 3D window ---
        import open3d as o3d
        if pts_all and i % ADD_EVERY == 0:
            P = np.vstack(pts_all); C = np.vstack(col_all)
            pcd.points = o3d.utility.Vector3dVector(P)
            pcd.colors = o3d.utility.Vector3dVector(C)
            if len(traj) > 1:
                ls.points = o3d.utility.Vector3dVector(np.array(traj))
                ls.lines = o3d.utility.Vector2iVector([[j, j + 1] for j in range(len(traj) - 1)])
                ls.colors = o3d.utility.Vector3dVector([[0.25, 0.84, 1.0]] * (len(traj) - 1))
            if not added:
                vis.add_geometry(pcd); vis.add_geometry(ls); added = True
            else:
                vis.update_geometry(pcd); vis.update_geometry(ls)
        vis.poll_events(); vis.update_renderer()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('b'):
            blind_mode = True
        while paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord(' '):
                paused = False
            if key == 27:
                paused = False; i = len(imgs)
            vis.poll_events(); vis.update_renderer()
        if blind_mode:
            break

    if a.selftest:
        print(f"[selftest ok] depth model: {net.name}", flush=True)
        return

    # --- finale: blind-spot recolor ---
    import open3d as o3d
    if pts_all:
        P = np.vstack(pts_all); C = np.vstack(col_all); KV = np.vstack(keys_all)
        counts = np.array([vox_count.get(tuple(k), 0) for k in KV])
        blind = counts < THR_VIEWS
        C2 = C * 0.55                                     # dim the seen tissue
        C2[blind] = [0.95, 0.15, 0.12]                    # blind spots in red
        pcd.colors = o3d.utility.Vector3dVector(C2)
        vis.update_geometry(pcd)
        print(f"[blind spots] {100*blind.mean():.1f}% of map flagged (<{THR_VIEWS} views). "
              f"ESC in 3D window to exit.", flush=True)
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        if (cv2.waitKey(30) & 0xFF) == 27:
            break
    vis.destroy_window(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
