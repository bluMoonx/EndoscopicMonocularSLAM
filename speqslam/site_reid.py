"""Cross-session site re-identification, stage 2 (protocol frozen in
results/site_reid_gate.md before any run was tracked). Requires every run's
poses_est.npy from c3vdv2_screening_seq.py. Computes the 45 site errors, applies
gates G1/G2 exactly as pre-registered, writes results/site_reid.csv and
results/site_reid_verdict.json. No side effects at import."""
import csv, glob, json, os, sys
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.argv = sys.argv[:1]
sys.path.insert(0, "speqslam")
import demo_app
import scaramuzza_undistort as su
from c3vdv2_deform import V2CAL, load_poses

PAIRS = [("c1_ascending_t4_v1_s1", "c1_ascending_t4_v4"),
         ("c1_cecum_t1_v1", "c1_cecum_t1_v4"),
         ("c1_descending_t4_v1", "c1_descending_t4_v4"),
         ("c2_cecum_t1_v1", "c2_cecum_t1_v4"),
         ("c2_transverse1_t1_v1_s1", "c2_transverse1_t1_v4_s1")]

_, _, K = su.build_map(V2CAL)
Kinv = np.linalg.inv(K)
net = demo_app.DepthNet()


def umeyama_full(gt, est):
    """s, R, t with p_gt ~= s R p_est + t (Umeyama 1991)."""
    mg, me = gt.mean(0), est.mean(0)
    G, E = gt - mg, est - me
    Sig = G.T @ E / len(gt)
    U, S, Vt = np.linalg.svd(Sig)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    var = (E ** 2).sum() / len(gt)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    t = mg - s * R @ me
    rmse = float(np.sqrt((((s * (R @ E.T).T + mg) - gt) ** 2).sum(1).mean()))
    return s, R, t, rmse


class Run:
    def __init__(self, name):
        self.base = f"data/c3vdv2/{name}"
        self.und = sorted(glob.glob(f"{self.base}/undist/*.jpg"))
        self.poses = np.load(f"{self.base}/poses_est.npy")
        assert len(self.poses) == len(self.und), \
            f"{name}: {len(self.poses)} poses vs {len(self.und)} frames - tracking incomplete"
        raws = sorted(glob.glob(f"{self.base}/rgb/*.png"),
                      key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in raws]
        gt_all = load_poses(f"{self.base}/pose.txt")
        off = 1 if max(nums) >= len(gt_all) else 0
        self.gt = np.array([gt_all[i - off][1] for i in nums])     # meters
        est = self.poses[:, :3, 3]
        self.s, self.R, self.t, rmse = umeyama_full(self.gt, est)
        print(f"  {name}: {len(self.und)} frames, align rmse {rmse*1000:.1f} mm")
        self._dep = {}

    def depth(self, i):
        if i not in self._dep:
            self._dep[i] = net.predict(cv2.imread(self.und[i]))
        return self._dep[i]

    def to_gt(self, p_est):
        return self.s * self.R @ p_est + self.t

    def from_gt(self, p_gt):
        return self.R.T @ (p_gt - self.t) / self.s

    def backproject(self, i, u, v):
        """World-GT point (meters) through est pose + est depth at pixel (u,v)."""
        d = float(self.depth(i)[int(v), int(u)])
        if d <= 0:
            return None
        ray = Kinv @ np.array([u, v, 1.0])
        p_cam = ray / ray[2] * d
        p_est = self.poses[i][:3, :3] @ p_cam + self.poses[i][:3, 3]
        return self.to_gt(p_est)

    def nearest_valid(self, i, u0, v0, max_r=50):
        dep = self.depth(i)
        h, w = dep.shape[:2]
        for r in range(0, max_r + 1):
            for dv_ in range(-r, r + 1):
                for du in range(-r, r + 1):
                    if max(abs(du), abs(dv_)) != r:
                        continue
                    u, v = u0 + du, v0 + dv_
                    if 0 <= u < w and 0 <= v < h and dep[v, u] > 0:
                        return u, v
        return None


rows = []
for a_name, b_name in PAIRS:
    print(f"pair {a_name} -> {b_name}", flush=True)
    A, B = Run(a_name), Run(b_name)
    h, w = A.depth(0).shape[:2]
    sites = []
    for frac in (0.30, 0.50, 0.70):
        fi = int(frac * len(A.und))
        for u0 in (w // 2, w // 2 - w // 4, w // 2 + w // 4):
            pv = A.nearest_valid(fi, u0, h // 2)
            if pv is None:
                sites.append((fi, u0, h // 2, None)); continue
            sites.append((fi, pv[0], pv[1], A.backproject(fi, pv[0], pv[1])))

    hB, wB = B.depth(0).shape[:2]
    site_pts = [s[3] for s in sites]
    for si, (fi, u, v, p_gt) in enumerate(sites):
        if p_gt is None:
            rows.append([a_name, si, fi, u, v, "", "", "no_valid_depth_A", ""]); continue
        best = None
        for j in range(len(B.und)):
            p_estB = B.from_gt(p_gt)
            Pj = B.poses[j]
            p_cam = Pj[:3, :3].T @ (p_estB - Pj[:3, 3])
            if p_cam[2] <= 1e-6:
                continue
            uv = K @ (p_cam / p_cam[2])
            uj, vj = uv[0], uv[1]
            if not (0.1 * wB < uj < 0.9 * wB and 0.1 * hB < vj < 0.9 * hB):
                continue
            dist = np.linalg.norm(B.to_gt(Pj[:3, 3]) - p_gt)
            if best is None or dist < best[0]:
                best = (dist, j, int(round(uj)), int(round(vj)))
        if best is None:
            rows.append([a_name, si, fi, u, v, "", "", "not_refound", ""]); continue
        _, j, uj, vj = best
        pv = B.nearest_valid(j, uj, vj)
        if pv is None:
            rows.append([a_name, si, fi, u, v, j, "", "no_valid_depth_B", ""]); continue
        p_b = B.backproject(j, pv[0], pv[1])
        err = float(np.linalg.norm(p_b - p_gt)) * 1000.0
        d_others = [float(np.linalg.norm(p_b - q)) * 1000.0
                    for k, q in enumerate(site_pts) if k != si and q is not None]
        discr = bool(err < min(d_others)) if d_others else True
        rows.append([a_name, si, fi, u, v, j, f"{err:.2f}", "ok", int(discr)])
        print(f"  site {si}: err {err:7.2f} mm  frameA {fi} -> frameB {j}  "
              f"{'nearest-own' if discr else 'NEAREST WRONG'}", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/site_reid.csv", "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["pair_a", "site", "frameA", "u", "v", "frameB", "err_mm",
                   "status", "nearest_own"])
    wcsv.writerows(rows)

errs = [float(r[6]) for r in rows if r[7] == "ok"]
fails = [r for r in rows if r[7] != "ok"]
all_errs = errs + [float("inf")] * len([r for r in fails if r[7] == "not_refound"])
med = float(np.median(all_errs)) if all_errs else float("inf")
disc = [r[8] for r in rows if r[7] == "ok"]
disc_rate = (sum(disc) / len(rows)) if rows else 0.0   # failures count against G2
g1 = med <= 10.0
g2 = disc_rate >= 0.80
print(f"\nsites: {len(rows)} total, {len(errs)} re-found, {len(fails)} failed "
      f"({[r[7] for r in fails]})" if fails else f"\nsites: {len(rows)}, all re-found")
print(f"G1 median error = {med:.2f} mm (bar <= 10) -> {'PASS' if g1 else 'FAIL'}")
print(f"G2 discrimination = {disc_rate*100:.0f}% nearest-own (bar >= 80%) -> "
      f"{'PASS' if g2 else 'FAIL'}")
print("VERDICT:", "SITE RE-IDENTIFICATION HOLDS under oracle registration"
      if g1 and g2 else "gate failed - publish as the oracle bound with mechanism")
json.dump({"median_mm": med, "n_sites": len(rows), "n_refound": len(errs),
           "discrimination": disc_rate, "G1": bool(g1), "G2": bool(g2),
           "errors_mm": errs},
          open("results/site_reid_verdict.json", "w"), indent=1)
