"""
End-to-end ablation ladder for the monocular blind-spot system. The evaluation substrate is always
the ground-truth colon surface (built from GT depth + GT pose). We compute the blind-spot map four
ways and score each against the TRUE blind map (Stage A):

  A  GT depth   + GT pose      -> reference (defines the true blind spots)
  B  GT depth   + VO pose      -> tracking-only error
  C  learned depth + GT pose   -> depth-only error (off-the-shelf monocular depth)
  D  learned depth + VO-on-learned pose -> FULL monocular system (no depth sensor, no GT pose)

Reports, per stage: agreement with the true blind map, coverage correlation, and detected blind %.
"""
import sys, os, glob, csv, subprocess, numpy as np
sys.path.insert(0, "speqslam")
import validate_coverage as vc
from validate_coverage import load_poses, build_surface, coverage_on_surface, align_fullpose

vc.OCC_TOL = 0.008
vc.OCC_REL = float(os.environ.get("OCC_REL", "0.25"))  # =measured depth accuracy band (delta<1.25)
THR = vc.THR
EVO = "C:/Users/moonw/anaconda3/envs/slamfair/Scripts/evo_ape.exe"


def ate_mm(seq, est):
    o = subprocess.run([EVO, "tum", f"results/c3vd/{seq}_gt.txt", est, "-as", "--no_warnings"],
                       capture_output=True, text=True).stdout
    for l in o.splitlines():
        if l.strip().lower().startswith("rmse"):
            return float(l.split()[-1]) * 1000
    return float("nan")


def blind(counts):
    return counts < THR


def score(ref_blind, ref_counts, counts):
    b = blind(counts)
    agree = 100 * (ref_blind == b).mean()
    corr = np.corrcoef(ref_counts, counts)[0, 1]
    return agree, corr, 100 * b.mean()


def run(seq):
    undist = f"data/c3vd/{seq}_undist"; K = np.loadtxt(f"{undist}/K.txt")
    gt_deps = sorted(glob.glob(f"{undist}/depth01/*.png"))
    ln_deps = sorted(glob.glob(f"{undist}/depth01_learned/*.png"))
    ft_deps = sorted(glob.glob(f"{undist}/depth01_ftc/*.png"))
    gt = load_poses(f"results/c3vd/{seq}_gt.txt")
    vo = load_poses(f"results/c3vd/{seq}_raw.txt")
    ln = load_poses(f"results/c3vd/{seq}_learned.txt")
    vo_al, _ = align_fullpose(gt, vo, sorted(set(gt) & set(vo)))
    ln_al, _ = align_fullpose(gt, ln, sorted(set(gt) & set(ln)))

    S = build_surface(gt_deps, gt, K)
    cA = coverage_on_surface(S, gt_deps, gt, K)          # reference
    refb = blind(cA)
    print(f"\n=== {seq}: {len(S)} surface patches, true blind = {100*refb.mean():.1f}% ===", flush=True)
    print(f"{'stage':<52}{'agree%':>8}{'corr':>7}{'blind%':>8}", flush=True)
    rows = [("A  GT depth + GT pose (reference)", 100.0, 1.0, 100 * refb.mean())]
    rows.append(("B  GT depth + VO pose",) + score(refb, cA, coverage_on_surface(S, gt_deps, vo_al, K)))
    rows.append(("C  off-shelf depth + GT pose",) + score(refb, cA, coverage_on_surface(S, ln_deps, gt, K)))
    rows.append(("D  off-shelf depth + its VO pose (full monocular)",)
                + score(refb, cA, coverage_on_surface(S, ln_deps, ln_al, K)))
    have_ft = len(ft_deps) == len(gt_deps) and os.path.exists(f"results/c3vd/{seq}_ft.txt")
    if have_ft:
        ft = load_poses(f"results/c3vd/{seq}_ft.txt")
        ft_al, _ = align_fullpose(gt, ft, sorted(set(gt) & set(ft)))
        rows.append(("C' fine-tuned depth + GT pose",) + score(refb, cA, coverage_on_surface(S, ft_deps, gt, K)))
        rows.append(("D' fine-tuned depth + its VO pose (FULL MONOCULAR)",)
                    + score(refb, cA, coverage_on_surface(S, ft_deps, ft_al, K)))
    for name, ag, co, bl in rows:
        print(f"{name:<52}{ag:>8.1f}{co:>7.2f}{bl:>8.1f}", flush=True)
    ate_b, ate_d = ate_mm(seq, f"results/c3vd/{seq}_raw.txt"), ate_mm(seq, f"results/c3vd/{seq}_learned.txt")
    ate_ft = ate_mm(seq, f"results/c3vd/{seq}_ft.txt") if have_ft else float("nan")
    print(f"    ATE: GT-depth VO {ate_b:.2f}mm | off-shelf-depth VO {ate_d:.2f}mm | ft-depth VO {ate_ft:.2f}mm", flush=True)
    out = dict(seq=seq, patches=len(S), true_blind=100 * refb.mean(),
               agree_B=rows[1][1], agree_C=rows[2][1], agree_D=rows[3][1],
               corr_D=rows[3][2], blind_D=rows[3][3], ate_B=ate_b, ate_D=ate_d, ate_FT=ate_ft)
    if have_ft:
        out.update(agree_C2=rows[4][1], agree_D2=rows[5][1], corr_D2=rows[5][2], blind_D2=rows[5][3])
    return out


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
    out = [run(s) for s in seqs]
    fields = sorted({k for r in out for k in r}, key=lambda k: (k != "seq", k))
    with open("results/ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    have_ft = all("agree_D2" in r for r in out)
    print("\n=== SUMMARY (OCC_REL=%.2f) ===" % vc.OCC_REL, flush=True)
    hdr = f"{'seq':<12}{'ATE_FT':>8}{'agreeB%':>8}{'agreeC%':>8}{'agreeD%':>8}"
    if have_ft:
        hdr += f"{'agreeC2%':>9}{'agreeD2%':>9}{'corrD2':>8}"
    print(hdr, flush=True)
    for r in out:
        line = f"{r['seq']:<12}{r.get('ate_FT', float('nan')):>8.2f}{r['agree_B']:>8.1f}{r['agree_C']:>8.1f}{r['agree_D']:>8.1f}"
        if have_ft:
            line += f"{r['agree_C2']:>9.1f}{r['agree_D2']:>9.1f}{r['corr_D2']:>8.2f}"
        print(line, flush=True)
    if have_ft:
        print(f"{'MEAN':<12}{np.mean([r['ate_FT'] for r in out]):>8.2f}"
              f"{np.mean([r['agree_B'] for r in out]):>8.1f}{np.mean([r['agree_C'] for r in out]):>8.1f}"
              f"{np.mean([r['agree_D'] for r in out]):>8.1f}{np.mean([r['agree_C2'] for r in out]):>9.1f}"
              f"{np.mean([r['agree_D2'] for r in out]):>9.1f}{np.mean([r['corr_D2'] for r in out]):>8.2f}", flush=True)
    print("saved results/ablation.csv", flush=True)
