"""Shared red-team metric helpers. No side effects at import - keep it that way."""
import numpy as np


def horn_ate(gt, est):
    """RMSE after optimal similarity aligning est -> gt, via Horn's (1987) closed-form
    quaternion absolute orientation - deliberately NOT the project's Umeyama SVD."""
    gt = np.asarray(gt, float); est = np.asarray(est, float)
    cg, ce = gt.mean(0), est.mean(0)
    G, E = gt - cg, est - ce
    M = E.T @ G
    Sxx, Sxy, Sxz = M[0]; Syx, Syy, Syz = M[1]; Szx, Szy, Szz = M[2]
    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,        Sxy - Syx],
        [Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,        Szx + Sxz],
        [Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx,       Szx + Sxz,       Syz + Szy,        -Sxx - Syy + Szz]])
    w, v = np.linalg.eigh(N)
    q = v[:, np.argmax(w)]
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]])
    Er = E @ R.T
    denom = (E ** 2).sum()
    s = (Er * G).sum() / denom if denom > 0 else 1.0
    resid = s * Er - G
    return float(np.sqrt((resid ** 2).sum(1).mean()))


def tum_xyz(path):
    out = []
    for l in open(path):
        v = l.split()
        if len(v) >= 8:
            out.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(out)
