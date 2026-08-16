"""
Undistort C3VD fisheye (Scaramuzza / OcamCalib) frames to a virtual PINHOLE camera, so the
pinhole-based direct VO can use them. C3VD gives the cam2world polynomial f(rho)=a0+a2*rho^2+
a3*rho^3+a4*rho^4 (ray z-component for a pixel at radius rho from center) plus a small stretch
matrix [[c,d],[e,1]]. We build a remap: for each OUTPUT pinhole pixel we take its 3D ray, find
the matching fisheye radius (by inverting theta(rho)=atan2(rho,f(rho)) via a LUT), apply the
stretch, and sample the distorted image.
"""
import cv2, numpy as np


def load_calib(path):
    t = open(path).read().split()          # Width height cx cy a0 a2 a3 a4 e f g
    return dict(W=int(t[0]), H=int(t[1]), cx=float(t[2]), cy=float(t[3]),
                a=[float(t[4]), float(t[5]), float(t[6]), float(t[7])],
                c=float(t[8]), d=float(t[9]), e=float(t[10]))


def _f(rho, a):
    a0, a2, a3, a4 = a
    return a0 + a2 * rho**2 + a3 * rho**3 + a4 * rho**4


def build_map(cal, out_w=640, out_h=512, f_pin=250.0):
    cxp, cyp = out_w / 2.0, out_h / 2.0
    # LUT rho -> theta (angle of ray from optical axis); monotonic in rho
    rr = np.linspace(0, 0.8 * np.hypot(cal['cx'], cal['cy']), 6000)
    theta = np.arctan2(rr, _f(rr, cal['a']))
    keep = np.concatenate([[True], np.diff(theta) > 0])   # enforce monotonic for interp
    rr, theta = rr[keep], theta[keep]
    # output pinhole pixel rays
    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (us - cxp) / f_pin; y = (vs - cyp) / f_pin
    theta_D = np.arctan2(np.hypot(x, y), 1.0)
    phi = np.arctan2(y, x)
    rho = np.interp(theta_D.ravel(), theta, rr).reshape(theta_D.shape)
    xp, yp = rho * np.cos(phi), rho * np.sin(phi)
    mapu = (cal['c'] * xp + cal['d'] * yp + cal['cx']).astype(np.float32)
    mapv = (cal['e'] * xp + yp + cal['cy']).astype(np.float32)
    K = np.array([[f_pin, 0, cxp], [0, f_pin, cyp], [0, 0, 1.0]])
    return mapu, mapv, K


def undistort(img, mapu, mapv, nearest=False):
    return cv2.remap(img, mapu, mapv,
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
