"""
Generate an ORB-SLAM3 monocular .yaml for a Hamlyn rectified sequence, reading the shipped
intrinsics_<seq>.txt and the actual image size. Keeps ORB settings identical across all
sequences so baselines are comparable for the severity curve.

Usage: python make_config.py rectified04 [rectified06 ...]
"""
import sys, os, glob, cv2

TEMPLATE = """%YAML:1.0

# Argus - Hamlyn {seq}, LEFT camera, MONOCULAR. Auto-generated from intrinsics_{seq}.txt.
File.version: "1.0"

Camera.type: "PinHole"
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0
Camera.fps: 30
Camera.RGB: 1
Camera.width: {w}
Camera.height: {h}

ORBextractor.nFeatures: 4000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 10
ORBextractor.iniThFAST: 5
ORBextractor.minThFAST: 2

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""

def make(seq):
    toks = open(f"data/hamlyn/intrinsics_{seq}.txt").read().split()
    fx, cx, fy, cy = toks[0], toks[2], toks[5], toks[6]
    img = sorted(glob.glob(f"data/hamlyn/{seq}/image01/*.jpg"))[0]
    h, w = cv2.imread(img).shape[:2]
    out = f"configs/hamlyn_{seq}_mono.yaml"
    open(out, "w").write(TEMPLATE.format(seq=seq, fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h))
    print(f"{seq}: fx={fx} cx={cx} cy={cy} {w}x{h} -> {out}")

if __name__ == "__main__":
    for s in sys.argv[1:]:
        make(s)
