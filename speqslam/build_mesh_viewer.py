"""
Build a self-contained interactive TRIANGLE-MESH viewer (three.js, offline) from the
high-resolution fused reconstructions in results/mesh/*.ply. This is the renderer the
project always deserved: the Plotly station draws a point scatter, which makes a
0.6 mm-voxel colored mesh look like a blob; this one draws the actual lit surface.

Views: tissue (vertex colors from the video), coverage heat + blind spots (mapped from
the sequence's recon .npz coverage points onto the mesh via nearest neighbor), and the
camera path. Everything is embedded (three.js r128 + base64 geometry): one HTML file,
fully offline, double-click to open — fair-day safe.

Usage: python speqslam/build_mesh_viewer.py [seq] [mode]   (default cecum_t1_a ftmono)
Writes results/mesh3d_<seq>_<mode>.html
"""
import base64, json, os, sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

seq = sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a"
mode = sys.argv[2] if len(sys.argv) > 2 else "ftmono"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

mesh = o3d.io.read_triangle_mesh(os.path.join(ROOT, f"results/mesh/{seq}_{mode}.ply"))
tris0 = len(mesh.triangles)
TARGET = 300_000
if tris0 > TARGET:                       # keep browsers smooth; barely visible at 0.6mm voxels
    mesh = mesh.simplify_quadric_decimation(TARGET)
mesh.remove_unreferenced_vertices()
V = np.asarray(mesh.vertices, np.float32)
C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
T = np.asarray(mesh.triangles, np.uint32)
print(f"{seq}_{mode}: {tris0} -> {len(T)} tris, {len(V)} verts")

# coverage + blind spots from the recon point cloud, mapped onto mesh vertices.
# The npz and the PLY come from different pipeline stages and may differ by a rigid
# transform (frame convention / alignment step), so register the cloud to the mesh:
# best axis-flip + centroid init, then point-to-point ICP.
cov_v = None
npz = os.path.join(ROOT, f"results/{seq}_recon.npz")
if os.path.exists(npz):
    d = np.load(npz)
    pts = d["pts"].astype(np.float64)
    traj = d["traj"].astype(np.float64)
    kd = cKDTree(V)
    best = None
    for fl in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1),
               (1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, -1)]:
        F = np.diag(fl).astype(np.float64)
        p = pts @ F.T
        p = p - p.mean(0) + V.mean(0)
        err = np.mean(kd.query(p[::20], k=1)[0])
        if best is None or err < best[0]:
            best = (err, F)
    F = best[1]
    pts = pts @ F.T; traj = traj @ F.T
    shift = V.mean(0) - pts.mean(0)
    pts += shift; traj += shift
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[::5]))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V.astype(np.float64)))
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, 0.01, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    M = icp.transformation
    pts = pts @ M[:3, :3].T + M[:3, 3]
    traj = traj @ M[:3, :3].T + M[:3, 3]
    kd2 = cKDTree(pts)
    dist, idx = kd2.query(V, k=1)
    cov_v = d["coverage"][idx].astype(np.float32)
    cov_v[dist > 0.004] = -1              # vertex has no coverage sample within 4 mm
    traj = traj.astype(np.float32)
    print(f"registration: flip {np.diag(F).astype(int)}, icp fitness {icp.fitness:.2f}, "
          f"rmse {icp.inlier_rmse*1000:.1f} mm")
    print(f"coverage mapped: {np.mean(cov_v >= 0)*100:.0f}% of vertices sampled, "
          f"{np.mean((cov_v >= 0) & (cov_v < 3))*100:.1f}% blind")
else:
    traj = np.zeros((0, 3), np.float32)
    print("no recon npz - tissue view only")

def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

three = open(os.path.join(HERE, "vendor/three.r128.min.js"), encoding="utf-8").read()
orbit = open(os.path.join(HERE, "vendor/OrbitControls.r128.js"), encoding="utf-8").read()
payload = {"v": b64(V), "c": b64(C), "t": b64(T),
           "cov": b64(cov_v) if cov_v is not None else "",
           "traj": b64(traj), "nv": len(V), "nt": len(T), "ntr": len(traj)}

html = """<!doctype html><html><head><meta charset="utf-8">
<title>Argus 3D — %SEQ%</title>
<style>html,body{margin:0;height:100%%;background:#0b0e13;overflow:hidden;font-family:system-ui}
#hud{position:fixed;top:12px;left:50%%;transform:translateX(-50%%);z-index:9;display:flex;gap:8px}
#hud button{background:#1a1e24;color:#e9e5df;border:1px solid #333;border-radius:8px;
padding:8px 14px;cursor:pointer;font-size:14px}
#hud button.on{background:#2e4d3a;border-color:#51b06c}
#cap{position:fixed;bottom:10px;left:50%%;transform:translateX(-50%%);color:#8a8f98;font-size:12px}
</style></head><body>
<div id="hud"><button id="bt" class="on">Realistic tissue</button>
<button id="bc">Coverage + blind spots</button><button id="bp" class="on">Camera path</button></div>
<div id="cap">%SEQ% · %MODE% · fused at 0.6&thinsp;mm voxels from video alone · drag to orbit, wheel to zoom</div>
<script>%THREE%</script><script>%ORBIT%</script>
<script>
const P=%PAYLOAD%;
function f32(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new Float32Array(a.buffer)}
function u8(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}
function u32(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new Uint32Array(a.buffer)}
const V=f32(P.v),C=u8(P.c),T=u32(P.t),COV=P.cov?f32(P.cov):null,TR=f32(P.traj);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e13);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(V,3));
geo.setIndex(new THREE.BufferAttribute(T,1));geo.computeVertexNormals();
const tiss=new Float32Array(P.nv*3);for(let i=0;i<P.nv;i++){tiss[3*i]=C[3*i]/255;tiss[3*i+1]=C[3*i+1]/255;tiss[3*i+2]=C[3*i+2]/255}
let covc=null;
if(COV){covc=new Float32Array(P.nv*3);
 for(let i=0;i<P.nv;i++){const c=COV[i];
  if(c<0){covc[3*i]=.25;covc[3*i+1]=.25;covc[3*i+2]=.28}
  else if(c<3){covc[3*i]=.95;covc[3*i+1]=.12;covc[3*i+2]=.12}
  else{const t=Math.min(1,c/40);covc[3*i]=.1+.1*t;covc[3*i+1]=.35+.6*t;covc[3*i+2]=.9-.65*t}}}
geo.setAttribute('color',new THREE.BufferAttribute(tiss.slice(),3));
const mat=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.62,metalness:.05,side:THREE.DoubleSide});
const mesh=new THREE.Mesh(geo,mat);scene.add(mesh);
geo.computeBoundingBox();const bb=geo.boundingBox,ctr=bb.getCenter(new THREE.Vector3()),ext=bb.getSize(new THREE.Vector3()).length();
let path=null;
if(P.ntr>1){const pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(TR,3));
 path=new THREE.Line(pg,new THREE.LineBasicMaterial({color:0x3fd6ff,linewidth:2}));scene.add(path)}
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,ext/1000,ext*10);
cam.position.set(ctr.x+ext*.55,ctr.y+ext*.35,ctr.z+ext*.55);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2));document.body.appendChild(ren.domElement);
const ctl=new THREE.OrbitControls(cam,ren.domElement);ctl.target.copy(ctr);ctl.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x334,0.85));
const key=new THREE.DirectionalLight(0xffffff,0.9);key.position.set(1,1.2,.8);scene.add(key);
const head=new THREE.PointLight(0xffe8d0,0.7,ext*3);scene.add(head);
function setColors(a){geo.getAttribute('color').copyArray(a);geo.getAttribute('color').needsUpdate=true}
const bt=document.getElementById('bt'),bc=document.getElementById('bc'),bp=document.getElementById('bp');
bt.onclick=()=>{setColors(tiss);bt.classList.add('on');bc.classList.remove('on')};
bc.onclick=()=>{if(covc){setColors(covc);bc.classList.add('on');bt.classList.remove('on')}};
bp.onclick=()=>{if(path){path.visible=!path.visible;bp.classList.toggle('on')}};
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight)});
(function anim(){requestAnimationFrame(anim);head.position.copy(cam.position);ctl.update();ren.render(scene,cam)})();
</script></body></html>"""

html = (html.replace("%SEQ%", seq).replace("%MODE%", mode)
            .replace("%THREE%", three).replace("%ORBIT%", orbit)
            .replace("%PAYLOAD%", json.dumps(payload)))
out = os.path.join(ROOT, f"results/mesh3d_{seq}_{mode}.html")
open(out, "w", encoding="utf-8").write(html)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
