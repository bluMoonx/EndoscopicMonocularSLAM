"""
Argus Studio -- the booth console. One local web page consolidating every demo:
  - launch the live SLAM pipeline (phantom / in-vivo) as native windows
  - polyp finder: pick a sample or upload any colonoscopy image -> detection + size, in-browser
  - click-to-measure: click two points on a frame -> real distance in mm, in-browser
  - interactive 3D models, procedure report cards, results dashboard (served from results/)

Run:  python speqslam/argus_studio.py     then open http://localhost:7860
Models (depth + polyp) load lazily on first use; everything runs locally.
"""
import os, sys, glob, base64, subprocess
import numpy as np, cv2
from flask import Flask, request, jsonify, send_from_directory
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
app = Flask(__name__)
DS = 100.0 / 65535.0 / 1000.0
_polyp = _depth = None

MEASURE_FRAMES = [("cecum_t1_a", 40), ("cecum_t1_a", 150), ("desc_t4_a", 60), ("trans_t2_a", 90)]
SAMPLES = sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/images/*.jpg"))[10:22]


def polyp_model():
    global _polyp
    if _polyp is None:
        from polyp_detect import load_model
        _polyp = load_model()
    return _polyp


def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()


@app.route("/api/samples")
def samples():
    out = []
    for i, p in enumerate(SAMPLES):
        t = cv2.resize(cv2.imread(p), (140, 116))
        out.append(dict(id=i, thumb=b64(t)))
    return jsonify(out)


@app.route("/api/polyp", methods=["POST"])
def api_polyp():
    from polyp_detect import segment, overlay
    from polyp_measure import diameter_mm
    import demo_app
    if "file" in request.files:
        buf = np.frombuffer(request.files["file"].read(), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(SAMPLES[int(request.form["sample"])])
    mask = segment(bgr, polyp_model())
    vis = overlay(bgr, mask)
    net = demo_app.DepthNet()
    depth = net.predict(bgr, relative=True)
    m = depth > 0
    if m.any():
        depth[m] = 0.012 + (depth[m] - depth[m].min()) / max(depth[m].ptp(), 1e-6) * 0.033
    h, w = bgr.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    dia = diameter_mm(mask, depth, K)
    found = bool((mask > 0.5).mean() > 0.001)
    return jsonify(image=b64(vis), found=found,
                   size=None if not (found and dia) else round(dia, 1))


@app.route("/api/measure_frame")
def measure_frame():
    seq, fr = MEASURE_FRAMES[int(request.args["id"])]
    img = sorted(glob.glob(f"data/c3vd/{seq}_undist/image01/*.jpg"))[fr]
    return jsonify(image=b64(cv2.imread(img)), label=f"{seq} frame {fr}")


@app.route("/api/measure", methods=["POST"])
def api_measure():
    from measure import dist_mm
    d = request.get_json()
    seq, fr = MEASURE_FRAMES[int(d["id"])]
    u = f"data/c3vd/{seq}_undist"
    dep = cv2.imread(sorted(glob.glob(f"{u}/depth01_ftc/*.png"))[fr], cv2.IMREAD_UNCHANGED).astype(np.float32) * DS
    K = np.loadtxt(f"{u}/K.txt")
    mm = dist_mm((int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"])), dep, K)
    return jsonify(mm=None if mm is None else round(mm, 1))


@app.route("/launch/<mode>")
def launch(mode):
    seq = {"phantom": "cecum_t1_a", "invivo": "rectified01"}.get(mode)
    if seq is None:
        return jsonify(ok=False), 404
    subprocess.Popen([PY, "speqslam/demo_app.py", "--seq", seq], cwd=ROOT)
    return jsonify(ok=True, msg=f"live pipeline launching ({seq}) - watch for two windows")


@app.route("/results/<path:p>")
def results(p):
    return send_from_directory(os.path.join(ROOT, "results"), p)


@app.route("/")
def index():
    reports = [os.path.basename(p) for p in sorted(glob.glob("results/report_*.html"))]
    rep_links = "".join(f'<a class="chip" href="/results/{r}" target="_blank">{r.replace("report_","").replace(".html","")}</a>' for r in reports)
    return HTML.replace("__REPORTS__", rep_links or "<i>none generated yet</i>")


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Argus Studio</title>
<style>
 :root{--ink:#191c22;--paper:#f5f3ef;--card:#fff;--line:#e6e0d8;--muted:#6c655e;--teal:#2c7b74;--coral:#bc5346}
 @media(prefers-color-scheme:dark){:root{--ink:#e9e5df;--paper:#121519;--card:#1a1e24;--line:#2a2f37;--muted:#a29a90;--teal:#5fb5ad;--coral:#e0796b}}
 body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 'Segoe UI',system-ui,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:28px 18px 60px}
 h1{font-family:Georgia,serif;margin:0} .sub{color:var(--muted);margin:2px 0 24px}
 h2{font-family:Georgia,serif;font-size:1.25rem;border-bottom:1px solid var(--line);padding-bottom:6px;margin:34px 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
 button{background:var(--teal);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:.95rem;cursor:pointer;margin:3px 6px 3px 0}
 button.alt{background:var(--coral)} button:disabled{opacity:.5}
 .chip{display:inline-block;background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:4px 12px;margin:3px;color:var(--ink);text-decoration:none;font-size:.88rem}
 .thumb{width:140px;height:116px;border-radius:8px;cursor:pointer;margin:3px;border:2px solid transparent}
 .thumb:hover{border-color:var(--teal)}
 #polypOut img,#measureImg{max-width:100%;border-radius:10px;margin-top:10px}
 .big{font-size:1.25rem;font-weight:700;color:var(--teal)}
 .note{color:var(--muted);font-size:.83rem}
 #measureWrap{position:relative;display:inline-block}
 #measureCanvas{position:absolute;left:0;top:0;pointer-events:none}
 .status{color:var(--muted);font-size:.9rem;margin-left:8px}
</style></head><body><div class="wrap">
 <h1>Argus Studio</h1>
 <p class="sub">Fully monocular endoscopy analysis — every demo in one place. Everything below runs live on this machine.</p>

 <h2>1 · Live SLAM pipeline (native windows)</h2>
 <div class="card">
  <button onclick="launch('phantom')">Launch — phantom (validated)</button>
  <button class="alt" onclick="launch('invivo')">Launch — real tissue (qualitative)</button>
  <span class="status" id="launchStatus"></span>
  <p class="note">Opens two windows: live video panels (raw | glare mask | learned depth) and the growing 3D reconstruction. SPACE pause · B blind-spot reveal · ESC quit.</p>
 </div>

 <h2>2 · Polyp finder + auto-sizing</h2>
 <div class="card">
  <div id="gallery"></div>
  <input type="file" id="upl" accept="image/*"> <span class="note">or upload any colonoscopy image</span>
  <div id="polypOut"></div>
 </div>

 <h2>3 · Click-to-measure (validated: median 0.4–1.3 mm error)</h2>
 <div class="card">
  <span id="frameBtns"></span>
  <div id="measureWrap"><img id="measureImg"><canvas id="measureCanvas"></canvas></div>
  <div id="measureOut" class="big"></div>
  <p class="note">Click two points on the image — the distance is computed from the system's own monocular depth.</p>
 </div>

 <h2>4 · Interactive 3D models</h2>
 <div class="card">
  <a class="chip" href="/results/colon_3d_cecum_t1_a.html" target="_blank">3D colon + blind-spot heatmap (cecum)</a>
  <a class="chip" href="/results/colon_3d_desc_t4_a.html" target="_blank">3D colon (descending)</a>
  <a class="chip" href="/results/mono_recon_cecum_t1_a.html" target="_blank">monocular vs ground truth overlay</a>
 </div>

 <h2>5 · Procedure report cards</h2>
 <div class="card">__REPORTS__</div>

 <h2>6 · Full results dashboard</h2>
 <div class="card"><a class="chip" href="/results/dashboard.html" target="_blank">open dashboard</a></div>

<script>
const $=id=>document.getElementById(id);
function launch(m){fetch('/launch/'+m).then(r=>r.json()).then(d=>$('launchStatus').textContent=d.msg||'error')}
fetch('/api/samples').then(r=>r.json()).then(list=>{
  $('gallery').innerHTML=list.map(s=>`<img class="thumb" src="data:image/jpeg;base64,${s.thumb}" onclick="runPolyp(${s.id})">`).join('');
});
function showPolyp(d){
  $('polypOut').innerHTML=(d.found?`<div class="big">polyp detected — est. size ~${d.size} mm <span class="note">(illustrative scale; sizing geometry validated ±0.6 mm)</span></div>`:'<div class="big">no polyp detected</div>')+`<img src="data:image/jpeg;base64,${d.image}">`;
}
function runPolyp(i){
  $('polypOut').innerHTML='<span class="status">analyzing…</span>';
  const f=new FormData(); f.append('sample',i);
  fetch('/api/polyp',{method:'POST',body:f}).then(r=>r.json()).then(showPolyp);
}
$('upl').onchange=e=>{
  const f=new FormData(); f.append('file',e.target.files[0]);
  $('polypOut').innerHTML='<span class="status">analyzing…</span>';
  fetch('/api/polyp',{method:'POST',body:f}).then(r=>r.json()).then(showPolyp);
};
const FRAMES=4; let curFrame=0, pts=[];
$('frameBtns').innerHTML=Array.from({length:FRAMES},(_,i)=>`<button onclick="loadFrame(${i})">frame ${i+1}</button>`).join('');
function loadFrame(i){curFrame=i;pts=[];$('measureOut').textContent='';
  fetch('/api/measure_frame?id='+i).then(r=>r.json()).then(d=>{
    $('measureImg').src='data:image/jpeg;base64,'+d.image;
    $('measureImg').onload=()=>{const im=$('measureImg'),c=$('measureCanvas');c.width=im.width;c.height=im.height;c.getContext('2d').clearRect(0,0,c.width,c.height);};
  });}
$('measureImg').addEventListener('click',e=>{
  const im=$('measureImg'),r=im.getBoundingClientRect();
  const sx=im.naturalWidth/r.width, sy=im.naturalHeight/r.height;
  pts.push({dx:e.clientX-r.left, dy:e.clientY-r.top, x:(e.clientX-r.left)*sx, y:(e.clientY-r.top)*sy});
  const ctx=$('measureCanvas').getContext('2d');
  ctx.fillStyle='#3fd6a0'; ctx.beginPath(); ctx.arc(pts[pts.length-1].dx,pts[pts.length-1].dy,4,0,7); ctx.fill();
  if(pts.length===2){
    ctx.strokeStyle='#3fd6a0'; ctx.lineWidth=2; ctx.beginPath();
    ctx.moveTo(pts[0].dx,pts[0].dy); ctx.lineTo(pts[1].dx,pts[1].dy); ctx.stroke();
    fetch('/api/measure',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:curFrame,x1:pts[0].x,y1:pts[0].y,x2:pts[1].x,y2:pts[1].y})})
      .then(r=>r.json()).then(d=>{$('measureOut').textContent=d.mm?d.mm+' mm':'no depth at that point — try again';pts=[];});
  }
});
loadFrame(0);
</script></div></body></html>"""


if __name__ == "__main__":
    print("Argus Studio -> http://localhost:7860", flush=True)
    app.run(host="127.0.0.1", port=7860, debug=False)
