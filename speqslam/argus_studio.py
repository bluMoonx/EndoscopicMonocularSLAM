"""
Argus Studio v2 -- the booth console. One local web page consolidating every demo:
  1 live SLAM pipeline launcher (5 phantom + 4 in-vivo sequences, native windows)
  2 polyp finder + auto-sizing (sample gallery or upload, in-browser)
  3 click-to-measure (validated monocular mm measurement, in-browser)
  4 glare lab (interactive specular-detection threshold on real in-vivo frames)
  5 depth X-ray (off-the-shelf vs fine-tuned depth, side by side -- THE key result, visual)
  6 trajectory viewer (estimated vs ground-truth camera path, interactive 3D)
  7 evidence graphs (ablation / robustness / withdrawal-speed / measurement, interactive)
  8 interactive 3D models   9 report cards   10 dashboard

Run:  python speqslam/argus_studio.py   ->  http://localhost:7860
Fully offline once models are on disk (plotly.js served from the local package).
"""
import os, sys, glob, csv, base64, socket, subprocess
import numpy as np, cv2
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
app = Flask(__name__)
DS = 100.0 / 65535.0 / 1000.0
_polyp = _depthnet = _offshelf = None

PHANTOM = ["cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b"]
INVIVO = ["rectified01", "rectified04", "rectified05", "rectified06"]
MEASURE_FRAMES = [("cecum_t1_a", 40), ("cecum_t1_a", 150), ("desc_t4_a", 60), ("trans_t2_a", 90)]
GLARE_FRAMES = [("rectified05", 120), ("rectified05", 500), ("rectified04", 300), ("rectified01", 200)]
SAMPLES = sorted(glob.glob("data/kvasir_seg/Kvasir-SEG/images/*.jpg"))[10:22]


def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1]).decode()


def polyp_model():
    global _polyp
    if _polyp is None:
        from polyp_detect import load_model
        _polyp = load_model()
    return _polyp


def depthnet():
    global _depthnet
    if _depthnet is None:
        import demo_app
        _depthnet = demo_app.DepthNet()
    return _depthnet


def offshelf_depth(bgr):
    """Off-the-shelf DA-V2-Small relative depth, for the before/after comparison."""
    global _offshelf
    import torch
    if _offshelf is None:
        from transformers import AutoModelForDepthEstimation
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _offshelf = (AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf").to(dev).eval(), dev)
    model, dev = _offshelf
    rgb = cv2.cvtColor(cv2.resize(bgr, (392, 392)), cv2.COLOR_BGR2RGB)
    x = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 -
         torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    with torch.no_grad():
        d = model(pixel_values=x.unsqueeze(0).to(dev)).predicted_depth[0]
    d = torch.nn.functional.interpolate(d[None, None], size=bgr.shape[:2], mode="bilinear")[0, 0]
    return d.cpu().numpy()


def depth_vis(d):
    v = d[np.isfinite(d) & (d != 0)]
    lo, hi = (np.percentile(v, 3), np.percentile(v, 97)) if len(v) else (0, 1)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_MAGMA)


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


# ------------------------------ API ------------------------------

# The Argus mark: one great eye (Panoptes, the all-seeing watchman) -- iris of point-cloud dots
# (the reconstruction), a camera-path swoosh through the pupil, a ring of small watcher-eyes
# (coverage), and exactly ONE of them red: the blind spot it exists to catch.
FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="14" fill="#0b0e13"/>
<circle cx="32" cy="32" r="17" fill="none" stroke="#5fb5ad" stroke-width="2.6"/>
<g fill="#e0796b">
 <circle cx="32" cy="20.5" r="1.7"/><circle cx="40.1" cy="23.9" r="1.7"/><circle cx="43.5" cy="32" r="1.7"/>
 <circle cx="40.1" cy="40.1" r="1.7"/><circle cx="32" cy="43.5" r="1.7"/><circle cx="23.9" cy="40.1" r="1.7"/>
 <circle cx="20.5" cy="32" r="1.7"/><circle cx="23.9" cy="23.9" r="1.7"/>
</g>
<circle cx="32" cy="32" r="6.5" fill="#10151c" stroke="#5fb5ad" stroke-width="1"/>
<path d="M27.5 33.5 Q30 29.5 33 31.5 T36.5 30" fill="none" stroke="#3fd6ff" stroke-width="1.6" stroke-linecap="round"/>
<g fill="#5fb5ad">
 <circle cx="32" cy="6.5" r="2.1"/><circle cx="50" cy="14" r="2.1"/><circle cx="57.5" cy="32" r="2.1"/>
 <circle cx="50" cy="50" r="2.1"/><circle cx="32" cy="57.5" r="2.1"/><circle cx="14" cy="50" r="2.1"/>
 <circle cx="6.5" cy="32" r="2.1"/>
</g>
<circle cx="14" cy="14" r="2.4" fill="#e63946"/>
</svg>"""


@app.route("/favicon.svg")
def favicon():
    return Response(FAVICON, mimetype="image/svg+xml")


@app.route("/plotly.js")
def plotlyjs():
    import plotly
    return send_file(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"))


@app.route("/api/samples")
def samples():
    return jsonify([dict(id=i, thumb=b64(cv2.resize(cv2.imread(p), (140, 116))))
                    for i, p in enumerate(SAMPLES)])


@app.route("/api/polyp", methods=["POST"])
def api_polyp():
    from polyp_detect import segment, overlay
    from polyp_measure import diameter_mm
    if "file" in request.files:
        bgr = cv2.imdecode(np.frombuffer(request.files["file"].read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(SAMPLES[int(request.form["sample"])])
    mask = segment(bgr, polyp_model())
    vis = overlay(bgr, mask)
    depth = depthnet().predict(bgr, relative=True)
    m = depth > 0
    if m.any():
        depth[m] = 0.012 + (depth[m] - depth[m].min()) / max(depth[m].ptp(), 1e-6) * 0.033
    h, w = bgr.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    dia = diameter_mm(mask, depth, K)
    found = bool((mask > 0.5).mean() > 0.001)
    return jsonify(image=b64(vis), found=found, size=None if not (found and dia) else round(dia, 1))


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


@app.route("/api/glare")
def api_glare():
    from glare_removal import specular_mask
    seq, fr = GLARE_FRAMES[int(request.args["id"])]
    thr = int(request.args.get("thr", 225))
    bgr = cv2.imread(sorted(glob.glob(f"data/hamlyn/{seq}/image01/*.jpg"))[fr])
    mask = specular_mask(bgr, method="fixed", fixed_thr=thr, dilate=1)
    vis = bgr.copy(); vis[mask > 0] = (40, 40, 230)
    pct = 100 * (mask > 0).mean()
    return jsonify(image=b64(vis), pct=round(pct, 2), label=f"{seq} frame {fr}")


@app.route("/api/depthcompare")
def api_depthcompare():
    seq, fr = MEASURE_FRAMES[int(request.args["id"])]
    bgr = cv2.imread(sorted(glob.glob(f"data/c3vd/{seq}_undist/image01/*.jpg"))[fr])
    off = depth_vis(offshelf_depth(bgr))
    ft = depth_vis(-depthnet().predict(bgr))          # negate: nearer = brighter, both panels
    h = 300
    def rs(im): return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
    return jsonify(image=b64(np.hstack([rs(bgr), rs(off), rs(ft)])),
                   label=f"{seq} frame {fr}: raw | off-the-shelf | YOUR fine-tuned")


@app.route("/api/traj")
def api_traj():
    from validate_coverage import load_poses, align_fullpose
    seq = request.args.get("seq", "cecum_t1_a")
    if seq not in PHANTOM:
        return jsonify(error="unknown"), 404
    gt = load_poses(f"results/c3vd/{seq}_gt.txt")
    ft = load_poses(f"results/c3vd/{seq}_ft.txt")
    ft_al, _ = align_fullpose(gt, ft, sorted(set(gt) & set(ft)))
    G = np.array([gt[t][:3, 3] for t in sorted(gt)]) * 1000
    F = np.array([ft_al[t][:3, 3] for t in sorted(ft_al)]) * 1000
    return jsonify(gt=G.tolist(), est=F.tolist())


@app.route("/api/charts")
def api_charts():
    from make_dashboard import DEPTH_D1
    return jsonify(
        ablation=read_rows("results/ablation.csv"),
        robustness=read_rows("results/robustness.csv"),
        robustness_illum=[r for r in csv.reader(open("results/robustness_illum.csv"))][1:]
        if os.path.exists("results/robustness_illum.csv") else [],
        withdrawal=read_rows("results/withdrawal_speed.csv"),
        measure=read_rows("results/measure_validation.csv"),
        depth=[dict(seq=s, before=a, after=b) for s, (a, b) in DEPTH_D1.items()])


@app.route("/launch/<seq>")
def launch(seq):
    if seq not in PHANTOM + INVIVO:
        return jsonify(ok=False), 404
    subprocess.Popen([PY, "speqslam/demo_app.py", "--seq", seq], cwd=ROOT)
    return jsonify(ok=True, msg=f"live pipeline launching ({seq}) - watch for two windows")


@app.route("/results/<path:p>")
def results(p):
    return send_from_directory(os.path.join(ROOT, "results"), p)


# ---------------------- Argus Handheld: the phone becomes the scope ----------------------
_rec = {"on": False, "dir": None, "n": 0}


@app.route("/api/handheld_frame", methods=["POST"])
def handheld_frame():
    """Phone posts a camera frame; the real pipeline runs on it; a processed strip goes back.
    Unknown camera -> relative depth, adaptive glare detector, everything labeled demo."""
    from glare_removal import specular_mask
    bgr = cv2.imdecode(np.frombuffer(request.files["frame"].read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="bad frame"), 400
    if _rec["on"]:
        cv2.imwrite(f"{_rec['dir']}/image01/{_rec['n']:06d}.jpg", bgr)
        _rec["n"] += 1
    mask = specular_mask(bgr, method="adaptive", k=2.0, lo=225, hi=245, dilate=1)
    glr = bgr.copy(); glr[mask > 0] = (40, 40, 230)
    depth = depthnet().predict(bgr, relative=True)
    h = 240
    def rs(im): return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
    strip = np.hstack([rs(bgr), rs(glr), rs(depth_vis(-depth))])
    return jsonify(image=b64(strip), glare_pct=round(100 * (mask > 0).mean(), 1),
                   recording=_rec["on"], frames=_rec["n"])


@app.route("/api/handheld_record", methods=["POST"])
def handheld_record():
    import time
    if _rec["on"]:
        _rec["on"] = False
        return jsonify(recording=False, frames=_rec["n"], session=os.path.basename(_rec["dir"] or ""))
    sid = time.strftime("scan_%Y%m%d_%H%M%S")
    _rec.update(on=True, dir=f"data/handheld/{sid}", n=0)
    os.makedirs(f"{_rec['dir']}/image01", exist_ok=True)
    return jsonify(recording=True, session=sid)


@app.route("/handheld")
def handheld():
    return HANDHELD_HTML


HANDHELD_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Argus Handheld</title><link rel="icon" href="/favicon.svg">
<style>
 body{margin:0;background:#0b0e13;color:#e9e5df;font:16px/1.4 'Segoe UI',system-ui,sans-serif;text-align:center}
 h1{font-family:Georgia,serif;font-size:1.3rem;margin:12px 0 2px}
 .sub{color:#a29a90;font-size:.8rem;margin:0 0 10px}
 #view{width:96vw;max-width:900px;border-radius:12px;background:#10151c;min-height:130px}
 video{width:44vw;max-width:320px;border-radius:10px;border:1px solid #2a2f37;margin-top:8px}
 button{background:#2c7b74;color:#fff;border:0;border-radius:12px;padding:14px 26px;font-size:1.05rem;margin:12px 8px}
 button.rec{background:#bc5346}
 #stat{color:#a29a90;font-size:.85rem;min-height:22px}
 .tag{display:inline-block;border:1px solid #2a2f37;border-radius:12px;padding:2px 10px;font-size:.72rem;color:#a29a90;margin:4px 2px}
</style></head><body>
 <h1>Argus Handheld</h1>
 <p class="sub">your phone is the scope — the workstation is the brain · demo mode, uncalibrated camera</p>
 <img id="view"><video id="cam" autoplay playsinline muted></video><br>
 <button id="start">Start camera</button>
 <button id="flip" style="display:none">Flip camera</button>
 <button id="rec" class="rec" style="display:none">● Record scan</button>
 <div id="stat">the three panels: raw | glare mask | depth</div>
 <div><span class="tag">raw</span><span class="tag">glare (red)</span><span class="tag">AI depth</span></div>
<script>
const v=document.getElementById('cam'),view=document.getElementById('view'),stat=document.getElementById('stat');
let busy=false, on=false, cams=[], camIdx=0;
async function openCam(constraint){
 if(v.srcObject) v.srcObject.getTracks().forEach(t=>t.stop());
 const s=await navigator.mediaDevices.getUserMedia({video:constraint});
 v.srcObject=s;
 await v.play();
 await new Promise(r=>{ if(v.videoWidth) r(); else v.onloadeddata=r; });
 stat.textContent=`camera live: ${v.videoWidth}x${v.videoHeight} — small view below is the RAW feed`;
}
document.getElementById('start').onclick=async()=>{
 try{
  await openCam({facingMode:'environment',width:{ideal:640}});
  cams=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
  on=true;
  document.getElementById('start').style.display='none';
  document.getElementById('rec').style.display='inline-block';
  if(cams.length>1) document.getElementById('flip').style.display='inline-block';
  loop();
 }catch(e){stat.textContent='camera blocked: '+e.message+' — allow the camera in the browser permission popup';}
};
document.getElementById('flip').onclick=async()=>{
 camIdx=(camIdx+1)%cams.length;
 try{await openCam({deviceId:{exact:cams[camIdx].deviceId},width:{ideal:640}});}
 catch(e){stat.textContent='that lens refused: '+e.message}
};
async function loop(){
 if(!on)return;
 if(!busy && v.videoWidth){
  busy=true;
  const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(async b=>{
   const f=new FormData();f.append('frame',b,'f.jpg');
   try{
    const d=await fetch('/api/handheld_frame',{method:'POST',body:f}).then(r=>r.json());
    view.src='data:image/jpeg;base64,'+d.image;
    stat.textContent=`glare ${d.glare_pct}%`+(d.recording?`  ·  ● REC ${d.frames} frames`:'');
   }catch(e){stat.textContent='lost connection — is the studio running?'}
   busy=false;
  },'image/jpeg',0.8);
 }
 setTimeout(loop,120);
}
document.getElementById('rec').onclick=async()=>{
 const d=await fetch('/api/handheld_record',{method:'POST'}).then(r=>r.json());
 document.getElementById('rec').textContent=d.recording?'■ Stop recording':'● Record scan';
 if(!d.recording&&d.frames)stat.textContent=`saved ${d.frames} frames as ${d.session} — reconstruct it from the workstation`;
};
</script></body></html>"""


@app.route("/")
def index():
    reports = [os.path.basename(p) for p in sorted(glob.glob("results/report_*.html"))]
    rep = "".join(f'<a class="chip" href="/results/{r}" target="_blank">{r.replace("report_","").replace(".html","")}</a>' for r in reports)
    return HTML.replace("__REPORTS__", rep or "<i>none yet</i>")


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Argus Studio</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<script src="/plotly.js"></script>
<style>
 :root{--ink:#191c22;--paper:#f5f3ef;--card:#fff;--card2:#fbf9f6;--line:#e6e0d8;--muted:#6c655e;--teal:#2c7b74;--coral:#bc5346;--cyan:#1c7f9e;--shadow:0 1px 2px rgba(25,28,34,.05),0 6px 20px rgba(25,28,34,.06)}
 @media(prefers-color-scheme:dark){:root{--ink:#e9e5df;--paper:#121519;--card:#1a1e24;--card2:#171b20;--line:#2a2f37;--muted:#a29a90;--teal:#5fb5ad;--coral:#e0796b;--cyan:#3fd6ff;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 26px rgba(0,0,0,.34)}}
 *{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 'Segoe UI',system-ui,sans-serif}
 .wrap{max-width:1040px;margin:0 auto;padding:0 18px 70px}
 header{display:flex;align-items:center;gap:18px;padding:34px 0 10px}
 header img{width:64px;height:64px;border-radius:14px}
 h1{font-family:Georgia,serif;font-size:2.2rem;margin:0;letter-spacing:.01em}
 h1 small{display:block;font-family:'Segoe UI',sans-serif;font-size:.95rem;font-weight:400;color:var(--muted);letter-spacing:0}
 nav{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(6px);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:6px}
 nav a{display:inline-block;color:var(--ink);text-decoration:none;font-size:.82rem;border:1px solid var(--line);background:var(--card);border-radius:16px;padding:3px 11px;margin:2px 3px}
 nav a:hover{border-color:var(--teal);color:var(--teal)}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0 8px}
 .stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:var(--shadow)}
 .stat b{display:block;font-family:Georgia,serif;font-size:1.5rem;color:var(--teal)}
 .stat span{color:var(--muted);font-size:.76rem}
 h2{font-family:Georgia,serif;font-size:1.3rem;margin:38px 0 10px}
 h2 .num{font-family:Consolas,monospace;font-size:.8rem;color:var(--coral);font-weight:700;margin-right:9px;letter-spacing:.05em}
 h2 .note{font-weight:400}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px;box-shadow:var(--shadow)}
 button,select{background:var(--teal);color:#fff;border:0;border-radius:8px;padding:9px 14px;font-size:.92rem;cursor:pointer;margin:3px 6px 3px 0;transition:filter .15s}
 button:hover{filter:brightness(1.12)}
 select{background:var(--card);color:var(--ink);border:1px solid var(--line)}
 button.alt{background:var(--coral)}
 .chip{display:inline-block;background:var(--card2);border:1px solid var(--line);border-radius:16px;padding:5px 13px;margin:3px;color:var(--ink);text-decoration:none;font-size:.88rem}
 .chip:hover{border-color:var(--teal);color:var(--teal)}
 .thumb{width:140px;height:116px;border-radius:8px;cursor:pointer;margin:3px;border:2px solid transparent;object-fit:cover}
 .thumb:hover{border-color:var(--teal)}
 #polypOut img,#measureImg,#glareImg,#depthImg{max-width:100%;border-radius:10px;margin-top:10px}
 .big{font-size:1.2rem;font-weight:700;color:var(--teal)} .note{color:var(--muted);font-size:.83rem}
 #measureWrap{position:relative;display:inline-block}
 #measureCanvas{position:absolute;left:0;top:0;pointer-events:none}
 .status{color:var(--muted);font-size:.9rem;margin-left:8px}
 .chart{height:340px;margin-top:8px} input[type=range]{width:220px;vertical-align:middle;accent-color:var(--teal)}
 footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
 footer i{font-family:Georgia,serif}
</style></head><body><div class="wrap">
 <header><img src="/favicon.svg" alt="Argus">
  <h1>Argus Studio<small>the hundred-eyed watchman for colonoscopy — one camera in, blind spots out</small></h1>
 </header>
 <nav>
  <a href="#live">live pipeline</a><a href="/handheld">📱 handheld</a><a href="#polyp">polyp finder</a><a href="#measure">measure</a>
  <a href="#glare">glare lab</a><a href="#depth">depth x-ray</a><a href="#traj">trajectory</a>
  <a href="#graphs">evidence</a><a href="#models">3D models</a><a href="#reports">reports</a><a href="#dash">dashboard</a>
 </nav>
 <div class="stats">
  <div class="stat"><b>0.91<small> mm</small></b><span>tracking error, fully monocular</span></div>
  <div class="stat"><b>98.6<small> %</small></b><span>blind-spot agreement with truth</span></div>
  <div class="stat"><b>1.3&ndash;2.4<small> mm</small></b><span>3D surface accuracy from video</span></div>
  <div class="stat"><b>0.60<small> mm</small></b><span>median sizing error (3&ndash;30mm)</span></div>
  <div class="stat"><b>0.96</b><span>polyp segmentation Dice</span></div>
 </div>

 <h2 id="live"><span class="num">01</span>Live SLAM pipeline <span class="note">(native windows)</span></h2>
 <div class="card">
  <select id="seqSel">
   <optgroup label="phantom (validated)"><option>cecum_t1_a</option><option>desc_t4_a</option><option>trans_t1_a</option><option>trans_t2_a</option><option>trans_t2_b</option></optgroup>
   <optgroup label="real tissue (qualitative)"><option>rectified01</option><option>rectified04</option><option>rectified05</option><option>rectified06</option></optgroup>
  </select>
  <button onclick="launch()">Launch live pipeline</button><span class="status" id="launchStatus"></span>
  <p class="note">Two windows: live panels (raw | glare mask | learned depth, with the LIVE COVERAGE readout) and the growing 3D map. SPACE pause · B blind-spot reveal · ESC quit.<br>📱 <b>Argus Handheld:</b> run the studio with <code>--https</code>, open <code>/handheld</code> on a phone on the same WiFi — the phone becomes the scope, this machine becomes the brain.</p>
 </div>

 <h2 id="polyp"><span class="num">02</span>Polyp finder + auto-sizing</h2>
 <div class="card"><div id="gallery"></div>
  <input type="file" id="upl" accept="image/*"> <span class="note">or upload any colonoscopy image</span>
  <div id="polypOut"></div></div>

 <h2 id="measure"><span class="num">03</span>Click-to-measure <span class="note">(validated: median 0.4–1.3 mm error)</span></h2>
 <div class="card"><span id="frameBtns"></span>
  <div id="measureWrap"><img id="measureImg"><canvas id="measureCanvas"></canvas></div>
  <div id="measureOut" class="big"></div>
  <p class="note">Click two points — distance computed from the system's own monocular depth.</p></div>

 <h2 id="glare"><span class="num">04</span>Glare lab <span class="note">(Act I: the specular detector, on real in-vivo frames)</span></h2>
 <div class="card"><span id="glareBtns"></span>
  brightness threshold <input type="range" id="glareThr" min="180" max="254" value="225" oninput="glareRun()">
  <span id="glareStat" class="big"></span><br><img id="glareImg"></div>

 <h2 id="depth"><span class="num">05</span>Depth X-ray <span class="note">(the key result, visually: generic AI vs YOUR fine-tuned AI)</span></h2>
 <div class="card"><span id="depthBtns"></span>
  <div id="depthLbl" class="note"></div><img id="depthImg"></div>

 <h2 id="traj"><span class="num">06</span>Trajectory viewer <span class="note">(estimated camera path vs ground truth, mm)</span></h2>
 <div class="card">
  <select id="trajSel" onchange="trajRun()"><option>cecum_t1_a</option><option>desc_t4_a</option><option>trans_t1_a</option><option>trans_t2_a</option><option>trans_t2_b</option></select>
  <div id="trajChart" class="chart"></div></div>

 <h2 id="graphs"><span class="num">07</span>Evidence graphs <span class="note">(interactive; every number vs ground truth)</span></h2>
 <div class="card"><div id="chAbl" class="chart"></div><div id="chDepth" class="chart"></div>
  <div id="chRob" class="chart"></div><div id="chWd" class="chart"></div></div>

 <h2 id="models"><span class="num">08</span>Interactive 3D models</h2>
 <div class="card">
  <b>Surface viewers</b> — the real fused triangle meshes (0.6&thinsp;mm voxels, lit, vertex-colored;
  built by speqslam/build_mesh_viewer.py). The point-cloud pages below are kept for comparison:
  <br>
  <a class="chip" href="/results/mesh3d_cecum_t1_a_ftmono.html" target="_blank">SURFACE: cecum + coverage/blind overlay</a>
  <a class="chip" href="/results/mesh3d_desc_t4_a_ftmono.html" target="_blank">SURFACE: descending + coverage</a>
  <a class="chip" href="/results/mesh3d_trans_t1_a_ftmono.html" target="_blank">SURFACE: transverse</a>
  <a class="chip" href="/results/mesh3d_cecum_t1_a_mono.html" target="_blank">SURFACE: cecum, OFF-THE-SHELF depth (the before)</a>
  <a class="chip" href="/results/mesh3d_cecum_t1_a_gt.html" target="_blank">SURFACE: cecum, ground-truth depth (the ceiling)</a>
  <br>
  <a class="chip" href="/results/colon_3d_cecum_t1_a.html" target="_blank">point cloud: cecum + blind spots</a>
  <a class="chip" href="/results/colon_3d_desc_t4_a.html" target="_blank">point cloud: descending</a>
  <a class="chip" href="/results/mono_recon_cecum_t1_a.html" target="_blank">monocular vs ground truth</a></div>

 <h2 id="reports"><span class="num">09</span>Procedure report cards</h2><div class="card">__REPORTS__</div>
 <h2 id="dash"><span class="num">10</span>Full results dashboard</h2><div class="card"><a class="chip" href="/results/dashboard.html" target="_blank">open dashboard</a></div>

 <footer>ARGUS &mdash; named for <i>Argus Panoptes</i>, the hundred-eyed watchman who never fully slept. This system watches every patch of colon wall so nothing goes unseen. Built by BluMooon &middot; GSDSEF 2026 &middot; every number validated against C3VD ground truth.</footer>

<script>
const $=id=>document.getElementById(id);
const TEAL='#2c7b74', CORAL='#bc5346', AMBER='#b67f26';
const LAYOUT={margin:{l:50,r:10,t:40,b:45},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{size:11}};
function launch(){fetch('/launch/'+$('seqSel').value).then(r=>r.json()).then(d=>$('launchStatus').textContent=d.msg||'error')}
fetch('/api/samples').then(r=>r.json()).then(l=>$('gallery').innerHTML=l.map(s=>`<img class="thumb" src="data:image/jpeg;base64,${s.thumb}" onclick="runPolyp(${s.id})">`).join(''));
function showPolyp(d){$('polypOut').innerHTML=(d.found?`<div class="big">polyp detected — est. size ~${d.size} mm <span class="note">(illustrative scale; geometry validated ±0.6 mm)</span></div>`:'<div class="big">no polyp detected</div>')+`<img src="data:image/jpeg;base64,${d.image}">`}
function runPolyp(i){$('polypOut').innerHTML='<span class="status">analyzing…</span>';const f=new FormData();f.append('sample',i);fetch('/api/polyp',{method:'POST',body:f}).then(r=>r.json()).then(showPolyp)}
$('upl').onchange=e=>{const f=new FormData();f.append('file',e.target.files[0]);$('polypOut').innerHTML='<span class="status">analyzing…</span>';fetch('/api/polyp',{method:'POST',body:f}).then(r=>r.json()).then(showPolyp)};
let curFrame=0,pts=[];
$('frameBtns').innerHTML=[0,1,2,3].map(i=>`<button onclick="loadFrame(${i})">frame ${i+1}</button>`).join('');
function loadFrame(i){curFrame=i;pts=[];$('measureOut').textContent='';fetch('/api/measure_frame?id='+i).then(r=>r.json()).then(d=>{$('measureImg').src='data:image/jpeg;base64,'+d.image;$('measureImg').onload=()=>{const im=$('measureImg'),c=$('measureCanvas');c.width=im.width;c.height=im.height;c.getContext('2d').clearRect(0,0,c.width,c.height)}})}
$('measureImg').addEventListener('click',e=>{const im=$('measureImg'),r=im.getBoundingClientRect();const sx=im.naturalWidth/r.width,sy=im.naturalHeight/r.height;pts.push({dx:e.clientX-r.left,dy:e.clientY-r.top,x:(e.clientX-r.left)*sx,y:(e.clientY-r.top)*sy});const ctx=$('measureCanvas').getContext('2d');ctx.fillStyle='#3fd6a0';ctx.beginPath();ctx.arc(pts[pts.length-1].dx,pts[pts.length-1].dy,4,0,7);ctx.fill();
 if(pts.length===2){ctx.strokeStyle='#3fd6a0';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pts[0].dx,pts[0].dy);ctx.lineTo(pts[1].dx,pts[1].dy);ctx.stroke();fetch('/api/measure',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:curFrame,x1:pts[0].x,y1:pts[0].y,x2:pts[1].x,y2:pts[1].y})}).then(r=>r.json()).then(d=>{$('measureOut').textContent=d.mm?d.mm+' mm':'no depth there — try again';pts=[]})}});
let curGlare=0;
$('glareBtns').innerHTML=[0,1,2,3].map(i=>`<button onclick="curGlare=${i};glareRun()">frame ${i+1}</button>`).join('');
function glareRun(){fetch(`/api/glare?id=${curGlare}&thr=${$('glareThr').value}`).then(r=>r.json()).then(d=>{$('glareImg').src='data:image/jpeg;base64,'+d.image;$('glareStat').textContent=d.pct+'% flagged'})}
$('depthBtns').innerHTML=[0,1,2,3].map(i=>`<button onclick="depthRun(${i})">frame ${i+1}</button>`).join('');
function depthRun(i){$('depthLbl').textContent='computing both models…';fetch('/api/depthcompare?id='+i).then(r=>r.json()).then(d=>{$('depthImg').src='data:image/jpeg;base64,'+d.image;$('depthLbl').textContent=d.label})}
function trajRun(){fetch('/api/traj?seq='+$('trajSel').value).then(r=>r.json()).then(d=>{
 const mk=(P,name,color,w)=>({type:'scatter3d',mode:'lines',name,x:P.map(p=>p[0]),y:P.map(p=>p[1]),z:P.map(p=>p[2]),line:{color,width:w}});
 Plotly.newPlot('trajChart',[mk(d.gt,'ground truth',AMBER,6),mk(d.est,'Argus (monocular)',TEAL,4)],{...LAYOUT,scene:{aspectmode:'data'},title:'camera path (mm): estimated vs truth'},{displayModeBar:false})})}
fetch('/api/charts').then(r=>r.json()).then(d=>{
 const seqs=d.ablation.map(r=>r.seq.replace('_t','-t'));
 Plotly.newPlot('chAbl',[
  {type:'bar',name:'off-the-shelf depth (full mono)',x:seqs,y:d.ablation.map(r=>+r.agree_D),marker:{color:CORAL}},
  {type:'bar',name:'fine-tuned depth (full mono)',x:seqs,y:d.ablation.map(r=>+r.agree_D2),marker:{color:TEAL}}],
  {...LAYOUT,title:'blind-spot map agreement with ground truth (%) — the fine-tuning fix',barmode:'group',yaxis:{range:[0,105]}},{displayModeBar:false});
 Plotly.newPlot('chDepth',[
  {type:'bar',name:'off-the-shelf',x:d.depth.map(r=>r.seq.replace('_t','-t')),y:d.depth.map(r=>r.before),marker:{color:CORAL}},
  {type:'bar',name:'fine-tuned (held out)',x:d.depth.map(r=>r.seq.replace('_t','-t')),y:d.depth.map(r=>r.after),marker:{color:TEAL}}],
  {...LAYOUT,title:'depth accuracy: % pixels within 25% of truth',barmode:'group',yaxis:{range:[0,105]}},{displayModeBar:false});
 const rb=k=>d.robustness.filter(r=>r.condition===k);
 const fl=rb('flicker'),base=+rb('baseline')[0]?.ate_mm||0.67;
 const ill=d.robustness_illum.filter(r=>r[0].startsWith('flicker'));
 Plotly.newPlot('chRob',[
  {x:[0,...fl.map(r=>+r.level)],y:[base,...fl.map(r=>+r.ate_mm)],name:'tracker as-is',line:{color:CORAL,width:3},mode:'lines+markers'},
  {x:[0,...ill.map(r=>+r[1])],y:[0.75,...ill.map(r=>+r[2])],name:'+ illumination compensation',line:{color:TEAL,width:3,dash:'dash'},mode:'lines+markers'}],
  {...LAYOUT,title:'brightness-flicker stress: weakness found → fixed (tracking error, mm)',xaxis:{title:'flicker ±fraction'}},{displayModeBar:false});
 const wseqs=[...new Set(d.withdrawal.map(r=>r.seq))];
 Plotly.newPlot('chWd',wseqs.map(s=>({x:d.withdrawal.filter(r=>r.seq===s).map(r=>+r.speed_x),y:d.withdrawal.filter(r=>r.seq===s).map(r=>+r.blind_pct),name:s.replace('_t','-t'),mode:'lines+markers'})),
  {...LAYOUT,title:'the cost of rushing: blind spots vs withdrawal speed',xaxis:{title:'× real speed'},yaxis:{title:'blind %'}},{displayModeBar:false});
});
loadFrame(0);glareRun();trajRun();
</script></div></body></html>"""


if __name__ == "__main__":
    try:                                  # real LAN IP (hostnames don't resolve from phones)
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); _s.connect(("8.8.8.8", 80))
        host = _s.getsockname()[0]; _s.close()
    except OSError:
        host = socket.gethostname()
    https = "--https" in sys.argv
    proto = "https" if https else "http"
    print(f"Argus Studio -> {proto}://localhost:7860", flush=True)
    print(f"  on your network -> {proto}://{host}:7860  (any device on the same WiFi)", flush=True)
    if https:
        print(f"  HANDHELD (phone camera needs https): {proto}://{host}:7860/handheld", flush=True)
        print("  your phone will warn about the self-signed certificate - tap Advanced > Proceed", flush=True)
    app.run(host="0.0.0.0", port=7860, debug=False, ssl_context="adhoc" if https else None)
