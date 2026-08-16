"""
Build a self-contained HTML results dashboard for Argus: embeds the result images (as base64)
into one page you can open in any browser. Re-run any time to refresh with the latest results.
Output: results/dashboard.html
"""
import base64, os

# (title, caption, path) -- missing files are skipped
PANELS = [
    ("The problem", "A raw endoscopic frame. Smooth, low-texture tissue with bright specular "
     "glare (the white blobs) - exactly what breaks feature-based SLAM.",
     "data/hamlyn/rectified05/glariest_frame.jpg"),
    ("Stage 1: glare removal", "Left: original. Middle: what Argus flags as glare (red). "
     "Right: cleaned (glare inpainted, tissue preserved).",
     "results/glare_before_after.png"),
    ("Smooth vs AI fill", "original | glare | Telea (smooth mush) | LaMa (AI textured). The AI "
     "fill restores features where the smooth fill leaves a bald patch.",
     "results/telea_vs_lama.png"),
    ("HEADLINE: severity curve", "Glare severity (x) vs change in tracking (y), 5 runs/condition. "
     "The core scientific result - does removing glare help, and does AI beat smooth fill?",
     "results/severity_curve.png"),
    ("Per-sequence comparison", "Mean keyframes tracked per sequence, each condition side by side.",
     "results/baseline_vs_argus.png"),
    ("THE POSITIVE RESULT: direct methods", "For direct/photometric tracking (unlike feature-based "
     "ORB-SLAM3), glare removal HELPS - and only on glary sequences, scaling with glare. Temporal "
     "(real tissue) wins. Whether removal helps depends on the SLAM paradigm.",
     "results/direct_severity_curve.png"),
]


def data_uri(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def main():
    cards = []
    for title, cap, path in PANELS:
        if not os.path.exists(path):
            continue
        cards.append(f"""<figure class="card">
      <figcaption><span class="t">{title}</span><span class="c">{cap}</span></figcaption>
      <img src="{data_uri(path)}" alt="{title}">
    </figure>""")
    html = f"""<title>Argus - Results</title>
<style>
  :root{{--ink:#191c22;--paper:#f5f3ef;--surface:#fff;--muted:#6c655e;--line:#e6e0d8;
    --coral:#bc5346;--teal:#2c7b74;}}
  @media(prefers-color-scheme:dark){{:root{{--ink:#e9e5df;--paper:#121519;--surface:#1a1e24;
    --muted:#a29a90;--line:#2a2f37;--coral:#e0796b;--teal:#5fb5ad;}}}}
  *{{box-sizing:border-box}} body{{margin:0}}
  .wrap{{max-width:900px;margin:0 auto;padding:48px 20px 80px;background:var(--paper);
    color:var(--ink);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6}}
  .eyebrow{{font-family:Consolas,monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;
    color:var(--coral);font-weight:600;margin:0 0 10px}}
  h1{{font-family:Georgia,serif;font-size:2.4rem;margin:0 0 6px}}
  .sub{{color:var(--muted);margin:0 0 36px;font-size:1.05rem}}
  .card{{margin:0 0 34px;background:var(--surface);border:1px solid var(--line);border-radius:14px;
    overflow:hidden;box-shadow:0 6px 20px rgba(25,28,34,.06)}}
  figcaption{{padding:18px 22px 14px}}
  .t{{display:block;font-family:Georgia,serif;font-weight:600;font-size:1.25rem;margin-bottom:4px}}
  .c{{display:block;color:var(--muted);font-size:.95rem}}
  img{{display:block;width:100%;height:auto;border-top:1px solid var(--line);background:#000}}
  footer{{color:var(--muted);font-size:.9rem;border-top:1px solid var(--line);padding-top:20px}}
</style>
<div class="wrap">
  <p class="eyebrow">Argus - live results</p>
  <h1>Argus results</h1>
  <p class="sub">Specular-glare removal for monocular endoscopic SLAM. Every image below is a real
    output from the pipeline; refresh as experiments complete.</p>
  {''.join(cards)}
  <footer>Generated from <code>results/</code> by <code>speqslam/make_dashboard.py</code>.</footer>
</div>"""
    os.makedirs("results", exist_ok=True)
    with open("results/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote results/dashboard.html  ({len(cards)} panels, {os.path.getsize('results/dashboard.html')//1024} KB)")


if __name__ == "__main__":
    main()
