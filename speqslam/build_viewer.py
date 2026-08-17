"""
Build a self-contained interactive 3D viewer (Plotly) from a reconstruction .npz. Rotatable colon
model with a toggle between realistic tissue color and the coverage/blind-spot heatmap, plus the
camera path.
"""
import sys, numpy as np, plotly.graph_objects as go

seq = sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a"
d = np.load(f"results/{seq}_recon.npz")
pts, colors, cov, traj = d["pts"], d["colors"], d["coverage"], d["traj"]
thr = 3
covered = cov >= thr
rgb = ['rgb(%d,%d,%d)' % (c[0], c[1], c[2]) for c in colors]

real = go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
                    marker=dict(size=1.8, color=rgb), name='tissue', hoverinfo='skip', visible=True)
covtr = go.Scatter3d(x=pts[covered, 0], y=pts[covered, 1], z=pts[covered, 2], mode='markers',
                     marker=dict(size=1.8, color=np.clip(cov[covered], 0, 60), colorscale='Viridis',
                                 cmin=0, cmax=60, colorbar=dict(title='camera<br>views')),
                     name='coverage', hoverinfo='skip', visible=False)
blind = go.Scatter3d(x=pts[~covered, 0], y=pts[~covered, 1], z=pts[~covered, 2], mode='markers',
                     marker=dict(size=4.5, color='red'), name='blind spot', hoverinfo='skip', visible=False)
path = go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines',
                    line=dict(color='#3fd6ff', width=4), name='camera path', visible=True)

fig = go.Figure([real, covtr, blind, path])
fig.update_layout(
    title=dict(text=f'Argus &#8212; interactive 3D colon reconstruction &amp; blind-spot map', x=0.5),
    paper_bgcolor='#0b0e13', font_color='#e9e5df',
    scene=dict(aspectmode='data', bgcolor='#0b0e13',
               xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    updatemenus=[dict(type='buttons', direction='right', x=0.5, y=1.06, xanchor='center',
                      bgcolor='#1a1e24', font=dict(color='#e9e5df'),
                      buttons=[
                          dict(label='Realistic tissue', method='update',
                               args=[{'visible': [True, False, False, True]}]),
                          dict(label='Coverage + blind spots', method='update',
                               args=[{'visible': [False, True, True, True]}]),
                      ])],
    margin=dict(l=0, r=0, t=50, b=0), showlegend=False)
fig.write_html(f"results/colon_3d_{seq}.html", include_plotlyjs=True, full_html=True)
# div-only body (no <html>/<head>/<body>) for publishing as a hosted Artifact
frag = fig.to_html(include_plotlyjs=True, full_html=False)
open(f"results/colon_3d_{seq}_artifact.html", "w", encoding="utf-8").write(
    '<title>Argus 3D Colon</title>\n<div style="background:#0b0e13;min-height:100vh">\n'
    + frag + "\n</div>\n")
mb = round(__import__("os").path.getsize(f"results/colon_3d_{seq}.html") / 1e6, 1)
print(f"saved results/colon_3d_{seq}.html  ({mb} MB)  blind={100*(~covered).mean():.1f}% of {len(pts)} points")
