"""
Interactive viewer for the FULLY-MONOCULAR colon reconstruction (TSDF mesh from learned depth +
learned-VO poses) with a toggle to overlay the ground-truth mesh. Shows that a reconstruction built
from video alone matches the ground-truth surface. Meshes are decimated for the web.
"""
import sys, numpy as np, open3d as o3d, plotly.graph_objects as go


def load_mesh(path, target_tris=70000):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles) > target_tris:
        m = m.simplify_quadric_decimation(target_tris)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices); tri = np.asarray(m.triangles)
    c = np.asarray(m.vertex_colors)
    return v, tri, c


def mesh_trace(v, tri, colorspec, name, visible, opacity=1.0):
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                     name=name, visible=visible, opacity=opacity, **colorspec)


def build(seq, mode="ftmono", traj="ft"):
    vm, tm, cm = load_mesh(f"results/mesh/{seq}_{mode}.ply")
    vg, tg, cg = load_mesh(f"results/mesh/{seq}_gt.ply")
    # align GT mesh into the mono frame for overlay via camera-center Sim(3)
    sys.path.insert(0, "speqslam")
    from validate_coverage import load_poses, umeyama
    from scipy.spatial.transform import Rotation as Rot
    gt, ln = load_poses(f"results/c3vd/{seq}_gt.txt"), load_poses(f"results/c3vd/{seq}_{traj}.txt")
    common = sorted(set(gt) & set(ln))
    Pv = np.array([ln[t][:3, 3] for t in common]); Pg = np.array([gt[t][:3, 3] for t in common])
    Rg = Rot.from_matrix(np.array([gt[t][:3, :3] @ ln[t][:3, :3].T for t in common])).mean().as_matrix()
    _, s = umeyama(Pv, Pg)
    vm2 = (s * (Rg @ vm.T).T) + (Pg.mean(0) - s * (Rg @ Pv.mean(0)))   # mono -> GT frame

    rgb_m = ['rgb(%d,%d,%d)' % tuple((c * 255).astype(int)) for c in cm]
    mono = mesh_trace(vm2, tm, dict(vertexcolor=rgb_m), 'monocular (realistic)', True)
    mono_plain = mesh_trace(vm2, tm, dict(color='#c98d6d'), 'monocular', False)
    gtmesh = mesh_trace(vg, tg, dict(color='#4a8f88'), 'ground truth', False, opacity=0.55)

    fig = go.Figure([mono, mono_plain, gtmesh])
    fig.update_layout(
        title=dict(text='Argus &#8212; fully-monocular colon reconstruction vs ground truth', x=0.5),
        paper_bgcolor='#0b0e13', font_color='#e9e5df',
        scene=dict(aspectmode='data', bgcolor='#0b0e13',
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        updatemenus=[dict(type='buttons', direction='right', x=0.5, y=1.06, xanchor='center',
                          bgcolor='#1a1e24', font=dict(color='#e9e5df'), buttons=[
                              dict(label='Monocular (realistic)', method='update',
                                   args=[{'visible': [True, False, False]}]),
                              dict(label='Monocular vs Ground Truth', method='update',
                                   args=[{'visible': [False, True, True]}]),
                          ])],
        margin=dict(l=0, r=0, t=50, b=0), showlegend=False)
    out = f"results/mono_recon_{seq}.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    frag = fig.to_html(include_plotlyjs=True, full_html=False)
    open(f"results/mono_recon_{seq}_artifact.html", "w", encoding="utf-8").write(
        f'<title>Monocular Colon Recon</title>\n<div style="background:#0b0e13;min-height:100vh">\n{frag}\n</div>\n')
    mb = round(__import__("os").path.getsize(out) / 1e6, 1)
    print(f"saved {out} ({mb} MB)  mono {len(tm)} tris, gt {len(tg)} tris", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "cecum_t1_a")
