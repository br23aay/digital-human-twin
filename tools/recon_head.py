"""Reconstruct a textured 3D head+shoulders bust from 4 photos (front, left, right, back).
Visual hull from segmented silhouettes, rounded with superellipse cross-sections,
front face replaced by a MediaPipe face-mesh depth map, colours projected from the photos.
"""
import sys, json, numpy as np, trimesh
from PIL import Image
from scipy import ndimage as nd
from skimage import measure

SC = sys.argv[1]; P = sys.argv[2]; OUT = sys.argv[3]
K = 4            # world units (front-photo pixels) per voxel
PEXP = 2.4       # superellipse exponent

def load(n):
    c = np.load(SC + 'seg/' + n + '.npy')
    m = nd.binary_fill_holes(nd.binary_opening(c != 0, iterations=3))
    lab, k = nd.label(m); m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    img = np.asarray(Image.open(P + n + '.jpeg').convert('RGB')).astype(np.float32)
    return c, m, img

V = {}
for key, n in [('front', 'Front Head'), ('back', 'Back Head'), ('left', 'Left Head'), ('right', 'Right Head')]:
    c, m, img = load(n)
    top = np.where(m.any(1))[0][0]
    fy = np.where((c == 3).sum(1) > 15)[0]
    w = np.array([(np.where(r)[0][-1] - np.where(r)[0][0]) if r.any() else 0 for r in m])
    V[key] = dict(c=c, m=m, img=img, top=top, chin=fy[-1] if len(fy) else None, w=w)

f = V['front']; H0 = f['chin'] - f['top']
f['s'] = 1.0
V['back']['s'] = f['w'][f['top']:f['top'] + 700].max() / V['back']['w'][V['back']['top']:V['back']['top'] + 700].max()
for k in ('left', 'right'):
    V[k]['s'] = H0 / (V[k]['chin'] - V[k]['top'])
# horizontal centres at crown level (world row 0.3*H0)
for k, v in V.items():
    row = int(v['top'] + 0.3 * H0 / v['s'])
    xs = np.where(v['m'][row])[0]; v['cx'] = (xs[0] + xs[-1]) / 2
print({k: (round(v['s'], 3), int(v['top']), round(v['cx'], 1)) for k, v in V.items()})

Ymax = min((v['m'].shape[0] - 1 - v['top']) * v['s'] for v in V.values()) - 4
# stop above the row where the shoulders leave the frame in any view
for v in V.values():
    edge = np.where(v['m'][:, 0] | v['m'][:, -1])[0]
    if len(edge): Ymax = min(Ymax, (edge[0] - v['top']) * v['s'] - 12)
print('Ymax', Ymax)
Ys = np.arange(0, Ymax, K); Xs = np.arange(-460, 460, K); Zs = np.arange(-520, 520, K)

def sample_mask(v, Y, U):  # U = signed horizontal world coordinate for this view
    yy = np.clip((v['top'] + Y[:, None] / v['s']).round().astype(int), 0, v['m'].shape[0] - 1)
    uu = (v['cx'] + U[None, :] / v['s']).round().astype(int)
    ok = (uu >= 0) & (uu < v['m'].shape[1])
    return v['m'][yy, np.clip(uu, 0, v['m'].shape[1] - 1)] & ok

Mx = sample_mask(V['front'], Ys, Xs) & sample_mask(V['back'], Ys, -Xs)
Mz = sample_mask(V['left'], Ys, -Zs) & sample_mask(V['right'], Ys, Zs)

def intervals(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

# rounded hull -> front/back z surface per (y,x) column (indices into Zs)
nY, nX, nZ = len(Ys), len(Xs), len(Zs)
Fz = np.full((nY, nX), -1.0); Bz = np.full((nY, nX), -1.0)
zi = np.arange(nZ)
for yi in range(nY):
    xiv = [iv for iv in intervals(Mx[yi]) if iv[1] - iv[0] >= 2]
    ziv = [iv for iv in intervals(Mz[yi]) if iv[1] - iv[0] >= 2]
    if not xiv or not ziv: continue
    z0, z1 = min(a for a, b in ziv), max(b for a, b in ziv)
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2 + 0.5
    for a, b in xiv:
        xc, hx = (a + b) / 2, (b - a) / 2 + 0.5
        xs = np.arange(a, b + 1)
        t = np.clip(1 - np.abs((xs - xc) / hx) ** PEXP, 0, 1) ** (1 / PEXP)
        Fz[yi, xs] = zc + hz * t; Bz[yi, xs] = zc - hz * t

# ---- face depth from MediaPipe landmarks on the front photo ----
lm = np.array(json.load(open(SC + 'out/landmarks.json'))['landmarks'])
tris = np.array(json.load(open(SC + 'out/tris.json'))).reshape(-1, 3)
W, Hh = f['img'].shape[1], f['img'].shape[0]
LX = (lm[:, 0] * W - f['cx']) / K + (len(Xs) / 2)          # grid x index
LY = (lm[:, 1] * Hh - f['top']) / K                         # grid y index
LZ = -(lm[:, 2] - lm[1, 2]) * W / K                         # cells behind nose tip (<=0)
ZD = np.full((nY, nX), np.nan)
for t in tris:
    x, y, z = LX[t], LY[t], LZ[t]
    x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max())); y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max()))
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
    if abs(den) < 1e-9: continue
    a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
    b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
    c = 1 - a - b; ins = (a >= -1e-3) & (b >= -1e-3) & (c >= -1e-3)
    gx, gy, zz = gx[ins], gy[ins], (a * z[0] + b * z[1] + c * z[2])[ins]
    ok = (gx >= 0) & (gx < nX) & (gy >= 0) & (gy < nY)
    ZD[gy[ok], gx[ok]] = zz[ok]
fm = ~np.isnan(ZD)
nose_y, nose_x = int(round(LY[1])), int(round(LX[1]))
znose = Fz[nose_y, nose_x]
zface = znose + np.nan_to_num(ZD)
valid = Fz >= 0
# front-face region: face mesh where it is not too far behind the nose (avoid ear-side depths)
Rg = fm & (np.nan_to_num(ZD, nan=-999) > -60) & valid
Rg = nd.binary_erosion(Rg, iterations=2)
lab, nl = nd.label(Rg); Rg = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
# offset field: equals (ellipse - face) on the region border, diffused smoothly inside (Laplace)
ring = nd.binary_dilation(Rg, iterations=1) & ~Rg & valid
off = np.zeros_like(Fz); off[ring] = (Fz - zface)[ring]
known = ring.copy(); cur = off.copy()
box = nd.binary_dilation(Rg, iterations=2)
for it in range(1500):
    sm = nd.uniform_filter(cur, 3)
    cur = np.where(Rg, sm, off)
F2 = np.where(Rg, zface + cur, Fz)
# never leave the side-view silhouette: clamp to the profile's front extent per row
zmax = np.array([max([b for a, b in intervals(Mz[yi])] or [0]) for yi in range(nY)], float)
F2 = np.minimum(F2, zmax[:, None])
# soften the border a little
blend = nd.gaussian_filter(Rg.astype(float), 2.0)
F2 = np.where(valid, blend * F2 + (1 - blend) * Fz, Fz)
F2 = np.maximum(F2, Bz + 1)
print('nose z cell', znose, 'face depth range', np.nanmin(ZD), np.nanmax(ZD))

# ---- volume (signed-ish distance along z) and mesh ----
vol = np.zeros((nY, nX, nZ), np.float32)
for yi in range(nY):
    v = valid[yi]
    d = np.minimum(F2[yi][:, None] - zi[None, :], zi[None, :] - Bz[yi][:, None])
    vol[yi] = np.where(v[:, None], np.clip(d, -3, 3), -3)
vol[-1] = -3; vol[0] = -3
vol = nd.gaussian_filter(vol, 0.9)
verts, faces, normals, _ = measure.marching_cubes(vol, 0.0, allow_degenerate=False)
# world coordinates (front px): y down, x = his left, z = forward
Y = Ys[0] + verts[:, 0] * K; X = Xs[0] + verts[:, 1] * K; Z = Zs[0] + verts[:, 2] * K
mesh = trimesh.Trimesh(np.stack([X, -Y, Z], 1), faces[:, ::-1], process=True)
mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
trimesh.smoothing.filter_taubin(mesh, iterations=8)
N = mesh.vertex_normals; X, Y, Z = mesh.vertices[:, 0], -mesh.vertices[:, 1], mesh.vertices[:, 2]

# ---- colour projection ----
def sample(v, Yw, U):
    yy = v['top'] + Yw / v['s']; uu = v['cx'] + U / v['s']
    h, w = v['m'].shape
    ok = (yy >= 0) & (yy < h - 1) & (uu >= 0) & (uu < w - 1)
    yi = np.clip(yy.astype(int), 0, h - 2); ui = np.clip(uu.astype(int), 0, w - 2)
    fy, fx = (yy - yi)[:, None], (uu - ui)[:, None]
    im = v['img']
    col = (im[yi, ui] * (1 - fx) * (1 - fy) + im[yi, ui + 1] * fx * (1 - fy) + im[yi + 1, ui] * (1 - fx) * fy + im[yi + 1, ui + 1] * fx * fy)
    inside = ok & nd.binary_erosion(v['m'], iterations=4)[yi, ui]
    return col, inside
acc = np.zeros((len(X), 3)); wsum = np.zeros(len(X))
for key, U, nd_ in [('front', X, N[:, 2]), ('back', -X, -N[:, 2]), ('left', -Z, N[:, 0]), ('right', Z, -N[:, 0])]:
    col, inside = sample(V[key], Y, U)
    w = np.clip(nd_, 0, 1) ** 3 * inside + 1e-4 * inside
    if key == 'front': w = w * 2.0
    acc += col * w[:, None]; wsum += w
colors = np.clip(acc / np.maximum(wsum, 1e-6)[:, None], 0, 255)
nocol = wsum < 1e-5
if nocol.any():
    from scipy.spatial import cKDTree
    t = cKDTree(mesh.vertices[~nocol]); _, idx = t.query(mesh.vertices[nocol]); colors[nocol] = colors[~nocol][idx]

# metres: front-photo px -> m using top-of-head to chin ~= 0.235 m
mpp = 0.235 / H0
mesh.vertices = mesh.vertices * mpp
mesh.vertices -= [0, mesh.vertices[:, 1].min(), 0]
mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[colors.astype(np.uint8), np.full(len(colors), 255, np.uint8)])
mesh.export(OUT + '/head.glb'); mesh.export(OUT + '/head.ply')
print('verts', len(mesh.vertices), 'faces', len(mesh.faces), 'size m', mesh.extents.round(3))
