"""Full-body reconstruction from 4 photos (front, back, left side, right side).
Silhouette hull aligned with MediaPipe pose landmarks, superellipse cross-sections,
limb-aware depth, colours projected from the photos."""
import sys, json, numpy as np, trimesh
from PIL import Image
from scipy import ndimage as nd
from skimage import measure

SC, P, OUT = sys.argv[1:4]
HEIGHT_M = float(sys.argv[4]) if len(sys.argv) > 4 else 1.75
K = 3; PEXP = 2.2
pose = json.load(open(SC + 'seg/pose.json'))
views = {'front': 'Front Body', 'back': 'Back Side Body', 'left': 'Left Side Body', 'right': 'Right SIde Body'}
V = {}
for k, n in views.items():
    c = np.load(SC + 'seg/' + n + '.npy')
    m = nd.binary_fill_holes(nd.binary_opening(c != 0, iterations=2))
    lab, _ = nd.label(m)
    # keep the component that contains the hip landmark (removes furniture etc.)
    L = np.array(pose[n]); hx, hy = L[[23, 24], 0].mean(), L[[23, 24], 1].mean()
    lid = lab[int(hy), int(hx)] or (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    m = lab == lid
    img = np.asarray(Image.open(P + n + '.jpeg').convert('RGB')).astype(np.float32)
    V[k] = dict(c=c, m=m, img=img, L=L, hipv=hy, hipu=hx, shv=L[[11, 12], 1].mean())
f = V['front']; top_f = np.where(f['m'].any(1))[0][0]
for k, v in V.items():
    v['s'] = (f['hipv'] - f['shv']) / (v['hipv'] - v['shv'])
HIPW = f['hipv'] - top_f            # world y of hips (front px from top of head)
for k in ('left', 'right'):         # torso centre in side views, at chest level
    v = V[k]; row = int(v['shv'] + 0.3 * (v['hipv'] - v['shv']))
    lab, _ = nd.label(v['m'][row][None, :]); seg = lab[0]
    best = max(range(1, seg.max() + 1), key=lambda i: (seg == i).sum())
    xs = np.where(seg == best)[0]; v['cx'] = (xs[0] + xs[-1]) / 2
for k in ('front', 'back'): V[k]['cx'] = V[k]['hipu']
print({k: (round(v['s'], 3), round(v['cx'], 1)) for k, v in V.items()})

bottom_w = (f['m'].shape[0] - 1 - top_f)
Ys = np.arange(0, bottom_w, K); Xs = np.arange(-420, 420, K); Zs = np.arange(-330, 330, K)
nY, nX, nZ = len(Ys), len(Xs), len(Zs)

def rows(v, Y):
    return np.clip((v['hipv'] + (Y - HIPW) / v['s']).round().astype(int), 0, v['m'].shape[0] - 1)
def known(v, Y):
    r = v['hipv'] + (Y - HIPW) / v['s']; return (r >= 0) & (r <= v['m'].shape[0] - 3)
def sample_mask(v, Y, U):
    yy = rows(v, Y)[:, None]; uu = (v['cx'] + U[None, :] / v['s']).round().astype(int)
    ok = (uu >= 0) & (uu < v['m'].shape[1])
    m = v['m'][yy, np.clip(uu, 0, v['m'].shape[1] - 1)] & ok
    return m | ~known(v, Y)[:, None]   # rows outside the photo do not constrain the shape
Mx = sample_mask(V['front'], Ys, Xs) & sample_mask(V['back'], Ys, -Xs)
# side views disagree slightly (phone perspective), so average their extents instead of intersecting
Lz = sample_mask(V['left'], Ys, -Zs); Rz = sample_mask(V['right'], Ys, Zs)
kl, kr = known(V['left'], Ys), known(V['right'], Ys)
Mz = np.zeros_like(Lz)
def ext(row):
    w = np.where(row)[0]
    if len(w) < 2: return None
    lab, n = nd.label(row); sizes = np.bincount(lab)[1:]; big = np.where(sizes >= max(3, 0.15 * sizes.max()))[0] + 1
    w = np.where(np.isin(lab, big))[0]; return w[0], w[-1]
for yi in range(len(Ys)):
    e = [ext(m[yi]) for m, k in ((Lz, kl[yi]), (Rz, kr[yi])) if k]
    e = [x for x in e if x]
    if e: Mz[yi, int(round(np.mean([x[0] for x in e]))):int(round(np.mean([x[1] for x in e]))) + 1] = True
kz = known(V['left'], Ys) | known(V['right'], Ys)
last = np.where(kz)[0].max()
Mz[last + 1:] = Mz[last]            # below the side photos: continue the last seen leg depth
Mx = nd.binary_opening(Mx, iterations=1); Mz = nd.binary_opening(Mz, iterations=1)

def intervals(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))

vol = np.full((nY, nX, nZ), -3, np.float32)
zi = np.arange(nZ)[None, :]
for yi in range(nY):
    xiv = [iv for iv in intervals(Mx[yi]) if iv[1] - iv[0] >= 1]
    ziv = [iv for iv in intervals(Mz[yi]) if iv[1] - iv[0] >= 1]
    if not xiv or not ziv: continue
    widest = max(b - a for a, b in xiv)
    for a, b in xiv:
        xc, hx = (a + b) / 2, (b - a) / 2 + 0.5
        # z-interval overlapping best: use all z extent for torso/legs, circular for narrow limbs (arms)
        z0, z1 = min(p for p, q in ziv), max(q for p, q in ziv)
        zc, hz = (z0 + z1) / 2, (z1 - z0) / 2 + 0.5
        if len(xiv) >= 2 and (b - a) < 0.45 * widest:
            hz = min(hz, hx * 1.05)
        xs = np.arange(a, b + 1)[:, None]
        e = np.abs((xs - xc) / hx) ** PEXP + np.abs((zi - zc) / hz) ** PEXP
        d = (1 - e) * min(hx, hz) / PEXP
        vol[yi, a:b + 1] = np.maximum(vol[yi, a:b + 1], np.clip(d, -3, 3))
vol[0] = vol[-1] = -3; vol[:, 0] = vol[:, -1] = -3; vol[:, :, 0] = vol[:, :, -1] = -3
vol = nd.gaussian_filter(vol, 0.8)
verts, faces, _, _ = measure.marching_cubes(vol, 0.0, allow_degenerate=False)
Y = Ys[0] + verts[:, 0] * K; X = Xs[0] + verts[:, 1] * K; Z = Zs[0] + verts[:, 2] * K
mesh = trimesh.Trimesh(np.stack([X, -Y, Z], 1), faces[:, ::-1], process=True)
mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
trimesh.smoothing.filter_taubin(mesh, iterations=10)
N = mesh.vertex_normals; X, Y, Z = mesh.vertices[:, 0], -mesh.vertices[:, 1], mesh.vertices[:, 2]

def sample(v, Yw, U):
    yy = v['hipv'] + (Yw - HIPW) / v['s']; uu = v['cx'] + U / v['s']
    h, w = v['m'].shape
    ok = (yy >= 0) & (yy < h - 1) & (uu >= 0) & (uu < w - 1)
    yi = np.clip(yy.astype(int), 0, h - 2); ui = np.clip(uu.astype(int), 0, w - 2)
    fy, fx = (yy - yi)[:, None], (uu - ui)[:, None]; im = v['img']
    col = (im[yi, ui] * (1 - fx) * (1 - fy) + im[yi, ui + 1] * fx * (1 - fy) + im[yi + 1, ui] * (1 - fx) * fy + im[yi + 1, ui + 1] * fx * fy)
    inside = ok & nd.binary_erosion(v['m'], iterations=3)[yi, ui]
    return col, inside
acc = np.zeros((len(X), 3)); wsum = np.zeros(len(X))
for key, U, nd_ in [('front', X, N[:, 2]), ('back', -X, -N[:, 2]), ('left', -Z, N[:, 0]), ('right', Z, -N[:, 0])]:
    col, inside = sample(V[key], Y, U)
    w = np.clip(nd_, 0, 1) ** 3 * inside + 1e-4 * inside
    acc += col * w[:, None]; wsum += w
colors = np.clip(acc / np.maximum(wsum, 1e-6)[:, None], 0, 255)
nocol = wsum < 1e-5
if nocol.any():
    from scipy.spatial import cKDTree
    _, idx = cKDTree(mesh.vertices[~nocol]).query(mesh.vertices[nocol]); colors[nocol] = colors[~nocol][idx]
# scale: ankle level ~ 0.07 m above floor; head top..ankle maps to HEIGHT_M - 0.07
ank = f['L'][[27, 28], 1].mean() - top_f
mpp = (HEIGHT_M - 0.07) / ank
mesh.vertices *= mpp
mesh.vertices[:, 1] -= mesh.vertices[:, 1].max() - HEIGHT_M
mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.c_[colors.astype(np.uint8), np.full(len(colors), 255, np.uint8)])
mesh.export(OUT + '/body.glb'); mesh.export(OUT + '/body.ply')
json.dump({'mpp': mpp, 'top_f': int(top_f), 'height_m': HEIGHT_M}, open(OUT + '/body_meta.json', 'w'))
print('verts', len(mesh.vertices), 'faces', len(mesh.faces), 'extent m', mesh.extents.round(3))
