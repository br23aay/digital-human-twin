"""Fit a clean MakeHuman body to the person's 4 body photos (front/back/left/right silhouettes)."""
import sys, json, numpy as np
from scipy import ndimage as nd, optimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, sys.argv[1])
import mhcore as M, mhpose as P
SC = sys.argv[1]
pose_lm = json.load(open(SC + 'seg/pose.json'))
VIEWS = {'front': 'Front Body', 'back': 'Back Side Body', 'left': 'Left Side Body', 'right': 'Right SIde Body'}
DS = 4   # photo downsample for fitting

def person_mask(n):
    c = np.load(SC + 'seg/' + n + '.npy'); m = nd.binary_fill_holes(nd.binary_opening(c != 0, iterations=2))
    L = np.array(pose_lm[n]); hx, hy = L[[23, 24], 0].mean(), L[[23, 24], 1].mean()
    lab, _ = nd.label(m); m = lab == lab[int(hy), int(hx)]
    return m[::DS, ::DS]

PH = {}
for k, n in VIEWS.items():
    L = np.array(pose_lm[n]) / [DS, DS, 1]
    m = person_mask(n)
    top = np.where(m.any(1))[0][0]
    keys = {'top': top, 'sh': L[[11, 12], 1].mean(), 'hip': L[[23, 24], 1].mean(),
            'knee': L[[25, 26], 1].mean(), 'ank': L[[27, 28], 1].mean()}
    PH[k] = dict(m=m, L=L, keys=keys, cx=L[[23, 24], 0].mean())
    if k in ('left', 'right'):
        row = int(keys['sh'] + .3 * (keys['hip'] - keys['sh']))
        lab, _ = nd.label(m[row][None]); seg = lab[0]; best = np.bincount(seg)[1:].argmax() + 1
        xs = np.where(seg == best)[0]; PH[k]['cx'] = (xs[0] + xs[-1]) / 2

def angle_to(v, t):
    v = v / np.linalg.norm(v); t = t / np.linalg.norm(t)
    ax = np.cross(v, t); s = np.linalg.norm(ax)
    if s < 1e-8: return np.zeros(3)
    return ax / s * np.arctan2(s, np.dot(v, t))

def photo_dirs():
    L = np.array(pose_lm['Front Body'])
    d = lambda a, b: np.array([L[b, 0] - L[a, 0], -(L[b, 1] - L[a, 1]), 0.0])
    S = np.array(pose_lm['Left Side Body'])   # forward bend of forearms from side (face points image-left => +z = -u)
    fz = lambda a, b: -(S[b, 0] - S[a, 0]) / max(1, abs(S[b, 1] - S[a, 1]))
    out = {}
    for side, (s, e, w, h, k) in {'L': (11, 13, 15, 23, 25), 'R': (12, 14, 16, 24, 26)}.items():
        ua = d(s, e); fa = d(e, w); th = d(h, k)
        fa[2] = np.clip(fz(13, 15), -.2, .5) * np.linalg.norm(fa[:2])
        out[side] = (ua, fa, th)
    return out
PD = photo_dirs()

def posed(params):
    V = M.build(params)
    HT = P.heads_tails(V)
    rots = {}
    for side in ('L', 'R'):
        ua, fa, th = PD[side]
        sh, el = HT[f'upperarm01.{side}'][0], HT[f'lowerarm01.{side}'][0]
        wr = HT[f'wrist.{side}'][0]
        r1 = angle_to(el - sh, ua); rots[f'upperarm01.{side}'] = r1
        R1 = Rot.from_rotvec(r1).as_matrix()
        rots[f'lowerarm01.{side}'] = angle_to(wr - el, R1.T @ fa)
        hp, kn = HT[f'upperleg01.{side}'][0], HT[f'lowerleg01.{side}'][0]
        rots[f'upperleg01.{side}'] = angle_to(kn - hp, th)
    Vp, heads, _ = P.pose(V, rots)
    return Vp, heads

BODY_F, _ = M.tris(['body']); BODY_V = np.unique(BODY_F)
# dense surface samples (barycentric) for solid silhouettes
rng = np.random.default_rng(0)
bary = rng.dirichlet([1, 1, 1], size=(len(BODY_F), 3))

def model_keys(Vp, heads):
    return {'top': Vp[BODY_V, 1].max(), 'sh': (heads['upperarm01.L'][1] + heads['upperarm01.R'][1]) / 2,
            'hip': (heads['upperleg01.L'][1] + heads['upperleg01.R'][1]) / 2,
            'knee': (heads['lowerleg01.L'][1] + heads['lowerleg01.R'][1]) / 2,
            'ank': (heads['foot.L'][1] + heads['foot.R'][1]) / 2}
ORD = ['top', 'sh', 'hip', 'knee', 'ank']

def to_photo_rows(y, mk, pk):
    my = np.array([mk[o] for o in ORD]); py = np.array([pk[o] for o in ORD])
    # model y decreases downward; piecewise linear (extrapolate with end segments)
    out = np.interp(-y, -my, py)
    lo = y > my[0]; out[lo] = py[0] + (y[lo] - my[0]) * (py[1] - py[0]) / (my[1] - my[0])
    hi = y < my[-1]; out[hi] = py[-1] + (y[hi] - my[-1]) * (py[-1] - py[-2]) / (my[-1] - my[-2])
    return out

def local_scale(y, mk, pk):
    my = np.array([mk[o] for o in ORD]); py = np.array([pk[o] for o in ORD])
    sc = (py[1:] - py[:-1]) / (my[:-1] - my[1:])
    seg = np.clip(np.searchsorted(-my, -y) - 1, 0, 3)
    return sc[seg]

def silhouette(P3, view, mk):
    ph = PH[view]; pk = ph['keys']; m = ph['m']
    rows = to_photo_rows(P3[:, 1], mk, pk); sc = local_scale(P3[:, 1], mk, pk)
    if view == 'front': u = P3[:, 0]
    elif view == 'back': u = -P3[:, 0]
    elif view == 'left': u = -P3[:, 2]
    else: u = P3[:, 2]
    if view in ('left', 'right'):  # centre model torso at chest row like the photo
        y0 = mk['sh'] + .3 * (mk['hip'] - mk['sh'])
        band = np.abs(P3[:, 1] - y0) < .15
        uu = u[band]; c = (np.percentile(uu, 1) + np.percentile(uu, 99)) / 2
    else:
        c = 0.0
    cols = ph['cx'] + (u - c) * sc
    img = np.zeros_like(m)
    ri, ci = rows.round().astype(int), cols.round().astype(int)
    ok = (ri >= 0) & (ri < m.shape[0]) & (ci >= 0) & (ci < m.shape[1])
    img[ri[ok], ci[ok]] = True
    img = nd.binary_closing(nd.binary_dilation(img, iterations=1), iterations=2)
    return nd.binary_fill_holes(img)

PARAMS = ['weight', 'muscle', 'measure-shoulder-dist', 'measure-bust-circ', 'measure-underbust-circ', 'measure-waist-circ',
          'measure-hips-circ', 'measure-upperarm-circ', 'measure-lowerarm-length', 'measure-thigh-circ', 'measure-calf-circ',
          'measure-neck-circ', 'measure-frontchest-dist']
def to_params(x):
    p = {k: float(v) for k, v in zip(PARAMS, x)}
    p['weight'] = .5 + .5 * np.clip(p['weight'], -1, 1); p['muscle'] = .5 + .5 * np.clip(p['muscle'], -1, 1)
    for k in PARAMS[2:]: p[k] = float(np.clip(p[k], -1, 1))
    return p

def surface_points(Vp):
    T = Vp[BODY_F]                                 # (F,3,3)
    return np.einsum('fsk,fkd->fsd', bary, T).reshape(-1, 3)

def loss(x, verbose=False):
    Vp, heads = posed(to_params(x)); mk = model_keys(Vp, heads)
    S3 = surface_points(Vp)
    tot = 0; rep = {}
    for view in VIEWS:
        ph = PH[view]; m = ph['m']
        # only rows the photo actually covers (feet are cut in 3 views)
        sil = silhouette(S3, view, mk)
        valid_rows = np.zeros(m.shape[0], bool); valid_rows[:min(m.shape[0], int(ph['keys']['ank']) - 2)] = True
        b = m[valid_rows]; best = 0
        for sh in range(-10, 11, 2):
            a = np.roll(sil, sh, axis=1)[valid_rows]
            iou = (a & b).sum() / max(1, (a | b).sum()); best = max(best, iou)
        rep[view] = round(best, 4); tot += 1 - best
    if verbose: print(rep)
    return tot + 0.02 * np.sum(np.square(x))

if __name__ == '__main__':
    x0 = np.zeros(len(PARAMS))
    print('start'); loss(x0, True)
    res = optimize.minimize(loss, x0, method='Powell', options={'maxfev': 450, 'xtol': 1e-2, 'ftol': 1e-4})
    print('end'); loss(res.x, True)
    p = to_params(res.x); print(json.dumps(p, indent=1))
    json.dump(p, open(SC + 'recon/body_params.json', 'w'))
    # debug overlay of silhouettes
    from PIL import Image
    Vp, heads = posed(p); mk = model_keys(Vp, heads); S3 = surface_points(Vp)
    tiles = []
    for view in VIEWS:
        m = PH[view]['m']; s = silhouette(S3, view, mk)
        rgb = np.zeros(m.shape + (3,), np.uint8); rgb[m] = [60, 140, 255]; rgb[s] = [255, 120, 60]; rgb[m & s] = [230, 230, 230]
        tiles.append(Image.fromarray(rgb))
    c = Image.new('RGB', (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles: c.paste(t, (x, 0)); x += t.width
    c.save(SC + 'recon/fit_overlay.png')
