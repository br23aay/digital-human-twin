"""v3 (refines v2). Build the digital twin: fitted MakeHuman body + face fitted to 478 landmarks + skin texture baked from the 8 photos."""
import sys, json, numpy as np, trimesh
from PIL import Image
from scipy import ndimage as nd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
SC = sys.argv[1]; PHOTOS = sys.argv[2]; OUT = sys.argv[3]
HEIGHT_M = float(sys.argv[4]) if len(sys.argv) > 4 else 1.75
TEX = 2048
sys.argv = [sys.argv[0], SC]; sys.path.insert(0, SC)
import mhcore as M, mhpose as P
import fit_body as FB
import facefit as FF
REPORT = {}

params = json.load(open(SC + 'recon/body_params.json'))
V = M.build(params)

# ---------------- face shape from 478 landmarks ----------------
corr = json.load(open(SC + 'recon/mh_face_corr.json'))['vid']
lmj = json.load(open(SC + 'out/landmarks.json')); lm = np.array(lmj['landmarks'])
FW, FH = 1200, 1600
Ph = np.c_[lm[:, 0] * FW, -lm[:, 1] * FH, -lm[:, 2] * FW][:468]
vid = np.array(corr[:468])
uniq, inv = np.unique(vid, return_inverse=True)
Phu = np.array([Ph[inv == i].mean(0) for i in range(len(uniq))])
# v3: eyebrow landmarks sit on brow hair, not on skin geometry -> they are used for the camera, not the shape fit
lm_of_uniq = [set(np.where(vid == u)[0]) for u in uniq]
shape_cp = np.array([not ids <= FF.BROWS for ids in lm_of_uniq])
Q = V[uniq]
def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A)); d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d]); R = U @ Dm @ Vt; s = np.trace(np.diag(S) @ Dm) / A0.var(0).sum()
    return s, R, mb - s * R @ ma
s, R, t = umeyama(Phu, Q)
Dsp = (s * (R @ Phu.T).T + t) - Q
Dsp[:, 2] *= 0.75
print('face fit: mean landmark change (mm)', np.round(np.linalg.norm(Dsp, axis=1).mean() * 100 * (HEIGHT_M / 1.73), 1))
TGT = Q + Dsp                                   # target positions of the control vertices
neck_y = M.joint(V, 'neck')[1]
cand = np.where(V[:, 1] > neck_y - 0.6)[0]
Qs = Q[shape_cp]
dist, _ = cKDTree(Qs).query(V[cand])
w = np.clip(1 - (dist - 0.12) / 0.9, 0, 1); w = w * w * (3 - 2 * w)
rbf = RBFInterpolator(Qs, Dsp[shape_cp], kernel='thin_plate_spline', smoothing=0.5)
V[cand] += w[:, None] * rbf(V[cand])
# second, tighter pass on the remaining residual (face only)
res2 = TGT[shape_cp] - V[uniq[shape_cp]]
REPORT['landmark_residual_before_pass2_mm'] = round(float(np.linalg.norm(res2, axis=1).mean() * 100 * (HEIGHT_M / 1.73)), 2)
rbf2 = RBFInterpolator(V[uniq[shape_cp]], res2, kernel='thin_plate_spline', smoothing=0.03)
w2 = np.clip(1 - (dist - 0.05) / 0.5, 0, 1); w2 = w2 * w2 * (3 - 2 * w2)
V[cand] += w2[:, None] * rbf2(V[cand])
REPORT['landmark_residual_after_pass2_mm'] = round(float(np.linalg.norm(TGT[shape_cp] - V[uniq[shape_cp]], axis=1).mean() * 100 * (HEIGHT_M / 1.73)), 2)
print('3D landmark residual mm: before/after pass 2', REPORT['landmark_residual_before_pass2_mm'], REPORT['landmark_residual_after_pass2_mm'])

# v3: side-profile depth correction from the Left/Right Head photos (midline: forehead, nose, lips, chin)
A0 = FF.affine(V, uniq, Phu)
head_idx = np.where(V[:, 1] > neck_y - 0.2)[0]
ys, dz, prof_rep, _ = FF.profile_correction(SC, V, head_idx, A0, lm, corr)
FF.apply_profile(V, head_idx, ys, dz)
REPORT['profile_fit'] = prof_rep
print('profile correction (mm) by height:', prof_rep['dz_mm_samples'])

# v3: eyes - anatomical eyeballs at the photographed irises, lid openings fitted to the photographed eye contours
A0 = FF.affine(V, uniq, Phu)
k0 = np.linalg.norm(A0[:3, 0])
FBF, _ = M.tris(['body'])
scale_est = HEIGHT_M / (V[np.unique(FBF), 1].max() - V[np.unique(FBF), 1].min())
R_EYE = 0.012 / scale_est                           # 12 mm adult eyeball radius (not measurable from photos)
front_img = np.asarray(Image.open(PHOTOS + 'Front Head.jpeg').convert('RGB')).astype(np.float32)
EYES = []
for side in ('L', 'R'):
    ph = FF.photo_eye(lm, front_img, side)
    rim = np.array([corr[i] for i in FF.EYE_CONTOUR[side]])
    z_rim = V[rim, 2].mean()
    cz = z_rim + 0.012 - R_EYE                      # corneal apex ~1.2 mm in front of the lid-margin plane
    cxy = FF.unproject_xy(A0, ph['iris_c'][0], ph['iris_c'][1], cz + R_EYE)
    centre = np.array([cxy[0], cxy[1], cz])
    near = np.where(np.linalg.norm(V - centre, axis=1) < 0.45)[0]
    near = near[V[near, 2] > cz - 0.3 * R_EYE]
    loc = np.where(np.linalg.norm(V - centre, axis=1) < 0.75)[0]; locset = np.zeros(len(V), bool); locset[loc] = True
    faces_local = FBF[locset[FBF].all(1)]
    V, (iou0, iou1, xp) = FF.fit_eye(V, FBF, A0, ph, centre, R_EYE, faces_local, near)
    # lids must not cut into the eyeball: push any lid vertex inside the ball out to its surface (+0.3 mm)
    dv = V[near] - centre; dn = np.linalg.norm(dv, axis=1); inside = (dn < R_EYE + 0.003) & (V[near, 2] > cz)
    V[near[inside]] = centre + dv[inside] / dn[inside, None] * (R_EYE + 0.003)
    iris_ang = float(np.arcsin(min(0.9, (ph['iris_r'] / k0) / R_EYE)))
    EYES.append(dict(centre=centre, r=R_EYE, iris_ang=iris_ang, iris=ph['iris_rgb'], sclera=ph['sclera_rgb']))
    REPORT[f'eye_{side}'] = {'opening_iou_before': round(float(iou0), 3), 'opening_iou_after': round(float(iou1), 3),
                             'lid_fit_params': [round(float(v), 3) for v in xp], 'iris_angle_deg': round(np.degrees(iris_ang), 1),
                             'pushed_out_lid_vertices': int(inside.sum())}
    print('eye', side, REPORT[f'eye_{side}'])

# ---------------- pose like the photos ----------------
HT = P.heads_tails(V); rots = {}
for side in ('L', 'R'):
    ua, fa, th = FB.PD[side]
    sh, el, wr = HT[f'upperarm01.{side}'][0], HT[f'lowerarm01.{side}'][0], HT[f'wrist.{side}'][0]
    r1 = FB.angle_to(el - sh, ua); rots[f'upperarm01.{side}'] = r1
    rots[f'lowerarm01.{side}'] = FB.angle_to(wr - el, Rot.from_rotvec(r1).as_matrix().T @ fa)
    hp, kn = HT[f'upperleg01.{side}'][0], HT[f'lowerleg01.{side}'][0]
    rots[f'upperleg01.{side}'] = FB.angle_to(kn - hp, th)
Vp, heads, _ = P.pose(V, rots)
mk = FB.model_keys(Vp, heads)

FB_, FTB = M.tris(['body'])
FE, FTE = M.tris(['helper-l-eye', 'helper-r-eye'])
body = trimesh.Trimesh(Vp, FB_, process=False)
VN = body.vertex_normals

# ---------------- cameras ----------------
imgs = {n: np.asarray(Image.open(PHOTOS + n + '.jpeg').convert('RGB')).astype(np.float32)
        for n in ['Front Head', 'Left Head', 'Right Head', 'Back Head', 'Front Body', 'Back Side Body', 'Left Side Body', 'Right SIde Body']}
segs = {n: np.load(SC + 'seg/' + n + '.npy') for n in imgs}
# v3: photometric normalisation - match each photo's median skin colour to the front head photo (reduces seams)
def skin_median(n):
    m = (segs[n] == 2) | (segs[n] == 3); return np.median(imgs[n][m], 0)
ref_skin = skin_median('Front Head'); GAINS = {}
def hair_median(n): return np.median(imgs[n][segs[n] == 1], 0)
for n in imgs:
    sm = skin_median(n) if ((segs[n] == 2) | (segs[n] == 3)).sum() > 500 else None
    if sm is not None: g = ref_skin / np.maximum(sm, 1)
    else:   # back of head shows no skin: match its hair colour to the side photos' hair (already skin-matched)
        side = np.mean([hair_median(x) * (ref_skin / np.maximum(skin_median(x), 1)) for x in ('Left Head', 'Right Head')], 0)
        g = side / np.maximum(hair_median(n), 1)
    g = np.clip(np.nan_to_num(g, nan=1.0), 0.75, 1.35); GAINS[n] = [round(float(x), 3) for x in g]
for n in imgs: imgs[n] = np.clip(imgs[n] * np.array(GAINS[n]), 0, 255)
REPORT['photo_gains'] = GAINS
# front head: affine 3D -> 2D from landmark correspondences
X3 = np.c_[Vp[uniq], np.ones(len(uniq))]
img2 = np.c_[Phu[:, 0], -Phu[:, 1]]
A_front, *_ = np.linalg.lstsq(X3, img2, rcond=None)
k_front = np.linalg.norm(A_front[:3, 0])
def cam_front_head(P3): return (np.c_[P3, np.ones(len(P3))] @ A_front)  # (u,v)
REPROJ = float(np.linalg.norm(cam_front_head(Vp[uniq]) - img2, axis=1).mean())
print('face landmark reprojection error (px in 1200x1600 photo):', round(REPROJ, 2))
head_top_y = Vp[np.unique(FB_), 1].max(); chin_y = Vp[corr[152], 1]; nose = Vp[corr[1]]
def head_profile_cam(name, sign):
    seg = segs[name]; m = seg != 0; top = np.where(m.any(1))[0][0]
    fy = np.where((seg == 3).sum(1) > 15)[0]; chin_row = fy[-1]
    # chin row in photo corresponds to the lowest face-skin row (beard/jaw); model: chin landmark
    k = (chin_row - top) / (head_top_y - chin_y)
    nose_row = int(top + (head_top_y - nose[1]) * k)
    band = (seg[max(0, nose_row - 12):nose_row + 12] == 3) | (seg[max(0, nose_row - 12):nose_row + 12] == 2)
    cols = np.where(band.any(0))[0]
    u_nose = cols.min() if sign < 0 else cols.max()
    def cam(P3):
        u = u_nose + sign * (P3[:, 2] - nose[2]) * k
        v = top + (head_top_y - P3[:, 1]) * k
        return np.c_[u, v]
    return cam
cam_left_head = head_profile_cam('Left Head', -1)   # camera on his left (+x): face points image-left
cam_right_head = head_profile_cam('Right Head', +1)
def cam_back_head_make():
    seg = segs['Back Head']; m = seg != 0; top = np.where(m.any(1))[0][0]
    k = k_front / 1.023 * 1.0
    row = int(top + 0.3 * (head_top_y - chin_y) * k); xs = np.where(m[row])[0]; cx = (xs[0] + xs[-1]) / 2
    def cam(P3): return np.c_[cx - P3[:, 0] * k, top + (head_top_y - P3[:, 1]) * k]
    return cam
cam_back_head = cam_back_head_make()

# body cameras from the silhouette fit (piecewise vertical mapping, best horizontal shift)
S3 = FB.surface_points(Vp); SIL_IOU = {}
def body_cam(view):
    ph = FB.PH[view]; m = ph['m']
    sil = FB.silhouette(S3, view, mk)
    vr = np.zeros(m.shape[0], bool); vr[:int(ph['keys']['ank']) - 2] = True
    best, bsh = -1, 0
    for shv in range(-12, 13):
        a = np.roll(sil, shv, axis=1)[vr]; iou = (a & m[vr]).sum() / max(1, (a | m[vr]).sum())
        if iou > best: best, bsh = iou, shv
    if view in ('left', 'right'):
        y0 = mk['sh'] + .3 * (mk['hip'] - mk['sh']); band = np.abs(S3[:, 1] - y0) < .15
        uu = (-S3[:, 2] if view == 'left' else S3[:, 2])[band]; c = (np.percentile(uu, 1) + np.percentile(uu, 99)) / 2
    else: c = 0.0
    def cam(P3):
        rows = FB.to_photo_rows(P3[:, 1], mk, ph['keys']); sc = FB.local_scale(P3[:, 1], mk, ph['keys'])
        u = {'front': P3[:, 0], 'back': -P3[:, 0], 'left': -P3[:, 2], 'right': P3[:, 2]}[view]
        return np.c_[(ph['cx'] + (u - c) * sc + bsh) * FB.DS, rows * FB.DS]
    print(view, 'silhouette IoU', round(best, 3)); SIL_IOU[view] = round(float(best), 4)
    return cam
CAMS = [  # (photo, camera fn, view direction (towards camera), region)
    ('Front Head', cam_front_head, np.array([0, 0, 1.]), 'head'),
    ('Left Head', cam_left_head, np.array([1., 0, 0]), 'head'),
    ('Right Head', cam_right_head, np.array([-1., 0, 0]), 'head'),
    ('Back Head', cam_back_head, np.array([0, 0, -1.]), 'head'),
    ('Front Body', body_cam('front'), np.array([0, 0, 1.]), 'body'),
    ('Back Side Body', body_cam('back'), np.array([0, 0, -1.]), 'body'),
    ('Left Side Body', body_cam('left'), np.array([1., 0, 0]), 'body'),
    ('Right SIde Body', body_cam('right'), np.array([-1., 0, 0]), 'body'),
]

# visibility: orthographic depth maps along each view axis
def depth_map(d):
    ax = int(np.argmax(np.abs(d))); sgn = np.sign(d[ax]); o = [i for i in range(3) if i != ax]
    rng = np.random.default_rng(1)
    bw = rng.dirichlet([1, 1, 1], size=(len(FB_), 14))
    pts = np.einsum('fsk,fkd->fsd', bw, Vp[FB_]).reshape(-1, 3); res = 0.02
    lo = pts[:, o].min(0) - .1
    ij = ((pts[:, o] - lo) / res).astype(int); shp = ij.max(0) + 2
    D = np.full(shp, -np.inf); np.maximum.at(D, (ij[:, 0], ij[:, 1]), sgn * pts[:, ax])
    D = nd.maximum_filter(D, 3); Dm = np.where(np.isfinite(D), D, -1e3); D = np.where(np.isfinite(D), nd.gaussian_filter(Dm, 1.0), -np.inf)
    def visible(P3):
        q = ((P3[:, o] - lo) / res).astype(int); q = np.clip(q, 0, shp - 1)
        return np.clip((sgn * P3[:, ax] - D[q[:, 0], q[:, 1]] + 0.10) / 0.05, 0, 1)
    return visible
VIS = {tuple(c[2]): depth_map(c[2]) for c in CAMS}

def colour_points(P3, N3, head_w, vote_hook=None):
    samples = []
    front_face = np.clip(N3 @ np.array([0, 0, 1.]), 0, 1)
    for name, cam, d, region in CAMS:
        uv = cam(P3); img = imgs[name]; seg = segs[name]; h, w_ = seg.shape
        u, v = uv[:, 0], uv[:, 1]
        ok = (u >= 0) & (u < w_ - 1) & (v >= 0) & (v < h - 1)
        ui, vi = np.clip(u.astype(int), 0, w_ - 2), np.clip(v.astype(int), 0, h - 2)
        fx, fy = (u - ui)[:, None], (v - vi)[:, None]
        col = img[vi, ui] * (1 - fx) * (1 - fy) + img[vi, ui + 1] * fx * (1 - fy) + img[vi + 1, ui] * (1 - fx) * fy + img[vi + 1, ui + 1] * fx * fy
        person = nd.binary_erosion(seg != 0, iterations=3)[vi, ui]
        facing = np.clip(N3 @ d, 0, 1) ** 2.5
        rw = head_w if region == 'head' else (1 - head_w) * 0.999 + 0.001
        wt = facing * ok * person * VIS[tuple(d)](P3) * rw
        if name == 'Front Head': wt *= 3.0
        elif region == 'head': wt *= (1 - front_face) ** 2   # profiles/back only where the front photo is oblique
        cls = seg[np.clip(v.round().astype(int), 0, h - 1), np.clip(u.round().astype(int), 0, w_ - 1)]
        cls = np.where(cls == 3, 2, cls)                       # face skin and body skin count as one class
        samples.append((col, wt, cls))
    # semantic vote: keep only views that agree on the class (skin / hair / clothes) to stop colour bleeding
    votes = np.zeros((len(P3), 6))
    for col, wt, cls in samples: np.add.at(votes, (np.arange(len(P3)), cls), wt)
    win = vote_hook(votes) if vote_hook else votes.argmax(1)
    acc = np.zeros((len(P3), 3)); ws = np.zeros(len(P3))
    for col, wt, cls in samples:
        w2 = wt * (cls == win); acc += col * w2[:, None]; ws += w2
    return acc, ws

# ---------------- bake textures (body atlas + a dedicated high-resolution head atlas) ----------------
UV = M.VT
def pull_push(img, m):
    if min(m.shape) <= 4 or m.all(): return img
    w = m.astype(np.float32)
    c = (img * w[..., None]).reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2, 3).sum((1, 3))
    ww = w.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).sum((1, 3))
    small = np.where(ww[..., None] > 0, c / np.maximum(ww, 1e-6)[..., None], 0)
    small = pull_push(small, ww > 0)
    up = np.repeat(np.repeat(small, 2, 0), 2, 1)
    up = nd.uniform_filter(up, size=(3, 3, 1))
    return np.where(m[..., None], img, up)

def bake(F, FT, uvs, size, fname):
    posmap = np.zeros((size, size, 3), np.float32); nrmmap = np.zeros((size, size, 3), np.float32); used = np.zeros((size, size), bool)
    for f, ft in zip(F, FT):
        uv = uvs[ft] * (size - 1); uv[:, 1] = (size - 1) - uv[:, 1]
        x0, x1 = int(np.floor(uv[:, 0].min())), int(np.ceil(uv[:, 0].max())); y0, y1 = int(np.floor(uv[:, 1].min())), int(np.ceil(uv[:, 1].max()))
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        den = (uv[1, 1] - uv[2, 1]) * (uv[0, 0] - uv[2, 0]) + (uv[2, 0] - uv[1, 0]) * (uv[0, 1] - uv[2, 1])
        if abs(den) < 1e-12: continue
        a = ((uv[1, 1] - uv[2, 1]) * (gx - uv[2, 0]) + (uv[2, 0] - uv[1, 0]) * (gy - uv[2, 1])) / den
        b = ((uv[2, 1] - uv[0, 1]) * (gx - uv[2, 0]) + (uv[0, 0] - uv[2, 0]) * (gy - uv[2, 1])) / den
        c = 1 - a - b; ins = (a >= -0.02) & (b >= -0.02) & (c >= -0.02)
        gx, gy = gx[ins], gy[ins]; a, b, c = a[ins, None], b[ins, None], c[ins, None]
        ok = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
        gx, gy, a, b, c = gx[ok], gy[ok], a[ok], b[ok], c[ok]
        posmap[gy, gx] = a * Vp[f[0]] + b * Vp[f[1]] + c * Vp[f[2]]
        nrmmap[gy, gx] = a * VN[f[0]] + b * VN[f[1]] + c * VN[f[2]]
        used[gy, gx] = True
    ys, xs = np.where(used)
    P3 = posmap[ys, xs]; N3 = nrmmap[ys, xs]; N3 /= np.linalg.norm(N3, axis=1, keepdims=True) + 1e-9
    head_w = np.clip((P3[:, 1] - (mk['sh'] + 0.25)) / 0.35, 0, 1)
    def smooth_votes(votes):   # blur class votes in texture space so skin/hair/cloth borders are clean, not speckled
        vm = np.zeros((size, size, votes.shape[1]), np.float32); vm[ys, xs] = votes
        vm = nd.gaussian_filter(vm, (2.5, 2.5, 0)); return vm[ys, xs].argmax(1)
    acc, ws = colour_points(P3, N3, head_w, smooth_votes)
    tex = np.zeros((size, size, 3), np.float32); have = np.zeros((size, size), bool)
    good = ws > 1e-3
    tex[ys[good], xs[good]] = acc[good] / ws[good, None]; have[ys[good], xs[good]] = True
    print(fname, 'texels observed', round(good.mean() * 100, 1), '%')
    tex = pull_push(tex, have)
    Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)).save(OUT + '/' + fname, quality=93)
    return good.mean()

# UV islands (faces connected through shared UV vertices); the head island is the one holding the nose tip
parent = np.arange(len(UV))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
for ft in FTB:
    for k in (1, 2):
        a, b = find(ft[0]), find(ft[k])
        if a != b: parent[a] = b
roots = np.array([find(ft[0]) for ft in FTB])
nose_face = np.where((FB_ == corr[1]).any(1))[0][0]
is_head = roots == roots[nose_face]
FH_, FTH = FB_[is_head], FTB[is_head]
FBo, FTBo = FB_[~is_head], FTB[~is_head]
hu = np.unique(FTH); lo, hi = UV[hu].min(0), UV[hu].max(0); span = (hi - lo).max() * 1.02
UVH = UV.copy(); UVH[hu] = (UV[hu] - lo) / span + (1 - (hi - lo) / span) / 2
cov_head = bake(FH_, FTH, UVH, 2048, 'head_texture.jpg')
cov_body = bake(FBo, FTBo, UV, 2048, 'body_texture.jpg')

# eyes: procedural sclera / iris / pupil; iris colour sampled at the iris landmarks of the front photo
ev = np.unique(FE); img = imgs['Front Head']
eye_col = np.zeros((len(Vp), 4), np.uint8); eye_col[:, 3] = 255
irs = []
for i in range(468, 478):
    x, y = int(lm[i, 0] * FW), int(lm[i, 1] * FH); irs.append(img[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3))
iris_rgb = np.percentile(np.concatenate(irs), 12, axis=0)
eye_meshes = []; EYE_INFO = []
for E in EYES:
    c = E['centre'] + (Vp[corr[1]] - V[corr[1]])          # head is not posed; keep the same offset (zero) for safety
    sph = trimesh.creation.uv_sphere(radius=E['r'], count=[64, 64])
    P_ = sph.vertices; ang = np.arccos(np.clip(P_[:, 2] / E['r'], -1, 1))
    ia = E['iris_ang']; pa = ia * 0.38
    t_pupil = np.clip((ang - pa) / 0.03, 0, 1)[:, None]
    t_iris = np.clip((ang - ia) / 0.035, 0, 1)[:, None]
    limbus = np.exp(-((ang - ia) / 0.05) ** 2)[:, None]
    iris_c = E['iris'] * (1 - 0.35 * limbus)
    col = (E['iris'] * 0.15) * (1 - t_pupil) + iris_c * t_pupil
    col = col * (1 - t_iris) + E['sclera'] * t_iris
    sph.vertices = P_ + c
    sph.visual = trimesh.visual.ColorVisuals(sph, vertex_colors=np.c_[np.clip(col, 0, 255).astype(np.uint8), np.full(len(P_), 255, np.uint8)])
    eye_meshes.append(sph); EYE_INFO.append((c, E['r']))

# ---------------- export (metres, feet on the floor) ----------------
scale = HEIGHT_M / (head_top_y - Vp[np.unique(FB_), 1].min())
floor = Vp[np.unique(FB_), 1].min()
def to_m(X): return (X - [0, floor, 0]) * scale
VN_WELD = trimesh.Trimesh(Vp, FB_, process=False).vertex_normals   # one set of normals for head+body -> no shading seam at the neck
def textured(F, FT, uvs, texfile, rough):
    key = np.stack([F.ravel(), FT.ravel()], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    m = trimesh.Trimesh(to_m(Vp[uk[:, 0]]), inv.reshape(-1, 3), vertex_normals=VN_WELD[uk[:, 0]], process=False)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(OUT + '/' + texfile), roughnessFactor=rough, metallicFactor=0.0)
    m.visual = trimesh.visual.TextureVisuals(uv=uvs[uk[:, 1]], material=mat)
    return m
bodym = textured(FBo, FTBo, UV, 'body_texture.jpg', 0.8)
headm = textured(FH_, FTH, UVH, 'head_texture.jpg', 0.62)
eyem = trimesh.util.concatenate(eye_meshes); eyem.vertices = to_m(eyem.vertices)
scn = trimesh.Scene()
scn.add_geometry(bodym, node_name='body', geom_name='body'); scn.add_geometry(headm, node_name='head', geom_name='head'); scn.add_geometry(eyem, node_name='eyes', geom_name='eyes')
from trimesh.exchange.gltf import export_glb
open(OUT + '/digital_twin_v3.glb', 'wb').write(export_glb(scn, include_normals=True))
# connected-surface mesh (welded, no UV seams) for tests and shape-only rendering
trimesh.Trimesh(to_m(Vp), FB_, process=True).export(OUT + '/digital_twin_v3_shape.ply')
json.dump({'height_m': HEIGHT_M, 'params': params, 'scale_m_per_unit': float(scale), 'texture_coverage': {'head': float(cov_head), 'body': float(cov_body)},
           'silhouette_iou': SIL_IOU, 'face_reprojection_px': REPROJ, 'face_landmarks_used': int(len(uniq)), 'v3': REPORT}, open(OUT + '/digital_twin_v3.json', 'w'), indent=1)
print('done height', round(float(to_m(Vp[np.unique(FB_)])[:, 1].max()), 3))

np.savez(OUT + '/state.npz', Vp=Vp, uniq=uniq, Phu=Phu, A_front=A_front, corr=np.array(corr), FB=FB_, eye_c=np.array([e[0] for e in EYE_INFO]), eye_r=np.array([e[1] for e in EYE_INFO]), scale=scale, floor=floor)
