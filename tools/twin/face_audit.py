"""Quantitative face audit of a built twin against the reference photographs.
usage: face_audit.py SC PHOTOS STATE_DIR  -> prints metrics, writes STATE_DIR/face_audit.json and overlay images."""
import sys, json, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
SC, PH, D = sys.argv[1:4]
sys.path.insert(0, SC)
import mhcore as M, raster
st = np.load(D + '/state.npz')
Vp, A, corr = st['Vp'], st['A_front'], st['corr']
FB = st['FB']; eye_c, eye_r = st['eye_c'], st['eye_r']
lm = np.array(json.load(open(SC + 'out/landmarks.json'))['landmarks']); W, H = 1200, 1600
photo = lambda i: np.array([lm[i, 0] * W, lm[i, 1] * H])
proj = lambda P3: np.c_[P3, np.ones(len(P3))] @ A

REG = {
 'eye_L': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
 'eye_R': [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
 'brows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
 'nose': [1, 2, 98, 327, 4, 5, 195, 197, 6, 168, 48, 278, 64, 294, 129, 358],
 'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
 'jaw_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]}
out = {'front_residual_px': {}}
for k, idx in REG.items():
    e = [np.linalg.norm(proj(Vp[[corr[i]]])[0] - photo(i)) for i in idx]
    out['front_residual_px'][k] = round(float(np.mean(e)), 2)

# eye opening: width (corner to corner) and height (mid upper lid to mid lower lid)
EYE = {'L': (33, 133, 159, 145), 'R': (263, 362, 386, 374)}
out['eye_opening'] = {}
for s, (a, b, u, l) in EYE.items():
    pw, phh = np.linalg.norm(photo(a) - photo(b)), np.linalg.norm(photo(u) - photo(l))
    mp_ = proj(Vp[[corr[a], corr[b], corr[u], corr[l]]])
    mw, mh = np.linalg.norm(mp_[0] - mp_[1]), np.linalg.norm(mp_[2] - mp_[3])
    out['eye_opening'][s] = {'photo_w': round(pw, 1), 'model_w': round(mw, 1), 'photo_h': round(phh, 1), 'model_h': round(mh, 1)}

# visible eyeball vs photo eye opening (render in the front-photo camera)
fb = FB
spheres = []
for c, r in zip(eye_c, eye_r):
    import trimesh
    sph = trimesh.creation.uv_sphere(radius=r, count=[32, 32]); sph.vertices += c; spheres.append(sph)
Vall = np.r_[Vp, spheres[0].vertices, spheres[1].vertices]
Fall = np.r_[fb, spheres[0].faces + len(Vp), spheres[1].faces + len(Vp) + len(spheres[0].vertices)]
uv = proj(Vall)
# camera space: x=u, y=-v (image down), z = model z (front-facing camera)
Vc = np.c_[uv[:, 0], -uv[:, 1], Vall[:, 2]]
x0, x1 = photo(33)[0] - 120, photo(263)[0] + 120; ycen = (photo(159)[1] + photo(386)[1]) / 2
y0, y1 = -(ycen + 90), -(ycen - 90)
nfb = len(fb)
col = np.r_[np.tile([[.8, .6, .5]], (nfb, 1)), np.tile([[1, 1, 1]], (len(Fall) - nfb, 1))]
img, zb, fid, _ = raster.render(Vc, Fall, int(x1 - x0), x0, x1, y0, y1, col)
eyepix = (fid >= nfb)
ph_img = Image.new('L', img.shape[1::-1], 0); d = ImageDraw.Draw(ph_img)
for s in ('eye_L', 'eye_R'):
    poly = [tuple(photo(i) - [x0, -y1]) for i in REG[s]]
    d.polygon(poly, fill=255)
openm = np.array(ph_img) > 0
inter = (eyepix & openm).sum(); uni = (eyepix | openm).sum()
out['eyeball_visible_px'] = int(eyepix.sum()); out['photo_eye_opening_px'] = int(openm.sum())
out['eye_opening_iou'] = round(float(inter / max(uni, 1)), 3)
vis = img.copy(); vis[openm & ~eyepix] = [255, 60, 60]; vis[eyepix & ~openm] = [60, 160, 255]; vis[eyepix & openm] = [240, 240, 240]
crop = np.asarray(Image.open(PH + 'Front Head.jpeg').convert('RGB'))[int(-y1):int(-y1) + vis.shape[0], int(x0):int(x0) + vis.shape[1]]
Image.fromarray(np.vstack([crop, vis])).save(D + '/audit_eyes.png')

# side profile: frontmost contour of the face, model vs Left/Right Head photos.
# Profile camera: vertical scale/offset from two points visible in both photos (top of hair, nose tip),
# horizontal offset from the nose tip. Everything else (forehead, lips, chin) is then a genuine comparison.
seg_front = np.load(SC + 'seg/Front Head.npy'); top_f = np.where((seg_front != 0).any(1))[0][0]
nose_f = photo(1)[1]
rows_lm = {k: photo(i)[1] for k, i in {'glabella': 168, 'nose_bridge': 6, 'nose_tip': 1, 'subnasale': 2, 'upper_lip': 0, 'lower_lip': 17, 'chin_front': 199, 'chin': 152}.items()}
k = np.linalg.norm(A[:3, 0])
uv_all = proj(Vp)
def profile_view(name, sign):
    seg = np.load(SC + 'seg/' + name + '.npy'); top_p = np.where((seg != 0).any(1))[0][0]
    skin = nd.binary_opening((seg == 2) | (seg == 3), iterations=2)
    # photo nose tip: frontmost skin point between 35% and 60% of the head height
    hh = np.where((seg != 0).any(1))[0]; band = range(top_p + int(.3 * (nose_f - top_f) * 1.0), top_p + int(1.6 * (nose_f - top_f)))
    best = None
    for r in band:
        cs = np.where(skin[r])[0]
        if len(cs):
            c = cs.max() if sign > 0 else cs.min()
            if best is None or sign * c > sign * best[1]: best = (r, c)
    nose_r, nose_c = best
    s = (nose_f - top_f) / (nose_r - top_p)          # front px per profile px
    rowp = top_p + (uv_all[:, 1] - top_f) / s
    colp = sign * Vp[:, 2] * k / s
    depth = -sign * Vp[:, 0]                           # camera on his right (sign +1) sees small x
    Vc = np.c_[colp, -rowp, depth]
    x0, x1 = colp.min() - 5, colp.max() + 5
    r0, r1 = int(top_p + (rows_lm['glabella'] - top_f) / s), int(top_p + (rows_lm['chin'] - top_f) / s) + 25
    keep = (rowp > r0 - 60) & (rowp < r1 + 60)
    fk = FB[keep[FB].all(1)]
    img, zb, fid, _ = raster.render(Vc, fk, int((x1 - x0) * 1), x0, x1, -(r1 + 60), -(r0 - 60))
    m = fid >= 0
    rows = np.arange(r0, r1); mf = np.full(len(rows), np.nan); pf = np.full(len(rows), np.nan)
    for j, r in enumerate(rows):
        rr = int(round(r - (r0 - 60))); 
        if 0 <= rr < m.shape[0]:
            cs = np.where(m[rr])[0]
            if len(cs): mf[j] = x0 + (cs.max() if sign > 0 else cs.min())
        cs = np.where(skin[r])[0]
        if len(cs): pf[j] = cs.max() if sign > 0 else cs.min()
    # anchor horizontally at the nose tip
    jn = nose_r - r0
    model_nose = np.nanmax(mf[max(0, jn - 8):jn + 8]) if sign > 0 else np.nanmin(mf[max(0, jn - 8):jn + 8])
    off = nose_c - model_nose
    diff = sign * ((mf + off) - pf)                    # + = model protrudes further than the photo
    mm = lambda px: px * (s / k) * float(st['scale']) * 1000
    ok = ~np.isnan(diff)
    res = {'mae_mm': round(float(mm(np.mean(np.abs(diff[ok])))), 2), 'scale_front_per_profile_px': round(float(s), 3)}
    for lmk, rf in rows_lm.items():
        jj = int(top_p + (rf - top_f) / s) - r0
        if 0 <= jj < len(rows) and ok[jj]: res[lmk + '_mm'] = round(float(mm(diff[jj])), 1)
    pim = np.asarray(Image.open(PH + name + '.jpeg').convert('RGB')).copy()
    for j, r in enumerate(rows):
        if not np.isnan(mf[j]):
            c = int(round(mf[j] + off)); pim[r, max(0, c - 1):c + 2] = [80, 200, 255]
    Image.fromarray(pim[max(0, r0 - 40):r1 + 40, max(0, nose_c - 330):nose_c + 120 if sign > 0 else nose_c + 330][:, :: 1 if sign > 0 else 1]).save(D + f'/audit_profile_{name.split()[0].lower()}.png')
    return res, (rows, mf + off, pf, s, top_p, nose_r, nose_c, sign)
out['profile'] = {}
PROFILE_DATA = {}
for name, sign in (('Left Head', -1), ('Right Head', +1)):
    out['profile'][name], PROFILE_DATA[name] = profile_view(name, sign)
json.dump(out, open(D + '/face_audit.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
