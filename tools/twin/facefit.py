"""v3 face refinement against the reference photographs:
   - side-profile (midline) depth correction from the Left/Right Head photos
   - eye openings fitted to the photographed eye contours, anatomical eyeballs placed at the photographed irises
All measurements come from the photos; nothing is set from generic proportions except the eyeball radius
(adult average 12 mm, which the photos cannot measure)."""
import numpy as np, json
from scipy import ndimage as nd, optimize
from scipy.interpolate import interp1d
from PIL import Image, ImageDraw
import raster

W, H = 1200, 1600
EYE_CONTOUR = {'L': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
               'R': [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]}
BROWS = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276}

def affine(V, uniq, Phu):
    X3 = np.c_[V[uniq], np.ones(len(uniq))]; img2 = np.c_[Phu[:, 0], -Phu[:, 1]]
    A, *_ = np.linalg.lstsq(X3, img2, rcond=None); return A
def proj(A, P3): return np.c_[P3, np.ones(len(P3))] @ A

# ---------------- side profile ----------------
def photo_profile(SC, name, sign, nose_f, top_f):
    seg = np.load(SC + 'seg/' + name + '.npy'); top_p = np.where((seg != 0).any(1))[0][0]
    skin = nd.binary_opening((seg == 2) | (seg == 3), iterations=2)
    best = None
    for r in range(top_p + int(.3 * (nose_f - top_f)), top_p + int(1.6 * (nose_f - top_f))):
        cs = np.where(skin[r])[0]
        if len(cs):
            c = cs.max() if sign > 0 else cs.min()
            if best is None or sign * c > sign * best[1]: best = (r, c)
    rows = np.arange(top_p, min(seg.shape[0], top_p + int(2.6 * (best[0] - top_p))))
    cols = np.array([(np.where(skin[r])[0].max() if sign > 0 else np.where(skin[r])[0].min()) if skin[r].any() else np.nan for r in rows])
    return dict(top=top_p, nose=best, rows=rows, cols=cols, sign=sign)

def midline_curve(V, head_idx, step=0.02):
    P = V[head_idx]; zc = np.median(P[:, 2]); sel = (np.abs(P[:, 0]) < 0.07) & (P[:, 2] > zc)   # front midline only
    y, z = P[sel, 1], P[sel, 2]
    yb = np.arange(y.min(), y.max(), step); zb = np.full(len(yb), np.nan)
    idx = ((y - y.min()) / step).astype(int)
    for i in range(len(yb)):
        m = idx == i
        if m.any(): zb[i] = z[m].max()
    ok = ~np.isnan(zb)
    yb, zb = yb[ok], zb[ok]
    # drop isolated dips (bins whose only midline vertex lies inside a fold); keep the frontmost envelope
    zs = nd.maximum_filter1d(zb, 3)
    return yb, zs

def fit_side(prof, yb, zb, y_nose, z_nose, y_glab, y_subn, a0):
    """similarity (scale a, pitch th, translation) mapping photo contour -> model profile plane, fitted on the
    rigid upper face (glabella..subnasale) with the nose tip pinned."""
    sgn = prof['sign']; rows, cols = prof['rows'], prof['cols']; ok = ~np.isnan(cols)
    p = np.c_[sgn * cols[ok], -rows[ok]]                    # forward, up
    pn = np.array([sgn * prof['nose'][1], -prof['nose'][0]])
    zf = interp1d(yb, zb, bounds_error=False)
    def xf(x):
        a, th, ty, tz = x; c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        q = (a * (R @ (p - pn).T)).T + [z_nose + tz, y_nose + ty]   # (z, y)
        return q
    def loss(x):
        q = xf(x); band = (q[:, 1] < y_glab) & (q[:, 1] > y_subn)
        e = q[band, 0] - zf(q[band, 1]); e = e[~np.isnan(e)]
        return np.mean(e ** 2) + 50 * (x[2] ** 2 + x[3] ** 2) + 2 * ((x[0] - a0) / a0) ** 2
    res = optimize.minimize(loss, [a0, 0.0, 0.0, 0.0], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-8})
    q = xf(res.x)
    return q, res.x

def profile_correction(SC, V, head_idx, A, lm, corr, verbose=True, max_dz=0.15):
    """returns dz(y) (model units) = photo profile depth - model profile depth, averaged over both side photos"""
    seg_front = np.load(SC + 'seg/Front Head.npy'); top_f = np.where((seg_front != 0).any(1))[0][0]
    nose_f = lm[1, 1] * H
    yb, zb = midline_curve(V, head_idx)
    y_of = lambda i: V[corr[i], 1]
    y_nose, z_nose = y_of(1), np.interp(y_of(1), yb, zb)
    y_glab, y_subn, y_chin = y_of(168), y_of(2), y_of(152)
    k = np.linalg.norm(A[:3, 0])
    curves, report = [], {}
    for name, sign in (('Left Head', -1), ('Right Head', +1)):
        prof = photo_profile(SC, name, sign, nose_f, top_f)
        s = (nose_f - top_f) / (prof['nose'][0] - prof['top'])
        q, x = fit_side(prof, yb, zb, y_nose, z_nose, y_glab, y_subn, a0=s / k)
        order = np.argsort(q[:, 1]); qy, qz = q[order, 1], q[order, 0]
        sel = (qy > y_chin - 0.02) & (qy < y_glab + 0.15)
        zp = interp1d(qy[sel], qz[sel], bounds_error=False)
        curves.append(zp); report[name] = {'scale': round(float(x[0] * k / s), 3), 'pitch_deg': round(float(np.degrees(x[1])), 1)}
    ys = np.arange(y_chin - 0.02, y_glab + 0.15, 0.01)
    zm = np.interp(ys, yb, zb)
    dzs = np.array([c(ys) - zm for c in curves])
    # use a height only where both side photos agree (within 8 mm); elsewhere leave the geometry alone
    agree = np.all(~np.isnan(dzs), 0) & (np.abs(dzs[0] - dzs[1]) < 0.08)
    dz = np.where(agree, dzs.mean(0), 0.0)
    dz = nd.gaussian_filter1d(dz, 3.0)
    report['agreeing_heights_fraction'] = round(float(agree.mean()), 2)
    # fade the correction out at the ends of the band and around the (pinned) nose tip it stays ~0 by construction
    fade = np.clip((ys - (y_chin - 0.02)) / 0.12, 0, 1) * np.clip(((y_glab + 0.15) - ys) / 0.2, 0, 1)
    # lips: the side-photo contour there is moustache/beard, not lip surface -> do not push the lips outwards
    y_ll = y_of(17)
    lipband = np.clip(np.minimum((ys - (y_ll - 0.06)) / 0.05, ((y_subn + 0.02) - ys) / 0.05), 0, 1)
    fade = fade * (1 - lipband)
    dz = np.clip(dz * fade, -max_dz, max_dz)
    report['dz_mm_samples'] = {f'{yy:.2f}': round(float(d) * 100, 1) for yy, d in zip(ys[::6], dz[::6])}
    return ys, dz, report, dzs

def apply_profile(V, idx, ys, dz, sigma_x=0.30):
    P = V[idx]; d = np.interp(P[:, 1], ys, dz, left=0, right=0)
    w = np.exp(-(P[:, 0] / sigma_x) ** 2)
    zc = np.median(P[:, 2]); wf = np.clip((P[:, 2] - zc) / 0.3, 0, 1)   # front half of the head only
    V[idx, 2] += d * w * wf

# ---------------- eyes ----------------
def photo_eye(lm, img, side):
    cont = np.array([[lm[i, 0] * W, lm[i, 1] * H] for i in EYE_CONTOUR[side]])
    # iris landmarks: 468-472 and 473-477; pick the set inside this contour
    sets = [list(range(468, 473)), list(range(473, 478))]
    cen = cont.mean(0)
    ir = min(sets, key=lambda st: np.linalg.norm(np.array([lm[st[0], 0] * W, lm[st[0], 1] * H]) - cen))
    ic = np.array([lm[ir[0], 0] * W, lm[ir[0], 1] * H])
    rr = np.mean([np.linalg.norm(np.array([lm[i, 0] * W, lm[i, 1] * H]) - ic) for i in ir[1:]])
    # colours: iris = inside iris circle (excluding pupil/specular extremes); sclera = inside contour, outside iris
    mimg = Image.new('L', (W, H), 0); ImageDraw.Draw(mimg).polygon([tuple(p) for p in cont], fill=255); m = np.array(mimg) > 0
    yy, xx = np.mgrid[0:H, 0:W]; d = np.hypot(xx - ic[0], yy - ic[1])
    iris_px = img[(d < rr * 0.9) & (d > rr * 0.4) & m]; scl_px = img[m & (d > rr * 1.15)]
    br = scl_px.sum(1); scl = np.median(scl_px[br >= np.percentile(br, 50)], 0) if len(scl_px) > 10 else np.array([215, 205, 195.])
    iris = np.median(iris_px, 0) if len(iris_px) else np.array([60, 40, 30.])
    return dict(contour=cont, iris_c=ic, iris_r=rr, iris_rgb=iris, sclera_rgb=scl)

def unproject_xy(A, u, v, z):
    # solve [x y] from A^T [x y z 1] = [u v]
    M = A[:2, :].T; rhs = np.array([u, v]) - A[2, :] * z - A[3, :]
    return np.linalg.solve(M, rhs)

def eye_render_mask(Vc_faces, nfb, x0, x1, y0, y1):
    pass

def fit_eye(V, FB, A, eye, center_guess, r_eye, faces_local, side_idx_verts):
    """Deform the lid region (vertices side_idx_verts) with a weighted 2D affine in the photo plane so the visible
    eyeball matches the photographed eye opening. Returns new V and eyeball centre."""
    cont = eye['contour']
    x0, x1 = cont[:, 0].min() - 25, cont[:, 0].max() + 25; y0, y1 = cont[:, 1].min() - 25, cont[:, 1].max() + 25
    res = int(x1 - x0)
    mimg = Image.new('L', (res, int(round(res * (y1 - y0) / (x1 - x0)))), 0)
    sc = (res - 1) / (x1 - x0)
    ImageDraw.Draw(mimg).polygon([((p[0] - x0) * sc, (p[1] - y0) * sc) for p in cont], fill=255)
    target = np.array(mimg) > 0
    import trimesh
    sph = trimesh.creation.uv_sphere(radius=r_eye, count=[24, 24])
    base = V.copy(); P0 = base[side_idx_verts]
    uv0 = proj(A, P0); c_uv = cont.mean(0)
    du = uv0 - c_uv; dist = np.hypot(du[:, 0] / (np.ptp(cont[:, 0]) / 2), du[:, 1] / (np.ptp(cont[:, 1]) / 2))
    wgt = np.clip(1.9 - dist, 0, 1) ** 2
    def deformed(x):
        tx, ty, sx, sy = x
        Vn = base.copy()
        nu = c_uv + du * [sx, sy] + [tx, ty]
        uvn = uv0 + (nu - uv0) * wgt[:, None]
        for j, vi in enumerate(side_idx_verts):
            if wgt[j] > 0: Vn[vi, :2] = unproject_xy(A, uvn[j, 0], uvn[j, 1], P0[j, 2])
        return Vn
    def iou(x, return_img=False):
        Vn = deformed(x)
        S = sph.vertices + center_guess
        Vall = np.r_[Vn, S]; Fall = np.r_[faces_local, sph.faces + len(Vn)]
        uv = proj(A, Vall); Vc = np.c_[uv[:, 0], -uv[:, 1], Vall[:, 2]]
        img, zb, fid, _ = raster.render(Vc, Fall, res, x0, x1, -y1, -y0)
        eyev = fid >= len(faces_local)
        hh = min(eyev.shape[0], target.shape[0]); a, b = eyev[:hh], target[:hh]
        v = (a & b).sum() / max(1, (a | b).sum())
        return (v, a, b) if return_img else v
    x0p = np.array([0., 0., 1., 1.])
    r = optimize.minimize(lambda x: -iou(x) + 0.002 * np.sum((x[:2]) ** 2) + 8.0 * np.sum((x[2:] - 1) ** 2), x0p, method='Powell',
                          options={'maxfev': 160, 'xtol': 0.02, 'ftol': 1e-3})
    v0 = iou(x0p); v1 = iou(r.x)
    return deformed(r.x), (v0, v1, r.x)
