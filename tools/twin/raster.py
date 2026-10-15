import numpy as np

def render(V, F, res, x0, x1, y0, y1, colors=None, light=(0.3, 0.4, 1.0)):
    """Orthographic render looking down -z (camera at +z). Returns rgb, zbuf, face-id buffer."""
    W = res; H = int(round(res * (y1 - y0) / (x1 - x0)))
    sx = (V[:, 0] - x0) / (x1 - x0) * (W - 1); sy = (y1 - V[:, 1]) / (y1 - y0) * (H - 1); sz = V[:, 2]
    zb = np.full((H, W), -np.inf); fid = np.full((H, W), -1, int)
    T = V[F]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    l = np.array(light, float); l /= np.linalg.norm(l)
    shade = np.clip(n @ l, 0, 1) * .75 + .25
    for i, (a, b, c) in enumerate(F):
        xs, ys, zs = sx[[a, b, c]], sy[[a, b, c]], sz[[a, b, c]]
        if n[i, 2] <= 0: continue
        xa, xb = int(max(0, np.floor(xs.min()))), int(min(W - 1, np.ceil(xs.max())))
        ya, yb = int(max(0, np.floor(ys.min()))), int(min(H - 1, np.ceil(ys.max())))
        if xa > xb or ya > yb: continue
        gx, gy = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-12: continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        w2 = 1 - w0 - w1
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        gx, gy, z = gx[ins], gy[ins], z[ins]
        upd = z > zb[gy, gx]
        zb[gy[upd], gx[upd]] = z[upd]; fid[gy[upd], gx[upd]] = i
    base = np.ones((len(F), 3)) * 0.8 if colors is None else colors
    rgb = np.zeros((H, W, 3))
    ok = fid >= 0
    rgb[ok] = base[fid[ok]] * shade[fid[ok], None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), zb, fid, (sx, sy)
