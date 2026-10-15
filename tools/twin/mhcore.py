"""Minimal MakeHuman (CC0 assets) engine: base mesh + targets, macro modifiers, measure modifiers."""
import numpy as np, os, functools
D = os.path.join(os.path.dirname(__file__), 'mh/makehuman/data/')

def load_obj():
    V, VT, F, FT, G = [], [], [], [], []
    g = None
    for line in open(D + '3dobjs/base.obj'):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('vt '): VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith('g '): g = line.split()[1]
        elif line.startswith('f '):
            p = [t.split('/') for t in line.split()[1:]]
            F.append([int(a[0]) - 1 for a in p]); FT.append([int(a[1]) - 1 for a in p]); G.append(g)
    return np.array(V), np.array(VT), F, FT, np.array(G)

V0, VT, FQ, FTQ, FG = load_obj()
GROUPS = sorted(set(FG))

def tris(groups):
    f, ft = [], []
    for q, qt, g in zip(FQ, FTQ, FG):
        if g in groups:
            for k in range(1, len(q) - 1):
                f.append([q[0], q[k], q[k + 1]]); ft.append([qt[0], qt[k], qt[k + 1]])
    return np.array(f), np.array(ft)

def group_verts(g):
    s = set()
    for q, gg in zip(FQ, FG):
        if gg == g: s.update(q)
    return np.array(sorted(s))

@functools.lru_cache(None)
def target(path):
    idx, d = [], []
    for line in open(D + 'targets/' + path):
        if line.startswith('#') or not line.strip(): continue
        a = line.split(); idx.append(int(a[0])); d.append([float(a[1]), float(a[2]), float(a[3])])
    return np.array(idx, dtype=int), np.array(d, dtype=float).reshape(-1, 3)

def add(Vt, path, w):
    if abs(w) < 1e-6: return
    i, d = target(path); Vt[i] += w * d

def tri_w(v):  # value in [0,1] -> weights for (min, average, max)
    return (max(0, 1 - 2 * v), 1 - abs(2 * v - 1), max(0, 2 * v - 1))

def build(p):
    """p: dict with muscle, weight, height (0..1, 0.5 neutral) and measure-* values in [-1,1]."""
    V = V0.copy()
    mw = dict(zip(('minmuscle', 'averagemuscle', 'maxmuscle'), tri_w(p.get('muscle', .5))))
    ww = dict(zip(('minweight', 'averageweight', 'maxweight'), tri_w(p.get('weight', .5))))
    for e in ('african', 'asian', 'caucasian'):
        add(V, f'macrodetails/{e}-male-young.target', 1 / 3)
    for m, a in mw.items():
        for w, b in ww.items():
            add(V, f'macrodetails/universal-male-young-{m}-{w}.target', a * b)
            h = p.get('height', .5)
            if h > .5: add(V, f'macrodetails/height/male-young-{m}-{w}-maxheight.target', a * b * (h - .5) * 2)
            else: add(V, f'macrodetails/height/male-young-{m}-{w}-minheight.target', a * b * (.5 - h) * 2)
    for k, v in p.items():
        if k.startswith('measure-'):
            add(V, f'measure/{k}-{"incr" if v > 0 else "decr"}.target', abs(v))
    return V

def joint(V, name):
    return V[group_verts('joint-' + name)].mean(0)
