"""Linear-blend skinning for the MakeHuman default skeleton (CC0)."""
import json, numpy as np, os
from scipy.spatial.transform import Rotation as Rot
D = os.path.join(os.path.dirname(__file__), 'mh/makehuman/data/rigs/')
SK = json.load(open(D + 'default.mhskel'))
WT = json.load(open(D + 'default_weights.mhw'))['weights']
BONES = SK['bones']
ORDER = []
def _visit(b):
    if b in ORDER: return
    p = BONES[b]['parent']
    if p: _visit(p)
    ORDER.append(b)
for b in BONES: _visit(b)

def joints(V):
    return {k: V[np.array(v)].mean(0) for k, v in SK['joints'].items()}

def heads_tails(V):
    J = joints(V)
    return {b: (J[BONES[b]['head']], J[BONES[b]['tail']]) for b in BONES}

def skin_matrix(nv):
    W = np.zeros((nv, len(ORDER)), np.float32)
    for bi, b in enumerate(ORDER):
        for i, w in WT.get(b, []):
            if i < nv: W[i, bi] = w
    s = W.sum(1, keepdims=True); s[s == 0] = 1
    return W / s
_W = None

def pose(V, rots):
    """rots: {bone: rotvec (world axes, radians)} applied about the bone head in rest space."""
    global _W
    if _W is None or _W.shape[0] != len(V): _W = skin_matrix(len(V))
    HT = heads_tails(V)
    M = {}
    for b in ORDER:
        h = HT[b][0]
        L = np.eye(4)
        if b in rots:
            R = Rot.from_rotvec(rots[b]).as_matrix()
            L[:3, :3] = R; L[:3, 3] = h - R @ h
        p = BONES[b]['parent']
        M[b] = (M[p] @ L) if p else L
    Ms = np.stack([M[b] for b in ORDER])                     # (B,4,4)
    Vh = np.c_[V, np.ones(len(V))]
    out = np.einsum('vb,bij,vj->vi', _W, Ms, Vh)[:, :3]
    return out, {b: (M[b] @ np.r_[HT[b][0], 1])[:3] for b in ORDER}, M
