"""v3 checks against the reference-photo audit (run: python -m pytest tests -q)."""
import json, os, numpy as np, trimesh, pytest
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(ROOT, 'Output', 'digital_twin_v3'); V2 = os.path.join(ROOT, 'Output', 'digital_twin_v2')
@pytest.fixture(scope='module')
def scene(): return trimesh.load(os.path.join(ROOT, 'app', 'models', 'digital_twin_v3.glb'))
@pytest.fixture(scope='module')
def a3(): return json.load(open(os.path.join(V3, 'face_audit.json')))
@pytest.fixture(scope='module')
def a2(): return json.load(open(os.path.join(V2, 'face_audit_v2.json')))

def test_structure_height_floor(scene):
    assert set(scene.geometry) == {'body', 'head', 'eyes'}
    b = scene.bounds; assert abs(b[1][1] - b[0][1] - 1.75) < 0.01 and abs(b[0][1]) < 0.01

def test_textures_not_blank():
    for f in ('head_texture.jpg', 'body_texture.jpg'):
        assert np.asarray(Image.open(os.path.join(V3, f))).mean() > 20, f + ' is (nearly) black'

def test_single_connected_surface_and_closed_neck():
    m = trimesh.load(os.path.join(V3, 'digital_twin_v3_shape.ply'), process=True)
    assert len(m.split(only_watertight=False)) == 1
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True); y = m.vertices[u[c == 1]].mean(1)[:, 1]
    top = m.bounds[1][1]; assert not np.any((y > top - 0.35) & (y < top - 0.18))

def test_no_shading_seam_between_head_and_body(scene):
    h, b = scene.geometry['head'], scene.geometry['body']
    from scipy.spatial import cKDTree
    d, i = cKDTree(b.vertices).query(h.vertices)
    seam = d < 1e-7
    assert seam.sum() > 20
    dots = np.einsum('ij,ij->i', h.vertex_normals[seam], b.vertex_normals[i[seam]])
    assert dots.min() > 0.999, f'normals differ across the neck seam (min dot {dots.min():.4f})'

def test_eyes_anatomical_size_and_spacing(scene):
    parts = scene.geometry['eyes'].split(only_watertight=False); assert len(parts) == 2
    for p in parts:
        r = np.linalg.norm(p.vertices - p.vertices.mean(0), axis=1).mean(); assert 0.0115 < r < 0.0125
    cx = sorted(p.centroid[0] for p in parts); assert 0.05 < cx[1] - cx[0] < 0.075

def test_eye_openings_match_photo_better_than_v2(a3, a2):
    assert a3['eye_opening_iou'] >= 0.75 and a3['eye_opening_iou'] > a2['eye_opening_iou']
    for s in ('L', 'R'):
        e = a3['eye_opening'][s]; assert abs(e['model_w'] / e['photo_w'] - 1) < 0.05

def test_front_landmarks_not_worse_than_v2(a3, a2):
    for k in ('eye_L', 'eye_R', 'nose', 'lips', 'jaw_oval'):
        assert a3['front_residual_px'][k] <= a2['front_residual_px'][k] + 0.5, k

def test_side_profile_closer_to_photos_than_v2(a3, a2):
    for side in ('Left Head', 'Right Head'):
        assert a3['profile'][side]['mae_mm'] < a2['profile'][side]['mae_mm'], side
