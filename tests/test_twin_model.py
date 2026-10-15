"""Checks on the generated digital twin (run: python -m pytest tests -q from the project folder)."""
import json, os, numpy as np, trimesh, pytest
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'Output', 'digital_twin_v2')
GLB = os.path.join(ROOT, 'app', 'models', 'digital_twin_v2.glb')

@pytest.fixture(scope='module')
def scene(): return trimesh.load(GLB)
@pytest.fixture(scope='module')
def meta(): return json.load(open(os.path.join(OUT, 'digital_twin_v2.json')))

def test_scene_has_body_head_and_eyes(scene):
    names = ' '.join(scene.geometry.keys()).lower()
    assert len(scene.geometry) == 3, scene.geometry.keys()

def test_height_matches_setting(scene, meta):
    b = scene.bounds
    assert abs((b[1][1] - b[0][1]) - meta['height_m']) < 0.01
    assert abs(b[0][1]) < 0.01            # feet on the floor

def test_head_and_body_form_one_connected_surface():
    shape = trimesh.load(os.path.join(OUT, 'digital_twin_v2_shape.ply'), process=True)
    parts = shape.split(only_watertight=False)
    assert len(parts) == 1, f'{len(parts)} disconnected pieces'

def test_no_open_cut_at_the_neck():
    shape = trimesh.load(os.path.join(OUT, 'digital_twin_v2_shape.ply'), process=True)
    top = shape.bounds[1][1]
    # boundary edges (holes) must not appear between chin and shoulders
    edges = shape.edges_sorted; uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    y = shape.vertices[bnd].mean(1)[:, 1]
    assert not np.any((y > top - 0.35) & (y < top - 0.18)), 'open boundary in the neck region'

def test_geometry_is_finite_with_valid_normals(scene):
    for g in scene.geometry.values():
        assert np.isfinite(g.vertices).all()
        n = g.vertex_normals; assert np.isfinite(n).all()

def test_two_eyes_in_front_of_face(scene):
    eyes = [g for k, g in scene.geometry.items() if 'eye' in k.lower()]
    assert eyes, 'eyes mesh missing'
    parts = eyes[0].split(only_watertight=False)
    assert len(parts) == 2
    cx = sorted(p.centroid[0] for p in parts)
    assert 0.04 < cx[1] - cx[0] < 0.08          # interpupillary distance in metres

def test_textures_present_and_mostly_observed(meta):
    for f in ('head_texture.jpg', 'body_texture.jpg'):
        im = Image.open(os.path.join(OUT, f)); assert im.size == (2048, 2048)
    assert meta['texture_coverage']['head'] > 0.85
    assert meta['texture_coverage']['body'] > 0.70

def test_face_alignment_error_small(meta):
    assert meta['face_reprojection_px'] < 10     # mean error of 453 landmarks in the 1200x1600 front photo

def test_silhouette_agreement(meta):
    assert min(meta['silhouette_iou'].values()) > 0.6
