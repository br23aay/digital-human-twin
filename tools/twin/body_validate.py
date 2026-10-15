"""Body proportion check: ratios measured in the photos (MediaPipe Pose) vs the same ratios on the model.
All ratios are normalised by the shoulder->hip vertical span, so camera distance cancels; camera tilt does not."""
import sys, json, numpy as np
SC, D = sys.argv[1:3]; sys.argv = ['x', SC]; sys.path.insert(0, SC)
import mhcore as M, mhpose as P, fit_body as FB
st = np.load(D + '/state.npz'); Vp = st['Vp']
pose = json.load(open(SC + 'seg/pose.json'))
def photo_ratios(n):
    L = np.array(pose[n]); d = lambda a, b: np.hypot(*(L[a, :2] - L[b, :2]))
    span = L[[23, 24], 1].mean() - L[[11, 12], 1].mean()
    return {'shoulder_width': d(11, 12) / span, 'hip_width': d(23, 24) / span,
            'upper_arm': (d(11, 13) + d(12, 14)) / 2 / span, 'forearm': (d(13, 15) + d(14, 16)) / 2 / span,
            'thigh': (d(23, 25) + d(24, 26)) / 2 / span, 'shin': (d(25, 27) + d(26, 28)) / 2 / span}
# model joints (same anatomical points MediaPipe reports: shoulder, elbow, wrist, hip, knee, ankle)
J = P.joints(Vp); HT = P.heads_tails(Vp)
pt = {k: HT[b][0] for k, b in {'sh_l': 'upperarm01.L', 'sh_r': 'upperarm01.R', 'el_l': 'lowerarm01.L', 'el_r': 'lowerarm01.R',
      'wr_l': 'wrist.L', 'wr_r': 'wrist.R', 'hp_l': 'upperleg01.L', 'hp_r': 'upperleg01.R', 'kn_l': 'lowerleg01.L', 'kn_r': 'lowerleg01.R',
      'an_l': 'foot.L', 'an_r': 'foot.R'}.items()}
dd = lambda a, b: np.linalg.norm((pt[a] - pt[b])[:2])
span = (pt['sh_l'][1] + pt['sh_r'][1]) / 2 - (pt['hp_l'][1] + pt['hp_r'][1]) / 2
model = {'shoulder_width': dd('sh_l', 'sh_r') / span, 'hip_width': dd('hp_l', 'hp_r') / span,
         'upper_arm': (dd('sh_l', 'el_l') + dd('sh_r', 'el_r')) / 2 / span, 'forearm': (dd('el_l', 'wr_l') + dd('el_r', 'wr_r')) / 2 / span,
         'thigh': (dd('hp_l', 'kn_l') + dd('hp_r', 'kn_r')) / 2 / span, 'shin': (dd('kn_l', 'an_l') + dd('kn_r', 'an_r')) / 2 / span}
fr, bk = photo_ratios('Front Body'), photo_ratios('Back Side Body')
out = {k: {'photo_front': round(fr[k], 3), 'photo_back': round(bk[k], 3), 'model': round(model[k], 3),
           'model_vs_photo_%': round(100 * (model[k] / ((fr[k] + bk[k]) / 2) - 1), 1)} for k in model}
meta = json.load(open(D + '/' + [f for f in __import__('os').listdir(D) if f.endswith('.json') and f.startswith('digital_twin')][0]))
out['silhouette_iou'] = meta['silhouette_iou']; out['height_m_assumed'] = meta['height_m']
json.dump(out, open(D + '/body_validation.json', 'w'), indent=1); print(json.dumps(out, indent=1))
