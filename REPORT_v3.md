# Digital human v3: audit, fixes and validation (14 Oct 2026)

v3 builds on v2. It uses the same pipeline and assets, and adds targeted corrections measured against the 8 reference photographs. v1 and v2 assets are unchanged. The viewer can switch between v3, v2 and v1.

## 1. Audit findings (from the code and assets)

| Question | Finding |
|---|---|
| How was v1 made? | `tools/seg.py`, `recon_head.py`, `recon_body.py`, `merge.py`. Silhouettes from 4 views were carved into separate head and body meshes, with per-vertex colours from the photos. |
| How was v2 made? | `tools/twin/fit_body.py` and `build_twin.py`. A **fitted template**: the MakeHuman base mesh (CC0), shaped by silhouette-fitted body targets, posed with its own skeleton, face morphed to 453 MediaPipe landmarks, textures baked from the photos. No v1 geometry is used. |
| Head / neck / body | One connected surface (13,380 vertices), exported as two meshes (body and head) only so the head can have its own texture. They meet at a UV seam on the neck. |
| Eyes | Separate meshes. v2 used procedural 48×48 spheres (MakeHuman eye radius, about 15 mm) with fixed iris and sclera colours. |
| Eyelids | Modelled by the MakeHuman base mesh. Only moved by the landmark fit (thin-plate interpolation, smoothing 0.5). Never compared with the photographed eye opening. |
| Reference photos | Available (8 × 1200×1600). Plus segmentation masks, 478 face landmarks (front) and pose landmarks (body). |
| Camera alignment | Front head: least-squares affine from the 453 landmarks. Side and back head: scale from top-of-hair to chin. Body: landmark-based height mapping. |

### Measured v2 defects (new audit tool `tools/twin/face_audit.py`)

| Defect | Cause found |
|---|---|
| **Eyes look wide and staring** | Visible eyeball vs the photographed eye opening overlapped by only **0.60** (IoU). The white showed above the photographed upper lid and was missing at the lower lid, i.e. the opening sat too high. The eyeball was larger than anatomical (about 15 mm radius) with an over-bright procedural sclera, a grey-blue iris (colour sampling picked up reflections), and a clearcoat highlight. |
| **Eyebrow area distorted** | Brow landmarks (hair, not skin) were used as shape targets: 13.7 px residual. |
| **Profile wrong** | Nose too prominent relative to the face: compared with the side photos, the glabella sat 12 mm behind and the lips and chin 4–13 mm behind, with the nose tip matched. MediaPipe depth comes from one front photo, and the side photos were never used for shape. |
| **Visible neck line** | Head and body were exported with separately computed normals. At the seam the normals disagreed (minimum dot product **−0.27**), which drew a shading line across the neck. |
| **Colour seams between views** | The photos have different exposures, and nothing normalised them. |

## 2. Changes made in v3

All in `tools/twin/build_twin_v3.py` and `tools/twin/facefit.py`.

1. **Face landmark fit**
   - Eyebrow landmarks are no longer shape targets (they are still used for camera alignment).
   - A second, tighter interpolation pass corrects what the first one leaves behind.
   - Mean 3D residual: 2.36 → **1.50 mm**.
2. **Side-profile correction from the Left/Right Head photos**
   - Each side photo is aligned to the model's midline over the rigid upper face, with the nose tip pinned. Fitted scale is 1.00, and pitch is 0° (left photo) and −6.3° (right photo).
   - The depth difference is applied only at heights where **both side photos agree within 8 mm** (93% of heights).
   - Correction is smoothed, limited to ±15 mm and fades out sideways.
   - **The lips are excluded**: there the side-photo outline is moustache and beard, not lip surface.
3. **Eyes**
   - Each eyeball is placed so its iris centre matches the photographed iris, with the corneal apex 1.2 mm in front of the lid-margin plane.
   - Radius is 12 mm. That is the adult average, because the photos cannot measure it; the visible eye size comes from the lids.
   - Iris angular size comes from the photographed iris radius (30°). Iris and sclera colours are sampled from the photographed eyes.
   - The lid region is deformed (small 2D affine, regularised) to maximise overlap with the photographed eye opening polygon.
   - Lid vertices that cut into the eyeball are pushed out to its surface plus 0.3 mm.
4. **Neck integration**: head and body now share one set of normals, computed on the welded surface and exported explicitly. The minimum normal dot product at the seam is now **1.000**.
5. **Photometric normalisation**: each photo's skin median is matched to the front head photo. The back-of-head photo shows no skin, so it is matched through hair colour instead.

## 3. Results against the reference photographs (same audit for v2 and v3)

| Metric | v2 | v3 |
|---|---|---|
| Eye opening IoU (visible eyeball vs photographed opening) | 0.598 | **0.803** |
| Eye width, model / photo (px), L & R | 93.8/97.5, 96.1/101.6 | **97.1/97.5, 100.6/101.6** |
| Eye height, model / photo (px), L & R | 32.0/32.3, 31.7/32.2 | 35.2/32.3, 36.2/32.2 (**worse, about +1 mm**) |
| Front landmark error (px): eyes L/R | 7.8 / 7.4 | **5.8 / 4.4** |
| Front landmark error (px): nose / lips / jaw | 4.5 / 6.7 / 6.8 | **3.7 / 3.4 / 4.6** |
| Front landmark error (px): brows | 13.7 | 12.3 (geometry intentionally not fitted to brows) |
| Side profile mean error, left photo | 7.1 mm | **5.5 mm** |
| Side profile mean error, right photo | 8.2 mm | **7.3 mm** |
| Neck seam normal agreement (min dot) | −0.27 | **1.000** |

Overlays are in `Output/digital_twin_v3/audit_*.png` (v2 versions: `Output/digital_twin_v2/*_v2.png`).

The profile error uses nose-anchored rigid alignment. The correction uses a related alignment, so this metric is not fully independent of the fix. The lips and chin remain 3–10 mm behind the photo outline. That gap is partly the beard and moustache, which are not modelled as geometry.

## 4. Body validation

See `Output/digital_twin_v3/body_validation.json`.

- **Overall outline**: silhouette overlap with the photos is front 0.825, back 0.727, left 0.805, right 0.675. This is unchanged from v2; the body was intentionally not altered.
- **Lengths cannot be validated from these photos.** The same ratio measured in the front and back photos disagrees by up to 30% (forearm 0.46 vs 0.39; hip width 0.38 vs 0.26). The lower legs measure 0.51 of the shoulder-to-hip span in the photos against about 0.84 for an average adult. That is strong foreshortening from a close, downward-tilted camera.
- **No proportions were changed**, because the photos do not support a specific change. Height is still the assumed 1.75 m.

## 5. Files

New:
- `tools/twin/build_twin_v3.py`, `facefit.py`, `face_audit.py`, `body_validate.py`, `compare_render.html`
- `Output/digital_twin_v3/*`
- `Output/digital_twin_v2/*_v2.*` (audit of v2)
- `Output/comparisons/compare_{body,face}_{photo,clay}.jpg`
- `app/models/digital_twin_v3.glb`
- `tests/test_twin_v3.py`
- this report

Modified:
- `app/my-twin.html`: adds v3 (now the default) and keeps v2 and v1.
- `tests/viewer.test.js`: now also tests v3.

Unchanged: all v1 and v2 assets and `build_twin.py`.

## 6. Tests

- `python -m pytest tests -q`: **17 passed**. That is the 9 existing v2 tests plus 8 new v3 tests:
  - structure and height;
  - textures not blank (added after a NaN bug in the back-of-head gain made a first v3 build render black; that bug was found and fixed);
  - one connected surface with a closed neck;
  - no shading seam;
  - eye size and spacing;
  - eye opening better than v2 with widths within 5% of the photo;
  - front landmarks not worse than v2;
  - side profile closer than v2.
- `node tests/viewer.test.js` (Playwright, software WebGL): **55 checks, ALL PASSED** (`tests/viewer_test_log.txt`). It covers v3, v2 and v1 at 1400×900 and 420×860, in both views and all 3 render modes (framing and visibility), plus turntable, zoom and no page errors.

## 7. Before / after

`Output/comparisons/`: v1, v2 and v3 with identical cameras (fov 30°; 0°, 35°, 90° and 180°), lights, background, scale and render settings, each textured and with a neutral material, for both full body and face close-up.

## 8. Known limitations

- **Eye openings are about 1 mm taller than in the photo**, and some white still shows above the iris in the front view.
- **Lips and chin sit 3–10 mm behind the side-photo outline**, partly beard and moustache. Lip shape comes from front landmarks only.
- **Hair is painted on the scalp.** Volume and the length at the back are not modelled, and a vertical colour seam remains at the back of the head.
- **Ears are the template's ears**, coloured from the side photos. Their position and shape were not fitted; the photos give no ear landmarks.
- **The neck still shows the shirt collar and beard texture**, with uneven shading, though there is no longer a geometric or normal seam.
- **Body lengths and height are not measured** (see section 4). Fingers, toes and the soles of the feet come from the template.
- **Clothes are painted on.**

## 9. Photos needed for further accuracy

1. **Your height plus a few tape measurements**: shoulder width, chest, waist, hips, inside-leg, arm length. This lets me fix the body proportions straight away.
2. **Body photos from 3–4 m away**, with the camera at hip height, a 2× zoom, the phone held level, and feet in frame. This removes the foreshortening.
3. **Face**: 8–12 photos at eye level, 1 m away, in soft even light:
   - front, plus 30°, 60° and 90° to each side;
   - a slightly raised and a slightly lowered head;
   - hair tied back so the ears and hairline are visible;
   - a neutral expression with lips closed.
   This lets the face be fitted from several views instead of one.
4. **Ideally, a walk-around video** (photogrammetry) for exact hair and clothing.
