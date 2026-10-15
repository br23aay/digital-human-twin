# Digital human v2: audit, changes and results (14 Oct 2026)

## 1. Audit of v1 (what existed)

| Item | Finding |
|---|---|
| Sources | 8 phone photos, 1200×1600, `Pictures/Bharadwaj/` (4 head views: front, left, right, back; 4 body views). Feet are cut off at the bottom of 3 of the 4 body photos. |
| Algorithm | `tools/seg.py` (MediaPipe multiclass segmentation) → `recon_head.py` and `recon_body.py` (silhouette visual hull + superellipse cross-sections, marching cubes, Taubin smoothing, vertex colours projected from the photos) → `merge.py`. |
| Head vs body | **Two separate meshes.** The head bust was built at about 1.2 mm voxels; the body at about 3.5 mm. Both were cut flat near the chin and overlapped, not fused. |
| UV / material | No UVs. Colour stored per vertex (about 1–2 mm spacing). Single default material. |
| Scale / alignment | Head scaled by top-of-head→chin measured on a small face in the body photo (factor 1.41), aligned by centroid. Body height fixed at 1.75 m. |
| Viewer | `app/my-twin.html`: three.js r128, ambient light plus one directional and one rim light, vertex-colour material, fixed camera positions. |

## 2. Root causes, most important first

1. **Method ceiling.** A visual hull from 4 outlines cannot recover surfaces between the views. The result was boxy, lumpy cross-sections (the "crushed paper" look), irregular shirt and trousers, and no fingers. Tuning cannot fix this.
2. **Two independently reconstructed meshes** at different resolutions, joined by cut-and-overlap. This caused the disconnected head and neck and the visible cut edges.
3. **Phone perspective.** The photos were taken close and from above, so the head looked about 40% too big and the legs too short. Head scaling copied that distortion.
4. **Texture by vertex colour from 4 differently lit photos**, with no awareness of what each pixel shows. Shirt, background and hair bled onto skin, and seams appeared at the view borders.
5. **Face depth from a single photo**, blended into the hull. This caused jagged cheeks and the jaw step.
6. **Basic lighting.** No environment, no tone mapping, no shadows.

## 3. What was changed (v2)

The method changed because cause 1 is architectural. Instead of carving a shape out of outlines, v2 **fits a clean, complete human body model to the photos**. The base is MakeHuman, released as CC0, which includes a skeleton and skin weights.

- `tools/twin/mhcore.py`: loads the MakeHuman base mesh and applies its shape targets (overall build, muscle, weight, height, and 22 measurement shapes).
- `tools/twin/mhpose.py`: poses the body with the skeleton and skin weights (linear blend skinning).
- `tools/twin/fit_body.py`: poses the arms and legs to match the photos (MediaPipe Pose angles). It then optimises build, muscle and 11 measurements (shoulders, bust, waist, hips, arms, thighs, calves, neck…) to maximise how well the model's outline overlaps the person's outline in all 4 body photos. The comparison uses landmark-based height mapping, which compensates for perspective. Overlap with the photo outline: front 0.825, back 0.727, left 0.804, right 0.675.
- `tools/twin/build_twin.py`:
  - **Face shape:** MediaPipe found 478 points on a render of the base head and on the front photo. The base face is morphed to match the person's points (similarity alignment, then a smooth interpolation that fades out towards the back of the head). Mean error after fitting: **6.2 px** in the 1200×1600 photo.
  - **One connected surface.** Head, neck and body are a single mesh with consistent resolution. The neck needs no cut or overlap. Tests confirm one connected piece and no open edges at the neck.
  - **Textures** are baked into proper UV layouts: 2048 px for the body, plus a separate 2048 px head texture, giving about 8× more texture detail on the face. Each colour comes from 8 projections, with three safeguards:
    - each photo is weighted by how directly the surface faces that camera, and only where that camera can actually see it;
    - photo pixels only count where all the views agree on what they show (skin, hair or clothes);
    - unobserved areas are filled smoothly instead of with blocks.
  - **Eyes:** new 48×48 eyeballs, with the iris colour taken from the iris points in the photo.
- `app/my-twin.html`:
  - Lighting: room-reflection environment, film-like tone mapping, soft key light with shadow, fill light and cyan rim light.
  - Materials: skin roughness 0.58 on the head and 0.72 on the body; glossy, coated eyes.
  - All views kept: Photo, Hologram, Shape only, Full body, Face, Turntable. Added a **v2 fitted / v1 scan** switch.
  - Camera framing is computed from the model's size and the window shape (works on narrow phone screens). Zoom is limited so the model is never clipped.
- **v1 files are preserved unchanged**; v2 is added alongside them.

## 4. Files

New:
- `tools/twin/*`
- `Output/digital_twin_v2/*` (glb, shape.ply, head and body textures, json)
- `app/models/digital_twin_v2.glb`
- `app/lib/RoomEnvironment.js`
- `tests/test_twin_model.py`, `tests/viewer.test.js`, `tests/viewer_test_log.txt`, `tests/shot_*.png`
- `Output/before_after_v1_v2.jpg`
- this report

Modified:
- `app/my-twin.html`

## 5. Tests (actually run)

- `python -m pytest tests -q`: **9 passed**. Checks: 3 meshes present; height 1.75 m with feet on the floor; one connected surface; no open boundary at the neck; finite geometry and normals; two eyes with 4–8 cm between centres; textures present, with 92.5% (head) and 78.1% (body) directly observed; face error under 10 px; outline overlap above 0.6.
- `node tests/viewer.test.js` (Playwright, software WebGL): **ALL PASSED**. At 1400×900 and 420×860, for both models, both views and all 3 render modes: the model is framed inside the viewport and visible. Turntable toggles, zoom works, no page errors.
- The first viewer run reported "0 lit cells". That was a test-harness problem: the canvas could not be read back. It was fixed with `preserveDrawingBuffer:true`, and the screenshots show the model rendered correctly.

## 6. Before / after

`Output/before_after_v1_v2.jpg`: same four camera angles (0°, −34°, −90°, 180°). The framing rule is identical, but the framing comes from each model's own size.

## 7. Still unresolved (needs manual inspection or better data)

- **Hair is painted on the scalp.** Its volume and the length at the back are not modelled.
- **Clothes are painted on.** The shirt and trousers have no thickness, collar or folds, and *Shape only* shows the bare base body.
- **Body lengths are the model's average adult proportions.** Height is assumed (1.75 m); arm, leg and torso lengths are not measured.
- **Fingers, toes, ears, teeth, and the back and underside of the body** come from the base model, coloured from the photos. They are not reconstructed details of this person.
- Eye whites look slightly too large and bright. The neck and collar shading is uneven, and some arm edges pick up dark shirt and trouser colour.
- Face depth is MediaPipe's estimate from one front photo.
- Not rigged for animation in the exported file, although the skeleton exists in the pipeline.

## 8. Are the current photos enough?

**Enough for a recognisable, clean v2. Not enough for an exact twin.**

To get an exact twin:

1. **Give real measurements** (height, shoulder width, chest, waist, hips, inside-leg, arm length). This fixes proportions with the current photos.
2. **Do a new capture session.** 80–150 photos or a slow video walk-around, standing still in an A-pose with fingers spread and feet in frame, in even light and fitted clothes. Add 20–30 close face photos at different angles. Process it with photogrammetry (for example RealityScan), then fit this pipeline (or MetaHuman) to the scan.
