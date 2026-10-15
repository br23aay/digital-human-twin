# Embodied Personal Digital Twin

A research build of a **provenance-aware, photoreal digital twin of a real person** that becomes an **embodied agent** in a virtual environment. The project has two tracks that will eventually meet: a **digital-human reconstruction pipeline** (the body and face) and an **embodied-agent console** (perception, memory, reasoning and action).

> Status: active research build. The numbers below are measured against reference photographs; the known limitations are stated honestly rather than hidden.

---

## Why

Most "digital human" demos optimise for a flattering render. The harder, more useful problem is a twin that is **measurably faithful to a real person** and that **never presents a fabricated memory as fact** — a prerequisite for any assistant that acts on your behalf.

---

## Track 1 — Digital-human reconstruction (current version: v3)

Reconstructs a single connected head-and-body mesh (13,380 vertices) from **8 reference photographs**, built on the MakeHuman CC0 base template, shaped by silhouette-fitted body targets and **453 MediaPipe face landmarks**, with textures baked from the photos.

**v3 corrections, measured against the reference photos:**

| Metric | v2 | v3 |
|---|---|---|
| Eye-opening IoU (visible eyeball vs photographed opening) | 0.60 | **0.80** |
| Front landmark error — nose / lips / jaw (px) | 4.5 / 6.7 / 6.8 | **3.7 / 3.4 / 4.6** |
| Side-profile mean error — left / right photo | 7.1 / 8.2 mm | **5.5 / 7.3 mm** |
| Neck-seam normal agreement (min dot product) | −0.27 | **1.000** (seam removed) |

Key engineering in v3: a second tighter landmark-fit pass; side-profile depth correction applied only where both side photos agree within 8 mm; anatomically-placed eyeballs with lid-region deformation to match the photographed eye opening; welded normals across the neck seam; and per-photo photometric normalisation.

**Validation:** `pytest` — **17 passed** (9 v2 + 8 new v3 checks: structure, non-blank textures, closed neck, no shading seam, eye size/spacing, landmark and profile improvements). Viewer regression — **55 Playwright checks passed** across v1/v2/v3 at two resolutions and three render modes.

An interactive viewer (`app/my-twin.html`) switches between v1, v2 and v3 for side-by-side comparison.

---

## Track 2 — Embodied-agent console (v1 prototype)

A single-page Three.js vertical slice: one digital person in a small flat, four tracked objects, two household relationships. A **9-stage pipeline** — Perceive → Understand → Remember → Reason → Plan → Move → Interact → Verify → Communicate.

- **Provenance-aware memory:** every fact is tagged **Verified / User-provided / Inferred / Reconstructed / Unknown**. Memory-only questions answer with their provenance and **refuse to fabricate**.
- **Belief-driven search:** benchmarks "Where are my keys?" and "Daddy, can you get me the book?" — the agent searches last-seen → usual place → sweep, updates beliefs from perception, and **replans mid-route** when an object is moved out of sight.

---

## Architecture

```
reconstruction/
  tools/seg.py, recon_head.py, recon_body.py, merge.py   # v1 silhouette carving
  tools/twin/fit_body.py, build_twin.py                  # v2 fitted template
  tools/twin/build_twin_v3.py, facefit.py, face_audit.py, body_validate.py  # v3
console/
  app/my-twin.html                                       # Three.js embodied-agent prototype
tests/  test_twin_v3.py, viewer.test.js                  # pytest + Playwright
```

---

## Roadmap

- **Body accuracy:** replace assumed proportions with real tape measurements + photos taken from 3–4 m (removes camera foreshortening).
- **Face:** fit from 8–12 multi-angle photos instead of one front view; model ears from side landmarks.
- **Hair & clothing:** move from painted-on textures to modelled geometry (photogrammetry walk-around).
- **Console:** replace keyword intent parsing with an LLM + tool-calling; swap the stylised avatar for the rigged v3 mesh; real speech input; a persistent memory store with the provenance schema.
- **Convergence:** drive the embodied console with the reconstructed v3 digital human.

---

## Honest limitations (current)

- Eye openings are ~1 mm taller than the photo; some sclera still shows above the iris in front view.
- Lips/chin sit 3–10 mm behind the side-photo outline (partly beard/moustache, which aren't modelled as geometry).
- Hair is painted on the scalp; ears are the template's ears; body lengths and height are **not** measured (height is an assumed 1.75 m).
- The console runs on **sample data**; intent parsing is keyword-based and voice output uses browser TTS as a stand-in.

---

## Author

**Bharadwaj Rachuri** — ML & AI Engineer · [br23aay.github.io](https://br23aay.github.io) · [github.com/br23aay](https://github.com/br23aay)

*Independent research project, built and run locally.*
