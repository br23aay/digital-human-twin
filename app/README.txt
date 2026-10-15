Digital Human Studio (Stage 1) - local copy

1. Double-click start.bat (needs Python installed; it runs "python -m http.server 8000").
2. Your browser opens http://localhost:8000
3. Drop the photos from ..\Pictures\Bharadwaj onto the page.
   Files named like "Front Head" / "Left Side Body" go to the right slot.
4. Tick consent, open Generate, click "Build my digital human".

Everything runs in the browser. Photos are not uploaded anywhere.
The mp folder holds the MediaPipe face and pose models (Apache 2.0).

Output\face_v1 has your face as a textured 3D mesh (face.obj + face_texture.jpg).
It opens in Blender, or File > Import in Unreal Engine.
tools\facegen.py rebuilds it from any front photo:
  pip install mediapipe pillow numpy
  python facegen.py "Front Head.jpeg" face_landmarker.task canonical_face_model.obj outdir
