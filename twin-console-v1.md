# Twin Console v1 (interface prototype)

Artifact: https://claude.ai/artifact/NX5sEHXMPj6MePsrJtmvvi  (built 25 Sep 2026)

## What it is
Single-page vertical slice of the first demonstration: one digital person (sample twin "Ravi Kumar"), one small flat (living room, hallway, kitchen), four tracked objects (keys, The Gruffalo, phone, glasses), two family members (Anya, daughter; Meera, spouse). Three.js r128 isometric scene, all in one HTML file. All data is sample data.

## Layout
- Left: identity, consent chips, household relationships, observed style (with memory refs), 9 build stages, legacy mode off.
- Centre: 3D room (twin has a wireframe shell, amber ground ring, cyan field-of-view cone; planned route drawn as dashed amber line), 9-stage pipeline bar (Perceive → Understand → Remember → Reason → Plan → Move → Interact → Verify → Communicate), conversation, speaker picker + preset requests.
- Right tabs: Activity (task card with confidence + evidence chips, action log), Memory (filter by Verified / User-provided / Inferred / Reconstructed / Unknown), World (twin belief vs simulation truth), Versions (what changed and which memory caused it).

## Behaviour implemented
- Benchmarks: "Where are my keys?" (Meera) and "Daddy, can you get me the book?" (Anya).
- Search order from belief: last seen → usual place → sweep remaining surfaces by path length; perception (same room, 60° cone, 5.6 m) updates beliefs each step and can reroute mid-walk.
- Failed checks create verified negative observations and mark the belief stale; routine confidence (e.g. M-004) moves up/down when the routine holds or breaks; each task that adds memories bumps the model version.
- "Move the keys while the twin isn't looking" creates a belief/truth mismatch to demo replanning. "Reset room" keeps learned memories.
- Memory-only questions (yesterday, visitors, unknown Grandma story) answer with provenance and refuse to fabricate.
- Intent parsing is keyword-based; voice output uses browser TTS as a stand-in for the authorised voice.

## Next steps to consider
Replace keyword intent with an LLM + tool calls; swap stylised avatars for a rigged digital human; real speech input; persistent memory store with provenance schema.
