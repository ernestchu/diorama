# The car

Diorama draws one car model, for the ego car and for every lead vehicle. The app and the web page load it from
`DioramaKit/Sources/DioramaCore/Resources/diorama-assets.bin`. This folder holds where that file comes from: the
original model, the script that turns it into the asset, and a copy of the asset as built.

| Path | What |
|---|---|
| `source/29ad21b0-2e6c-427d-b510-0d369d939c77.obj`, `.mtl` and folder | the original model, as downloaded (OBJ, 37 MB, with its material file and two textures) |
| `assets.py` | the build script: reads the OBJ, edits and recolours it, and writes the asset |
| `diorama-assets.bin` | the asset as built, identical to the app's copy (sha256 `f27b4fc7…6173f0`) |

## Attribution

The car is **"2021 Tesla Model Y"** by **tonielpro520**,
<https://sketchfab.com/3d-models/2021-tesla-model-y-59e2ead369984b1a85c800ff6cf6789d>, licensed under
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.

**Changed for Diorama:**
- It is recoloured in Diorama's flat greys, by part.
- Its badges, interior, brakes and brake discs, its tail lamps' red lens, and the frames hidden behind its body are
  left out.
- It is turned into Diorama's frame and units, and its vertices are stored at 16-bit precision.

Its remaining triangles are the model's own, at full resolution. The Diorama code in this repo is MIT; the model, in
`source/` and in the built asset, stays under CC BY 4.0.

## What the build does

`assets.py` reads the OBJ and does the following:
1. **Axes:** turns the model's axes (y up, headlamps toward +z) into Diorama's local frame: x forward, y right, z up,
   origin on the ground under the car's centre, metres.
2. **Parts:** keeps each material in `PARTS` as a role, with one flat colour per role:
   - body (paint)
   - glass, trim, rims, tyres
   - tail lamps: light strip, dark red inside, housing, reflector
   - headlamps
3. **Dropped materials:** leaves out the materials listed under `PARTS`: the badges, the interior, the brakes, and the
   tail lamps' red lens. The page has no see-through, so the lens drawn solid would hide the lamps.
4. **Hidden parts:** leaves out pieces that are hidden or would flicker (`kept`):
   - the matte-black frames behind the body and glass (only the arch liners and lower trim stay);
   - the radiator behind the bumper;
   - a strip of the glass roof that lies flush on the tailgate.

   It also leaves out the tail lamps' clear lenses (`lenses`).
5. **Rims:** colours the dark trim inside each wheel as the rim.
6. **Emission:** marks the paint with emission −1, so the page and the app colour it themselves: grey for the lead,
   and for the ego car. The tail lamps glow, so a lead's brake lights show; the headlamps glow too.
7. **Packing:** computes normals and packs the meshes into parts under 65,536 vertices, for 16-bit indices.
8. **Writing:** writes `diorama-assets.bin`, described in `write_library`:
   - a u32 header length, then a JSON header with the dims `[4.76, 1.87, 1.66]` m, `span` 4 m, and each part's counts
     and offsets;
   - 16-byte vertices (position as int16 fractions of `span`, normal and emission as int8, colour as uint8);
   - uint16 triangles.

## Rebuilding it

Needs Python 3.10+, numpy and scipy (built with Python 3.10.21, numpy 2.2.6, scipy 1.15.3). From the repo root:

    python car/assets.py car/source/29ad21b0-2e6c-427d-b510-0d369d939c77.obj

This overwrites `DioramaKit/Sources/DioramaCore/Resources/diorama-assets.bin`, the copy the app bundles and the web
page serves. To write it elsewhere, give a second path, for example to compare with the copy here:

    python car/assets.py car/source/29ad21b0-2e6c-427d-b510-0d369d939c77.obj /tmp/diorama-assets.bin
    cmp /tmp/diorama-assets.bin car/diorama-assets.bin    # no output: identical

With the versions above, the build reproduces the committed asset byte for byte.

**If you change the look:**
- edit the colours, `PARTS` or the left-out materials in `assets.py`;
- rebuild;
- copy the new `DioramaKit/.../diorama-assets.bin` over `car/diorama-assets.bin`, so the two stay the same.

The page reads the file in `web/page/diorama-render.js`, and the app in
`DioramaKit/Sources/DioramaUI/DioramaRenderer.swift`.

**Where the OBJ came from:** the model's download page on Sketchfab, as the OBJ download. Its
`source/2021TeslaModelY_obj.zip`, unzipped, is what `source/` holds, as is.
