---
status: accepted; spec'd, not yet implemented (ROADMAP PR A)
---

# GPU-side obstacle rasterization; no CPU field mirrors

Obstacle rasterization moves into a compute shader (`rasterize_obstacle.wgsl`), and the CPU field mirrors (`interaction._uData` / `_vData` / `_sData`) are deleted outright. The mirrors were seeded at preset load and never refreshed from the GPU, so every `rasterizeObstacle()` ended by pushing a stale whole-field array over the live state — a drag restarted the flow instead of perturbing it (the field-reset defect, ROADMAP Known gaps, user-visible on the flagship demo). The shader takes center, velocity, angle, shape, and the previous bounding box as uniforms and runs one dispatch per rotation slot — three dispatches, each binding the solid mask, a boundary-mask buffer, one velocity pair, pressure, and one smoke buffer (6 storage buffers, under the 8-per-stage limit the solver documents). It writes the new footprint, restores the vacated footprint from the boundary mask, and zeroes pressure and Smoke there. Vacated cells get zero velocity and pressure and Smoke = absence; the projection refills them on the next dispatch. The same rule will govern the Draw-mode eraser (ROADMAP step 5). The inflow-slider path, which pushed the same stale mirror on every `input`, switches to a bounded column-1 write, matching the per-frame `writeInflowColumn` re-application.

## Considered options

- **CPU rasterize + bounded bounding-box upload** — smallest diff and keeps shape geometry on the CPU, but still pays hundreds of `writeBuffer` calls per `mousemove` at tier 1024, and leaves Draw mode to port the geometry to the GPU later.
- **Readback refresh of the CPU mirrors** — fixes correctness only, keeps the whole-array upload pattern, and adds a full-field round trip.

## Consequences

- Shape geometry (circle, square, NACA 0012, wedge, rotation) now lives in WGSL; the deleted CPU inside-tests survive as the test oracle, compared exactly on non-degenerate geometry.
- A solver-owned boundary-mask buffer is uploaded once per preset load (riding the `applyTier` → preset-reload path); the rasterizer restores vacated cells to it and cannot carve permanent boundary cells.
- `paintMode` is deleted — dead code (never enabled), and its per-cell `writeBuffer` dye loop is the upload pattern this change removes.
- Draw mode (ROADMAP step 5) rasterizes on this mechanism.
