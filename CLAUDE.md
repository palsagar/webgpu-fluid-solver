# CLAUDE.md

## Project

FlowLab — real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders, with the field drawn by a WebGPU render pass and overlays (streamlines, arrows, particles, obstacles) on a transparent 2D canvas above it (ADR-0005), served by FastAPI. Includes interactive obstacles, Lagrangian particle tracing, and multiple visualization modes.

## Quick Start

```bash
uv run uvicorn server:app --reload --port 8000
```
Open `http://localhost:8000` in Chrome (WebGPU required).

## Architecture

- **Compute**: 7 WGSL compute shaders dispatched via raw WebGPU compute pipelines — `pressure.wgsl`, `boundary.wgsl`, `diffuse.wgsl` (explicit five-point viscous pass, substepped), and four advection shaders: `advect.wgsl` (velocity semi-Lagrangian pass, serves as both MacCormack forward and backward), `advect_smoke.wgsl` (same for smoke), `maccormack.wgsl` (smoke combine), `maccormack_velocity.wgsl` (velocity combine). The combines are separate modules from their advect passes because bindings 3..6 mean different buffers, and WGSL forbids two module-scope bindings at the same `@group/@binding`. (`render_field.wgsl` is a render, not compute, shader — see Rendering.)
- **Rendering**: Hybrid (ADR-0005) — Field View drawn by a WebGPU render pass (`field-renderer.js` + `render_field.wgsl`: bilinear sampling, colormap LUT textures, in-shader solid cells) on a display-resolution canvas; overlays (streamlines, arrows, particles, obstacles) drawn on a transparent 2D canvas layered above, also at display resolution
- **Particles**: CPU-side Lagrangian particle system (`particles.js`) using velocity readback data — continuous emitters, fading trails
- **Backend**: FastAPI server with NoCacheMiddleware for development (serves static files)
- **No build step**: Vanilla ES modules
- **Deployment**: Dockerfile with health check, supports PORT env var for PaaS platforms

## Key Technical Decisions

### WebGPU Bind Group Layouts
Do NOT use `layout: 'auto'` for compute pipelines — auto-layout only includes bindings that are **statically used** by the entry point. If a shader declares bindings it doesn't reference (e.g., boundary.wgsl's `extrapolate_horizontal` doesn't use `v`), the auto-layout omits them, and bind group creation fails. Use explicit `GPUBindGroupLayout` with all declared bindings.

### Three-Slot Buffer Rotation
Velocity and smoke rotate through three slots: `solver.velPairs[0..2]` (each `{u, v}`) and `solver.smokeBufs[0..2]`, with `_velCur` / `_smokeCur` naming the live slot. Three rather than two because MacCormack's combine writes in place into the backward pair; a 2-cycle would need a fourth pair and a distinct combine output, exceeding `maxStorageBuffersPerShaderStage` (8).

Never write a rotation buffer directly from JS. Use `writeVelocityU` / `writeVelocityV` / `writeSmoke` / `writeInflowColumn` / `writeSmokeCell` — they write **every** slot, so no slot can hold stale data whichever one the rotation lands on.

The rotation governs the **pressure and boundary** bind groups, not just
advection. `pressure.wgsl` writes `u`/`v` in place, so a bind group pointing at
the wrong slot has its entire solve discarded by the next advection write. Any
new pass that reads or writes velocity must index its bind group by `_velCur`.

Smoke advection bind groups are a 3x3 table indexed `[velCur][smokeCur]`: the dye is carried by the velocity slot, which advances independently of the smoke slot. Both indices advance only **after** the whole command buffer is encoded, so smoke advects through the projected time-n velocity rather than the unprojected time-(n+1) field velocity advection just produced.

### Boundary Velocity Enforcement
Inflow velocities at column `i=1` survive advection because the left wall (`i=0`) is solid — the advection condition `s[(i-1)*n+j] != 0` fails, so the velocity isn't overwritten. However, they still need per-frame re-application **after** `step()` to prevent the pressure solver from drifting them.

### Smoke Field Convention
`m = 1.0` means clear (no dye), `m = 0.0` means dark dye. The renderer uses a fixed [0, 1] range for smoke (no auto-ranging) with the magma colormap.

### Solid Cell Rendering
Solid cells (`s = 0`) are rendered dark gray (50, 50, 60) in-shader by `render_field.wgsl`, which binds the solid buffer directly — no readback needed for display. The CPU-side solid readback (`renderer.invalidateSolid()`) still exists because the particle system needs `solidData` to kill particles entering solids.

### Obstacle Shape Switching
When changing obstacle shape, `rasterizeObstacle()` must clear smoke (`m=1.0`) in the old obstacle's bounding box cells. Without this, stale dye imprints persist. Also, `_prevBBox` must be nulled before rasterizing on a new grid size to avoid out-of-bounds buffer writes.

### Streamline/Arrow/Particle Caching
Streamline paths and velocity arrow geometry are computed once when new velocity readback data arrives (every 10 frames), then drawn from cache every frame. Particles are advected every frame but only when the solver is not paused.

### Particle System Design
- CPU-based advection reusing renderer's velocity readback (`uData`/`vData`) — no GPU compute needed for ~5000 particles
- Continuous emitters (3 particles/frame) instead of burst emission — produces visible steady streams
- Ice-blue trails (`rgba(100, 200, 255)`) — chosen for contrast against the warm magma colormap on both light and dark regions
- Mode switching via explicit UI toggle button — implicit interactions (click threshold, modifier keys) failed in practice because they interfered with existing drag behavior or weren't discoverable
- Particles freeze when solver is paused, cleared on preset change and grid resize
- Velocity readback gate must include `showParticles` so particles get velocity data even when streamlines/arrows are off

### Browser Caching During Development
ES modules are cached aggressively by browsers. The server includes `NoCacheMiddleware` that sends `Cache-Control: no-cache, no-store, must-revalidate` for `.js`, `.css`, `.html`, and `.wgsl` files. Without this, code changes don't reach the browser and debugging becomes impossible. This was the root cause of multiple "fix doesn't work" cycles during particle system development.

### Resolution Control
Use discrete buttons (not a range slider) for grid resolution tiers. A continuous slider fires `input` events during drag, each triggering expensive GPU buffer destruction/recreation. Discrete buttons fire once per click. Tiers: 64–1024 (1024 added with GPU field rendering).

### Screenshots
`static/screenshots/` holds README images (karman-smoke, karman-pressure, windtunnel-streamlines). `.gitignore` blocks `*.png` globally but has `!static/screenshots/*.png` exception.

### Author Link
Title bar includes an "Author" link to sagar-pal.dev — same pattern as the Gray-Scott sibling project. Order: Author | GitHub pill | Guide(?).

## Git

- Use `git -c commit.gpgsign=false` for all commits (GPG agent has timeout issues in this environment)

## Presets

Three working presets (default: Kármán Vortex). All use `windTunnel` or `backwardStep` boundary types with inflow velocity at column `i=1`. No gravity — removed from solver for simplicity.

### Known Limitation: Lid-Driven Cavity
The solver's `extrapolate` boundary step copies interior velocities to wall cells, overwriting any forced velocity. This makes lid-driven cavity (which requires a fixed velocity at the top wall) infeasible without modifying the boundary shader. Removed from presets for now.

## Playwright Testing Notes

- Screenshots timeout due to the continuous `requestAnimationFrame` rendering loop — use `browser_evaluate` and `browser_run_code` instead
- Always disable browser cache via CDP (`Network.setCacheDisabled`) OR rely on the server's NoCacheMiddleware
- Canvas content must be inspected via `getImageData` pixel sampling, not screenshots
- When testing particle visibility, sample for the specific trail color (currently ice-blue: R<180, G>150, B>200)
- The Field View lives on the WebGPU canvas (`#field-canvas`); sample it by `drawImage`-ing into a temp 2D canvas, then `getImageData`. The overlay (`#overlay-canvas`) is transparent — `getImageData` it directly. `window.__flowlab` exposes `{ device, solver, renderer, interaction, ui, adaptive, particles }` for test scripting.
