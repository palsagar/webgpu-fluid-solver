# CLAUDE.md

## Project

FlowLab — real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders, rendered with a 2D canvas, served by FastAPI. Includes interactive obstacles, Lagrangian particle tracing, and multiple visualization modes.

## Quick Start

```bash
uv run uvicorn server:app --reload --port 8000
```
Open `http://localhost:8000` in Chrome (WebGPU required).

## Architecture

- **Compute**: 3 WGSL shaders (pressure, boundary, advect) dispatched via raw WebGPU compute pipelines
- **Rendering**: 2D canvas with `putImageData` + canvas drawing for overlays (streamlines, arrows, particles, obstacles)
- **Particles**: CPU-side Lagrangian particle system (`particles.js`) using velocity readback data — continuous emitters, fading trails
- **Backend**: FastAPI server with NoCacheMiddleware for development (serves static files)
- **No build step**: Vanilla ES modules
- **Deployment**: Dockerfile with health check, supports PORT env var for PaaS platforms

## Key Technical Decisions

### WebGPU Bind Group Layouts
Do NOT use `layout: 'auto'` for compute pipelines — auto-layout only includes bindings that are **statically used** by the entry point. If a shader declares bindings it doesn't reference (e.g., boundary.wgsl's `extrapolate_horizontal` doesn't use `v`), the auto-layout omits them, and bind group creation fails. Use explicit `GPUBindGroupLayout` with all declared bindings.

### Ping-Pong Buffers
Advection uses ping-pong buffer pairs (u/uNew, v/vNew, m/mNew). When writing boundary conditions or obstacle velocities from JS, **write to BOTH buffers** — the solver alternates which one it reads from based on `_advectVelFlip` / `_advectSmokeFlip` state.

### Boundary Velocity Enforcement
Inflow velocities at column `i=1` survive advection because the left wall (`i=0`) is solid — the advection condition `s[(i-1)*n+j] != 0` fails, so the velocity isn't overwritten. However, they still need per-frame re-application **after** `step()` to prevent the pressure solver from drifting them.

### Smoke Field Convention
`m = 1.0` means clear (no dye), `m = 0.0` means dark dye. The renderer uses a fixed [0, 1] range for smoke (no auto-ranging) with the magma colormap.

### Solid Cell Rendering
Solid cells (`s = 0`) are rendered as dark gray (50, 50, 60) in the field visualization by reading back the solid mask. The mask is re-read when presets change or obstacles are dragged (`renderer.invalidateSolid()`).

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
Use discrete buttons (not a range slider) for grid resolution tiers. A continuous slider fires `input` events during drag, each triggering expensive GPU buffer destruction/recreation. Discrete buttons fire once per click.

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
