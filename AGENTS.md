# AGENTS.md

## Project
FlowLab — real-time 2D incompressible Navier-Stokes solver on the GPU via WebGPU compute shaders. Field rendered by a WebGPU render pass; overlays (streamlines, arrows, particles, obstacles) on a transparent 2D canvas above it (ADR-0005). FastAPI backend. Interactive obstacles, Lagrangian particle tracing, multiple visualization modes.

## Quick Start
```bash
uv run uvicorn server:app --reload --port 8000
```
Open `http://localhost:8000` in Chrome (WebGPU required).

## Architecture
- **Compute**: 8 WGSL compute shader files producing 14 raw compute pipelines — `pressure.wgsl` (red-black Gauss-Seidel + pressure-gauge normalization via `main`, `reset_ref`, `find_ref`, `normalize`, `normalize_ref`, `clear`), `boundary.wgsl` (`extrapolate_horizontal`, `extrapolate_vertical`), `diffuse.wgsl`, `advect.wgsl`, `advect_smoke.wgsl`, `maccormack.wgsl`, `maccormack_velocity.wgsl`, and `rasterize_obstacle.wgsl`. Combines are separate because WGSL forbids duplicate `@group/@binding` declarations. (`render_field.wgsl` is a render shader.) Per-frame dispatches = 2·numIters + 12 + N (e.g. 526 at numIters=256, N=2).
- **Rendering**: Hybrid (ADR-0005) — field drawn by WebGPU render pass (`field-renderer.js` + `render_field.wgsl`: bilinear sampling, colormap LUTs, in-shader solid cells) on `#field-canvas`; overlays on transparent `#overlay-canvas` at display resolution.
- **Tour**: `static/js/tour.js` — `Tour` class with `STEPS` array; welcome modal + 12-step spotlight tour gated by `localStorage` key `flowlab.tour.v1`. Four-dim-rect spotlight hole, z-index 300-302, click-only progression, centered fallback for hidden targets, reduced-motion handling, resolution-tier restore on reset. Guide modal footer has "Replay the Tour".
- **Particles**: CPU-side Lagrangian system (`particles.js`) using velocity readback data; continuous emitters, fading trails.
- **Backend**: FastAPI with `NoCacheMiddleware` (development only; serves static files).
- **No build step**: vanilla ES modules.
- **Deployment**: Dockerfile with health check; supports `PORT` env var.
- **Analytics**: Umami tag injected at request time by `UmamiInjectionMiddleware` (env `UMAMI_DOMAIN`/`UMAMI_ID`, empty = inert; same pattern as the sibling site's nginx sub_filter). Custom events via `static/js/analytics.js` `trackEvent` — no-op when the tag is absent. Ops runbook: `DEPLOYMENT.md`.

## Key Technical Decisions
### WebGPU Bind Group Layouts
Never use `layout: 'auto'` for compute pipelines. Auto-layout includes only statically used bindings; unused declarations (e.g., `boundary.wgsl` `extrapolate_horizontal` doesn't reference `v`) are omitted, causing bind-group creation to fail. Use explicit `GPUBindGroupLayout` with every declared binding.

### Three-Slot Buffer Rotation
Velocity and smoke rotate through three slots: `solver.velPairs[0..2]` (`{u,v}`) and `solver.smokeBufs[0..2]`, with `_velCur` / `_smokeCur` live slots. Three slots are required because MacCormack's combine writes in-place into the backward pair; a 2-cycle would need a fourth pair and distinct combine output, exceeding `maxStorageBuffersPerShaderStage` (8).
Never write rotation buffers directly from JS. Use `writeVelocityU` / `writeVelocityV` / `writeSmoke` / `writeInflowColumn` / `writeSmokeCell` — they write **all** slots so no stale data survives whichever slot is active.
The rotation governs **pressure and boundary** bind groups, not just advection. `pressure.wgsl` writes `u`/`v` in place, so a mis-indexed bind group has its solve discarded by the next advection write. Any pass that reads/writes velocity must index its bind group by `_velCur`.
Smoke advection bind groups are a 3×3 table `[velCur][smokeCur]`: dye is carried by the velocity slot, which advances independently of smoke. Both indices advance only **after** the command buffer is encoded, so smoke advects through the projected time-n velocity, not the unprojected time-(n+1) velocity.

### Boundary Velocity Enforcement
Inflow at column `i=1` survives advection because the left wall (`i=0`) is solid — `s[(i-1)*n+j] != 0` fails. Re-apply inflow velocities **after** `step()` so pressure doesn't drift them.

### Smoke Field Convention
`m = 1.0` clear, `m = 0.0` dye. Renderer uses fixed [0,1] range with magma colormap.

### Solid Cell Rendering
Solid cells (`s = 0`) are dark gray `(50,50,60)` in-shader via `render_field.wgsl` (no readback for display). CPU solid readback `renderer.invalidateSolid()` remains for the particle system.

### Obstacle Shape Switching
`rasterizeObstacle()` clears smoke (`m=1.0`) in the old bbox to prevent stale dye imprints. Null `_prevBBox` before rasterizing on a new grid size to avoid out-of-bounds writes.

### Streamline/Arrow/Particle Caching
Streamline paths, arrow geometry, and particle advection use velocity readback data refreshed every 10 frames; paths/geometry are cached and redrawn each frame. Particles only advect while unpaused.

### Particle System Design
CPU advection reuses renderer readback `uData`/`vData` — no GPU compute for ~5000 particles. Continuous emitters (3 particles/frame) give steady streams. Ice-blue trails `rgba(100,200,255)` contrast with magma. Explicit UI toggle (implicit interactions interfered with drag behavior). Freeze on pause; clear on preset change and resize. Velocity-readback gate must include `showParticles` so particles get data when streamlines/arrows are off.

### Browser Caching During Development
`NoCacheMiddleware` sends `Cache-Control: no-cache, no-store, must-revalidate` for `.js`, `.css`, `.html`, `.wgsl`. Without it, module caching hides code changes and causes "fix doesn't work" loops.

### Resolution Control
Discrete buttons instead of a continuous slider: slider `input` events during drag destroy/recreate GPU buffers repeatedly; buttons fire once. Tiers 64–1024 (1024 manual-only).

### Screenshots
`static/screenshots/` holds `karman-smoke.png` (hero) and `guide.png`. `.gitignore` blocks `*.png` globally with `!static/screenshots/*.png` exception.

### Author Link
Title bar: Author | GitHub pill | Guide(?). Author links to sagar-pal.dev (same pattern as Gray-Scott sibling project).

### Onboarding Tour
Welcome modal + 12-step spotlight tour (`tour.js`). `localStorage` key `flowlab.tour.v1` gates first visit; tour element uses four-dim-rect spotlight hole, z-index 300-302. Click-only writes (no hover auto-advance), centered fallback for hidden targets, reduced-motion handling, early wiring for welcome-skip, and resolution-tier restore on reset. Guide modal footer includes "Replay the Tour".

## Presets
Two presets: default Kármán Vortex (`windTunnel`, obstacle visible) and Backward Step (`backwardStep`, obstacle-less). Both use inflow velocity at column `i=1` and no gravity.
**Insert-on-click**: `backwardStep` ships without an obstacle. On the first canvas `pointerdown` in an obstacle-less preset, `interaction.js:163-166` inserts the obstacle and sets `interaction.showObstacle = true`, activating the overlay ring, shape buttons, and Re/St badges. This relaxes ADR-0011's `d71bc9d` `showObstacle` rasterize guard on the mousedown path only; the `Shift+mousemove` phantom-rotation guard (`interaction.js:198`) stays.
**Known limitation**: lid-driven cavity is infeasible because the `extrapolate` boundary step copies interior velocities to wall cells, overwriting any forced wall velocity.

## Playwright Testing Notes
- Run headed on port 8321 with `--enable-unsafe-webgpu` for WebGPU in CI.
- 104/104 tests: 78 solver/diagnostics/render/perf-hud + 25 tour (`tests/tour.spec.js`) + 1 insert-on-click. `tests/tour.spec.js` tests the welcome modal, step progression, spotlight rendering, skip/complete flows, and `localStorage` gating.
- Screenshots time out due to the continuous `requestAnimationFrame` loop — use `page.evaluate` instead.
- Disable browser cache via CDP (`Network.setCacheDisabled`) or rely on `NoCacheMiddleware`.
- Canvas content via `getImageData` pixel sampling, not screenshots.
- Field View (`#field-canvas`) is sampled by `drawImage` to a temp 2D canvas, then `getImageData`; overlay (`#overlay-canvas`) is transparent and sampled directly.
- Particle tests sample ice-blue trails: `R<180, G>150, B>200`.
- `window.__flowlab` exposes `{ device, solver, renderer, interaction, ui, adaptive, particles, tour }` for test scripting.

## Git
- Use `git -c commit.gpgsign=false` for commits (GPG agent timeout issues).
