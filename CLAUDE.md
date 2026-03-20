# CLAUDE.md

## Project

WebGPU Eulerian fluid solver — real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders, rendered with a 2D canvas, served by FastAPI.

## Quick Start

```bash
uv run uvicorn server:app --reload --port 8000
```
Open `http://localhost:8000` in Chrome (WebGPU required).

## Architecture

- **Compute**: 4 WGSL shaders (integrate, pressure, boundary, advect) dispatched via raw WebGPU compute pipelines
- **Rendering**: 2D canvas with `putImageData` + canvas drawing for overlays (streamlines, arrows, obstacles)
- **Backend**: Minimal FastAPI server (~10 lines) serving static files
- **No build step**: Vanilla ES modules, Three.js import map removed (not used)

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
Solid cells (`s = 0`) are rendered as dark gray (40, 40, 48) in the field visualization by reading back the solid mask. The mask is re-read when presets change or obstacles are dragged (`renderer.invalidateSolid()`).

### Streamline/Arrow Caching
Streamline paths and velocity arrow geometry are computed once when new velocity readback data arrives (every 10 frames), then drawn from cache every frame. This prevents flicker and keeps fps high.

## Git

- Use `git -c commit.gpgsign=false` for all commits (GPG agent has timeout issues in this environment)

## Presets

Three working presets: Wind Tunnel, Karman Vortex, Backward Step. All use `windTunnel` or `backwardStep` boundary types with inflow velocity at column `i=1`.

### Known Limitation: Lid-Driven Cavity
The solver's `extrapolate` boundary step copies interior velocities to wall cells, overwriting any forced velocity. This makes lid-driven cavity (which requires a fixed velocity at the top wall) infeasible without modifying the boundary shader. Removed from presets for now.
