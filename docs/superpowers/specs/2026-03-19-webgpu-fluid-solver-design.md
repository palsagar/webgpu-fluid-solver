# WebGPU Eulerian Fluid Solver — Design Spec

## Overview

Rewrite of `fast_euler_solver_browser.html` (a CPU-based 2D Eulerian fluid solver) as a WebGPU compute shader application with Three.js rendering and a FastAPI backend. The app targets two audiences simultaneously: casual visitors who want a visually striking interactive demo, and power users who want to explore CFD numerics through an advanced control panel.

## Architecture

**Approach:** Raw WebGPU compute pipelines for the fluid solver + Three.js WebGPU renderer for visualization. The compute and render pipelines share GPU storage buffers — no CPU readback in the hot loop.

**Backend:** Minimal FastAPI server (~30 lines). Serves static files and a `/api/health` endpoint. No auth, no persistence, no database. Runs on Uvicorn.

### System Diagram

```
FastAPI (server.py)
  └── serves static/ directory
        │
        ▼
Browser Client
  ├── JS Orchestrator (main.js)
  │     ├── GPU device init + capability detection
  │     ├── Animation loop (requestAnimationFrame)
  │     └── Adaptive resolution controller
  │
  ├── Compute Pipeline (Raw WebGPU + WGSL)
  │     ├── integrate.wgsl — gravity/force integration
  │     ├── pressure.wgsl — Red-Black Gauss-Seidel projection
  │     ├── advect.wgsl — semi-Lagrangian vel + smoke advection
  │     ├── boundary.wgsl — extrapolation + obstacle mask
  │     └── minmax.wgsl — parallel reduction for colorbar
  │
  ├── Render Pipeline (Three.js)
  │     ├── Colormap quad (field visualization)
  │     ├── Streamline lines
  │     ├── Velocity arrow InstancedMesh
  │     ├── Obstacle overlay mesh
  │     ├── Colorbar (HTML overlay)
  │     └── Performance HUD (HTML overlay)
  │
  └── UI (HTML/CSS)
        ├── Preset bar (6 presets)
        ├── Shape picker (circle, square, airfoil, wedge)
        ├── Visualization toggles
        ├── Playback controls (play/pause/step/reset)
        └── Advanced panel (slide-out)
```

## GPU Data Layout

### Staggered MAC Grid

Same column-major layout as the original solver. Horizontal velocity `u` lives on left cell edges, vertical velocity `v` on bottom edges, pressure `p` and smoke `m` at cell centers. Solid mask `s` marks fluid (1.0) vs solid (0.0) cells.

Index: `cell(i, j) = i * numY + j`

### Storage Buffers

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `u_buf` | Float32 | numX × numY | Horizontal velocity |
| `v_buf` | Float32 | numX × numY | Vertical velocity |
| `p_buf` | Float32 | numX × numY | Pressure |
| `s_buf` | Float32 | numX × numY | Solid mask (0/1) |
| `m_buf` | Float32 | numX × numY | Smoke/dye density |
| `u_new` | Float32 | numX × numY | Ping-pong pair for u |
| `v_new` | Float32 | numX × numY | Ping-pong pair for v |
| `m_new` | Float32 | numX × numY | Ping-pong pair for m |

### Uniform Buffer

```
numX:    u32
numY:    u32
h:       f32  (cell size)
dt:      f32  (timestep)
gravity: f32
omega:   f32  (over-relaxation factor)
density: f32
```

## Compute Pipeline

### Workgroup Size

8×8 threads (64 per workgroup). For a 256×256 grid: 32×32 = 1024 workgroups per dispatch.

### Per-Frame Dispatch Order

1. **INTEGRATE** (`integrate.wgsl`) — Apply gravity to v-velocity for fluid cells. In-place write, each thread owns one cell. Dispatch: `(numX/8, numY/8, 1)`.

2. **PRESSURE SOLVE** (`pressure.wgsl`) — Red-Black Gauss-Seidel incompressibility projection. Two dispatches per iteration: red cells (i+j even), barrier, black cells (i+j odd), barrier. Repeated for `numIters` iterations. Each thread computes the divergence for its cell and adjusts the pressure and the four neighboring velocity components. This is safe within each color because no two same-colored cells share an edge velocity. Reads u, v, s, p. Writes u, v, p (in-place, no race within a color).

3. **EXTRAPOLATE** (`boundary.wgsl`) — Copy velocities from interior to boundary cells. Edge-only dispatch.

4. **ADVECT VELOCITY** (`advect.wgsl`, velocity mode) — Semi-Lagrangian advection. Reads u, v, s. Writes to u_new, v_new (ping-pong). Swap is a bind group rebind (pointer swap in JS), not a GPU copy.

5. **ADVECT SMOKE** (`advect.wgsl`, smoke mode) — Semi-Lagrangian advection of dye field. Reads u, v, m, s. Writes to m_new. Swap m↔m_new via bind group rebind.

Total dispatches per frame: `4 + 2 × numIters` (e.g., 84 for 40 iterations).

### Pressure Solver: Red-Black Gauss-Seidel

Replaces the original sequential sweep. Each color (red/black) can be updated fully in parallel because no two same-colored cells are neighbors. Converges faster than Jacobi at the same iteration count. The over-relaxation factor ω (default 1.9) accelerates convergence via SOR.

A Jacobi solver toggle is a stretch goal for post-v1 — simpler (one dispatch per iteration instead of two) but converges slower. For v1, only RB Gauss-Seidel is implemented.

## Rendering Pipeline

### Layer Stack (bottom to top)

1. **Field Quad** — `PlaneGeometry` spanning the domain with a custom `ShaderMaterial`. The fragment shader reads the scalar field from a texture and applies a colormap via a 1D LUT texture (256 entries). Zero CPU involvement per frame. Modes: pressure, smoke, or combined. Combined mode composites as: `color = colormap(pressure) × (1.0 - 0.5 × smoke)`.

2. **Velocity Arrows** — `InstancedMesh` with arrow glyphs at every 5th cell. Length proportional to velocity magnitude, rotation from `atan2(v, u)`. Single draw call. Dark gray, semi-transparent.

3. **Streamlines** — `LineSegments` with RK4 integration along the velocity field. Computed on the CPU using readback velocity data. Evenly spaced seed points, ~15 segments each. Recomputed every 5 frames (not every frame). Black, 1px.

4. **Obstacle Overlay** — 2D mesh matching the current shape. Gray fill, black stroke. Updated only on drag events.

5. **HUD Overlay** (HTML/CSS, outside Three.js) — Performance stats (frame time, fps, grid size, divergence error) in top-left. Colorbar with min/max values and units on the right edge.

### GPU → Three.js Buffer Sharing

**Primary path (zero-copy across passes):** The compute pass writes scalar field data to the `.r` channel of a storage texture (format: `rgba16float`). The subsequent render pass binds the same texture as a sampled texture (not storage) and applies the colormap LUT. These are separate passes with an implicit barrier — WebGPU guarantees ordering between passes within a command buffer.

**Fallback path (throttled readback):** For streamlines and velocity arrows, `mapAsync` reads u_buf and v_buf every 5 frames. Non-blocking, doesn't stall the compute pipeline.

### Min/Max for Colorbar

A reduction compute shader (`minmax.wgsl`) computes field min/max in parallel, writes to a 2-element buffer. The fragment shader reads these as uniforms for normalization. The colorbar HTML element reads them (via mapAsync, throttled) for label updates.

### Colormaps

Three scientific colormaps baked as 256×1 PNG textures:

- **viridis** — default for pressure
- **coolwarm** — diverging, good for pressure difference
- **magma** — smoke/density

## UI Design

### Layout

Dark theme (GitHub-style). Four horizontal zones:

1. **Title bar** — App name, WebGPU badge, keyboard shortcut help button.
2. **Preset bar** — 6 preset buttons. Active preset highlighted blue.
3. **Canvas** — Simulation fills most of the viewport. Colorbar docked right. Performance HUD top-left.
4. **Bottom toolbar** — Three groups: visualization toggles (left), shape picker (center), playback controls + Advanced button (right).

### Advanced Panel

Slides out from the right edge, overlaying the canvas. Three sections:

**Numerical Parameters** — Sliders with live value readout:

- Time step (dt): 1/240 → 1/30
- Over-relaxation (ω): 1.0 → 1.98
- Solver iterations: 10 → 200
- Inflow velocity: 0.5 → 5.0

**Flow Physics:**

- Inflow velocity (U): 0.5 → 5.0. This is the primary control for flow regime — higher velocity produces more turbulent-looking behavior. The solver has no explicit viscosity term; numerical viscosity comes from semi-Lagrangian advection and is resolution-dependent. A Reynolds number label is displayed as an approximate indicator (`Re ≈ U·D/h`, where h is cell size), but it is not a directly tunable physical parameter.
- Gravity: -20 → 0

**Solver Internals:**

- Pressure solver: RB Gauss-Seidel (Jacobi toggle is a post-v1 stretch goal)
- Grid resolution slider: 64 → 1024 (overrides adaptive auto-scaling while manually set)
- Reset to Defaults button

### Interaction

- **Mouse/touch drag** on canvas moves the active obstacle. Velocity of the drag is coupled to the fluid (solid cells get obstacle velocity).
- **Shape picker** changes the obstacle geometry. Rasterized on CPU into `s_buf`, uploaded via `writeBuffer()`.
- **Keyboard shortcuts:** P = pause/play, M = step one frame, 1-6 = load preset.

## Obstacle Shapes

Four shapes, rasterized on the CPU into the solid mask:

| Shape | Test | Parameters |
|-------|------|------------|
| Circle | `dx² + dy² < r²` | center, radius |
| Square | `|dx| < hw && |dy| < hw` (rotated) | center, half-width, rotation |
| Airfoil | NACA 4-digit thickness function | center, chord, angle of attack |
| Wedge | Half-angle containment test | center, half-angle, length |

Upload: `writeBuffer()` on the `s_buf` region only, triggered by drag or shape change — not every frame.

## Adaptive Resolution

### Resolution Tiers

64 → 128 → 256 (default) → 512 → 1024. `numY = tier`, `numX = round(tier × canvasWidth / canvasHeight)`. Aspect ratio matches the canvas.

### Controller Logic

- **Measurement:** Rolling average of the last 60 frame times via `performance.now()`. First 30 frames after resize are ignored (GPU warmup).
- **Downscale:** Immediate when avg frame time > 18ms (~55fps). No cooldown — don't let the user suffer.
- **Upscale:** When avg < 10ms and current tier is below max. 2-second cooldown to avoid thrashing.
- **Manual override:** When the user adjusts grid resolution in the advanced panel, adaptive auto-scaling pauses until Reset to Defaults is clicked.

### On Resize

1. `solver.resize(newNumX, newNumY)` — recreate all GPU buffers
2. Re-rasterize obstacle mask at new resolution
3. Re-apply current preset's boundary conditions
4. `renderer.resize()` — update quad geometry + camera
5. HUD briefly shows tier change indicator ("↑ 512×512")

## Presets

### 1. Wind Tunnel (default)

- Boundary: inflow left (u=2.0), open right, walls top/bottom
- Obstacle: circle at (0.4, 0.5), r=0.15
- Gravity: 0, dt: 1/60, iters: 40, ω: 1.9
- Show: smoke + pressure

### 2. Kármán Vortex Street

- Boundary: inflow left (u=1.0), open right, walls top/bottom
- Obstacle: circle at (0.3, 0.5), r=0.06
- Gravity: 0, dt: 1/120, iters: 80, ω: 1.9, Re: ~150
- Show: smoke

### 3. Lid-Driven Cavity

- Boundary: walls all sides, top wall moves right (u=1.0)
- No obstacle
- Gravity: 0, dt: 1/60, iters: 60, ω: 1.9
- Show: streamlines + pressure

### 4. Backward-Facing Step

- Boundary: inflow left (u=1.5) on upper half, walls top/bottom/step, open right
- Step geometry: solid block in lower-left quadrant (0→0.3 x, 0→0.5 y)
- Gravity: 0, dt: 1/60, iters: 60, ω: 1.9
- Show: streamlines + pressure
- Purpose: classic CFD benchmark — recirculation zone behind step, reattachment point visible in streamlines

### 5. Channel Flow

- Boundary: inflow left (u=1.5), open right, walls top/bottom
- 3 staggered circles: (0.25, 0.35), (0.45, 0.65), (0.65, 0.35), r=0.08
- Gravity: 0, dt: 1/60, iters: 40, ω: 1.9
- Show: smoke + streamlines

### 6. Sandbox

- Boundary: walls all sides
- No obstacle initially — user drags to place
- Gravity: 0, dt: 1/60, iters: 40, ω: 1.0
- Show: smoke
- Special: dragging paints oscillating dye (m = sin(frameNr × 0.1))

## Project Structure

```
webgpu_fluid/
├── server.py                    # FastAPI app
├── requirements.txt             # fastapi, uvicorn
├── static/
│   ├── index.html               # Single page, UI shell
│   ├── js/
│   │   ├── main.js              # Entry: GPU init, animation loop
│   │   ├── fluid-solver.js      # Compute pipelines, buffers, step()
│   │   ├── renderer.js          # Three.js scene, layers, colormaps
│   │   ├── interaction.js       # Mouse/touch, shape rasterization
│   │   ├── ui.js                # DOM bindings, panel logic
│   │   ├── adaptive.js          # Frame time monitor, auto-scaling
│   │   └── presets.js           # 6 preset configurations
│   ├── shaders/
│   │   ├── integrate.wgsl
│   │   ├── pressure.wgsl
│   │   ├── advect.wgsl
│   │   ├── boundary.wgsl
│   │   └── minmax.wgsl
│   ├── colormaps/
│   │   ├── viridis.png
│   │   ├── coolwarm.png
│   │   └── magma.png
│   └── css/
│       └── style.css
└── fast_euler_solver_browser.html  # Original reference
```

## Dependencies

**Frontend:** Three.js (via CDN or npm), no other JS dependencies. No build step — vanilla ES modules.

**Backend:** `fastapi`, `uvicorn`. Python 3.10+.

**Browser requirements:** WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly with flag). Fallback: if `navigator.gpu` is undefined, replace the canvas with a centered full-page message directing the user to a supported browser.

**GPU device loss:** On `device.lost`, pause the simulation and display an error banner with a reload button.
