# GPU Compute & Rendering Pipeline

How WebGPU compute shaders and the renderer simulate and visualize flow. For algorithms, see [Numerical Methods](numerical-methods.md). For how pieces fit together, see [Architecture](architecture.md).

Hub: [../README.md](../README.md) · Index: [README.md](README.md) · Related: [Architecture](architecture.md) · [Numerical Methods](numerical-methods.md) · [ADR index](adr/README.md)

## 1. Buffer Layout

All field buffers are `numX * numY` float32 values in column-major order: `idx = i * numY + j`. Created in `FluidSolver._createBuffers()`.

### Field Buffers

Velocity and smoke use a three-slot rotation (§4), so there are **11 field buffers**, not 8:

| Buffer | Count | Purpose |
|---|---|---|
| `velPairs[0..2].u` | 3 | Horizontal velocity per rotation slot |
| `velPairs[0..2].v` | 3 | Vertical velocity per rotation slot |
| `smokeBufs[0..2]` | 3 | Smoke density per rotation slot |
| `p` | 1 | Pressure |
| `s` | 1 | Solid mask (0 = solid, 1 = fluid) |

Usage: `STORAGE | COPY_SRC | COPY_DST`. `COPY_SRC` enables GPU readback; `COPY_DST` enables CPU writes via `queue.writeBuffer`.

### Uniform Buffers

**Five** uniform buffers, each 32 bytes, usage `UNIFORM | COPY_DST | COPY_SRC`. They share one `Params` struct:

| Buffer | Overrides | Used by |
|---|---|---|
| `uniformBuf` | — | boundary, forward advects, combines |
| `uniformBufRed` | `color = 0` | pressure red sweep |
| `uniformBufBlack` | `color = 1` | pressure black sweep |
| `uniformBufNegDt` | `dt = -dt` | backward advect passes |
| `uniformBufVisc` | `dt = dt/N`, `nu = nuEff` | diffusion |

`uniformBufNegDt` lets one advect shader run both directions; `uniformBufVisc` is rewritten each step.

**`Params` struct (32 bytes):** `numX` (u32), `numY` (u32), `h` (f32), `dt` (f32), `omega` (f32), `density` (f32), `color` (u32), `nu` (f32) at offsets 0/4/8/12/16/20/24/28.

## 2. Compute Pipeline Architecture

Each frame dispatches all compute passes into one `commandEncoder`, submitted once: pressure (red/black × `numIters`) → pressure gauge normalization (reset_ref → find_ref → normalize → normalize_ref) → boundary H → boundary V → velocity MacCormack (fwd/back/combine) → smoke MacCormack (fwd/back/combine) → diffuse (×N substeps).

**Eight shader files, fourteen pipelines:**

| Shader file | Entry point | Role |
|---|---|---|
| `pressure.wgsl` | `main` | Red-black SOR projection |
| `pressure.wgsl` | `reset_ref`, `find_ref`, `normalize`, `normalize_ref` | Gauge normalization: fix pressure constant (per frame) |
| `pressure.wgsl` | `clear` | Pressure clear on large obstacle teleport (not per frame) |
| `boundary.wgsl` | `extrapolate_horizontal`, `extrapolate_vertical` | Free-slip domain edges (two pipelines) |
| `advect.wgsl` | `advect_velocity` | Semi-Lagrangian trace for velocity, both directions |
| `advect_smoke.wgsl` | `advect_smoke` | Semi-Lagrangian trace for smoke, both directions |
| `maccormack_velocity.wgsl` | `maccormack_velocity` | Limited combine for velocity |
| `maccormack.wgsl` | `maccormack_smoke` | Limited combine for smoke |
| `diffuse.wgsl` | `diffuse` | Explicit five-point viscous update |
| `rasterize_obstacle.wgsl` | `rasterize` | Obstacle rasterization (not per-frame) |

Pressure gauge normalization uses a single u32 atomic buffer, `pRef`, shared across the four passes. Pressure is defined only up to an additive constant; the solver deterministically picks the lowest-index fluid cell, subtracts its value everywhere, and zeros that cell in a separate `normalize_ref` pass so the reference is never read and written in the same dispatch.

**Dispatch count per frame (canonical):**

```
total = 2*numIters + 4 + 2 + 3 + 3 + N = 2*numIters + 12 + N
```

Breakdown: `2*numIters` pressure red/black sweeps, 4 gauge passes, 2 boundary passes, 3 velocity MacCormack passes, 3 smoke MacCormack passes, and `N` viscous diffusion substeps. At Kármán defaults (256 iterations, N=2): **526 dispatches** in one submit. Measured frame times are in [Architecture §5](architecture.md#5-adaptive-resolution) and [ROADMAP](ROADMAP.md).

**Workgroup sizing:**

| Shader | Workgroup size | Dispatch |
|---|---|---|
| pressure | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |
| boundary H | `64` | `ceil(numX/64)` |
| boundary V | `64` | `ceil(numY/64)` |
| advect_velocity | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |
| advect_smoke | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |
| maccormack_velocity | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |
| maccormack_smoke | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |
| diffuse | `(8, 8)` | `ceil(numX/8) × ceil(numY/8)` |

## 3. Bind Group Strategy

The solver creates **9 explicit `GPUBindGroupLayout` objects**. The additional layout is `_pressureNormalizeBGL` (uniform + solid mask + pressure + atomic `pRef`). Explicit layouts are required because `layout: 'auto'` only includes statically used bindings; e.g., `boundary.wgsl`'s `extrapolate_horizontal` uses `u` but not `v`, so auto-layout would drop `v` and creation would fail. See [ADR-0002](adr/0002-explicit-bind-group-layouts.md).

### Storage-buffer budget

Layouts are designed against `maxStorageBuffersPerShaderStage`, whose guaranteed minimum is **8**. The tightest layouts are `_advectVelBGL` and `_rasterizeBGL`, both at **7** storage buffers.

`_advectVelBGL` reaches 7 because the backward pass needs advecting velocity (`u^n, v^n`), solid mask, field being advected (`phi^`), and origin field (`phi^n`) all bound at once.

`_mcVelBGL` carries no solid mask; the velocity limiter's `phi^` seed makes the clamp the identity wherever a face reverted, so a solid guard is a no-op.

### Bind groups indexed by rotation slot

| Table | Indexed by |
|---|---|
| `pressureRed`, `pressureBlack`, `boundary` | velocity slot |
| `velFwd`, `velBack`, `velCombine` | velocity slot |
| `smokeFwd`, `smokeBack`, `smokeCombine` | `[velCur][smokeCur]` |
| `diffuse` | `[src][dst]`; diagonal is `null` |

Smoke tables are 2D because the advecting velocity is not tied to the smoke rotation.

## 4. The Three-Slot Rotation

Advection cannot read and write the same buffer in one dispatch. MacCormack needs `phi^n`, `phi^` (forward result), and `phi~` (backward result) live simultaneously, so a two-buffer ping-pong is insufficient.

For live slot `c`:

```
phi^n = slot c
phi^  = slot (c + 1) % 3
phi~  = slot (c + 2) % 3
```

The combine writes `phi^{n+1}` in place into `tilde`, so the inviscid step advances the index by +2 and the sequence runs `0, 2, 1, 0, 2, 1, …`.

The viscous pass ping-pongs between the combine's output pair and the finished pair, so after `N` substeps the result lands on `tilde` when `N` is even and `hat` when `N` is odd. `step()` publishes the final source slot, which is the first thing that makes `_velCur` and `_smokeCur` diverge — hence the smoke bind groups need a `[velCur][smokeCur]` table.

`smokeBuffer` and `velocityBuffers` getters return the currently published slot.

**Critical rule:** when writing boundary conditions, inflow velocities, or obstacle velocities from JS, write to all three slots. `writeU`, `writeV`, and `writeSmoke` do this; the rasterizer's per-slot dispatches do it implicitly.

## 5. Solid Mask Through the Pipeline

The `s` buffer is initialized from the CPU at preset load via `writeSolidMask`, then rasterized on the GPU during interaction. `sBoundary` is uploaded once per preset load and read by the rasterizer so permanent wall cells are never carved or restored.

| Shader | How it reads `s` |
|---|---|
| **pressure** | Counts fluid neighbors via `sx0 + sx1 + sy0 + sy1`. Skips `s[idx] == 0` or `sTotal == 0`. Divergence correction divided by `sTotal`. |
| **boundary** | Does not read `s`; operates only on domain edges. |
| **advect / advect_smoke** | Skips faces/cells where an adjacent cell is solid. Velocity also checks the neighbor sharing the face. Revert condition is identical on forward/backward passes. |
| **maccormack** | Skips solid cells and drops solid corners from limiter bounds unconditionally. Bounds are seeded with `phi^`, so the widening is fluid-only and the inlet band survives. |
| **maccormack_velocity** | Binds no solid mask; `phi^` seed makes the clamp the identity wherever a face reverted. |
| **diffuse** | Classifies each velocity face as FLUID (both cells fluid, diffused), WALL (one solid, copied through), or BURIED. Mask-buried faces ghost to `w + (w - center)` with `w` the stored wall velocity. Index-buried faces (`i == 0` / `j == 0`) and top-row u-faces (`j == numY - 1`) ghost to `-center` (ADR-0011). |

## 6. Rendering Pipeline

Two stacked, display-resolution canvases (see [ADR-0005](adr/0005-hybrid-gpu-field-rendering.md)):

| Layer | Canvas | Context | Drawn by | Content |
|---|---|---|---|---|
| Bottom | `#field-canvas` | webgpu | `FieldRenderer` | Colormapped scalar field |
| Top | `#overlay-canvas` | 2d | `Renderer` | Streamlines, arrows, particles, obstacle |

### Field Render Pass

`FieldRenderer.draw(fieldBuffer, colormapName, minVal, maxVal)` runs every frame. It writes a 16-byte uniform (`numX`, `numY`, `minVal`, `maxVal`), then records one render pass drawing a 3-vertex fullscreen triangle.

`render_field.wgsl` samples the `storage, read` field buffer with manual bilinear interpolation in cell-center space; substitutes the nearest cell's value for solid neighbors. Solid cells are tested per-fragment against the nearest cell in `solid` and returned dark gray before sampling. Because the solid-cell early return makes fragment control flow non-uniform, the LUT lookup uses `textureSampleLevel` with an explicit LOD; implicit-derivative `textureSample` would fail WGSL uniformity validation. The normalized value indexes a 256x1 LUT **texture** via `textureSampleLevel`.

### Colormap LUTs

`_loadLuts(['magma', 'coolwarm'])` runs once in `FieldRenderer.create()`. PNGs are fetched, decoded with `createImageBitmap()`, and uploaded to 256x1 `rgba8unorm` textures via `copyExternalImageToTexture()`. Until textures resolve, `draw()` skips the frame.

- **Smoke:** `magma`, fixed `[0, 1]`. `m = 0` dye → dark; `m = 1` clear → bright.
- **Pressure:** `coolwarm`, auto-ranged symmetrically about the field mean. Requires throttled pressure readback.

### Readback Flow

Three independent readbacks feed the CPU.

1. **Pressure** — throttled temporary staging buffer, only when pressure view is active and `_frameCount % 10 === 1`. `copyBufferToBuffer` from `solver.pressureBuffer`; `_computePressureRange()` returns `[mean - range, mean + range]`. `readbackPending` limits to one in-flight map.
2. **Velocity** — every 10 frames, gated on `showStreamlines || showVelocities || showParticles || showProbe`. Two temporary staging buffers for `u` and `v`. Completion increments `_velDataGen`, triggering streamline/arrow geometry recompute.
3. **Solid mask** — lazy, one-shot. Read once after init and on `invalidateSolid()` (preset change, obstacle drag, resize). Particles need `solidData` on the CPU to kill particles entering solids.

### Device Loss

`device.lost.then()` in `main.js` shows `#device-lost-banner` with a reload button. Not recoverable without reload.

## 7. Overlay Rendering

All overlays are drawn with Canvas 2D on the transparent `#overlay-canvas`, cleared each frame. Stroke widths are multiplied by `_overlayScale` (`canvas.height / numY`).

### Velocity Readback

Same as §6 readback 2: two temporary staging buffers per cycle, throttled by `_velReadbackPending` every 10 frames. Gated when streamlines, arrows, particles, or the probe are active.

### Streamlines

`_computeStreamlines()`: seed every 5th cell starting at `(1, 1)`; 25 segments, step scale 0.01; bilinear `_sampleVel()` with staggered-grid offsets; stop at zero velocity or domain exit; cache in `_cachedStreamlines`; redrawn every frame. Style: white lines at 70% opacity, 1.5px width.

### Velocity Arrows

`_computeArrows()`: sample every 8th cell; length ∝ speed (max 12px), 1% threshold filter; color interpolated by speed from dark blue-green `rgb(30, 80, 120)` to bright cyan-green `rgb(0, 255, 255)`; triangular arrowhead (40% shaft length, min 3px, ±0.5 rad).

### Particle Trails

Rendered after velocity arrows. Particles are advected on the CPU using the same velocity readback as streamlines/arrows (see [Architecture §7](architecture.md#7-particle-tracer)). Each trail (last 20 positions) is drawn as a polyline with opacity fading by age.

### Obstacle Overlay

Drawn with Canvas 2D primitives based on `interaction.activeShape`: circle, square, airfoil (NACA 0012), wedge (15° half-angle).

## 8. Performance Characteristics

**Single outstanding readback.** `readbackPending` ensures at most one `mapAsync` is in flight for the field buffer.

**Batched compute submission.** All dispatches for one step are recorded into one `GPUCommandEncoder` and submitted once.

**Dispatch-count formula (canonical):**

```
total = 2*numIters + 4 + 2 + 3 + 3 + N = 2*numIters + 12 + N
```

At Kármán defaults (256 iterations, N=2): **526 dispatches**. Measured frame times and the upscale-threshold rationale are in [Architecture §5](architecture.md#5-adaptive-resolution) and [ROADMAP](ROADMAP.md).

**Timestep does not change frame cost.** One `solver.step()` runs per rAF frame regardless of `dt`.

**Cached overlay geometry.** Streamline paths and arrow geometry are computed once per velocity readback (every 10 frames) and cached. Every frame, the cache is redrawn with cheap Canvas 2D calls.

**Lazy solid mask readback.** `s` is read back once after init and again only on `invalidateSolid()`.

**Velocity readback is temporary.** Two staging buffers created/destroyed per cycle, acceptable at most every 10 frames and only when overlays are enabled.
