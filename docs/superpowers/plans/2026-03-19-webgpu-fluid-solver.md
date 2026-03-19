# WebGPU Eulerian Fluid Solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite a CPU-based 2D Eulerian fluid solver as a WebGPU compute shader app with Three.js rendering and FastAPI backend.

**Architecture:** Raw WebGPU compute pipelines for fluid simulation (5 WGSL shaders), Three.js WebGPU renderer for visualization (colormap quad, streamlines, arrows, obstacles), minimal FastAPI server for hosting. Compute and render share GPU storage buffers — no CPU readback in the hot loop.

**Tech Stack:** WebGPU + WGSL (compute), Three.js WebGPU renderer (visualization), vanilla ES modules (no build step), FastAPI + Uvicorn (backend), Python 3.10+.

**Spec:** `docs/superpowers/specs/2026-03-19-webgpu-fluid-solver-design.md`

**Reference:** `fast_euler_solver_browser.html` (original CPU solver)

---

## File Map

| File | Responsibility | Created in Task |
|------|----------------|-----------------|
| `server.py` | FastAPI app, serves static files + health endpoint | 1 |
| `requirements.txt` | Python deps: fastapi, uvicorn | 1 |
| `static/index.html` | Single page: UI shell, script imports, WebGPU fallback | 2 |
| `static/css/style.css` | Dark theme, layout, panels, overlays | 2 |
| `static/shaders/integrate.wgsl` | Gravity/force integration compute shader | 3 |
| `static/shaders/pressure.wgsl` | Red-Black Gauss-Seidel pressure projection | 3 |
| `static/shaders/boundary.wgsl` | Boundary extrapolation compute shader | 3 |
| `static/shaders/advect.wgsl` | Semi-Lagrangian advection (velocity + smoke) | 3 |
| `static/shaders/minmax.wgsl` | Parallel reduction for field min/max | deferred post-v1 |
| `static/js/fluid-solver.js` | GPU buffer management, compute pipelines, step() | 4 |
| `static/js/renderer.js` | Three.js scene, colormap quad, layers | 5 |
| `static/js/main.js` | Entry point: GPU init, animation loop, wiring | 5 |
| `static/js/interaction.js` | Mouse/touch drag, shape rasterization, obstacle upload | 7 |
| `static/js/presets.js` | 6 preset configurations + loadPreset() | 8 |
| `static/js/ui.js` | DOM bindings, panel toggles, slider logic | 9 |
| `static/js/adaptive.js` | Frame time monitor, resolution auto-scaling | 10 |
| `static/colormaps/viridis.png` | 256x1 viridis colormap LUT | 6 |
| `static/colormaps/coolwarm.png` | 256x1 coolwarm colormap LUT | 6 |
| `static/colormaps/magma.png` | 256x1 magma colormap LUT | 6 |

---

### Task 1: Project Scaffold & FastAPI Server

**Files:**
- Create: `server.py`
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
fastapi
uvicorn
```

- [ ] **Step 2: Create server.py**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 3: Create static/ directory structure**

```bash
mkdir -p static/js static/shaders static/css static/colormaps
```

- [ ] **Step 4: Create a placeholder index.html to verify serving works**

```html
<!DOCTYPE html>
<html><body><h1>WebGPU Fluid Solver</h1></body></html>
```

Save to `static/index.html`.

- [ ] **Step 5: Verify server starts and serves the page**

```bash
cd /Users/sagarpal/random_shit/webgpu_test && uv run uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000` — should see "WebGPU Fluid Solver". Hit `http://localhost:8000/api/health` — should return `{"status": "ok"}`.

- [ ] **Step 6: Commit**

```bash
git add server.py requirements.txt static/
git commit -m "feat: scaffold project with FastAPI server and static directory"
```

---

### Task 2: HTML Shell & CSS Theme

**Files:**
- Create: `static/index.html` (replace placeholder)
- Create: `static/css/style.css`

- [ ] **Step 1: Write index.html with full UI structure**

The HTML should contain:
- WebGPU fallback: a `<div id="no-webgpu">` with a centered message, hidden by default
- Title bar with app name and "WebGPU" badge
- Preset bar with 6 buttons: Wind Tunnel (active by default), Karman Vortex, Lid Cavity, Backward Step, Channel Flow, Sandbox
- Canvas container `<div id="canvas-container">` (Three.js will mount here)
- Colorbar `<div id="colorbar">` positioned absolute right with min/max labels and gradient strip
- Performance HUD `<div id="perf-hud">` positioned absolute top-left
- Bottom toolbar with three groups:
  - Left: checkboxes for Pressure, Smoke (checked), Streamlines, Velocities
  - Center: shape buttons — Circle (active), Square, Airfoil, Wedge
  - Right: Play/Pause, Step, Reset, Advanced toggle
- Advanced panel `<div id="advanced-panel" class="hidden">` slide-out from right with:
  - Numerical params: dt slider (0.004-0.033, default 0.0167), omega slider (1.0-1.98, default 1.9), iterations slider (10-200, default 40), inflow velocity slider (0.5-5.0, default 2.0)
  - Flow physics: gravity slider (-20-0, default 0), Re display label (read-only)
  - Solver internals: grid resolution slider (64-1024, default 256), Reset to Defaults button
- Import map for Three.js via CDN (e.g. `esm.sh` or `unpkg`):
  ```html
  <script type="importmap">{ "imports": { "three": "https://esm.sh/three@0.171.0", "three/addons/": "https://esm.sh/three@0.171.0/examples/jsm/" } }</script>
  ```
- Script tag: `<script type="module" src="js/main.js"></script>`
- WebGPU detection inline script: if `!navigator.gpu`, show `#no-webgpu`, hide `#app`

- [ ] **Step 2: Write style.css**

Dark GitHub-style theme:
- Body: `background: #0d1117; color: #e0e0e0; font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0;`
- `#app`: flex column, full viewport height
- Title bar: `background: #161b22; border-bottom: 1px solid #30363d; padding: 10px 16px;`
- Preset bar: `background: #161b22; border-bottom: 1px solid #30363d; padding: 8px 16px; display: flex; gap: 8px;`
- Preset button: `padding: 4px 10px; background: #21262d; border: 1px solid #30363d; border-radius: 4px; font-size: 12px; color: #c9d1d9; cursor: pointer;`
- Preset button active: `background: #1f6feb; border-color: #1f6feb; color: white;`
- Canvas container: `flex: 1; position: relative; background: #000; overflow: hidden;`
- `canvas`: `width: 100%; height: 100%; display: block;`
- Colorbar: `position: absolute; right: 20px; top: 20px; bottom: 20px; width: 18px;`
- Perf HUD: `position: absolute; left: 12px; top: 12px; background: rgba(0,0,0,0.7); padding: 6px 10px; border-radius: 4px; font-family: monospace; font-size: 11px; color: #7ee787;`
- Bottom toolbar: `background: #161b22; border-top: 1px solid #30363d; padding: 10px 16px; display: flex; justify-content: space-between;`
- Advanced panel: `position: absolute; right: 0; top: 0; bottom: 0; width: 280px; background: #161b22; border-left: 1px solid #30363d; padding: 16px; transform: translateX(100%); transition: transform 0.3s; z-index: 10;`
- Advanced panel visible: `transform: translateX(0);`
- Slider styling: custom range inputs matching the dark theme
- `.hidden`: `display: none;`
- Button styles: match the mockup from brainstorming (green for play, dark for step/reset, red accent for advanced)

- [ ] **Step 3: Verify in browser**

Run server, open `http://localhost:8000`. Verify:
- Dark theme renders correctly
- All UI elements visible (preset bar, bottom toolbar, canvas area)
- Advanced panel slides out when toggling `.hidden` class via devtools
- WebGPU fallback shows if tested in a non-WebGPU browser

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/css/style.css
git commit -m "feat: add HTML shell with dark theme UI layout"
```

---

### Task 3: WGSL Compute Shaders

**Files:**
- Create: `static/shaders/integrate.wgsl`
- Create: `static/shaders/pressure.wgsl`
- Create: `static/shaders/boundary.wgsl`
- Create: `static/shaders/advect.wgsl`

**Reference:** The original solver in `fast_euler_solver_browser.html` lines 106-302 contains the CPU implementations of all four operations. The WGSL shaders are direct ports of this logic to GPU compute.

- [ ] **Step 1: Write integrate.wgsl**

Port of `Fluid.integrate()` (reference lines 106-113). All shaders share the same Params struct with 8 fields (the `color` field is only used by the pressure shader but must be present in all for uniform buffer layout consistency):

```wgsl
struct Params {
    numX: u32,
    numY: u32,
    h: f32,
    dt: f32,
    gravity: f32,
    omega: f32,
    density: f32,
    color: u32,   // only used by pressure.wgsl, 0 in all others
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i < 1u || i >= params.numX || j < 1u || j >= n - 1u) { return; }
    let idx = i * n + j;
    if (s[idx] != 0.0 && s[i * n + j - 1u] != 0.0) {
        v[idx] += params.gravity * params.dt;
    }
}
```

- [ ] **Step 2: Write pressure.wgsl**

Port of `Fluid.solveIncompressibility()` (reference lines 116-151). Red-Black Gauss-Seidel — each dispatch handles one color. The color is passed via an extra uniform field.

Add `color: u32` to the Params struct (0 = red where (i+j) is even, 1 = black where (i+j) is odd).

```wgsl
struct Params {
    numX: u32,
    numY: u32,
    h: f32,
    dt: f32,
    gravity: f32,
    omega: f32,
    density: f32,
    color: u32,   // 0 = red, 1 = black
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;

    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }

    // Red-black check: skip if this cell doesn't match the current color
    if ((i + j) % 2u != params.color) { return; }

    let idx = i * n + j;
    if (s[idx] == 0.0) { return; }

    let sx0 = s[(i - 1u) * n + j];
    let sx1 = s[(i + 1u) * n + j];
    let sy0 = s[i * n + j - 1u];
    let sy1 = s[i * n + j + 1u];
    let sTotal = sx0 + sx1 + sy0 + sy1;
    if (sTotal == 0.0) { return; }

    let div = u[(i + 1u) * n + j] - u[idx] + v[i * n + j + 1u] - v[idx];
    let pCorr = -div / sTotal * params.omega;

    p[idx] += params.density * params.h / params.dt * pCorr;

    u[idx]           -= sx0 * pCorr;
    u[(i + 1u) * n + j] += sx1 * pCorr;
    v[idx]           -= sy0 * pCorr;
    v[i * n + j + 1u]   += sy1 * pCorr;
}
```

- [ ] **Step 3: Write boundary.wgsl**

Port of `Fluid.extrapolate()` (reference lines 154-163). Two entry points for horizontal and vertical boundary extrapolation.

```wgsl
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn extrapolate_horizontal(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = params.numY;
    if (i >= params.numX) { return; }
    u[i * n + 0u] = u[i * n + 1u];
    u[i * n + n - 1u] = u[i * n + n - 2u];
}

@compute @workgroup_size(64)
fn extrapolate_vertical(@builtin(global_invocation_id) id: vec3u) {
    let j = id.x;
    let n = params.numY;
    if (j >= n) { return; }
    v[0u * n + j] = v[1u * n + j];
    v[(params.numX - 1u) * n + j] = v[(params.numX - 2u) * n + j];
}
```

Note: This requires two separate pipelines (one per entry point).

- [ ] **Step 4: Write advect.wgsl**

Port of `Fluid.advectVel()` and `Fluid.advectSmoke()` (reference lines 221-288). Two entry points: `advect_velocity` and `advect_smoke`. Both use semi-Lagrangian backtracing with bilinear interpolation.

The bilinear sampling helper (`sampleField` in the original, lines 166-204) is inlined as a WGSL function. Note on `sample_field`: WGSL does not support passing storage buffer references as function parameters in all implementations. If this doesn't compile, inline the sampling logic directly in each entry point, or use separate functions for each field (sample_u, sample_v, sample_m). The implementer should test compilation and fall back to inlining if needed.

`advect_velocity` entry point: for each interior cell, backtrace using current velocity, sample u and v at the departure point, write to `u_new` / `v_new`.

`advect_smoke` entry point: for each interior cell, backtrace using cell-center averaged velocity, sample smoke at the departure point, write to `m_new`.

Bind groups for advect_velocity:
- binding 0: params (uniform)
- binding 1: u (storage, read)
- binding 2: v (storage, read)
- binding 3: s (storage, read)
- binding 4: u_new (storage, read_write)
- binding 5: v_new (storage, read_write)

Bind groups for advect_smoke:
- binding 0: params (uniform)
- binding 1: u (storage, read)
- binding 2: v (storage, read)
- binding 3: s (storage, read)
- binding 4: m (storage, read)
- binding 5: m_new (storage, read_write)

- [ ] **Step 5: Commit**

```bash
git add static/shaders/
git commit -m "feat: add WGSL compute shaders for fluid solver"
```

---

### Task 4: Fluid Solver JS Module (GPU Buffer Management)

**Files:**
- Create: `static/js/fluid-solver.js`

**Dependencies:** Task 3 (shaders must exist to be fetched)

- [ ] **Step 1: Write fluid-solver.js — buffer creation**

Export a `FluidSolver` class. Constructor takes `(device, numX, numY, h)`. Creates:
- 8 storage buffers: `u`, `v`, `p`, `s`, `m`, `uNew`, `vNew`, `mNew` — each `Float32Array(numX * numY)`, usage `STORAGE | COPY_SRC | COPY_DST`
- 1 uniform buffer: 32 bytes (see Params struct), usage `UNIFORM | COPY_DST`
- A `params` JS object: `{ numX, numY, h, dt: 1/60, gravity: 0, omega: 1.9, density: 1000, color: 0 }`

Method `writeParams()`: packs params into an `ArrayBuffer` (u32, u32, f32, f32, f32, f32, f32, u32) and calls `device.queue.writeBuffer()`.

Method `destroy()`: destroys all buffers.

- [ ] **Step 2: Write fluid-solver.js — pipeline creation**

Async factory method `static async create(device, numX, numY, h)`:
1. Fetch all 4 shader files via `fetch()` then `.text()`
2. Create `GPUShaderModule` for each
3. Create compute pipelines:
   - `integratePipeline` — entry point `main`
   - `pressurePipeline` — entry point `main`
   - `boundaryHPipeline` — entry point `extrapolate_horizontal`
   - `boundaryVPipeline` — entry point `extrapolate_vertical`
   - `advectVelPipeline` — entry point `advect_velocity`
   - `advectSmokePipeline` — entry point `advect_smoke`
4. Create bind group layouts and bind groups for each pipeline
5. For pressure pipeline: create two separate uniform buffers (one with color=0, one with color=1) and two bind groups. This avoids per-iteration CPU-to-GPU uniform writes.

For the ping-pong swap: maintain two sets of bind groups for advection. After dispatch, swap the references: `[this.advectVelBindGroupA, this.advectVelBindGroupB] = [this.advectVelBindGroupB, this.advectVelBindGroupA]`. Same for smoke.

- [ ] **Step 3: Write fluid-solver.js — step() method**

`step(numIters)`:
1. Write current params to uniform buffer (call `writeParams()`)
2. Create a `GPUCommandEncoder`
3. Begin compute pass
4. **Integrate:** set pipeline + bind group, dispatch `(ceil(numX/8), ceil(numY/8), 1)`
5. End compute pass (implicit barrier)
6. **Pressure solve loop** (numIters times):
   - Begin compute pass, set pipeline + red bind group (color=0), dispatch, end pass (barrier)
   - Begin compute pass, set pipeline + black bind group (color=1), dispatch, end pass (barrier)
7. **Boundary extrapolation:**
   - Begin compute pass, set horizontal pipeline, dispatch `(ceil(numX/64), 1, 1)`, end pass
   - Begin compute pass, set vertical pipeline, dispatch `(ceil(numY/64), 1, 1)`, end pass
8. **Advect velocity:** begin pass, dispatch `(ceil(numX/8), ceil(numY/8), 1)`, end pass.
9. **Advect smoke:** begin pass, dispatch, end pass.
10. `device.queue.submit([encoder.finish()])`
11. **After submit:** swap ping-pong bind group references in JS (`[this.advectVelBGA, this.advectVelBGB] = [this.advectVelBGB, this.advectVelBGA]`). This must happen after submit, not between encoder commands, since commands are recorded before submission.

- [ ] **Step 4: Write fluid-solver.js — resize() method**

`resize(numX, numY, h)`:
1. Destroy all existing buffers
2. Recreate buffers and bind groups at new size
3. Update params
4. Call `writeParams()`

- [ ] **Step 5: Write fluid-solver.js — setParams() and buffer accessors**

`setParams(overrides)`: merge overrides into `this.params`, call `writeParams()`.

Convenience methods for initial condition setup:
- `writeSolidMask(data)`: `device.queue.writeBuffer(this.s, 0, data)`
- `writeVelocityU(data)`: `device.queue.writeBuffer(this.u, 0, data)`
- `writeVelocityV(data)`: `device.queue.writeBuffer(this.v, 0, data)`
- `writeSmoke(data)`: `device.queue.writeBuffer(this.m, 0, data)`

Expose getters:
- `get pressureBuffer()` returns `this.p`
- `get smokeBuffer()` returns `this.m`
- `get velocityBuffers()` returns `{ u: this.u, v: this.v }`
- `get solidBuffer()` returns `this.s`
- `numX`, `numY`, `h` as readonly properties

- [ ] **Step 6: Commit**

```bash
git add static/js/fluid-solver.js
git commit -m "feat: add FluidSolver class with GPU buffer management and compute dispatch"
```

---

### Task 5: Basic Renderer & Main Loop (First Visual Output)

**Files:**
- Create: `static/js/renderer.js`
- Create: `static/js/main.js`

**Dependencies:** Task 2 (HTML shell), Task 4 (FluidSolver)

**Goal:** Get the first visual output — a colormap quad showing the pressure or smoke field updating in real time.

- [ ] **Step 1: Write renderer.js — scene setup**

Export a `Renderer` class. Constructor takes `(container, device, solver)` where `container` is the `#canvas-container` DOM element.

Setup:
1. Create `THREE.WebGPURenderer` with `antialias: false`
2. Set renderer size to container size
3. Create `THREE.OrthographicCamera` matching the simulation domain (0 to numX*h width, 0 to numY*h height)
4. Create `THREE.Scene`
5. Create the field quad: `THREE.PlaneGeometry(numX*h, numY*h)` centered in the domain
6. For the colormap material: use a simple grayscale shader for now: `color = vec3(value)` — colormap LUT comes in Task 6

The field `DataTexture` is updated each frame by reading the storage buffer via CPU readback. Use a staging buffer with `mapAsync` to read the active field (pressure or smoke) into a `Float32Array`, then update the DataTexture.

Note: The mapAsync is async. Strategy: fire the readback, render the previous frame's data, update the texture when the readback resolves. This keeps the render loop non-blocking.

Note: For the Three.js WebGPU renderer, shader materials may need to use WGSL or TSL instead of GLSL. Check Three.js r160+ docs for `WebGPURenderer` ShaderMaterial compatibility. If GLSL isn't supported, use `THREE.NodeMaterial` with TSL nodes, or a `RawShaderMaterial` with WGSL. The implementer should check the Three.js WebGPU examples for the correct approach.

- [ ] **Step 2: Write renderer.js — draw() method**

`draw()`:
1. Copy the active field buffer (pressure or smoke, based on `this.showPressure`/`this.showSmoke` flags) to a staging buffer via `commandEncoder.copyBufferToBuffer()`
2. Map the staging buffer, read into a Float32Array
3. Compute min/max from the read data (CPU-side, simple loop)
4. Update the DataTexture with the new data
5. Call `renderer.render(scene, camera)`

- [ ] **Step 3: Write renderer.js — resize() method**

`resize(numX, numY, h)`:
1. Recreate the DataTexture at new size
2. Update camera frustum
3. Recreate the plane geometry
4. Recreate the staging buffer

- [ ] **Step 4: Write main.js — GPU init and animation loop**

```javascript
import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';

async function init() {
    if (!navigator.gpu) {
        document.getElementById('no-webgpu').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { /* show error */ return; }
    const device = await adapter.requestDevice();

    device.lost.then((info) => {
        console.error('GPU device lost:', info.message);
        const banner = document.getElementById('device-lost-banner');
        banner.style.display = 'block';
    });

    const container = document.getElementById('canvas-container');
    const numY = 256;
    const numX = Math.round(numY * container.clientWidth / container.clientHeight);
    const h = 1.0 / numY;

    const solver = await FluidSolver.create(device, numX, numY, h);
    const renderer = new Renderer(container, device, solver);

    // Set up initial conditions: wind tunnel (hardcoded for now, presets come in Task 8)
    initWindTunnel(solver);

    function frame() {
        solver.step(40);
        renderer.draw();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

function initWindTunnel(solver) {
    const { numX, numY } = solver;
    const n = numY;
    const sData = new Float32Array(numX * numY);
    const uData = new Float32Array(numX * numY);

    for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
            let s = 1.0;
            if (i === 0 || j === 0 || j === numY - 1) s = 0.0;
            sData[i * n + j] = s;
            if (i === 1) uData[i * n + j] = 2.0;
        }
    }
    solver.writeSolidMask(sData);
    solver.writeVelocityU(uData);
    solver.setParams({ gravity: 0, omega: 1.9, dt: 1/60, density: 1000 });
}

init();
```

- [ ] **Step 5: Verify in browser**

Run the server, open `http://localhost:8000`. You should see:
- The simulation running with a grayscale visualization
- Fluid flowing from left to right
- The field updating in real time

If the screen is black/white/frozen, check browser console for WebGPU errors. Common issues: shader compilation errors, buffer size mismatches, missing bindings.

- [ ] **Step 6: Commit**

```bash
git add static/js/renderer.js static/js/main.js
git commit -m "feat: add renderer and main loop with first visual output"
```

---

### Task 6: Colormaps, Min/Max & Colorbar

**Files:**
- Create: `static/colormaps/viridis.png`, `coolwarm.png`, `magma.png`
- Modify: `static/js/renderer.js` (add colormap material, colorbar updates)

**Dependencies:** Task 5 (renderer must exist)

- [ ] **Step 1: Generate colormap PNG files**

Write a small Python script (throwaway, not committed) that generates 256x1 PNG images for viridis, coolwarm, and magma using matplotlib:

```python
import numpy as np
from matplotlib import colormaps
from PIL import Image

for name in ['viridis', 'coolwarm', 'magma']:
    cmap = colormaps[name]
    colors = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colors.reshape(1, 256, 3))
    img.save(f'static/colormaps/{name}.png')
```

Run: `uv run python generate_colormaps.py` (requires `matplotlib`, `Pillow`, `numpy`). Delete the script after.

- [ ] **Step 2: Update renderer.js — add colormap material**

1. Load the 3 colormap PNGs as `THREE.Texture` objects via `THREE.TextureLoader`
2. Update the field quad's shader material fragment shader to:
   - Sample the scalar field DataTexture
   - Normalize using min/max uniforms (computed CPU-side from readback data)
   - Look up the colormap LUT texture: `color = texture(colormapTex, vec2(t, 0.5)).rgb`
3. Add combined mode support: if both pressure and smoke are shown, read both fields and composite as `color = colormap(pressure) * (1.0 - 0.5 * smoke)`. This requires a second DataTexture for the smoke field.

Min/max is computed CPU-side from the readback data (already available from Task 5's draw method). No GPU reduction shader needed for v1.

- [ ] **Step 3: Update renderer.js — colorbar DOM updates**

In `draw()`, after computing min/max from the readback data:
1. Update `#colorbar-max` text content to `maxVal.toFixed(0)`
2. Update `#colorbar-min` text content to `minVal.toFixed(0)`
3. Update `#colorbar-unit` based on active field: "N/m2" for pressure, "" for smoke
4. Update the CSS gradient of `#colorbar-gradient` to match the active colormap

- [ ] **Step 4: Verify in browser**

Run server, open app. Should see:
- Viridis-colored pressure field (or smoke field) instead of grayscale
- Colorbar on the right with correct min/max values updating each frame
- Switch between pressure/smoke by toggling flags in devtools

- [ ] **Step 5: Commit**

```bash
git add static/colormaps/ static/js/renderer.js
git commit -m "feat: add scientific colormaps with colorbar and min/max tracking"
```

---

### Task 7: Interaction — Obstacle Drag & Shape Rasterization

**Files:**
- Create: `static/js/interaction.js`
- Modify: `static/js/main.js` (wire up interaction)

**Dependencies:** Task 5 (main loop + solver running)

- [ ] **Step 1: Write interaction.js — mouse/touch event handling**

Export an `Interaction` class. Constructor takes `(canvas, solver)`.

Sets up event listeners:
- `mousedown` on canvas element: `startDrag(x, y)`
- `mousemove` on canvas element: `drag(x, y)`
- `mouseup` on canvas element: `endDrag()`
- `touchstart` on canvas element: `startDrag(touch.clientX, touch.clientY)`
- `touchmove` on canvas element: `drag(touch.clientX, touch.clientY)` with `preventDefault()`
- `touchend` on canvas element: `endDrag()`

Coordinate conversion: canvas pixel to simulation domain coords:
```javascript
screenToSim(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    const x = mx / rect.width * this.solver.numX * this.solver.h;
    const y = (1.0 - my / rect.height) * this.solver.numY * this.solver.h;
    return { x, y };
}
```

- [ ] **Step 2: Write interaction.js — shape rasterization**

`rasterizeObstacle(centerX, centerY)`:
1. Read current `activeShape` (default: `'circle'`)
2. For each cell `(i, j)` in the interior (skip edges):
   - Compute cell center: `cx = (i + 0.5) * h`, `cy = (j + 0.5) * h`
   - Test if inside shape:
     - Circle: `(cx-x)^2 + (cy-y)^2 < r^2`
     - Square: `|dx| < hw && |dy| < hw` (with rotation via cos/sin transform)
     - Airfoil: NACA 0012 profile (symmetric). Compute upper/lower surface at the x-position, test if y is between them. Transform to obstacle-local coords accounting for angle of attack.
     - Wedge: rotate to local coords, test if `|local_y| < local_x * tan(halfAngle)` and `local_x < length`
   - If inside: `s[i*n+j] = 0.0` (solid), set `u` and `v` to obstacle velocity
   - If outside but was solid from obstacle (not boundary): `s[i*n+j] = 1.0`
3. Write the updated `s`, `u`, `v` buffers to GPU via `device.queue.writeBuffer()`

Important: preserve boundary walls (i=0, j=0, j=numY-1, etc.) — only toggle cells that were set by the obstacle, not by preset boundary conditions. Strategy: keep a separate `boundaryMask` array that marks which cells are boundary walls (set during preset loading). Only modify cells where `boundaryMask[idx] == 0`.

- [ ] **Step 3: Write interaction.js — velocity coupling on drag**

In `drag(x, y)`:
1. Compute obstacle velocity: `vx = (x - prevX) / dt`, `vy = (y - prevY) / dt`
2. Call `rasterizeObstacle(x, y)` which sets solid cell velocities to `(vx, vy)`
3. Update `prevX = x`, `prevY = y`

In `startDrag(x, y)`:
1. Set `prevX = x`, `prevY = y`, `dragging = true`
2. Call `rasterizeObstacle(x, y)` with zero velocity (reset)

- [ ] **Step 4: Wire up in main.js**

```javascript
import { Interaction } from './interaction.js';
// after solver and renderer are created:
const interaction = new Interaction(renderer.canvas, solver);
```

- [ ] **Step 5: Verify in browser**

Run server, open app. Should be able to:
- Click and drag on the canvas to move a circular obstacle
- See the fluid react to the obstacle (flow around it, vortices behind it)
- Obstacle velocity couples to the fluid (dragging fast creates visible disturbance)

- [ ] **Step 6: Commit**

```bash
git add static/js/interaction.js static/js/main.js
git commit -m "feat: add obstacle drag interaction with shape rasterization"
```

---

### Task 8: Presets

**Files:**
- Create: `static/js/presets.js`
- Modify: `static/js/main.js` (use presets for init)

**Dependencies:** Task 7 (interaction must exist for obstacle setup)

- [ ] **Step 1: Write presets.js**

Export a `PRESETS` object and a `loadPreset(name, solver, interaction)` function.

Define 6 presets as config objects:
- `windTunnel`: inflow left u=2.0, open right, walls top/bottom, circle obstacle at (0.4,0.5) r=0.15, gravity=0, dt=1/60, iters=40, omega=1.9, show smoke+pressure
- `karmanVortex`: inflow left u=1.0, circle at (0.3,0.5) r=0.06, dt=1/120, iters=80, show smoke only
- `lidCavity`: walls all sides, top wall moves right u=1.0, no obstacle, dt=1/60, iters=60, show streamlines+pressure
- `backwardStep`: inflow upper-left u=1.5, step block in lower-left (0-0.3 x, 0-0.5 y), no draggable obstacle, dt=1/60, iters=60, show streamlines+pressure
- `channelFlow`: inflow left u=1.5, 3 staggered circles, dt=1/60, iters=40, show smoke+streamlines
- `sandbox`: walls all sides, no obstacle initially, gravity=0, omega=1.0, paintMode=true, show smoke

`loadPreset(name, solver, interaction)`:
1. Get preset config
2. Call `solver.setParams({ dt, gravity, omega, density: 1000 })`
3. Reset all field buffers to zero
4. Apply boundary conditions based on `boundaryType`
5. Apply smoke injection for wind tunnel presets (central 10% of cells at x=0 column set to m=0.0)
6. Rasterize obstacle(s) via `interaction.rasterizeObstacle()`
7. Return the `show` config for the renderer

- [ ] **Step 2: Update main.js to use presets**

Replace the hardcoded `initWindTunnel()` with:
```javascript
import { loadPreset } from './presets.js';
const showConfig = loadPreset('windTunnel', solver, interaction);
renderer.setVisualization(showConfig);
```

- [ ] **Step 3: Verify each preset in browser**

Manually change the preset name in main.js and reload to test each:
- Wind Tunnel: flow around circle, smoke + pressure visible
- Karman Vortex: small circle, alternating vortex wake in smoke
- Lid Cavity: verify pressure field (streamlines come in Task 11)
- Backward Step: recirculation zone behind step
- Channel Flow: 3 obstacles with interacting wakes
- Sandbox: static, no flow until user drags

- [ ] **Step 4: Commit**

```bash
git add static/js/presets.js static/js/main.js
git commit -m "feat: add 6 simulation presets with boundary condition setup"
```

---

### Task 9: UI Wiring

**Files:**
- Create: `static/js/ui.js`
- Modify: `static/js/main.js` (wire up UI module)

**Dependencies:** Task 8 (presets), Task 7 (interaction)

- [ ] **Step 1: Write ui.js — preset buttons**

Export a `UI` class. Constructor takes `(solver, renderer, interaction)`.

Preset buttons: query all `[data-preset]` buttons, add click handlers. On click:
1. Call `loadPreset(name, solver, interaction)`
2. Update active button styling (remove `.active` from all, add to clicked)
3. Update visualization toggles to match preset's `show` config
4. Update checkbox states in the DOM

- [ ] **Step 2: Write ui.js — visualization toggles and shape picker**

Bind checkbox change events for pressure, smoke, streamlines, velocities to renderer flags.

Bind shape button clicks to set `interaction.activeShape`.

- [ ] **Step 3: Write ui.js — playback controls**

- Play/Pause button: toggle `solver.paused`. Update button text.
- Step button: if paused, call `solver.step(numIters)` once, `renderer.draw()` once.
- Reset button: reload current preset via `loadPreset(currentPreset, ...)`

- [ ] **Step 4: Write ui.js — advanced panel**

- Advanced button click: toggle `#advanced-panel` visibility via CSS class
- Slider bindings:
  - dt slider: `solver.setParams({ dt: parseFloat(value) })`, update value label
  - omega slider: `solver.setParams({ omega: parseFloat(value) })`, update label
  - iterations slider: store in `ui.numIters`, used by main loop
  - inflow velocity slider: update inflow boundary cells, update Re label
  - gravity slider: `solver.setParams({ gravity: parseFloat(value) })`
  - grid resolution slider: trigger resize (calls solver.resize + renderer.resize + re-apply preset)
- Re label: compute `Re = U * D / h` (approximate, display only)
- Reset to Defaults button: reload current preset, reset all sliders

- [ ] **Step 5: Write ui.js — keyboard shortcuts**

```javascript
document.addEventListener('keydown', (e) => {
    switch (e.key) {
        case 'p': this.togglePause(); break;
        case 'm': this.stepOnce(); break;
        case '1': case '2': case '3': case '4': case '5': case '6':
            this.loadPresetByIndex(parseInt(e.key) - 1); break;
    }
});
```

- [ ] **Step 6: Wire up in main.js**

```javascript
import { UI } from './ui.js';
const ui = new UI(solver, renderer, interaction);
```

Update animation loop to respect `solver.paused` and use `ui.numIters`.

- [ ] **Step 7: Verify in browser**

- Click preset buttons: simulation resets correctly
- Toggle checkboxes: field display changes
- Shape buttons work
- Play/Pause/Step/Reset work
- Advanced panel slides out, sliders adjust parameters live
- Keyboard shortcuts (P, M, 1-6)

- [ ] **Step 8: Commit**

```bash
git add static/js/ui.js static/js/main.js
git commit -m "feat: wire up UI controls, presets, and keyboard shortcuts"
```

---

### Task 10: Adaptive Resolution Controller

**Files:**
- Create: `static/js/adaptive.js`
- Modify: `static/js/main.js` (integrate adaptive controller)
- Modify: `static/js/ui.js` (manual override connection)

**Dependencies:** Task 9 (UI must exist for manual override)

- [ ] **Step 1: Write adaptive.js**

Export an `AdaptiveController` class. Constructor takes `(solver, renderer, interaction, ui)`.

State:
- `tiers = [64, 128, 256, 512, 1024]`
- `currentTierIndex = 2` (256 default)
- `frameTimes = []` (rolling window of 60)
- `warmupFrames = 0` (ignore first 30 after resize)
- `lastUpscaleTime = 0`
- `manualOverride = false`

- [ ] **Step 2: Write adaptive.js — tick() method**

Called each frame with the current frame time (ms):
1. If `manualOverride`, return early
2. Increment `warmupFrames`, skip if <= 30
3. Push frame time, maintain window of 60
4. Compute rolling average
5. If avg > 18ms and not at minimum tier: downscale (immediate)
6. If avg < 10ms and not at maximum tier and 2s since last upscale: upscale

- [ ] **Step 3: Write adaptive.js — applyTier() method**

1. Compute `numY = tier`, `numX = round(tier * canvas.clientWidth / canvas.clientHeight)`
2. Call `solver.resize(numX, numY, h)`
3. Re-rasterize obstacle via `interaction`
4. Re-apply current preset boundary conditions via `ui.reapplyCurrentPreset()`
5. Call `renderer.resize(numX, numY, h)`
6. Reset frame times and warmup counter
7. Show brief tier change indicator in HUD

- [ ] **Step 4: Integrate into main.js**

```javascript
import { AdaptiveController } from './adaptive.js';
const adaptive = new AdaptiveController(solver, renderer, interaction, ui);

function frame() {
    const t0 = performance.now();
    if (!solver.paused) solver.step(ui.numIters);
    renderer.draw();
    adaptive.tick(performance.now() - t0);
    requestAnimationFrame(frame);
}
```

- [ ] **Step 5: Connect manual override from UI**

In ui.js, when grid resolution slider changes: set `adaptive.manualOverride = true` and directly resize.
When "Reset to Defaults" is clicked: set `adaptive.manualOverride = false`.

- [ ] **Step 6: Verify in browser**

- Normal operation: resolution stays at 256 or upscales on fast hardware
- Set iterations to 200: should downscale, HUD shows indicator
- Manual grid slider: overrides auto-scaling
- Reset to Defaults: re-enables auto-scaling

- [ ] **Step 7: Commit**

```bash
git add static/js/adaptive.js static/js/main.js static/js/ui.js
git commit -m "feat: add adaptive resolution controller with manual override"
```

---

### Task 11: Streamlines & Velocity Arrows

**Files:**
- Modify: `static/js/renderer.js` (add streamline + arrow layers)

**Dependencies:** Task 6 (renderer with colormap must exist)

- [ ] **Step 1: Add velocity readback to renderer**

Add method `readbackVelocity()`:
1. Copy `solver.velocityBuffers.u` and `.v` to staging buffers
2. Map and read into Float32Arrays `this.uData` and `this.vData`
3. Called every 5th frame (use a frame counter), only when streamlines or velocities are enabled

- [ ] **Step 2: Add streamline computation**

Method `computeStreamlines()`:
1. Create seed points: every 5th cell in x and y, at cell centers
2. For each seed, integrate with RK4 for 15 steps:
   - At each step: evaluate velocity at current position, compute 4 RK stages, advance position
   - Break if position exits domain
   - Accumulate points
3. Helper `sampleVelocity(x, y, field)`: bilinear interpolation from `this.uData`/`this.vData` — same logic as original `sampleField()` (reference lines 166-204)
4. Create/update `THREE.LineSegments` from accumulated points
5. Material: `LineBasicMaterial({ color: 0x000000 })`, renderOrder 3

- [ ] **Step 3: Add velocity arrows**

Method `updateVelocityArrows()`:
1. Create arrow geometry: simple triangle + line (or cone + cylinder)
2. Create `THREE.InstancedMesh` with arrow geometry
3. For each arrow (every 5th cell):
   - Position at cell center
   - Rotation from `Math.atan2(v, u)`
   - Scale proportional to velocity magnitude (clamped)
4. Update instance matrices each readback
5. Material: `MeshBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.6 })`, renderOrder 2

- [ ] **Step 4: Integrate into draw()**

```javascript
this.frameCount++;
if (this.frameCount % 5 === 0 && (this.showStreamlines || this.showVelocities)) {
    await this.readbackVelocity();
    if (this.showStreamlines) this.computeStreamlines();
    if (this.showVelocities) this.updateVelocityArrows();
}
this.streamlinesMesh.visible = this.showStreamlines;
this.arrowsMesh.visible = this.showVelocities;
```

- [ ] **Step 5: Verify in browser**

- Check "Streamlines": black lines showing flow direction, updating every 5 frames
- Check "Velocities": gray arrows scaled by speed
- Uncheck both: no lines or arrows

- [ ] **Step 6: Commit**

```bash
git add static/js/renderer.js
git commit -m "feat: add streamline and velocity arrow visualization layers"
```

---

### Task 12: Obstacle Overlay & Performance HUD

**Files:**
- Modify: `static/js/renderer.js` (add obstacle mesh)
- Modify: `static/js/main.js` (perf HUD updates)
- Modify: `static/css/style.css` (tier indicator styling)

**Dependencies:** Task 11 (renderer layers)

- [ ] **Step 1: Add obstacle overlay to renderer**

Create obstacle meshes:
- Circle: `THREE.CircleGeometry(radius, 32)`
- Square: `THREE.PlaneGeometry(size, size)` with rotation
- Airfoil: custom `THREE.Shape` tracing NACA 0012 profile, `THREE.ShapeGeometry`
- Wedge: custom `THREE.Shape` triangular profile

Two meshes per shape:
- Fill: `MeshBasicMaterial({ color: 0xdddddd })` (or `0x000000` when showing pressure)
- Outline: `THREE.LineLoop` with `LineBasicMaterial({ color: 0x000000 })`

RenderOrder 4. Method `updateObstacle(shape, x, y, params)`: show correct shape, position it.
Wire to interaction.js: call `renderer.updateObstacle()` on drag.

- [ ] **Step 2: Add performance HUD updates**

In main.js, update `#perf-hud` every 10th frame:

```javascript
const perfHud = document.getElementById('perf-hud');
let frameTimeSmoothed = 0;
let hudCounter = 0;

// inside frame():
frameTimeSmoothed = frameTimeSmoothed * 0.9 + frameTime * 0.1;
hudCounter++;
if (hudCounter % 10 === 0) {
    perfHud.textContent =
        frameTimeSmoothed.toFixed(1) + ' ms/frame | ' +
        Math.round(1000/frameTimeSmoothed) + ' fps\n' +
        'grid: ' + solver.numX + 'x' + solver.numY +
        ' | iters: ' + ui.numIters;
}
```

- [ ] **Step 3: Add tier change indicator CSS**

```css
.tier-indicator {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 24px;
    font-family: monospace;
    color: #7ee787;
    background: rgba(0,0,0,0.7);
    padding: 8px 16px;
    border-radius: 8px;
    animation: fadeOut 1.5s forwards;
}
@keyframes fadeOut { to { opacity: 0; } }
```

In `renderer.showTierChange(tier)`: create a div with class `tier-indicator`, append to canvas container, remove after 1.5s via `setTimeout`.

- [ ] **Step 4: Verify in browser**

- Drag obstacle: shape overlay follows cursor
- Perf HUD shows frame time and fps, updates smoothly
- Trigger adaptive resize: tier indicator appears briefly

- [ ] **Step 5: Commit**

```bash
git add static/js/renderer.js static/js/main.js static/css/style.css
git commit -m "feat: add obstacle overlay and performance HUD"
```

---

### Task 13: Smoke Injection & Final Integration

**Files:**
- Modify: `static/js/fluid-solver.js` (smoke injection convenience)
- Modify: `static/js/main.js` (final wiring)
- Modify: `static/js/interaction.js` (sandbox paint mode)

**Dependencies:** Task 12 (all visual layers exist)

- [ ] **Step 1: Add smoke injection**

For wind tunnel presets, smoke is injected each frame at the inlet. Add a `injectSmoke(cells)` method to FluidSolver that writes specific cells of the `m` buffer.

In the animation loop, before `solver.step()`, if the current preset has a smoke inlet, write `m = 0.0` for the central 10% of cells at x=0 (dark dye at inlet).

Inlet definition matches reference code lines 386-391.

- [ ] **Step 2: Add sandbox paint mode**

In interaction.js, if `currentPreset.paintMode === true`:
- Dragging writes smoke values at obstacle cells: `m[i*n+j] = 0.5 + 0.5 * Math.sin(0.1 * frameNr)`
- This creates the oscillating dye effect from the original solver (reference line 662)

- [ ] **Step 3: End-to-end verification**

Open `http://localhost:8000` and verify the complete experience:

1. Wind Tunnel loads by default — smoke flows past circle, pressure colormap visible
2. Click each preset: simulation resets correctly, appropriate visualization
3. Drag obstacle: fluid reacts, obstacle overlay follows cursor
4. Toggle visualization checkboxes: layers appear/disappear
5. Change shape: next drag uses new shape
6. Open advanced panel: sliders adjust in real time
7. Keyboard shortcuts work (P, M, 1-6)
8. Performance HUD shows reasonable numbers
9. Colorbar updates with correct min/max
10. Sandbox mode: dragging paints colorful dye

- [ ] **Step 4: Commit**

```bash
git add static/js/
git commit -m "feat: add smoke injection, sandbox paint mode, and final integration"
```

---

## Post-v1 Stretch Goals (not in this plan)

- Jacobi solver toggle in advanced panel
- `minmax.wgsl` GPU reduction (replace CPU min/max)
- Zero-copy storage texture path (eliminate staging buffer readback for field quad)
- Inferno colormap
- More NACA airfoil profiles (currently only 0012)
- WebSocket-based live parameter control from server
- Snapshot save/load via FastAPI endpoints
