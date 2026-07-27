# GPU-Side Obstacle Rasterization (PR A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move obstacle rasterization into a GPU compute shader and delete the stale CPU field mirrors, fixing the field-reset defect (every drag/slider-input restarts the flow) and the drag-time upload-cost gap.

**Architecture:** New `rasterize_obstacle.wgsl` compute shader dispatched once per rotation slot (3 dispatches, 6 storage buffers each — under the documented 8-per-stage limit). A solver-owned boundary-mask buffer (`sBoundary`), uploaded once per preset load, tells the shader what to restore vacated cells to. The CPU mirrors `_uData`/`_vData`/`_sData`, the dead `paintMode` path, and the per-cell `writeSmokeCell` API are deleted. The inflow-slider path switches to a bounded column-1 write.

**Tech Stack:** WebGPU compute (WGSL), vanilla JS modules, Playwright (headed Chromium, `--enable-unsafe-webgpu`) for tests.

**Spec:** `docs/ROADMAP.md` PR A bullet (7 build notes) + `docs/adr/0010-gpu-side-obstacle-rasterization.md`. Read both before Task 1.

## Global Constraints

- Work on branch `feat/gpu-obstacle-rasterization` (already exists; the spec commit `6467e94` is its tip).
- Run tests with `npx playwright test tests/solver.spec.js -g "<test name substring>"` (the config auto-starts the server on port 8321). Tests run HEADED Chromium — on a headless machine prefix with `xvfb-run -a`.
- Full suite before finishing: `npm test` (or `xvfb-run -a npm test`).
- **Port the CPU geometry exactly.** The WGSL inside-tests must reproduce `interaction.js`'s pre-PR-A cell-center tests (circle, square, NACA 0012, wedge, compass-needle rotation). The deleted CPU code survives as the test oracle in Task 2.
- **Newly-fluid cell semantics (spec note 3):** vacated cells get zero velocity, zero pressure, Smoke = 1.0; boundary cells keep their BC values and are never carved.
- Tests use the house style: `boot(page)`, `page.evaluate` against `window.__flowlab`, inline `readBuf` helpers, bit-exact assertions where the mechanism is exact, and a comment recording what mutation each assertion catches.
- No new measured-number claims anywhere. This PR changes no physics.
- `interaction.rasterizeObstacle(centerX, centerY, vx, vy)` keeps its exact signature — `tests/diagnostics.spec.js:1337` and `ui.js`'s shape picker call it.

---

### Task 1: Boundary-mask buffer on the GPU

**Files:**
- Modify: `static/js/fluid-solver.js` — `_createBuffers` (~line 85), `destroy` (~line 202), `writeBoundaryMask` next to `writeSolidMask` (~line 691)
- Modify: `static/js/presets.js` — `loadPreset`, right after `solver.writeSolidMask(sData)` (~line 161)
- Test: `tests/solver.spec.js`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `solver.sBoundary` (GPUBuffer, `numX*numY*4` bytes, same usage flags as `solver.s`); `solver.writeBoundaryMask(data: Float32Array)`. Later tasks and tests rely on both names.

- [ ] **Step 1: Write the failing test**

Append to `tests/solver.spec.js`:

```js
test('the boundary mask buffer holds the preset boundary mask after load', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    const { numX, numY } = solver;
    const n = numY;
    const size = numX * numY * 4;
    const readBuf = async (src) => {
      const staging = device.createBuffer({
        size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    };
    const sb = await readBuf(solver.sBoundary);
    const s = await readBuf(solver.solidBuffer);

    // s == boundary mask everywhere except the obstacle footprint, which is
    // rasterized into s only.
    let diffs = 0;
    for (let k = 0; k < sb.length; k++) if (sb[k] !== s[k]) diffs++;
    // Every diff must be sBoundary fluid (1) -> s solid (0): the rasterizer
    // adds solids, it never removes the boundary's.
    let badDiff = 0;
    for (let k = 0; k < sb.length; k++) {
      if (sb[k] !== s[k] && !(sb[k] === 1 && s[k] === 0)) badDiff++;
    }
    // The permanent walls: i = 0 column is all solid in the boundary mask.
    let col0Solid = true;
    for (let j = 0; j < n; j++) if (sb[0 * n + j] !== 0) col0Solid = false;
    return { diffs, badDiff, col0Solid };
  });
  // Guards against a vacuous test: the Kármán circle is ~46 cells even at
  // tier 64, so a missing obstacle readback can't sneak past.
  expect(r.diffs).toBeGreaterThan(10);
  expect(r.badDiff).toBe(0);
  expect(r.col0Solid).toBe(true);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test tests/solver.spec.js -g "boundary mask buffer"`
Expected: FAIL — `solver.sBoundary` is `undefined`, so `copyBufferToBuffer` throws inside `page.evaluate`.

- [ ] **Step 3: Implement the buffer and upload**

In `static/js/fluid-solver.js` `_createBuffers`, after the `this.s = ...` line, add:

```js
    // Boundary mask: the preset's permanent solids (walls, step), uploaded
    // once per preset load. The obstacle rasterizer reads it to restore
    // vacated cells and to never carve permanent boundary cells.
    this.sBoundary = device.createBuffer({ size: size * 4, usage: storageUsage });
```

In `destroy`, after `this.s.destroy();` add:

```js
    this.sBoundary.destroy();
```

Next to `writeSolidMask` add:

```js
  writeBoundaryMask(data) { this.device.queue.writeBuffer(this.sBoundary, 0, data); }
```

In `static/js/presets.js` `loadPreset`, immediately after `solver.writeSolidMask(sData);` add:

```js
  solver.writeBoundaryMask(sData); // permanent solids only — obstacle not yet rasterized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx playwright test tests/solver.spec.js -g "boundary mask buffer"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add static/js/fluid-solver.js static/js/presets.js tests/solver.spec.js
git commit -m "feat: solver-owned boundary-mask buffer, uploaded per preset load"
```

---

### Task 2: The rasterizer shader and `solver.rasterizeObstacle()`

**Files:**
- Create: `static/shaders/rasterize_obstacle.wgsl`
- Modify: `static/js/fluid-solver.js` — `create` (~line 224 fetch list, ~line 330 BGL block, ~line 346 pipeline block), `_createBuffers` (uniform buffer, ~line 118), `destroy` (~line 207), `_createBindGroups` (end, ~line 476), new method next to `writeInflowColumn` (~line 710)
- Test: `tests/solver.spec.js`

**Interfaces:**
- Consumes: `solver.sBoundary`, `solver.writeBoundaryMask` (Task 1).
- Produces: `solver.rasterizeObstacle(o)` where `o = { shape: number, centerX: number, centerY: number, vx: number, vy: number, radius: number, angle: number, prevBBox: [iMin, iMax, jMin, jMax] | null }`. Shape enum: `0 = circle, 1 = square, 2 = airfoil, 3 = wedge`. `prevBBox: null` (or iMin > iMax) means "no previous footprint — skip restore"; callers must ensure `s` already equals the boundary mask in that case (preset load does). Task 3's interaction cutover and the Task 2/3 tests call exactly this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/solver.spec.js` (two tests, one shared inline oracle):

```js
// ---------------------------------------------------------------------------
// GPU obstacle rasterizer (PR A). The oracle below is the CPU inside-test
// code deleted from interaction.js, transcribed verbatim: same cell-center
// coordinates, same rotation, same NACA 0012 coefficients. Positions and
// angles are chosen off grid lines and away from axis alignment so no cell
// center sits within f32/f64 disagreement of a shape boundary — exact-match
// comparison pins the GEOMETRY, not floating-point noise.
// ---------------------------------------------------------------------------

test('the GPU rasterizer matches the CPU oracle mask and wall velocity', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, interaction } = window.__flowlab;
    solver.paused = true;
    const { numX, numY, h } = solver;
    const n = numY;
    const size = numX * numY * 4;
    const readBuf = async (src) => {
      const staging = device.createBuffer({
        size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    };

    const sBoundaryArr = await readBuf(solver.sBoundary);

    // CPU oracle — verbatim port of the pre-PR-A interaction.js inside-tests.
    const insideAt = (shapeIdx, centerX, centerY, radius, angle, i, j) => {
      const r = radius, chord = r * 4, wedgeLen = r * 3;
      const tanHA = Math.tan(15 * Math.PI / 180);
      const cosA = Math.cos(-angle), sinA = Math.sin(-angle);
      const dx = (i + 0.5) * h - centerX;
      const dy = (j + 0.5) * h - centerY;
      const ldx = dx * cosA - dy * sinA;
      const ldy = dx * sinA + dy * cosA;
      if (shapeIdx === 0) return dx * dx + dy * dy < r * r;
      if (shapeIdx === 1) return Math.abs(ldx) < r && Math.abs(ldy) < r;
      if (shapeIdx === 2) {
        const lx = ldx + chord * 0.5;
        if (lx < 0 || lx > chord) return false;
        const xc = lx / chord;
        const yt = 5 * 0.12 * chord * (0.2969 * Math.sqrt(xc) - 0.1260 * xc
          - 0.3516 * xc * xc + 0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
        return Math.abs(ldy) < yt;
      }
      const lx = ldx + wedgeLen * 0.5;
      return lx >= 0 && lx < wedgeLen && Math.abs(ldy) < lx * tanHA;
    };

    const W = numX * h, H = numY * h;
    const cases = [
      { shape: 0, cx: 0.62 * W, cy: 0.37 * H, r: 0.055, angle: 0.73 },
      { shape: 1, cx: 0.55 * W, cy: 0.61 * H, r: 0.070, angle: 0.50 },
      { shape: 2, cx: 0.48 * W, cy: 0.42 * H, r: 0.045, angle: -0.31 },
      { shape: 3, cx: 0.70 * W, cy: 0.55 * H, r: 0.060, angle: 1.19 },
    ];

    // Same conservative bounding-box formula interaction.js uses.
    const bboxOf = (cx, cy, r) => {
      const maxExtent = Math.max(r, r * 4 * 0.5, r * 3 * 0.5);
      return [
        Math.max(1, Math.floor((cx - maxExtent) / h - 1)),
        Math.min(numX - 2, Math.ceil((cx + maxExtent) / h + 1)),
        Math.max(1, Math.floor((cy - maxExtent) / h - 1)),
        Math.min(numY - 2, Math.ceil((cy + maxExtent) / h + 1)),
      ];
    };

    const results = [];
    // Thread prevBBox through the cases exactly as interaction.js will:
    // the boot obstacle's bbox first, then each case's own bbox, so every
    // case starts from the clean boundary mask.
    let prevBB = (() => {
      const p = interaction._prevBBox;
      return [p.iMin, p.iMax, p.jMin, p.jMax];
    })();
    for (const c of cases) {
      solver.rasterizeObstacle({
        shape: c.shape, centerX: c.cx, centerY: c.cy, vx: 0.7, vy: -0.3,
        radius: c.r, angle: c.angle, prevBBox: prevBB,
      });
      prevBB = bboxOf(c.cx, c.cy, c.r);
      const s = await readBuf(solver.solidBuffer);
      const u = await readBuf(solver.velPairs[solver._velCur].u);
      const v = await readBuf(solver.velPairs[solver._velCur].v);

      let maskMismatch = 0, carvedBoundary = 0, uMismatch = 0, vMismatch = 0;
      for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
          const idx = i * n + j;
          const bnd = sBoundaryArr[idx] === 0;
          const inHere = !bnd && insideAt(c.shape, c.cx, c.cy, c.r, c.angle, i, j);
          const inLeft = i > 0 && sBoundaryArr[(i - 1) * n + j] !== 0
            && insideAt(c.shape, c.cx, c.cy, c.r, c.angle, i - 1, j);
          const expectS = bnd ? 0 : (inHere ? 0 : 1);
          if (s[idx] !== expectS) maskMismatch++;
          if (bnd && s[idx] !== 0) carvedBoundary++;
          // u faces: inside cells AND faces right of an inside cell carry vx.
          if (inHere || inLeft) { if (u[idx] !== 0.7) uMismatch++; }
          // v: cell-owned only — the CPU rasterizer writes no neighbour v face.
          if (inHere) { if (v[idx] !== -0.3) vMismatch++; }
        }
      }
      results.push({ shape: c.shape, maskMismatch, carvedBoundary, uMismatch, vMismatch });
    }
    return results;
  });
  for (const res of r) {
    expect(res.maskMismatch, `shape ${res.shape} mask`).toBe(0);
    expect(res.carvedBoundary, `shape ${res.shape} boundary`).toBe(0);
    expect(res.uMismatch, `shape ${res.shape} u`).toBe(0);
    expect(res.vMismatch, `shape ${res.shape} v`).toBe(0);
  }
});

test('a vacated footprint is restored: fluid, zero velocity/pressure, smoke cleared', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, interaction } = window.__flowlab;
    solver.paused = true;
    const { numX, numY, h } = solver;
    const n = numY;
    const size = numX * numY * 4;
    const readBuf = async (src) => {
      const staging = device.createBuffer({
        size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    };

    const sBoundaryArr = await readBuf(solver.sBoundary);

    // Square oracle (same port as the test above).
    const insideSquare = (centerX, centerY, radius, angle, i, j) => {
      const cosA = Math.cos(-angle), sinA = Math.sin(-angle);
      const dx = (i + 0.5) * h - centerX;
      const dy = (j + 0.5) * h - centerY;
      const ldx = dx * cosA - dy * sinA;
      const ldy = dx * sinA + dy * cosA;
      return Math.abs(ldx) < radius && Math.abs(ldy) < radius;
    };

    const W = numX * h, H = numY * h;
    const rSq = 0.07;
    const A = { cx: 0.50 * W, cy: 0.50 * H, angle: 0.50 };
    const B = { cx: 0.56 * W, cy: 0.57 * H, angle: 0.50 };

    const bboxOf = (cx, cy) => {
      const maxExtent = Math.max(rSq, rSq * 4 * 0.5, rSq * 3 * 0.5);
      return [
        Math.max(1, Math.floor((cx - maxExtent) / h - 1)),
        Math.min(numX - 2, Math.ceil((cx + maxExtent) / h + 1)),
        Math.max(1, Math.floor((cy - maxExtent) / h - 1)),
        Math.min(numY - 2, Math.ceil((cy + maxExtent) / h + 1)),
      ];
    };
    const bbA = bboxOf(A.cx, A.cy);
    const bbB = bboxOf(B.cx, B.cy);

    // Dirty the state first: non-zero pressure and smoke=0 (dye) everywhere,
    // so "restored to zero / cleared" cannot pass vacuously.
    const pDirty = new Float32Array(numX * numY).fill(3.25);
    device.queue.writeBuffer(solver.p, 0, pDirty);
    const smokeDirty = new Float32Array(numX * numY).fill(0.0);
    for (const b of solver.smokeBufs) device.queue.writeBuffer(b, 0, smokeDirty);

    const prev = interaction._prevBBox;
    solver.rasterizeObstacle({
      shape: 1, centerX: A.cx, centerY: A.cy, vx: 0.7, vy: -0.3, radius: rSq,
      angle: A.angle, prevBBox: [prev.iMin, prev.iMax, prev.jMin, prev.jMax],
    });
    solver.rasterizeObstacle({
      shape: 1, centerX: B.cx, centerY: B.cy, vx: 0.4, vy: 0.2, radius: rSq,
      angle: B.angle, prevBBox: bbA,
    });

    const s = await readBuf(solver.solidBuffer);
    const u = await readBuf(solver.velPairs[solver._velCur].u);
    const v = await readBuf(solver.velPairs[solver._velCur].v);
    const p = await readBuf(solver.pressureBuffer);
    const smoke = await readBuf(solver.smokeBufs[solver._smokeCur]);

    let sBad = 0, uBad = 0, vBad = 0, pBad = 0, smokeBad = 0, boundaryCarved = 0, checked = 0;
    for (let i = bbA[0]; i <= bbA[1]; i++) {
      for (let j = bbA[2]; j <= bbA[3]; j++) {
        const inB = i >= bbB[0] && i <= bbB[1] && j >= bbB[2] && j <= bbB[3]
          && insideSquare(B.cx, B.cy, rSq, B.angle, i, j);
        // A cell in A∩B stays solid — checked by the oracle test. A cell
        // whose LEFT neighbour is inside B legitimately carries vx on its u
        // face (the wall-velocity face write), so it is not a "vacated, zero"
        // cell either.
        const leftInB = i > 0 && insideSquare(B.cx, B.cy, rSq, B.angle, i - 1, j);
        if (inB || leftInB) continue;
        const idx = i * n + j;
        const bnd = sBoundaryArr[idx] === 0;
        if (bnd) {
          if (s[idx] !== 0) boundaryCarved++;
          continue;
        }
        checked++;
        if (s[idx] !== 1) sBad++;
        if (u[idx] !== 0) uBad++;
        if (v[idx] !== 0) vBad++;
        if (p[idx] !== 0) pBad++;
        // Smoke cleared only where the OLD mask was solid obstacle.
        if (insideSquare(A.cx, A.cy, rSq, A.angle, i, j) && smoke[idx] !== 1.0) smokeBad++;
      }
    }
    return { sBad, uBad, vBad, pBad, smokeBad, boundaryCarved, checked };
  });
  expect(r.checked).toBeGreaterThan(50);
  expect(r.sBad).toBe(0);
  expect(r.uBad).toBe(0);
  expect(r.vBad).toBe(0);
  expect(r.pBad).toBe(0);
  expect(r.smokeBad).toBe(0);
  expect(r.boundaryCarved).toBe(0);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx playwright test tests/solver.spec.js -g "GPU rasterizer|vacated footprint"`
Expected: FAIL — `solver.rasterizeObstacle is not a function`.

- [ ] **Step 3: Create the shader**

Create `static/shaders/rasterize_obstacle.wgsl`:

```wgsl
// ============================================================================
// Obstacle rasterizer — writes the Solid Mask, obstacle velocity, and the
// vacated-footprint restore entirely on the GPU (ADR-0010). Replaces the CPU
// rasterizer whose stale whole-field mirrors reset the flow on every drag.
//
// One dispatch per rotation slot (three total), each binding that slot's
// velocity pair and smoke buffer; writes to `s` and `p` are idempotent across
// the three. 6 storage buffers per dispatch — under the 8-per-stage limit.
//
// OWNERSHIP RULE (what makes a single pass race-free): each thread writes
// ONLY its own idx in every array. The CPU rasterizer's neighbour write —
// `u[(i+1)*n+j] = vx` for an inside cell, and the symmetric zeroing of that
// face on restore — becomes a READ of the left neighbour's inside/restore
// state here. Face (i,j) is owned by cell (i,j) and decided by cells (i,j)
// and (i-1,j).
//
// RESTORE SEMANTICS (spec note 3): a vacated non-boundary cell returns to
// fluid with zero velocity and zero pressure; smoke is cleared to 1.0
// (absence) only where the OLD mask was solid. Boundary cells
// (sBoundary == 0) are never carved and never restored — they keep their BC
// values. Pressure is zeroed across the whole previous bbox, boundary cells
// included (the CPU path zeroed full column slices).
// ============================================================================

struct RasterParams {
    numX: u32,
    numY: u32,
    h: f32,
    shape: u32,      // 0 circle, 1 square, 2 airfoil (NACA 0012), 3 wedge
    center: vec2f,   // obstacle centre, sim units
    vel: vec2f,      // obstacle (drag) velocity, sim units/s
    radius: f32,
    angle: f32,      // compass-needle rotation, radians
    prevBBox: vec4i, // iMin, iMax, jMin, jMax; iMin > iMax = no previous
};

@group(0) @binding(0) var<uniform> params: RasterParams;
@group(0) @binding(1) var<storage, read_write> s: array<f32>;
@group(0) @binding(2) var<storage, read> sBoundary: array<f32>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: array<f32>;
@group(0) @binding(6) var<storage, read_write> smoke: array<f32>;

// Cell-centre inside-test, ported 1:1 from the deleted CPU rasterizer.
fn inside(i: u32, j: u32) -> bool {
    let dx = (f32(i) + 0.5) * params.h - params.center.x;
    let dy = (f32(j) + 0.5) * params.h - params.center.y;
    let cosA = cos(-params.angle);
    let sinA = sin(-params.angle);
    let ldx = dx * cosA - dy * sinA;
    let ldy = dx * sinA + dy * cosA;
    let r = params.radius;
    switch params.shape {
        case 0u: {
            return dx * dx + dy * dy < r * r;
        }
        case 1u: {
            return abs(ldx) < r && abs(ldy) < r;
        }
        case 2u: {
            let chord = r * 4.0;
            let lx = ldx + chord * 0.5;
            if (lx < 0.0 || lx > chord) { return false; }
            let xc = lx / chord;
            let yt = 5.0 * 0.12 * chord * (
                0.2969 * sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc
                + 0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
            return abs(ldy) < yt;
        }
        default: { // 3u: wedge, apex at centre, pointing right in local frame
            let wedgeLen = r * 3.0;
            let tanHA = tan(15.0 * 3.14159265 / 180.0);
            let lx = ldx + wedgeLen * 0.5;
            return lx >= 0.0 && lx < wedgeLen && abs(ldy) < lx * tanHA;
        }
    }
}

fn inPrevBBox(i: u32, j: u32) -> bool {
    let b = params.prevBBox;
    if (b.x > b.y) { return false; } // sentinel: no previous footprint
    return i >= u32(b.x) && i <= u32(b.y) && j >= u32(b.z) && j <= u32(b.w);
}

@compute @workgroup_size(8, 8)
fn rasterize(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i >= params.numX || j >= n) { return; }
    let idx = i * n + j;

    let oldS = s[idx];
    let boundaryHere = sBoundary[idx] == 0.0;
    let insideHere = !boundaryHere && inside(i, j);
    let zeroHere = !boundaryHere && inPrevBBox(i, j);

    // The left neighbour decides this thread's u face as its right face.
    var insideLeft = false;
    var zeroLeft = false;
    if (i > 0u) {
        let leftIdx = (i - 1u) * n + j;
        if (sBoundary[leftIdx] != 0.0) {
            insideLeft = inside(i - 1u, j);
            zeroLeft = inPrevBBox(i - 1u, j);
        }
    }

    // Solid mask
    if (!boundaryHere) {
        if (insideHere) { s[idx] = 0.0; }
        else if (zeroHere) { s[idx] = 1.0; }
    }

    // u: inside cells and faces right of an inside cell carry the wall
    // velocity; the restore zeroes the same set. Obstacle wins over restore
    // where they overlap (the CPU ordering: restore first, then rasterize).
    if (insideHere || insideLeft) { u[idx] = params.vel.x; }
    else if (zeroHere || zeroLeft) { u[idx] = 0.0; }

    // v: cell-owned only — the CPU rasterizer writes no neighbour v face.
    if (!boundaryHere) {
        if (insideHere) { v[idx] = params.vel.y; }
        else if (zeroHere) { v[idx] = 0.0; }
    }

    // Pressure: zeroed across the whole previous bbox, boundary included.
    if (inPrevBBox(i, j)) { p[idx] = 0.0; }

    // Smoke: cleared only where the OLD mask was solid obstacle.
    if (zeroHere && oldS == 0.0) { smoke[idx] = 1.0; }
}
```

- [ ] **Step 4: Wire the pipeline and method into the solver**

In `static/js/fluid-solver.js`:

1. In `create`, extend the fetch list (add an 8th entry) and destructure:

```js
    const [pressureWgsl, boundaryWgsl, advectWgsl, advectSmokeWgsl, maccormackWgsl, maccormackVelWgsl, diffuseWgsl, rasterizeWgsl] = await Promise.all([
      fetch('/shaders/pressure.wgsl').then(r => r.text()),
      fetch('/shaders/boundary.wgsl').then(r => r.text()),
      fetch('/shaders/advect.wgsl').then(r => r.text()),
      fetch('/shaders/advect_smoke.wgsl').then(r => r.text()),
      fetch('/shaders/maccormack.wgsl').then(r => r.text()),
      fetch('/shaders/maccormack_velocity.wgsl').then(r => r.text()),
      fetch('/shaders/diffuse.wgsl').then(r => r.text()),
      fetch('/shaders/rasterize_obstacle.wgsl').then(r => r.text()),
    ]);
```

```js
    const rasterizeMod     = device.createShaderModule({ code: rasterizeWgsl });
```

2. In the BGL block (after `_diffuseBGL`):

```js
    // Obstacle rasterizer: uniform + s(rw) + sBoundary(ro) + u,v(rw) + p(rw)
    // + smoke(rw) = 6 storage buffers, under maxStorageBuffersPerShaderStage
    // (8). One dispatch per rotation slot; writes to s and p are idempotent.
    solver._rasterizeBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, STORAGE), bglEntry(4, STORAGE), bglEntry(5, STORAGE),
                bglEntry(6, STORAGE)],
    });
```

3. In the pipeline block:

```js
    solver.rasterizePipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._rasterizeBGL), compute: { module: rasterizeMod,   entryPoint: 'rasterize' } });
```

4. In `_createBuffers`, next to the other uniform buffers:

```js
    // Rasterizer uniforms: 64 bytes. Layout must match RasterParams in
    // rasterize_obstacle.wgsl: numX(0) numY(4) h(8) shape(12) center(16,20)
    // vel(24,28) radius(32) angle(36) pad(40-48) prevBBox vec4i(48-64).
    this.uniformBufRaster = device.createBuffer({ size: 64, usage: uniformUsage });
```

5. In `destroy`, after `this.uniformBufVisc.destroy();`:

```js
    this.uniformBufRaster.destroy();
```

6. At the end of `_createBindGroups` (before the `this._velCur = 0;` lines):

```js
    // One rasterizer bind group per rotation slot.
    this.rasterize = [];
    for (let k = 0; k < 3; k++) {
      this.rasterize.push(device.createBindGroup({
        layout: this._rasterizeBGL,
        entries: [entry(0, this.uniformBufRaster), entry(1, this.s), entry(2, this.sBoundary),
                  entry(3, this.velPairs[k].u), entry(4, this.velPairs[k].v),
                  entry(5, this.p), entry(6, this.smokeBufs[k])],
      }));
    }
```

7. New method, next to `writeInflowColumn`:

```js
  /**
   * Rasterizes the obstacle on the GPU: writes the new footprint into the
   * solid mask with the drag velocity, restores the previous footprint from
   * the boundary mask (zero velocity/pressure, smoke cleared), in every
   * rotation slot. One uniform upload + three dispatches; no CPU field
   * arrays are involved, so the live field outside the footprints is
   * untouched (the field-reset defect, ADR-0010).
   *
   * @param {Object} o
   * @param {number} o.shape - 0 circle, 1 square, 2 airfoil, 3 wedge
   * @param {number} o.centerX - obstacle centre X, sim units
   * @param {number} o.centerY - obstacle centre Y, sim units
   * @param {number} o.vx - obstacle velocity X (drag), sim units/s
   * @param {number} o.vy - obstacle velocity Y (drag), sim units/s
   * @param {number} o.radius - shape radius, sim units
   * @param {number} o.angle - rotation, radians
   * @param {number[]|null} o.prevBBox - [iMin,iMax,jMin,jMax] or null for
   *   "no previous footprint" (first rasterize after preset load; `s` must
   *   already equal the boundary mask).
   */
  rasterizeObstacle(o) {
    const ab = new ArrayBuffer(64);
    const dv = new DataView(ab);
    dv.setUint32(0, this.numX, true);
    dv.setUint32(4, this.numY, true);
    dv.setFloat32(8, this.h, true);
    dv.setUint32(12, o.shape, true);
    dv.setFloat32(16, o.centerX, true);
    dv.setFloat32(20, o.centerY, true);
    dv.setFloat32(24, o.vx, true);
    dv.setFloat32(28, o.vy, true);
    dv.setFloat32(32, o.radius, true);
    dv.setFloat32(36, o.angle, true);
    const bb = o.prevBBox ?? [1, 0, 0, 0]; // iMin > iMax = no previous
    dv.setInt32(48, bb[0], true);
    dv.setInt32(52, bb[1], true);
    dv.setInt32(56, bb[2], true);
    dv.setInt32(60, bb[3], true);
    this.device.queue.writeBuffer(this.uniformBufRaster, 0, ab);

    const encoder = this.device.createCommandEncoder();
    const dx = Math.ceil(this.numX / 8);
    const dy = Math.ceil(this.numY / 8);
    for (let k = 0; k < 3; k++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rasterizePipeline);
      pass.setBindGroup(0, this.rasterize[k]);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx playwright test tests/solver.spec.js -g "GPU rasterizer|vacated footprint"`
Expected: PASS (both)

- [ ] **Step 6: Commit**

```bash
git add static/shaders/rasterize_obstacle.wgsl static/js/fluid-solver.js tests/solver.spec.js
git commit -m "feat: GPU obstacle rasterizer — shader, pipeline, solver.rasterizeObstacle"
```

---

### Task 3: Bounded inflow-slider write

**Files:**
- Modify: `static/js/ui.js` — `_setInflowVelocity` (~line 606-628)
- Test: `tests/solver.spec.js`

**Interfaces:**
- Consumes: `solver.writeInflowColumn(col, data, srcOffset, count)` (existing).
- Produces: `_setInflowVelocity(inVel)` with no reference to `interaction._uData` or `interaction.boundaryMask` — Task 4 deletes both, so this task must land first.

- [ ] **Step 1: Write the failing test**

Append to `tests/solver.spec.js`:

```js
test('the inflow slider writes only column 1, in every rotation slot', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;
    const { numX, numY } = solver;
    const n = numY;
    const size = numX * numY * 4;
    const readBuf = async (src) => {
      const staging = device.createBuffer({
        size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    };

    const before = [];
    for (const pair of solver.velPairs) before.push(await readBuf(pair.u));

    ui._setInflowVelocity(2.5);

    const after = [];
    for (const pair of solver.velPairs) after.push(await readBuf(pair.u));

    // The regression this pins: _setInflowVelocity used to rebuild a whole
    // field from the STALE CPU mirror and push it over the live one — a
    // second instance of the field-reset defect. Outside column 1 every slot
    // must be bit-identical to its own pre-slider state.
    let outsideDrift = 0;
    for (let k = 0; k < 3; k++) {
      for (let idx = 0; idx < before[k].length; idx++) {
        const i = Math.floor(idx / n);
        if (i === 1) continue;
        if (after[k][idx] !== before[k][idx]) outsideDrift++;
      }
    }
    // Column 1 itself: the new inflow in all three slots, and the persistent
    // boundaryVelData slice the per-frame re-application reads.
    let col1Bad = 0;
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < n; j++) if (after[k][1 * n + j] !== 2.5) col1Bad++;
    }
    let bvBad = 0;
    for (let j = 0; j < n; j++) {
      if (ui.boundaryVelData.uData[1 * n + j] !== 2.5) bvBad++;
    }
    return { outsideDrift, col1Bad, bvBad };
  });
  expect(r.outsideDrift).toBe(0);
  expect(r.col1Bad).toBe(0);
  expect(r.bvBad).toBe(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test tests/solver.spec.js -g "inflow slider writes only column 1"`
Expected: FAIL — `outsideDrift` is large (the stale-mirror push resets the field).

- [ ] **Step 3: Replace `_setInflowVelocity`**

In `static/js/ui.js`, replace the whole `_setInflowVelocity` method with:

```js
    /**
     * Update the inflow velocity at column i=1 on the GPU and in the
     * persistent boundaryVelData so it survives across frames. Bounded: one
     * column write per rotation slot via writeInflowColumn — the old path
     * rebuilt a whole field from interaction's stale CPU mirror and pushed
     * it over the live one (a second instance of the field-reset defect).
     * @param {number} inVel - New inflow velocity value
     */
    _setInflowVelocity(inVel) {
        if (!this.boundaryVelData) return;
        const n = this.solver.numY;
        const col = this.boundaryVelData.uData;
        for (let j = 0; j < n; j++) col[1 * n + j] = inVel;
        this.solver.writeInflowColumn(1, col, 1 * n, n);
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx playwright test tests/solver.spec.js -g "inflow slider writes only column 1"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add static/js/ui.js tests/solver.spec.js
git commit -m "fix: inflow slider writes a bounded column, not the stale whole field"
```

---

### Task 4: Interaction cutover — delete the CPU mirrors

**Files:**
- Modify: `static/js/interaction.js` — constructor (~line 10-45), `rasterizeObstacle` (~line 79-246)
- Modify: `static/js/presets.js` — `loadPreset` mirror block (~line 172-181)
- Modify: `static/js/fluid-solver.js` — delete `writeSmokeCell` (~line 716-726)
- Test: `tests/solver.spec.js`

**Interfaces:**
- Consumes: `solver.rasterizeObstacle(o)` (Task 2); Task 3's `_setInflowVelocity` (removes the last consumer of `interaction._uData` / `boundaryMask`).
- Produces: `interaction.rasterizeObstacle(centerX, centerY, vx = 0, vy = 0)` — same signature as today, called by `ui.js`'s shape picker and `tests/diagnostics.spec.js:1337`. `Interaction.SHAPES = ['circle', 'square', 'airfoil', 'wedge']` — index order is the WGSL shape enum.

- [ ] **Step 1: Write the failing test**

Append to `tests/solver.spec.js`:

```js
test('an obstacle drag leaves the field outside both bounding boxes bit-identical', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, interaction } = window.__flowlab;
    solver.paused = true;
    const { numX, numY, h } = solver;
    const n = numY;
    const size = numX * numY * 4;
    const readBuf = async (src) => {
      const staging = device.createBuffer({
        size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    };

    // Snapshot every buffer the rasterizer may touch, per slot.
    const bufs = [];
    for (let k = 0; k < 3; k++) bufs.push({ u: solver.velPairs[k].u, v: solver.velPairs[k].v, m: solver.smokeBufs[k] });
    const before = { s: await readBuf(solver.solidBuffer), p: await readBuf(solver.pressureBuffer), slots: [] };
    for (const b of bufs) before.slots.push({ u: await readBuf(b.u), v: await readBuf(b.v), m: await readBuf(b.m) });

    // A mid-field drag, like a user dragging the circle right and down.
    interaction.rasterizeObstacle(
      interaction.obstacleX + 0.15 * numX * h,
      interaction.obstacleY - 0.10 * numY * h,
      0.5, 0.25,
    );

    // Union of old and new bounding boxes (same formula interaction uses).
    // interaction.obstacleX/Y already hold the NEW centre post-drag; the old
    // centre is the drag delta back.
    const r0 = interaction.obstacleRadius;
    const maxExtent = Math.max(r0, r0 * 4 * 0.5, r0 * 3 * 0.5);
    const newBB = interaction._prevBBox;
    const oldBB = {
      iMin: Math.max(1, Math.floor((interaction.obstacleX - 0.15 * numX * h - maxExtent) / h - 1)),
      iMax: Math.min(numX - 2, Math.ceil((interaction.obstacleX - 0.15 * numX * h + maxExtent) / h + 1)),
      jMin: Math.max(1, Math.floor((interaction.obstacleY + 0.10 * numY * h - maxExtent) / h - 1)),
      jMax: Math.min(numY - 2, Math.ceil((interaction.obstacleY + 0.10 * numY * h + maxExtent) / h + 1)),
    };
    const inUnion = (i, j) =>
      (i >= newBB.iMin && i <= newBB.iMax && j >= newBB.jMin && j <= newBB.jMax) ||
      (i >= oldBB.iMin && i <= oldBB.iMax && j >= oldBB.jMin && j <= oldBB.jMax);

    const after = { s: await readBuf(solver.solidBuffer), p: await readBuf(solver.pressureBuffer), slots: [] };
    for (const b of bufs) after.slots.push({ u: await readBuf(b.u), v: await readBuf(b.v), m: await readBuf(b.m) });

    // THE field-reset regression test: outside the union bbox every buffer
    // must be bit-identical to its own pre-drag state. On the pre-PR-A code
    // the stale CPU mirrors were pushed over the whole field, so this fails
    // on essentially every non-initial cell.
    let drift = 0;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        if (inUnion(i, j)) continue;
        const idx = i * n + j;
        if (after.s[idx] !== before.s[idx]) drift++;
        if (after.p[idx] !== before.p[idx]) drift++;
        for (let k = 0; k < 3; k++) {
          if (after.slots[k].u[idx] !== before.slots[k].u[idx]) drift++;
          if (after.slots[k].v[idx] !== before.slots[k].v[idx]) drift++;
          if (after.slots[k].m[idx] !== before.slots[k].m[idx]) drift++;
        }
      }
    }
    // Three-slot fan-out: inside the new bbox the three slots agree.
    let slotSkew = 0;
    for (let i = newBB.iMin; i <= newBB.iMax; i++) {
      for (let j = newBB.jMin; j <= newBB.jMax; j++) {
        const idx = i * n + j;
        if (after.slots[0].u[idx] !== after.slots[1].u[idx] ||
            after.slots[1].u[idx] !== after.slots[2].u[idx]) slotSkew++;
        if (after.slots[0].m[idx] !== after.slots[1].m[idx] ||
            after.slots[1].m[idx] !== after.slots[2].m[idx]) slotSkew++;
      }
    }
    return { drift, slotSkew };
  });
  expect(r.drift).toBe(0);
  expect(r.slotSkew).toBe(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx playwright test tests/solver.spec.js -g "outside both bounding boxes"`
Expected: FAIL — `drift` is enormous (the whole interior resets to initial conditions).

- [ ] **Step 3: Rewrite `interaction.js`**

In the constructor, delete these lines: `this.paintMode = false;`, `this._paintFrame = 0;`, the `const size = ...` / `_sData` / `_uData` / `_vData` block, and `this.boundaryMask = null;`. Add the shape table to the class:

```js
export class Interaction {
    /** Shape enum order — index is the WGSL `shape` in rasterize_obstacle.wgsl. */
    static SHAPES = ['circle', 'square', 'airfoil', 'wedge'];
```

Replace the entire `rasterizeObstacle` method with:

```js
    /**
     * Rasterizes the active obstacle shape onto the solver's grid at the given
     * center position — on the GPU, via solver.rasterizeObstacle (ADR-0010).
     * The previous bounding box is handed over as uniforms; the shader
     * restores it from the boundary mask and writes the new footprint with
     * the drag velocity, in every rotation slot. No CPU field mirrors exist:
     * the live field outside both footprints is untouched.
     *
     * @param {number} centerX - Obstacle center X in simulation units.
     * @param {number} centerY - Obstacle center Y in simulation units.
     * @param {number} [vx=0] - Obstacle velocity X (from drag motion).
     * @param {number} [vy=0] - Obstacle velocity Y (from drag motion).
     */
    rasterizeObstacle(centerX, centerY, vx = 0, vy = 0) {
        this.obstacleX = centerX;
        this.obstacleY = centerY;

        const { numX, numY, h } = this.solver;
        const r = this.obstacleRadius;

        // Same conservative bounding extent the CPU rasterizer used
        const maxExtent = Math.max(r, r * 4 * 0.5, r * 3 * 0.5);
        const iMin = Math.max(1, Math.floor((centerX - maxExtent) / h - 1));
        const iMax = Math.min(numX - 2, Math.ceil((centerX + maxExtent) / h + 1));
        const jMin = Math.max(1, Math.floor((centerY - maxExtent) / h - 1));
        const jMax = Math.min(numY - 2, Math.ceil((centerY + maxExtent) / h + 1));

        this.solver.rasterizeObstacle({
            shape: Interaction.SHAPES.indexOf(this.activeShape),
            centerX, centerY, vx, vy,
            radius: r,
            angle: this.obstacleAngle,
            prevBBox: this._prevBBox
                ? [this._prevBBox.iMin, this._prevBBox.iMax, this._prevBBox.jMin, this._prevBBox.jMax]
                : null,
        });

        this._prevBBox = { iMin, iMax, jMin, jMax };
        if (this._renderer) this._renderer.invalidateSolid();
    }
```

(Everything else in the file — `screenToSim`, the pointer handlers, `_startDrag`, `_onPointerMove`, `_endDrag`, `_rotate` — is unchanged.)

In `static/js/presets.js` `loadPreset`, delete the mirror block:

```js
  // Resize interaction arrays if grid size changed
  const iSize = numX * numY;
  if (!interaction._sData || interaction._sData.length !== iSize) {
    interaction._sData = new Float32Array(iSize);
    interaction._uData = new Float32Array(iSize);
    interaction._vData = new Float32Array(iSize);
  }
  interaction.boundaryMask = sData.slice();
  interaction._uData.set(uData);
```

(Keep the `interaction._prevBBox = null;` / `interaction.obstacleAngle = 0;` lines right after it.)

In `static/js/fluid-solver.js`, delete the `writeSmokeCell` method — its only caller was the deleted paint/restore path. Verify first: `grep -n "writeSmokeCell" static/js tests` returns only the definition.

- [ ] **Step 4: Run the test, then the touched suites**

Run: `npx playwright test tests/solver.spec.js -g "outside both bounding boxes"`
Expected: PASS

Run: `npx playwright test tests/diagnostics.spec.js`
Expected: PASS (it calls `interaction.rasterizeObstacle` at line 1337 and exercises the probe-clearing path through `invalidateSolid`)

- [ ] **Step 5: Commit**

```bash
git add static/js/interaction.js static/js/presets.js static/js/fluid-solver.js tests/solver.spec.js
git commit -m "feat: interaction rasterizes on the GPU; CPU field mirrors deleted"
```

---

### Task 5: Rotation invariant across `applyTier`

**Files:**
- Test: `tests/solver.spec.js`

**Interfaces:**
- Consumes: everything above (the test exercises `resize` → buffer recreation → preset reload → rasterize on the new grid).
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `tests/solver.spec.js`:

```js
test('the three-slot rotation and boundary mask survive an applyTier buffer recreation', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, adaptive, ui, device } = window.__flowlab;
    solver.paused = true;

    adaptive.currentTierIndex = 0; // tier 64 — cheap; any tier exercises the path
    adaptive.applyTier(); // resize -> reapplyCurrentPreset -> resetFlipState

    const afterReset = { vel: solver._velCur, smoke: solver._smokeCur };
    const seq = [];
    for (let k = 0; k < 3; k++) {
      solver.step(ui.numIters);
      seq.push([solver._velCur, solver._smokeCur]);
    }

    // The boundary-mask buffer was recreated at the new grid size and
    // re-uploaded by loadPreset: its i=0 column is solid on the NEW grid.
    const size = solver.numX * solver.numY * 4;
    const staging = device.createBuffer({
      size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(solver.sBoundary, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const sb = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    let col0Solid = true;
    for (let j = 0; j < solver.numY; j++) if (sb[0 * solver.numY + j] !== 0) col0Solid = false;

    // And the rasterizer works on the recreated buffers: rasterize once and
    // count solid non-boundary cells (the obstacle).
    solver.rasterizeObstacle({
      shape: 0, centerX: 0.4 * solver.numX * solver.h, centerY: 0.5 * solver.numY * solver.h,
      vx: 0, vy: 0, radius: 0.06, angle: 0,
      prevBBox: [1, 0, 0, 0], // s is already the fresh boundary mask post-reload
    });
    const staging2 = device.createBuffer({
      size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(solver.solidBuffer, 0, staging2, 0, size);
    device.queue.submit([enc2.finish()]);
    await staging2.mapAsync(GPUMapMode.READ);
    const s = new Float32Array(staging2.getMappedRange().slice(0));
    staging2.unmap();
    staging2.destroy();
    let extraSolids = 0;
    for (let k = 0; k < s.length; k++) if (s[k] === 0 && sb[k] !== 0) extraSolids++;

    return { afterReset, seq, col0Solid, extraSolids, sBoundarySize: solver.sBoundary.size, expectSize: size };
  });
  expect(r.afterReset).toEqual({ vel: 0, smoke: 0 });
  expect(r.seq).toEqual([[2, 2], [1, 1], [0, 0]]);
  expect(r.col0Solid).toBe(true);
  expect(r.sBoundarySize).toBe(r.expectSize);
  expect(r.extraSolids).toBeGreaterThan(10); // ~46 cells at tier 64
});
```

- [ ] **Step 2: Run test to verify it passes**

This test pins existing behavior (rotation law, `resetFlipState`) plus the new `sBoundary` recreation — it should pass immediately. If `col0Solid` or `sBoundarySize` fails, `resize`/`_createBuffers` missed the new buffer.

Run: `npx playwright test tests/solver.spec.js -g "applyTier buffer recreation"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/solver.spec.js
git commit -m "test: rotation and boundary-mask invariant across applyTier buffer recreation"
```

---

### Task 6: Full suite + docs sweep

**Files:**
- Modify: `docs/ROADMAP.md` — the two Known-gaps entries
- Modify: `docs/adr/README.md` — ADR-0010 Implemented column
- Modify: `docs/architecture.md` — lines ~94-96 (preset-load steps), ~185-220 (interaction section)
- Modify: `docs/gpu-pipeline.md` — lines ~160-172 (`s` buffer description, "Critical rule")

**Interfaces:**
- Consumes: all previous tasks.
- Produces: nothing new.

- [ ] **Step 1: Run the full suite**

Run: `npm test` (or `xvfb-run -a npm test` headless)
Expected: PASS — all four spec files. The pre-existing moving-wall tests (`velocity limiter holds the wall BC...`, `velocity advection leaves the inflow BC...`, `a step leaves solid-cell values untouched...`) must pass unchanged: the wall velocity is now written by the shader, with identical values.

- [ ] **Step 2: Update ROADMAP known gaps**

In `docs/ROADMAP.md`, rewrite the **field-reset defect** entry: prefix with `~~…~~ ✅ fixed in PR A (ADR-0010)` and keep the original text struck through, following the roadmap's own convention for completed items (see steps 0–3). Do the same for the **3-slot rotation upload-cost** entry, noting the drag path is now three dispatches + one 64-byte uniform upload per `mousemove`.

- [ ] **Step 3: Update the ADR index**

In `docs/adr/README.md`, change ADR-0010's Implemented column to `Yes` and its Status frontmatter in `0010-gpu-side-obstacle-rasterization.md` to plain `accepted`.

- [ ] **Step 4: Update the prose docs**

- `docs/architecture.md`: the preset-load list (drop "resize interaction arrays"; add the boundary-mask upload) and the Obstacle Interaction section — `rasterizeObstacle` is now a uniform upload + GPU dispatch; the restore/rasterize/smoke-clear steps happen in `rasterize_obstacle.wgsl`.
- `docs/gpu-pipeline.md`: the `s` buffer is rasterized on the GPU by `rasterize_obstacle.wgsl`; the "Critical rule" still holds for `writeU`/`writeV`/`writeSmoke`/the rasterizer's per-slot dispatches, but delete the `writeSmokeCell` mention (the method is gone).

- [ ] **Step 5: Commit and push**

```bash
git add docs/ROADMAP.md docs/adr/README.md docs/adr/0010-gpu-side-obstacle-rasterization.md docs/architecture.md docs/gpu-pipeline.md
git commit -m "docs: PR A shipped — close the field-reset and upload-cost gaps"
git push -u origin feat/gpu-obstacle-rasterization
```
