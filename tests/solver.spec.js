import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';

/**
 * Solver correctness tests.
 *
 * These drive solver.step() directly with the render loop's own stepping
 * disabled, so the number of steps between measurements is exact.
 */

async function boot(page) {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  // Let the flow develop under the normal loop before we take control
  await page.evaluate(() => new Promise((resolve) => {
    let seen = 0;
    const tick = () => (++seen >= 120 ? resolve() : requestAnimationFrame(tick));
    requestAnimationFrame(tick);
  }));
}

/**
 * Steps the solver `steps` times, returning the summed |divergence| over fluid
 * cells after each step. Divergence matches pressure.wgsl:58 exactly.
 *
 * `iters` overrides the pressure iteration count used for each step; defaults
 * to the UI's current setting (`ui.numIters`).
 */
function readDivergenceSeries(page, steps, iters) {
  return page.evaluate(async ({ steps, iters }) => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true; // stop the render loop from stepping too
    const numIters = iters ?? ui.numIters;

    const n = solver.numY, numX = solver.numX;
    const size = numX * n * 4;

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

    const sBuf = await readBuf(solver.solidBuffer);
    const series = [];

    for (let k = 0; k < steps; k++) {
      solver.step(numIters);
      const { u, v } = solver.velocityBuffers;
      const uD = await readBuf(u);
      const vD = await readBuf(v);
      let sum = 0;
      for (let i = 1; i < numX - 1; i++) {
        for (let j = 1; j < n - 1; j++) {
          if (sBuf[i * n + j] === 0) continue;
          sum += Math.abs(
            uD[(i + 1) * n + j] - uD[i * n + j] +
            vD[i * n + j + 1] - vD[i * n + j],
          );
        }
      }
      series.push(sum);
    }
    return series;
  }, { steps, iters });
}

// NOTE: this test's even/odd split was written against the old 2-cycle
// ping-pong and only distinguishes parity. Under the 3-slot rotation
// (a9b78f5), a rotation off-by-one (e.g. advancing by 2 instead of 1) produces
// period-3 corruption that aliases only weakly into an even/odd split against
// the 0.15 threshold below -- so a pass here is much weaker evidence about the
// rotation than it looks. See the _velCur/_smokeCur and smoke bind group
// tests below for assertions that pin the rotation itself.
test('projection is applied every step, not every other step', async ({ page }) => {
  await boot(page);
  const series = await readDivergenceSeries(page, 20);

  // Discard the first two entries: the transition into manual stepping can
  // land on either parity, and the very first step inherits loop state.
  const s = series.slice(2);
  const even = s.filter((_, i) => i % 2 === 0);
  const odd  = s.filter((_, i) => i % 2 === 1);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

  const overall = mean(s);
  expect(overall).toBeGreaterThan(0); // guard: a dead field makes this vacuous

  // With the bug, projection lands on alternate steps, so post-step divergence
  // alternates between "just projected" and "two advections since projection".
  const imbalance = Math.abs(mean(even) - mean(odd)) / overall;
  expect(imbalance).toBeLessThan(0.15);
});

test('the three MacCormack smoke bind group tables index velocity independently of smoke', async ({ page }) => {
  await boot(page);
  const report = await page.evaluate(() => {
    const { solver, device } = window.__flowlab;

    // A GPUBindGroup is opaque -- nothing on it reports which buffer landed on
    // which binding -- so re-create them with createBindGroup patched to stash
    // each descriptor. Recording the descriptors is the only way to assert on
    // the wiring rather than on object identity. Comparing bind group objects
    // (`a !== b`) proves nothing: createBindGroup returns a fresh object every
    // call, so such a comparison holds for ANY implementation, including one
    // that aliases the velocity binding to the smoke index.
    const orig = device.createBindGroup.bind(device);
    device.createBindGroup = (desc) => {
      const bg = orig(desc);
      bg.__desc = desc;
      return bg;
    };
    try { solver._createBindGroups(); } finally { device.createBindGroup = orig; }
    const buf = (bg, binding) =>
      bg.__desc.entries.find((e) => e.binding === binding).resource.buffer;

    const TABLES = ['smokeFwd', 'smokeBack', 'smokeCombine'];
    const shape = TABLES.map((t) => [solver[t].length, solver[t][0].length]);

    // Bindings 1 and 2 are the advecting velocity in all three layouts.
    const wrongVelocity = [];
    // Slot roles the smoke axis must follow: n = sc, hat = sc+1, tilde = sc+2.
    const wrongSmoke = [];
    for (let vc = 0; vc < 3; vc++) {
      for (let sc = 0; sc < 3; sc++) {
        const n = solver.smokeBufs[sc];
        const hat = solver.smokeBufs[(sc + 1) % 3];
        const tilde = solver.smokeBufs[(sc + 2) % 3];
        for (const t of TABLES) {
          const bg = solver[t][vc][sc];
          if (buf(bg, 1) !== solver.velPairs[vc].u || buf(bg, 2) !== solver.velPairs[vc].v) {
            wrongVelocity.push(`${t}[${vc}][${sc}]`);
          }
        }
        const fwd = solver.smokeFwd[vc][sc];
        const back = solver.smokeBack[vc][sc];
        const comb = solver.smokeCombine[vc][sc];
        if (buf(fwd, 4) !== n || buf(fwd, 5) !== n || buf(fwd, 6) !== hat) wrongSmoke.push(`fwd[${vc}][${sc}]`);
        if (buf(back, 4) !== hat || buf(back, 5) !== n || buf(back, 6) !== tilde) wrongSmoke.push(`back[${vc}][${sc}]`);
        if (buf(comb, 3) !== n || buf(comb, 4) !== hat || buf(comb, 5) !== tilde) wrongSmoke.push(`combine[${vc}][${sc}]`);
      }
    }

    // The backward pass is the only one that may run with negated dt; if it
    // binds the ordinary uniform buffer the correction term is identically
    // zero and MacCormack silently degrades to first-order SL.
    const backUsesNegDt = [0, 1, 2].every((vc) => [0, 1, 2].every((sc) =>
      buf(solver.smokeBack[vc][sc], 0) === solver.uniformBufNegDt));
    const fwdUsesPosDt = [0, 1, 2].every((vc) => [0, 1, 2].every((sc) =>
      buf(solver.smokeFwd[vc][sc], 0) === solver.uniformBuf &&
      buf(solver.smokeCombine[vc][sc], 0) === solver.uniformBuf));

    return { shape, wrongVelocity, wrongSmoke, backUsesNegDt, fwdUsesPosDt };
  });

  expect(report.shape).toEqual([[3, 3], [3, 3], [3, 3]]);
  // Pins that the velocity axis is genuine, not aliased to the smoke axis --
  // the exact property the 3x3 tables (vs. 3-element arrays indexed by
  // smokeCur alone) exist to provide. Reads back the actual bound buffers, so
  // binding velPairs[sc] instead of velPairs[vc] fails here.
  expect(report.wrongVelocity).toEqual([]);
  expect(report.wrongSmoke).toEqual([]);
  expect(report.backUsesNegDt).toBe(true);
  expect(report.fwdUsesPosDt).toBe(true);
});

test('_velCur and _smokeCur each advance by 2, visiting all three slots', async ({ page }) => {
  await boot(page);
  const { velSeq, smokeSeq } = await page.evaluate(() => {
    const { solver, ui } = window.__flowlab;
    solver.paused = true; // stop the render loop from stepping too
    // The +2 rotation below is the INVISCID invariant. Task 9's Re control
    // gives nu a nonzero default, and an odd substep count deliberately leaves
    // the result on the hat pair instead (+1) -- that case is pinned by
    // 'an odd substep count leaves the result on the hat pair'. State the
    // precondition rather than let the default viscosity decide which is tested.
    solver.setParams({ nu: 0 });
    solver.resetFlipState(); // deterministic start: slot 0
    const numIters = ui.numIters;

    const velSeq = [];
    const smokeSeq = [];
    for (let i = 0; i < 6; i++) {
      velSeq.push(solver._velCur);
      smokeSeq.push(solver._smokeCur);
      solver.step(numIters);
    }
    return { velSeq, smokeSeq };
  });

  // Both fields now advance by 2. MacCormack's forward pass writes
  // hat = (c+1)%3; the backward pass and the in-place combine land phi^{n+1}
  // in tilde = (c+2)%3, so tilde is the slot that becomes live. Starting at 0:
  // 0 -> 2 -> 1 -> 0 -> 2 -> 1. Still visits all three slots, since 2 is
  // coprime with 3 -- a solver that advanced by 1 would read phi^ (the
  // uncorrected forward pass) as the next step's phi^n.
  //
  // Velocity moved from +1 to +2 in Task 6, when its single semi-Lagrangian
  // dispatch became the same three-pass chain smoke already used.
  expect(velSeq).toEqual([0, 2, 1, 0, 2, 1]);
  expect(smokeSeq).toEqual([0, 2, 1, 0, 2, 1]);
});

test('the three MacCormack velocity bind group tables keep phi^n on the advecting bindings', async ({ page }) => {
  await boot(page);
  const report = await page.evaluate(() => {
    const { solver, device } = window.__flowlab;

    // Same descriptor-capture trick as the smoke wiring test above: a
    // GPUBindGroup is opaque, so re-create them with createBindGroup patched.
    const orig = device.createBindGroup.bind(device);
    device.createBindGroup = (desc) => {
      const bg = orig(desc);
      bg.__desc = desc;
      return bg;
    };
    try { solver._createBindGroups(); } finally { device.createBindGroup = orig; }
    const buf = (bg, binding) =>
      bg.__desc.entries.find((e) => e.binding === binding).resource.buffer;

    const wrong = [];
    for (let c = 0; c < 3; c++) {
      const nPair = solver.velPairs[c];
      const hat = solver.velPairs[(c + 1) % 3];
      const tilde = solver.velPairs[(c + 2) % 3];
      const fwd = solver.velFwd[c], back = solver.velBack[c], comb = solver.velCombine[c];

      // THE assertion this test exists for. All three passes must keep phi^n on
      // bindings 1/2. advect.wgsl's unreliable-trace revert writes u[idx] from
      // binding 1, so binding hat there instead -- the obvious way to build a
      // backward pass out of the old single-role kernel -- would make a reverted
      // face write phi^ and leave a spurious correction on exactly the faces
      // carrying the inflow BC. Bindings 1/2 ARE the phi^n that a separate
      // mOrig binding provides on the smoke path.
      for (const [name, bg] of [['fwd', fwd], ['back', back], ['combine', comb]]) {
        if (buf(bg, 1) !== nPair.u || buf(bg, 2) !== nPair.v) wrong.push(`${name}[${c}] phi^n`);
      }
      // Forward advects phi^n into hat; the field bindings alias phi^n.
      if (buf(fwd, 4) !== nPair.u || buf(fwd, 5) !== nPair.v ||
          buf(fwd, 6) !== hat.u || buf(fwd, 7) !== hat.v) wrong.push(`fwd[${c}]`);
      // Backward advects phi^ into tilde.
      if (buf(back, 4) !== hat.u || buf(back, 5) !== hat.v ||
          buf(back, 6) !== tilde.u || buf(back, 7) !== tilde.v) wrong.push(`back[${c}]`);
      // Combine reads phi^ and rewrites tilde in place.
      if (buf(comb, 3) !== hat.u || buf(comb, 4) !== hat.v ||
          buf(comb, 5) !== tilde.u || buf(comb, 6) !== tilde.v) wrong.push(`combine[${c}]`);
    }

    const backUsesNegDt = [0, 1, 2].every((c) =>
      buf(solver.velBack[c], 0) === solver.uniformBufNegDt);
    // The combine must re-trace FORWARD to recover the stencil the forward pass
    // interpolated over, so it takes the positive-dt uniform, not the negated one.
    const othersUsePosDt = [0, 1, 2].every((c) =>
      buf(solver.velFwd[c], 0) === solver.uniformBuf &&
      buf(solver.velCombine[c], 0) === solver.uniformBuf);

    return {
      lengths: [solver.velFwd.length, solver.velBack.length, solver.velCombine.length],
      wrong, backUsesNegDt, othersUsePosDt,
    };
  });

  expect(report.lengths).toEqual([3, 3, 3]);
  expect(report.wrong).toEqual([]);
  expect(report.backUsesNegDt).toBe(true);
  expect(report.othersUsePosDt).toBe(true);
});

test('velocity advection leaves the inflow BC and solid-cell velocities bit-exact', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;

    // Encode just the three velocity MacCormack passes -- no pressure, no
    // boundary. Isolating advection is what lets this test attribute a change
    // to the combine: a full step() also runs the pressure solve, which
    // legitimately moves the very faces under test.
    const isolatedStep = () => {
      solver._writeAllParams();
      const c = solver._velCur;
      const dx = Math.ceil(solver.numX / 8), dy = Math.ceil(solver.numY / 8);
      const enc = device.createCommandEncoder();
      for (const [pipeline, group] of [
        [solver.advectVelPipeline, solver.velFwd[c]],
        [solver.advectVelPipeline, solver.velBack[c]],
        [solver.mcVelPipeline,     solver.velCombine[c]],
      ]) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      return solver.velPairs[(c + 2) % 3];
    };

    const numX = solver.numX, n = solver.numY;
    const size = numX * n * 4;
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

    // Simulate a drag: give the solid cells a velocity of their own, the case
    // where a solid's value is a genuine moving-wall BC rather than zero.
    const sMask = await readBuf(solver.solidBuffer);
    const solids = [];
    for (let i = 1; i < numX; i++) {
      for (let j = 1; j < n; j++) if (sMask[i * n + j] === 0) solids.push(i * n + j);
    }
    for (const p of solver.velPairs) {
      for (const idx of solids) {
        device.queue.writeBuffer(p.u, idx * 4, new Float32Array([3.5]));
        device.queue.writeBuffer(p.v, idx * 4, new Float32Array([-2.25]));
      }
    }

    const c = solver._velCur;
    const uBefore = await readBuf(solver.velPairs[c].u);
    const vBefore = await readBuf(solver.velPairs[c].v);
    const out = isolatedStep();
    const uAfter = await readBuf(out.u);
    const vAfter = await readBuf(out.v);

    // Column i=1 carries the inflow (presets.js) and is preserved only
    // because i=0 is solid (presets.js), so the u-face there never advects.
    let inflowDrift = 0, inflowMag = 0;
    for (let j = 1; j < n; j++) {
      const k = 1 * n + j;
      inflowDrift = Math.max(inflowDrift, Math.abs(uAfter[k] - uBefore[k]));
      inflowMag = Math.max(inflowMag, Math.abs(uBefore[k]));
    }

    let solidDrift = 0;
    for (const idx of solids) {
      solidDrift = Math.max(solidDrift, Math.abs(uAfter[idx] - uBefore[idx]));
      solidDrift = Math.max(solidDrift, Math.abs(vAfter[idx] - vBefore[idx]));
    }

    return { inflowDrift, inflowMag, solidDrift, solidCount: solids.length };
  });

  // Guards: a zero inflow or an obstacle-free grid would make this vacuous.
  expect(r.inflowMag).toBeGreaterThan(0.1);
  expect(r.solidCount).toBeGreaterThan(50);

  // Bit-exact, not approximate. Both advect passes revert these faces to phi^n
  // (the face guard is `s[idx] != 0 && s[(i-1)*n+j] != 0`, and s[0*n+j] == 0),
  // so phi^ == phi~ == phi^n and `corrected` is exactly phi^.
  //
  // What holds the INFLOW in PRODUCTION is that revert alone -- NOT the
  // limiter's phi^ seed. u_stencil clamps x to [h, nx*h], so its i0 is never 0
  // and the i=1 u-face never samples column 0; its corner range always contains
  // u[1, j0] and u[1, j1], and presets.js writes inVel into u[1, j] for every
  // j. Both corners therefore equal the face's own value, so the clamp is the
  // identity with or without the seed.
  //
  // Removing the seed DOES fail `inflowDrift` -- by exactly 2.5 -- but read that
  // number carefully, because Task 6's report originally misread it as "the
  // entire inflow velocity". Karman's inVel is 1.0, not 2.5. The 2.5 is
  // 3.5 - 1.0, an artifact of THIS TEST's own synthetic injection above: the
  // v = -2.25 written into the solid left wall makes cv = -1.125 at every i=1
  // u-face, bending the re-traced y up ~1.2 cells until the stencil collapses
  // into the solid top row, where the test itself wrote u = 3.5. So this
  // assertion's mutation sensitivity is about the injected wall values, not
  // about the inflow mechanism. The seed's genuine production case is the
  // dragged-obstacle face class, pinned by the next test down.
  //
  // Note the revert is FACE-based while cell (1, j) is FLUID, so porting the
  // smoke combine's `if (s[idx] == 0.0)` guard would NOT fire here.
  expect(r.inflowDrift).toBe(0);
  expect(r.solidDrift).toBe(0);
});

// The one production case the phi^ seed is genuinely load-bearing for.
//
// interaction.js:202-207 rasterises a dragged obstacle by writing the drag
// velocity vx into every solid cell AND into the u-face one column to its right
// (`uData[(i+1)*n+j] = vx`). That face is FLUID -- s[(i+1)*n+j] != 0 -- so no
// cell-based solid test reaches it, and the test above does not either: it
// iterates solid CELL indices only. But the face still reverts on both advect
// passes, because the cell to its LEFT is solid, so `corrected == vx` exactly.
//
// The re-trace is where the seed earns its keep. cu = u[idx] == vx, so the
// departure point is dt*|vx| away from the face -- out in the free stream,
// where nothing is near vx. The corner range misses vx entirely, and without
// the phi^ seed the clamp snaps the face to the nearest free-stream value,
// silently destroying the moving-wall BC every drag frame.
test('the velocity limiter holds the wall BC on fluid faces right of a dragged obstacle', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;

    // Drag the obstacle UPSTREAM. The sign matters: the combine re-traces with
    // POSITIVE dt (x = i*h - dt*cu), so vx < 0 sends the departure point to the
    // RIGHT, downstream into developed wake, where u is nowhere near vx. A
    // positive vx would trace back INTO the obstacle, whose cells also hold vx,
    // and the corner range would contain vx by accident -- a vacuous test.
    const vx = -2.0, vy = 0.0;

    // See the inflow-BC test above for why advection is isolated from step().
    const isolatedStep = () => {
      solver._writeAllParams();
      const c = solver._velCur;
      const dx = Math.ceil(solver.numX / 8), dy = Math.ceil(solver.numY / 8);
      const enc = device.createCommandEncoder();
      for (const [pipeline, group] of [
        [solver.advectVelPipeline, solver.velFwd[c]],
        [solver.advectVelPipeline, solver.velBack[c]],
        [solver.mcVelPipeline,     solver.velCombine[c]],
      ]) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      return solver.velPairs[(c + 2) % 3];
    };

    const numX = solver.numX, n = solver.numY;
    const size = numX * n * 4;
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

    const sMask = await readBuf(solver.solidBuffer);

    // Replicate interaction.js's drag write exactly: vx/vy into solid cells,
    // vx into the u-face one column right of each solid cell.
    const solids = [], dragFaces = [];
    for (let i = 1; i < numX - 1; i++) {
      for (let j = 1; j < n - 1; j++) {
        if (sMask[i * n + j] !== 0) continue;
        solids.push(i * n + j);
        // The face class under test: fluid cell, solid left neighbour.
        if (sMask[(i + 1) * n + j] !== 0) dragFaces.push((i + 1) * n + j);
      }
    }
    for (const p of solver.velPairs) {
      for (const idx of solids) {
        device.queue.writeBuffer(p.u, idx * 4, new Float32Array([vx]));
        device.queue.writeBuffer(p.v, idx * 4, new Float32Array([vy]));
      }
      for (const k of dragFaces) {
        device.queue.writeBuffer(p.u, k * 4, new Float32Array([vx]));
      }
    }

    const c = solver._velCur;
    const uBefore = await readBuf(solver.velPairs[c].u);
    const vBefore = await readBuf(solver.velPairs[c].v);
    const out = isolatedStep();
    const uAfter = await readBuf(out.u);

    let dragDrift = 0;
    for (const k of dragFaces) {
      dragDrift = Math.max(dragDrift, Math.abs(uAfter[k] - uBefore[k]));
    }

    // Non-vacuity guard. Rather than replicate u_stencil (which would be a
    // fourth copy of shader logic, free to drift), locate the departure point
    // from u_departure's formula -- a 4-term average, no stencil -- and show vx
    // sits below every u the bilinear stencil there could touch.
    //
    //   x = i*h - dt*cu, cu == u[idx] == vx exactly  -> i0 = fi + floor(dxCells)
    //   y = j*h + h/2 - dt*cv                        -> j0 = fj + floor(dyCells)
    // with i1 = i0+1, j1 = j0+1. One cell of slack each way absorbs any f32-vs-
    // f64 disagreement in the floor. If every u in that 4x4 box exceeds vx by a
    // margin then lo > vx at that face, so an unseeded clamp MUST move it.
    const h = solver.h, dt = solver.params.dt;
    const dragSet = new Set(dragFaces);
    let guarded = 0, minMargin = Infinity;
    for (const k of dragFaces) {
      const fi = Math.floor(k / n), fj = k % n;
      if (fj + 1 >= n) continue;
      const cu = uBefore[k];
      const cv = 0.25 * (vBefore[(fi - 1) * n + fj] + vBefore[k] +
                         vBefore[(fi - 1) * n + fj + 1] + vBefore[fi * n + fj + 1]);
      const i0 = fi + Math.floor((-dt * cu) / h);
      const j0 = fj + Math.floor((-dt * cv) / h);
      if (i0 - 1 < 1 || i0 + 2 >= numX || j0 - 1 < 1 || j0 + 2 >= n) continue;

      // Every reachable cell must be fluid -- a solid one holds vx itself, and
      // would bracket vx for an uninteresting reason.
      let clean = true, lo = Infinity;
      for (let i = i0 - 1; i <= i0 + 2 && clean; i++) {
        for (let j = j0 - 1; j <= j0 + 2; j++) {
          // A solid cell holds vx itself. So does another drag face -- the
          // block above wrote vx into every one of them -- and it is fluid, so
          // the mask test alone does not catch it. Either bracket vx for the
          // same uninteresting reason, so both are excluded.
          if (sMask[i * n + j] === 0 || dragSet.has(i * n + j)) { clean = false; break; }
          lo = Math.min(lo, uBefore[i * n + j]);
        }
      }
      if (!clean) continue;
      guarded++;
      minMargin = Math.min(minMargin, lo - vx);
    }

    return { dragDrift, dragFaceCount: dragFaces.length, guarded, minMargin };
  });

  // Guards: no faces, or a reach box that already contains vx, would make this
  // vacuous. `minMargin > 0` is the proof that the corner range excludes vx at
  // every guarded face, so the seed -- not the corner range -- is what holds
  // them. Removing the phi^ seed from maccormack_velocity.wgsl fails this test.
  expect(r.dragFaceCount).toBeGreaterThan(10);
  expect(r.guarded).toBeGreaterThan(5);
  expect(r.minMargin).toBeGreaterThan(0.25);

  // Bit-exact, same argument as the inflow test: the face reverts on both
  // passes, so corrected == vx exactly, and the phi^ seed makes the clamp the
  // identity despite a corner range that provably excludes vx.
  expect(r.dragDrift).toBe(0);
});

test('the velocity limiter keeps the combine within the range of the field it advected', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;

    // See the inflow-BC test above for why advection is isolated from step().
    const isolatedStep = () => {
      solver._writeAllParams();
      const c = solver._velCur;
      const dx = Math.ceil(solver.numX / 8), dy = Math.ceil(solver.numY / 8);
      const enc = device.createCommandEncoder();
      for (const [pipeline, group] of [
        [solver.advectVelPipeline, solver.velFwd[c]],
        [solver.advectVelPipeline, solver.velBack[c]],
        [solver.mcVelPipeline,     solver.velCombine[c]],
      ]) {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      return solver.velPairs[(c + 2) % 3];
    };

    const numX = solver.numX, n = solver.numY;
    const size = numX * n * 4;
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

    // Let the street develop so there are sharp velocity gradients for the
    // unlimited correction to overshoot on.
    for (let k = 0; k < 60; k++) solver.step(window.__flowlab.ui.numIters);

    const c = solver._velCur;
    const uN = await readBuf(solver.velPairs[c].u);
    const vN = await readBuf(solver.velPairs[c].v);
    const out = isolatedStep();
    const uOut = await readBuf(out.u);
    const vOut = await readBuf(out.v);

    // The limited combine cannot leave the global range of phi^n: every bound
    // is a min/max over stencil values of phi^n together with phi^, and phi^ is
    // itself either a convex blend of phi^n or a reverted phi^n[idx].
    const range = (a) => {
      let lo = Infinity, hi = -Infinity;
      for (const x of a) { if (x < lo) lo = x; if (x > hi) hi = x; }
      return { lo, hi };
    };
    const ru = range(uN), rv = range(vN);

    // Only the region advect_velocity writes: i=0 and j=0 keep stale values
    // from earlier rotations, which the boundary pass refreshes each step.
    let uOver = 0, vOver = 0;
    for (let i = 1; i < numX; i++) {
      for (let j = 1; j < n; j++) {
        const k = i * n + j;
        uOver = Math.max(uOver, uOut[k] - ru.hi, ru.lo - uOut[k]);
        vOver = Math.max(vOver, vOut[k] - rv.hi, rv.lo - vOut[k]);
      }
    }
    return { uOver, vOver, uSpread: ru.hi - ru.lo, vSpread: rv.hi - rv.lo };
  });

  // Guard: a uniform velocity field would make the bounds check vacuous.
  expect(r.uSpread).toBeGreaterThan(0.5);
  expect(r.vSpread).toBeGreaterThan(0.1);

  // MUTATION TARGET. Replacing `clamp(corrected, lo, hi)` with `corrected` in
  // maccormack_velocity.wgsl must fail this. If it still passes, the correction
  // is not firing and the combine is a silent no-op.
  expect(r.uOver).toBeLessThanOrEqual(1e-5);
  expect(r.vOver).toBeLessThanOrEqual(1e-5);
});

test('the velocity combine applies phi^ + (phi^n - phi~)/2, not a pass-through', async ({ page }) => {
  await boot(page);
  const stats = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    const numX = solver.numX, n = solver.numY;
    const size = numX * n * 4;
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

    // Dispatch ONE pass at a time, so phi^ and phi~ can be read back before the
    // next pass overwrites them. That is what makes the expected combine output
    // computable here without re-implementing the stencil in JS.
    const runPass = (pipeline, group) => {
      solver._writeAllParams();
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(numX / 8), Math.ceil(n / 8), 1);
      pass.end();
      device.queue.submit([enc.finish()]);
    };

    for (let k = 0; k < 60; k++) solver.step(ui.numIters);

    const c = solver._velCur;
    const hatPair = solver.velPairs[(c + 1) % 3];
    const tildePair = solver.velPairs[(c + 2) % 3];

    const uN = await readBuf(solver.velPairs[c].u);
    runPass(solver.advectVelPipeline, solver.velFwd[c]);
    const uHat = await readBuf(hatPair.u);
    runPass(solver.advectVelPipeline, solver.velBack[c]);
    const uTilde = await readBuf(tildePair.u);
    runPass(solver.mcVelPipeline, solver.velCombine[c]);
    const uOut = await readBuf(tildePair.u);

    // Faces where the clamp was inactive must equal the raw MacCormack value
    // exactly. Counting only the ones where the correction is non-trivial is
    // what distinguishes a live combine from one that writes phi^ through.
    let matched = 0, corrected = 0, passthrough = 0, stale = 0;
    for (let i = 1; i < numX; i++) {
      for (let j = 1; j < n - 1; j++) {
        const k = i * n + j;
        const target = uHat[k] + 0.5 * (uN[k] - uTilde[k]);
        const delta = Math.abs(target - uHat[k]);
        if (delta <= 1e-4) continue;   // correction too small to be evidence
        corrected++;
        if (Math.abs(uOut[k] - target) <= 1e-5) matched++;
        if (Math.abs(uOut[k] - uHat[k]) <= 1e-9) passthrough++;
        if (Math.abs(uOut[k] - uTilde[k]) <= 1e-9) stale++;
      }
    }
    return { matched, corrected, passthrough, stale };
  });

  // Guard: no non-trivial corrections at all would make the rest vacuous.
  expect(stats.corrected).toBeGreaterThan(200);

  // THE regression assertion for this task, and the counterpart to the limiter
  // test above. Task 5's first attempt shipped a combine whose correction was
  // silently a no-op, and an early gate passed anyway -- because a no-op
  // combine leaves the field bounded, in range, and visually plausible. Only a
  // direct comparison against phi^ + (phi^n - phi~)/2 catches it.
  //
  // Fails if the combine writes phi^ through (passthrough), if the combine
  // never runs so phi~ survives (stale), or if the 0.5 factor drifts.
  expect(stats.matched / stats.corrected).toBeGreaterThan(0.5);
  expect(stats.passthrough / stats.corrected).toBeLessThan(0.05);
  expect(stats.stale / stats.corrected).toBeLessThan(0.05);
});

test('the velocity forward pass traces upstream and honours the staggered offsets', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;

    const numX = solver.numX, n = solver.numY, h = solver.h;
    const dt = solver.params.dt;
    const size = numX * n * 4;
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

    // All-fluid, so no face guard fires and the trace is unobstructed.
    solver.writeSolidMask(new Float32Array(numX * n).fill(1));

    // Half a cell per step, so a dropped h/2 stagger offset (which costs
    // exactly half a cell) shows up as a shift of 0 instead of 0.5.
    const SHIFT = 0.5;
    const SPEED = SHIFT * h / dt;
    const AMP = 0.05 * SPEED;          // small, so self-advection stays negligible
    const SIG = 3.0;                   // gaussian half-width in cells
    const gauss = (d) => AMP * Math.exp(-0.5 * (d / SIG) * (d / SIG));

    // Each case builds a field varying along ONE axis only, so the other axis
    // cannot contribute and the measurement isolates a single offset.
    const iC = Math.floor(numX / 3), jC = Math.floor(n / 2);
    const WIN = 12;

    const runForward = () => {
      solver._writeAllParams();
      const c = solver._velCur;
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(solver.advectVelPipeline);
      pass.setBindGroup(0, solver.velFwd[c]);
      pass.dispatchWorkgroups(Math.ceil(numX / 8), Math.ceil(n / 8), 1);
      pass.end();
      device.queue.submit([enc.finish()]);
      return solver.velPairs[(c + 1) % 3];
    };

    // Amplitude-weighted centroid of the bump, in cell units.
    const centroid = (arr, at, centre) => {
      let w = 0, m = 0;
      for (let k = centre - WIN; k <= centre + WIN; k++) {
        const a = at(arr, k);
        w += a; m += a * k;
      }
      return m / w;
    };

    const out = {};

    // Case A -- u carried in x by a uniform base flow. Pins the SIGN of the
    // trace: a flipped `-params.dt` moves the bump upstream instead.
    {
      const u = new Float32Array(numX * n), v = new Float32Array(numX * n);
      for (let i = 0; i < numX; i++) {
        const b = SPEED + gauss(i - iC);
        for (let j = 0; j < n; j++) u[i * n + j] = b;
      }
      solver.writeVelocityU(u); solver.writeVelocityV(v);
      const before = centroid(u, (a, i) => a[i * n + jC] - SPEED, iC);
      const res = await readBuf(runForward().u);
      out.xShift = centroid(res, (a, i) => a[i * n + jC] - SPEED, iC) - before;
    }

    // Case B -- u carried in y by a uniform vertical flow. The u-face's
    // advecting cv is the four-way average of v, and the sample point is
    // y = j*h + h/2 - dt*cv, so this pins u's h/2 Y offset (and would fail if
    // u were given v's h/2 X offset instead).
    {
      const u = new Float32Array(numX * n), v = new Float32Array(numX * n);
      for (let j = 0; j < n; j++) {
        const b = gauss(j - jC);
        for (let i = 0; i < numX; i++) { u[i * n + j] = b; v[i * n + j] = SPEED; }
      }
      solver.writeVelocityU(u); solver.writeVelocityV(v);
      const before = centroid(u, (a, j) => a[iC * n + j], jC);
      const res = await readBuf(runForward().u);
      out.uYShift = centroid(res, (a, j) => a[iC * n + j], jC) - before;
    }

    // Case C -- v carried in x by a uniform base flow. Mirror of case B:
    // the sample point is x = i*h + h/2 - dt*cu, pinning v's h/2 X offset.
    {
      const u = new Float32Array(numX * n), v = new Float32Array(numX * n);
      for (let i = 0; i < numX; i++) {
        const b = gauss(i - iC);
        for (let j = 0; j < n; j++) { v[i * n + j] = b; u[i * n + j] = SPEED; }
      }
      solver.writeVelocityU(u); solver.writeVelocityV(v);
      const before = centroid(v, (a, i) => a[i * n + jC], iC);
      const res = await readBuf(runForward().v);
      out.vXShift = centroid(res, (a, i) => a[i * n + jC], iC) - before;
    }

    return { ...out, expected: SHIFT };
  });

  // Nothing else in this file pins the trace DIRECTION for velocity: a flipped
  // `-params.dt` inside u_departure/v_departure survives the drift test (both
  // copies flip together), the uniform-readback test (which only inspects the
  // uploaded bytes) and the limiter test (a backwards trace stays in range).
  // A positive shift is the assertion that catches it.
  expect(r.xShift).toBeGreaterThan(0);
  expect(r.uYShift).toBeGreaterThan(0);
  expect(r.vXShift).toBeGreaterThan(0);

  // And the magnitude pins the staggered offsets: each h/2 offset is worth
  // exactly half a cell, which is this whole displacement.
  expect(r.xShift).toBeCloseTo(r.expected, 1);
  expect(r.uYShift).toBeCloseTo(r.expected, 1);
  expect(r.vXShift).toBeCloseTo(r.expected, 1);
});

test('MacCormack keeps smoke inside [0,1]', async ({ page }) => {
  await boot(page);
  const range = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    const numX = solver.numX, numY = solver.numY;
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

    // Re-inject the inlet each step exactly as main.js's frame() does. Without
    // it the dye flushes downstream in ~200 steps at this CFL and the field
    // goes uniform, which would make the bounds below vacuous.
    for (let k = 0; k < 200; k++) {
      if (ui.smokeInletData) {
        device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
      }
      solver.step(ui.numIters);
    }

    const m = await readBuf(solver.smokeBuffer);
    const sMask = await readBuf(solver.solidBuffer);

    // Fluid cells only. Solid cells are painted dark grey in-shader by
    // render_field.wgsl and never displayed as dye, AND the inlet band is
    // written into the solid wall at i=0 -- so a whole-buffer scan reports a
    // 0-to-1 spread even when the interior holds no dye whatsoever.
    let lo = Infinity, hi = -Infinity, dyed = 0, front = 0, fluid = 0;
    for (let i = 1; i < numX - 1; i++) {
      for (let j = 1; j < numY - 1; j++) {
        const k = i * numY + j;
        if (sMask[k] === 0) continue;
        const x = m[k];
        fluid++;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        if (x < 0.5) dyed++;
        if (x > 0.02 && x < 0.98) front++;
      }
    }
    return { lo, hi, dyed, front, fluid };
  });

  // Guards: a uniform field would make the bounds check vacuous. The Karman
  // preset injects dye through the solid wall at i=0, so the interior must
  // hold both saturated dye and a partially-mixed front for the limiter to
  // have anything to bound.
  expect(range.fluid).toBeGreaterThan(1000);
  expect(range.dyed).toBeGreaterThan(500);
  expect(range.front).toBeGreaterThan(500);

  // The renderer uses a fixed [0,1] range with no auto-ranging, so any
  // overshoot clips into visible halos at dye fronts.
  expect(range.lo).toBeGreaterThanOrEqual(-1e-4);
  expect(range.hi).toBeLessThanOrEqual(1 + 1e-4);
});

/**
 * Strips comments and collapses whitespace, so the shader-drift comparison
 * below survives reformatting and comment edits but not a change to any token.
 */
function normalizeWgsl(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Extracts a brace-delimited `<kind> name...{ ... }` block from normalized WGSL. */
function extractBlock(src, decl, name) {
  const needle = decl === 'fn' ? `fn ${name}(` : `${decl} ${name} `;
  const start = src.indexOf(needle);
  if (start === -1) throw new Error(`${decl} ${name} not found`);
  let depth = 0;
  for (let k = src.indexOf('{', start); k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}' && --depth === 0) return src.slice(start, k + 1);
  }
  throw new Error(`unbalanced braces in ${decl} ${name}`);
}

const extractFn = (src, name) => extractBlock(src, 'fn', name);
const extractStruct = (src, name) => extractBlock(src, 'struct', name);

/**
 * Each MacCormack combine is correct ONLY because its re-trace reproduces the
 * matching forward pass's stencil exactly. WGSL has no modules, so the shared
 * declarations are duplicated across file pairs. Drift between copies still
 * yields plausible in-range values, so no behavioural test in this file can
 * catch it -- it would be a silent wrong-answer path. Hence these textual
 * checks.
 *
 * Playwright transpiles specs to CJS, so import.meta is unavailable and
 * config.rootDir points at testDir. npm test runs from the repo root.
 */
const SHADER_DIR = path.join(process.cwd(), 'static', 'shaders');
const readShader = (f) => normalizeWgsl(readFileSync(path.join(SHADER_DIR, f), 'utf8'));

test('the smoke combine re-traces with the same stencil code advect_smoke traced with', () => {
  const fwd = readShader('advect_smoke.wgsl');
  const comb = readShader('maccormack.wgsl');

  for (const fn of ['scalar_stencil', 'smoke_departure']) {
    expect(
      extractFn(comb, fn),
      `${fn} has drifted between advect_smoke.wgsl and maccormack.wgsl`,
    ).toBe(extractFn(fwd, fn));
  }
});

test('the velocity combine re-traces with the same stencil code advect_velocity traced with', () => {
  const fwd = readShader('advect.wgsl');
  const comb = readShader('maccormack_velocity.wgsl');

  // The velocity path has its OWN copies -- it uses staggered offsets the
  // smoke path does not (u: no x offset + h/2 y offset; v: h/2 x offset + no
  // y offset), so the smoke assertions above say nothing about it. Getting
  // those offsets backwards is the single most likely bug in this shader pair.
  for (const fn of ['u_stencil', 'v_stencil', 'u_departure', 'v_departure']) {
    expect(
      extractFn(comb, fn),
      `${fn} has drifted between advect.wgsl and maccormack_velocity.wgsl`,
    ).toBe(extractFn(fwd, fn));
  }
});

test('the duplicated Params and Stencil structs agree across every advection shader', () => {
  // Function bodies matching is not enough. `Params` is re-declared in all four
  // files and read from ONE uniform buffer, so a field reorder in a single copy
  // (swapping h and dt, say) makes that shader re-trace with wrong values --
  // wrong bounds, still-plausible output, invisible to every behavioural test.
  // `Stencil` is the same hazard for the i0/i1/j0/j1/tx/ty tuple.
  //
  // diffuse.wgsl carries the same 8-field Params and is the ONLY shader that
  // reads `nu`, so a field reorder there silently corrupts the Reynolds control
  // that rides on it -- it would diffuse with `color` reinterpreted as a float
  // and still produce a smooth, plausible field. It has no Stencil struct (it
  // is a five-point update, not a semi-Lagrangian trace), so the two structs
  // check over different file sets.
  const perStruct = {
    Params: [
      'advect.wgsl', 'advect_smoke.wgsl', 'maccormack.wgsl', 'maccormack_velocity.wgsl',
      'diffuse.wgsl',
    ],
    Stencil: [
      'advect.wgsl', 'advect_smoke.wgsl', 'maccormack.wgsl', 'maccormack_velocity.wgsl',
    ],
  };

  for (const [struct, files] of Object.entries(perStruct)) {
    const srcs = files.map((f) => [f, readShader(f)]);
    const ref = extractStruct(srcs[0][1], struct);
    for (const [file, src] of srcs.slice(1)) {
      expect(
        extractStruct(src, struct),
        `struct ${struct} has drifted between ${files[0]} and ${file}`,
      ).toBe(ref);
    }
  }
});

test('a step leaves solid-cell values untouched, even when the solid carries velocity', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    const numX = solver.numX, numY = solver.numY;
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

    const sMask = await readBuf(solver.solidBuffer);
    let idx = -1;
    for (let i = 2; i < numX - 2 && idx < 0; i++) {
      for (let j = 2; j < numY - 2; j++) {
        if (sMask[i * numY + j] === 0) { idx = i * numY + j; break; }
      }
    }
    if (idx < 0) return { found: false };

    // Stamp a marker no fluid cell is carrying, into every smoke slot so it is
    // present whichever slot this step reads as phi^n.
    const MARK = 0.25;
    for (const b of solver.smokeBufs) {
      device.queue.writeBuffer(b, idx * 4, new Float32Array([MARK]));
    }
    // Simulate a drag: rasterizeObstacle writes the obstacle's own velocity
    // into its solid cells, which sends the departure point far away.
    for (const p of solver.velPairs) {
      device.queue.writeBuffer(p.u, idx * 4, new Float32Array([5.0]));
      device.queue.writeBuffer(p.v, idx * 4, new Float32Array([5.0]));
    }

    solver.step(ui.numIters);
    const m = await readBuf(solver.smokeBuffer);
    return { found: true, mark: MARK, after: m[idx] };
  });

  expect(r.found).toBe(true);
  // A solid cell carries a boundary condition, not a transported field. For
  // smoke this is invisible (solids are painted in-shader by render_field.wgsl)
  // but Task 6 puts VELOCITY here, where the value in a solid cell IS the
  // moving-wall BC and rewriting it corrupts the boundary condition.
  //
  // Two independent mechanisms hold this: the combine's explicit solid guard,
  // and the fact that the limiter bounds are seeded with phi^ -- which equals
  // phi^n at solids, since both advect passes revert there -- so the interval
  // always contains the value being clamped. This pins the invariant, not
  // either mechanism, so it survives Task 6 rewiring one of them.
  expect(r.after).toBeCloseTo(r.mark, 6);
});

test('the backward pass uniform really holds a negated dt, and nu reaches offset 28', async ({ page }) => {
  await boot(page);
  const got = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    // nu has no producer yet (viscosity lands in a later task), so give it a
    // value the ArrayBuffer's zero-fill cannot fake before re-uploading.
    solver.params.nu = 0.375;
    solver.step(ui.numIters); // step() re-writes every uniform buffer

    const read = async (buf) => {
      const staging = device.createBuffer({
        size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, staging, 0, 32);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const dv = new DataView(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return { dt: dv.getFloat32(12, true), nu: dv.getFloat32(28, true) };
    };

    const out = { fwd: await read(solver.uniformBuf), back: await read(solver.uniformBufNegDt) };
    delete solver.params.nu;
    return out;
  });

  // The sharpness test below proves a correction FIRES; it does not pin the
  // correction's SIGN. Dropping the negation leaves the backward pass tracing
  // upstream, so phi~ = SL(SL(phi^n)) and the combine still applies a non-zero
  // (but wrong) anti-diffusive correction -- dye still saturates and every
  // behavioural assertion stays green. Only the uploaded bytes show it.
  expect(got.fwd.dt).toBeGreaterThan(0);
  expect(got.back.dt).toBe(-got.fwd.dt);

  // Offset 28 is written, not merely zero-filled by the ArrayBuffer.
  expect(got.fwd.nu).toBeCloseTo(0.375, 6);
  expect(got.back.nu).toBeCloseTo(0.375, 6);
});

test('the uniform buffer packs every field at its documented offset', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;

    // A lost device returns an all-zero buffer, which fits nothing here (every
    // sentinel is non-zero) but must be reported as device loss, not as a
    // confusing field mismatch.
    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    // A distinct sentinel in EVERY field, each exactly representable in its type
    // (u32 for numX/numY/color, f32 for the rest). Because all eight differ, a
    // swap of any two DataView writes moves a value to the wrong offset and the
    // read-back below mismatches — which the drift test on the WGSL structs and
    // the offset-28-only readback above both miss.
    const saved = { ...solver.params };
    solver.params.numX    = 61;
    solver.params.numY    = 62;
    solver.params.h       = 8.5;
    solver.params.dt      = 12.25;
    solver.params.omega   = 16.5;
    solver.params.density = 20.75;
    solver.params.color   = 3;
    solver.params.nu      = 28.125;

    // Exercise _writeParamsTo directly, no overrides, so color/dt/nu flow from
    // params. This is the hand-packed 32-byte layout every shader's Params
    // struct depends on — the single source of truth no other test pins
    // field-by-field.
    solver._writeParamsTo(solver.uniformBuf);

    const staging = device.createBuffer({
      size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(solver.uniformBuf, 0, staging, 0, 32);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const dv = new DataView(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Restore real params to every uniform buffer before handing the field back.
    Object.assign(solver.params, saved);
    solver._writeAllParams();

    return {
      deviceLost,
      numX:    dv.getUint32(0, true),
      numY:    dv.getUint32(4, true),
      h:       dv.getFloat32(8, true),
      dt:      dv.getFloat32(12, true),
      omega:   dv.getFloat32(16, true),
      density: dv.getFloat32(20, true),
      color:   dv.getUint32(24, true),
      nu:      dv.getFloat32(28, true),
    };
  });

  expect(r.deviceLost).toBe(false);

  // Each field at its documented offset (0,4,8,12,16,20,24,28), read as its
  // documented type. Swap any two setFloat32 offsets in _writeParamsTo and the
  // two fields trade values here.
  expect(r.numX).toBe(61);
  expect(r.numY).toBe(62);
  expect(r.h).toBeCloseTo(8.5, 6);
  expect(r.dt).toBeCloseTo(12.25, 6);
  expect(r.omega).toBeCloseTo(16.5, 6);
  expect(r.density).toBeCloseTo(20.75, 6);
  expect(r.color).toBe(3);
  expect(r.nu).toBeCloseTo(28.125, 6);
});

test('MacCormack actually corrects: dye fronts reach saturation, unlike first-order SL', async ({ page }) => {
  await boot(page);
  const stats = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    const numX = solver.numX, numY = solver.numY;
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

    for (let k = 0; k < 200; k++) {
      if (ui.smokeInletData) {
        device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
      }
      solver.step(ui.numIters);
    }

    const m = await readBuf(solver.smokeBuffer);
    const sMask = await readBuf(solver.solidBuffer);

    let saturated = 0, smeared = 0, fluid = 0;
    for (let i = 1; i < numX - 1; i++) {
      for (let j = 1; j < numY - 1; j++) {
        const k = i * numY + j;
        if (sMask[k] === 0) continue;
        fluid++;
        // m = 0 is fully dark dye, m = 1 is clear (inverted convention).
        if (m[k] <= 1e-3) saturated++;
        else if (m[k] < 1 - 1e-3) smeared++;
      }
    }
    return { saturated, smeared, fluid };
  });

  expect(stats.fluid).toBeGreaterThan(1000);

  // THE regression assertion for this task. First-order semi-Lagrangian
  // diffuses every dye front, so NO interior fluid cell ever reaches full
  // saturation -- measured at exactly 0 under the pre-MacCormack solver. The
  // second-order correction is what carries dye to m = 0.
  //
  // Anything that SUPPRESSES the correction -- a combine that writes phi^
  // through, a backward pass that never runs, a limiter clamped to phi^ --
  // collapses this to 0 and fails. That is invisible to every other test in
  // this file: the wiring test asserts only which buffer OBJECTS are bound,
  // and the bounds test passes trivially under first-order SL.
  //
  // It does NOT pin the correction's sign; see the negated-dt test above,
  // which is the other half of this pair.
  expect(stats.saturated).toBeGreaterThan(100);
});

test('smoke advection dispatches the bind groups for the live velocity slot, not the smoke slot', async ({ page }) => {
  await boot(page);
  const dispatched = await page.evaluate(() => {
    const { solver, ui } = window.__flowlab;
    solver.paused = true;
    // The split must be the forced one below, not one the default viscosity's
    // substep parity happens to introduce -- otherwise the expected counter
    // advance depends on where the Re slider sits.
    solver.setParams({ nu: 0 });
    solver.resetFlipState();

    // Force velCur and smokeCur apart. Under normal operation the two counters
    // advance from a shared reset, so a step() that dispatched smoke keyed by
    // [smokeCur][smokeCur] instead of [velCur][smokeCur] would agree with
    // correct code on the first step. Forcing a split here makes that bug
    // observable; the 3x3 tables exist so step() stays correct once something
    // (e.g. viscous substepping) causes this split to happen on its own.
    solver._velCur = 0;
    solver._smokeCur = 1;

    // Key off the pipeline, not the pass's position in step(). The old version
    // captured "the last setBindGroup wins", which only worked while smoke
    // advection happened to be the final pass; MacCormack adds two more.
    const captured = [];
    const proto = GPUComputePassEncoder.prototype;
    const origSetPipeline = proto.setPipeline;
    const origSetBindGroup = proto.setBindGroup;
    let current = null;
    proto.setPipeline = function (pipeline, ...rest) {
      current = pipeline;
      return origSetPipeline.call(this, pipeline, ...rest);
    };
    proto.setBindGroup = function (index, bindGroup, ...rest) {
      captured.push({ pipeline: current, bindGroup });
      return origSetBindGroup.call(this, index, bindGroup, ...rest);
    };
    try {
      solver.step(ui.numIters);
    } finally {
      proto.setPipeline = origSetPipeline;
      proto.setBindGroup = origSetBindGroup;
    }

    // GPUBindGroup cannot cross the evaluate boundary, so name each one by
    // where it sits in the tables.
    const label = (bg) => {
      for (const t of ['smokeFwd', 'smokeBack', 'smokeCombine']) {
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) if (solver[t][a][b] === bg) return `${t}[${a}][${b}]`;
        }
      }
      return 'unknown';
    };
    const forPipeline = (p) =>
      captured.filter((c) => c.pipeline === p).map((c) => label(c.bindGroup));

    return {
      advect: forPipeline(solver.advectSmokePipeline),
      combine: forPipeline(solver.mcSmokePipeline),
      advanced: { vc: solver._velCur, sc: solver._smokeCur },
    };
  });

  // The two advect_smoke dispatches are the forward and backward passes, in
  // that order, both keyed [velCur=0][smokeCur=1] -- not [1][1].
  expect(dispatched.advect).toEqual(['smokeFwd[0][1]', 'smokeBack[0][1]']);
  expect(dispatched.combine).toEqual(['smokeCombine[0][1]']);
  // And the step advanced both counters by 2, from (0, 1).
  expect(dispatched.advanced).toEqual({ vc: 2, sc: 0 });
});

// ---------------------------------------------------------------------------
// Taylor-Green calibration: the scheme's own numerical viscosity.
//
// A single Taylor-Green mode is an exact STEADY solution of the 2D Euler
// equations (vorticity is a linear function of the streamfunction), so with
// physical viscosity off the exact answer is "nothing happens". Every bit of
// kinetic energy the solver loses is therefore its own numerical dissipation,
// and fitting log(KE) against time gives that dissipation as a viscosity:
// the mode decays as exp(-2 nu k^2 t), so KE decays as exp(-4 nu k^2 t).
//
// Geometry. The mode must not push fluid through the walls, which pins both
// the phase and the domain. u = A sin(kX) cos(kY), v = -A cos(kX) sin(kY) with
// X = x - h, Y = y - h and k = pi/L vanishes in the wall-normal direction on
// all four walls of the square box of side L. The box is therefore the SQUARE
// sub-grid (cells 1..numY-2 in both axes), not the full rectangular grid --
// the grid is ~2.2:1, so using its full width would leave sin(k x) nonzero at
// the right wall and drive flow straight into it. (Swapping the sin/cos phase,
// as the task brief's snippet does, puts the full amplitude A on all four
// walls; it is divergence-free but it is not a closed-box Taylor-Green.)
//
// This choice is also exactly divergence-free on the MAC grid, not merely to
// truncation order: the u and v difference terms cancel identically because
// both reduce to 2A cos(k X_v) cos(k Y_u) sin(k h / 2). The test asserts that.
// ---------------------------------------------------------------------------

/**
 * Runs the Taylor-Green decay experiment and fits the numerical viscosity.
 *
 * Rewrites the complete solver state on every call -- solid mask, all three
 * velocity slots, all three smoke slots, pressure, and the rotation index --
 * so repeated calls on one page start from a byte-identical field and cannot
 * carry state forward from an earlier measurement.
 *
 * With `nu > 0` the same fit measures the TOTAL dissipation, numerical plus
 * physical, so `nuNum(nu) - nuNum(0)` at a fixed numIters isolates what the
 * viscous operator actually delivered.
 *
 * NOTE the fit is only a clean viscosity measurement at nu = 0. The closed box
 * is analytically FREE-SLIP -- the mode's wall-normal component vanishes on all
 * four walls but its TANGENTIAL component does not -- while diffuse.wgsl's
 * ghost imposes NO-SLIP. With nu > 0 the mismatch grows wall shear layers that
 * are not part of the mode and that dominate the box's KE budget, so the fitted
 * value overshoots nu_num + nu by 3-4x. Restricting the fit to an interior
 * window does not rescue it either: the window is not a closed subsystem, and
 * the fitted value then swings between 0.25x and 4.2x the true value depending
 * on the margin and the run length. The operator's accuracy is therefore
 * measured directly (see the one-step difference test), and nu > 0 is used here
 * only to drive the solver at a viscosity and assert the field survives it.
 *
 * `dt` is a parameter, not a constant, because `nu_num` is LINEAR in it (Task 7)
 * -- so the fit has to be re-run whenever a preset's timestep changes. The
 * fitted value is converted to physical time via `solver.params.dt`, so a run at
 * half the timestep and twice the steps covers the same simulated interval.
 *
 * @param {import('@playwright/test').Page} page
 * @param {{steps?: number, numIters?: number, sample?: number, nu?: number,
 *          dt?: number}} opts
 * @returns {Promise<Object>} nuNum, r2, and the setup-validity diagnostics
 */
function measureNuNum(page, { steps = 300, numIters = 80, sample = 20, nu = 0, dt = 1 / 120 } = {}) {
  return page.evaluate(async ({ steps, numIters, sample, nu, dt }) => {
    const { solver, device, interaction } = window.__flowlab;

    // A lost device makes every later readback return zeros, which fits a
    // straight line perfectly and would look like a flawless measurement.
    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const numX = solver.numX, n = solver.numY, h = solver.h;
    const Nc = n - 2;              // square fluid box: cells i, j in [1, Nc]
    const L = Nc * h;
    const k = Math.PI / L;
    const A = 1.0;

    const sData = new Float32Array(numX * n);
    const uData = new Float32Array(numX * n);
    const vData = new Float32Array(numX * n);
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < n; j++) {
        sData[i * n + j] = (i >= 1 && i <= Nc && j >= 1 && j <= Nc) ? 1 : 0;
        const Xu = i * h - h,           Yu = j * h + 0.5 * h - h;
        const Xv = i * h + 0.5 * h - h, Yv = j * h - h;
        uData[i * n + j] =  A * Math.sin(k * Xu) * Math.cos(k * Yu);
        vData[i * n + j] = -A * Math.cos(k * Xv) * Math.sin(k * Yv);
      }
    }

    // Setup validity, measured on the field before it is uploaded.
    let cpuMaxDiv = 0, cpuWallMax = 0;
    for (let i = 1; i <= Nc; i++)
      for (let j = 1; j <= Nc; j++)
        cpuMaxDiv = Math.max(cpuMaxDiv, Math.abs(
          uData[(i + 1) * n + j] - uData[i * n + j] + vData[i * n + j + 1] - vData[i * n + j]));
    for (let j = 1; j <= Nc; j++)
      cpuWallMax = Math.max(cpuWallMax, Math.abs(uData[n + j]), Math.abs(uData[(Nc + 1) * n + j]));
    for (let i = 1; i <= Nc; i++)
      cpuWallMax = Math.max(cpuWallMax, Math.abs(vData[i * n + 1]), Math.abs(vData[i * n + Nc + 1]));

    interaction.showObstacle = false;
    solver.paused = true;
    solver.resetFlipState();
    // nu = 0 measures the scheme's own dissipation; nu > 0 measures that plus
    // the viscous pass, which is what makes the two runs differenceable.
    solver.setParams({ nu, dt, omega: 1.9, density: 1000 });
    solver.writeSolidMask(sData);
    solver.writeVelocityU(uData);
    solver.writeVelocityV(vData);
    solver.writeSmoke(new Float32Array(numX * n).fill(1));
    device.queue.writeBuffer(solver.pressureBuffer, 0, new Float32Array(numX * n));

    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    // KE over the fluid box only. The surrounding solid ring is excluded: the
    // boundary shader writes extrapolated values there that are not part of the
    // flow, and including them would add a spurious constant to the fit.
    //
    const probe = async () => {
      const { u, v } = solver.velocityBuffers;
      const uD = await readBuf(u), vD = await readBuf(v);
      let e = 0, nBad = 0;
      for (let i = 1; i <= Nc; i++)
        for (let j = 1; j <= Nc; j++) {
          const a = uD[i * n + j], b = vD[i * n + j];
          if (!Number.isFinite(a) || !Number.isFinite(b)) { nBad++; continue; }
          e += a * a + b * b;
        }
      return { ke: e, nBad };
    };

    // Mean of sin^2 cos^2 over a full period is 1/4 per component, so the
    // initial KE is Nc^2 / 2 exactly. Checking it makes a zeroed buffer or a
    // dead device impossible to mistake for a physical decay.
    const keAnalytic = (Nc * Nc) / 2;
    const p0 = await probe();
    const ke0Rel = Math.abs(p0.ke - keAnalytic) / keAnalytic;

    // Read back from the solver rather than reusing the argument: this is the
    // dt the run actually used, so the time axis cannot drift from it.
    const dtUsed = solver.params.dt;
    const ts = [], ys = [];
    for (let s = 0; s <= steps; s++) {
      if (s % sample === 0) {
        const p = s === 0 ? p0 : await probe();
        if (p.nBad > 0) throw new Error(`non-finite velocity at step ${s}: ${p.nBad} cells`);
        if (!(p.ke > 0)) throw new Error(`kinetic energy collapsed to ${p.ke} at step ${s}`);
        ts.push(s * dtUsed); ys.push(Math.log(p.ke));
      }
      if (s < steps) {
        solver.step(numIters);
        // Draining the queue each step keeps a single submit under the GPU
        // watchdog. Without it, high iteration counts at the large tiers lose
        // the device outright. Purely a scheduling change: fitted values are
        // bit-identical with and without.
        await device.queue.onSubmittedWorkDone();
      }
    }
    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);

    const nP = ts.length;
    const mt = ts.reduce((a, b) => a + b, 0) / nP;
    const my = ys.reduce((a, b) => a + b, 0) / nP;
    let num = 0, den = 0;
    for (let q = 0; q < nP; q++) { num += (ts[q] - mt) * (ys[q] - my); den += (ts[q] - mt) ** 2; }
    const slope = num / den;                        // = -4 nu k^2

    let ssTot = 0, ssRes = 0;
    for (let q = 0; q < nP; q++) {
      const pred = my + slope * (ts[q] - mt);
      ssRes += (ys[q] - pred) ** 2;
      ssTot += (ys[q] - my) ** 2;
    }

    return {
      nuNum: -slope / (4 * k * k),
      r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
      points: nP, logDrop: ys[0] - ys[nP - 1],
      cpuMaxDiv, cpuWallMax, ke0Rel, k, h, numX, numY: n, Nc, dt: dtUsed, numIters,
      nu,
      substeps: solver.viscSubsteps, clamped: solver.viscClamped,
      nuEff: solver.viscNuEff, nuMax: solver.viscNuMax,
    };
  }, { steps, numIters, sample, nu, dt });
}

test('the Taylor-Green initial field is closed-box and divergence-free on the MAC grid', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);
  const r = await measureNuNum(page, { steps: 0, sample: 20 });

  // Exact cancellation, not truncation order: 5.96e-8 is float32 rounding on
  // values of order 1, i.e. the discrete divergence is zero to machine epsilon.
  expect(r.cpuMaxDiv).toBeLessThan(1e-6);
  // No flow through any of the four walls of the square box.
  expect(r.cpuWallMax).toBeLessThan(1e-9);
  // What the GPU holds after upload matches the analytic KE of the mode.
  expect(r.ke0Rel).toBeLessThan(1e-6);
});

test('Taylor-Green decay yields a numerical viscosity', async ({ page }) => {
  test.setTimeout(400_000);
  await boot(page);

  // Every measurement here runs at the Karman preset's dt = 1/240 and 600 steps
  // -- the same 2.5 s of simulated time the 300-step dt = 1/120 runs covered, so
  // the fit window is the physical one and not a step count.
  const DT = 1 / 240, STEPS = 600, SAMPLE = 40;
  const d = await page.evaluate(() => import('/js/diagnostics.js').then((m) => ({
    perDt: m.NU_NUM_PER_DT,
    converged: m.nuNumConverged(1 / 240),
    opIters: m.PROJECTION_ITERS_MEASURED,
    op_256: m.NU_NUM_ITERS256[256],
    opDt: m.NU_NUM_ITERS256_DT,
  })));
  expect(d.opDt).toBeCloseTo(DT, 10);

  // Operating point: the pressure iteration count the Karman preset ships, read
  // from the preset rather than hard-coded, so the measurement follows the app.
  const shippedIters = await page.evaluate(() =>
    import('/js/presets.js').then((m) => m.PRESETS.karmanVortex.numIters));
  expect(shippedIters).toBe(d.opIters);

  const op = await measureNuNum(page, { numIters: shippedIters, dt: DT, steps: STEPS, sample: SAMPLE });
  console.log(`nu_num(iters=${shippedIters})  = ${op.nuNum.toExponential(4)}  R^2 = ${op.r2.toFixed(5)}  n = ${op.points}`);

  expect(op.points).toBeGreaterThan(5);
  expect(op.r2).toBeGreaterThan(0.99);   // a bad fit means the decay is not exponential
  expect(op.nuNum).toBeGreaterThan(0);   // energy must decay, not grow
  expect(op.nuNum).toBeLessThan(1e-1);   // sanity ceiling

  // THE tie between the GPU and the shipped table. diagnostics.js drives a real
  // ceiling off NU_NUM_ITERS256[256]; if the solver drifts away from it, the
  // badge starts lying and only this assertion notices. 5% covers the fit's own
  // scatter and nothing like a scheme regression.
  expect(op.nuNum).toBeGreaterThan(0.95 * d.op_256);
  expect(op.nuNum).toBeLessThan(1.05 * d.op_256);

  // The escalation must actually have bought something: 256 iterations has to
  // sit well below the 80-iteration value this table replaced (2.0324e-3), or
  // the frame-rate cost was paid for nothing. Measured 7.7808e-4, a 2.6x drop.
  expect(op.nuNum).toBeLessThan(0.6 * 2.0324e-3);

  // With the projection converged, what is left is the advection scheme's own
  // dissipation. At 2048 iterations tier 256 is converged to 5 significant
  // figures (2048 and 4096 agree), and the value is 5.06e-4. Pin it against the
  // shipped coefficient the same way -- a MacCormack regression shows up here
  // as a rise toward 1.5e-3.
  const conv = await measureNuNum(page, { numIters: 2048, dt: DT, steps: STEPS, sample: SAMPLE });
  console.log(`nu_num(iters=2048) = ${conv.nuNum.toExponential(4)}  R^2 = ${conv.r2.toFixed(5)}`);

  expect(conv.r2).toBeGreaterThan(0.99);
  expect(conv.nuNum).toBeGreaterThan(0.95 * d.converged);
  expect(conv.nuNum).toBeLessThan(1.05 * d.converged);
  // Converging the projection must lower the measured viscosity, never raise
  // it -- the opposite ordering would mean the decay is not projection-limited
  // and the operating-point number means something else entirely.
  expect(conv.nuNum).toBeLessThan(op.nuNum);

  // LINEARITY IN dt, which is the whole basis for shipping a per-dt coefficient
  // instead of a constant. Halving dt must roughly halve nu_num. Measured ratio
  // is 1.943 (9.8230e-4 at dt = 1/120 vs 5.0564e-4 at 1/240); the bounds below
  // would reject both a flat nu_num (ratio 1.0) and an exact-2.0 assumption
  // being silently substituted for the measurement.
  const conv120 = await measureNuNum(page, { numIters: 2048, dt: 1 / 120, steps: 300, sample: 20 });
  const ratio = conv120.nuNum / conv.nuNum;
  console.log(`nu_num(dt=1/120)   = ${conv120.nuNum.toExponential(4)}  ratio = ${ratio.toFixed(3)}`);

  expect(conv120.r2).toBeGreaterThan(0.99);
  expect(ratio).toBeGreaterThan(1.8);
  expect(ratio).toBeLessThan(2.1);

  // And the shipped coefficient must be anchored at 1/240, not 1/120: anchoring
  // at the larger dt would make the ceiling OPTIMISTIC at the preset's own dt.
  //
  // This used to read `expect(d.perDt * DT).toBeCloseTo(d.converged, 12)`, with
  // `d.converged = nuNumConverged(1/240) = NU_NUM_PER_DT * (1/240)` — the same
  // expression on both sides. It asserted `x === x` and could not fail for any
  // value of the constant. Saying what it MEANT needs both measurements.
  //
  // The two candidate anchors are `conv.nuNum * 240` (shipped, 0.121354) and
  // `conv120.nuNum * 120` (0.117878). They differ only because linearity is
  // close but not exact — ratio 1.943, not 2.000 — so the ENTIRE content of the
  // anchoring choice lives in that ~2.9% gap, and any assertion that cannot
  // resolve 2.9% cannot test the choice. A first attempt at this asserted only
  // that the shipped anchor fits BETTER than the alternative; re-anchoring the
  // constant at 1/120 passed it, because the two errors then differ by float
  // noise rather than by anything meaningful. Hence explicit bounds.
  const errShipped = Math.abs(d.perDt * DT - conv.nuNum) / conv.nuNum;
  const errAt120   = Math.abs(conv120.nuNum * 120 * DT - conv.nuNum) / conv.nuNum;
  console.log(`anchor error: shipped(1/240) = ${(100 * errShipped).toFixed(2)}%  ` +
              `alternative(1/120) = ${(100 * errAt120).toFixed(2)}%`);

  // First: the gap must still be real. Both sides here are GPU measurements —
  // if the scheme ever became exactly linear in dt the two anchors would
  // coincide, the bound below would stop discriminating between them, and this
  // assertion says so instead of passing quietly. Measured 2.86%.
  expect(errAt120).toBeGreaterThan(0.02);

  // Second: the shipped coefficient must reproduce the 1/240 measurement to
  // well inside that gap — measured 0.00%, bounded at half the gap so a
  // 1/120-anchored constant (2.86% off) cannot satisfy it. This is tighter than
  // the 5% the surrounding assertions use, and deliberately: 5% cannot tell the
  // two anchors apart, which is the one thing this is here to check.
  expect(errShipped).toBeLessThan(0.015);

  // Third, the direction: evaluated at the LARGER dt the shipped model must
  // OVERSTATE nu_num, never understate it, because overstating nu_num
  // understates the Re ceiling — the safe direction for a claim about honesty.
  // A 1/120-anchored coefficient lands ON the measurement rather than above it,
  // so the margin is what separates them, not the inequality.
  expect(d.perDt * (1 / 120) / conv120.nuNum).toBeGreaterThan(1.01);
});

test('the viscous operator delivers the kinematic viscosity it is given', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);

  // THE calibration assertion for the viscous pass. Every other viscous test in
  // this file checks a SHAPE -- a boundary layer forms, a ring is not read, a
  // substep count is right -- and every one of them passes under a 2x
  // coefficient error, an h-vs-h/2 error, or a per-step-instead-of-per-substep
  // dt. Task 9 ships a LABELLED Re = U*D/nu, so the number printed on that
  // control is honest only if the operator delivers the nu it was handed.
  //
  // METHOD, and why it is not the Taylor-Green fit.
  //
  // The obvious route is to reuse measureNuNum and check
  // nu_measured ~ nu_num + nu. It does not work: the closed box is analytically
  // FREE-SLIP (the mode's wall-normal component vanishes on all four walls, its
  // tangential component does not), while diffuse.wgsl's ghost imposes NO-SLIP.
  // With nu > 0 the mismatch grows wall shear layers that are not part of the
  // mode and that dominate the box's KE budget. Measured over the full box the
  // fit returns 3.5-4.2x the true nu; restricting it to an interior window does
  // not rescue it, because the window is not a closed subsystem -- sweeping
  // margin over {0, 24, 48, 72} and run length over {100, 200} moves the fitted
  // value between 0.25x and 4.2x. Any single config that happens to land near
  // 1.0 does so by luck, and shipping it would be fitting the test to the
  // answer.
  //
  // So measure the operator directly instead, which is both exact and stronger.
  // Diffusion is the LAST thing step() encodes. Two steps from a byte-identical
  // field, one at nu = 0 and one at nu = NU, therefore share their pressure
  // solve, extrapolation and advection exactly -- those are the same dispatches
  // over the same inputs -- and differ by precisely the viscous increment:
  //
  //     w      = A(u0)                 (the nu = 0 result)
  //     w_visc = (I + c*L)^N A(u0)     (the nu = NU result)
  //     w_visc - w  =  N*c*L*w + O((c*L)^2)  =  nu*dt*lap5(w) + O(...)
  //
  // since N * c = N * nu * (dt/N) / h^2 = nu*dt/h^2. The neglected term is
  // C(N,2)(cL)^2, which for this field is ~0.1% of the first -- far below the
  // 2x / 4x / Nx errors this is here to catch.
  const NU = 1.2e-2;   // just under viscNuMax at this tier, and an ODD substep count
  const MARGIN = 20;   // keeps the no-slip wall layer out of the window; see below

  const r = await page.evaluate(async ({ NU, MARGIN }) => {
    const { solver, device, interaction, ui } = window.__flowlab;
    solver.paused = true;

    // A lost device returns zeros from every readback, and zeros give a
    // perfectly correlated slope of 0/0. Guarded here and asserted below.
    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const numX = solver.numX, n = solver.numY, h = solver.h;
    const Nc = n - 2;                    // square fluid box, cells 1..Nc
    const k = Math.PI / (Nc * h), A = 1.0;

    // The Taylor-Green mode from the Task 7 harness: smooth, divergence-free on
    // the MAC grid, and closed-box, so the projection has almost nothing to do
    // and the field stays clean enough for a discrete Laplacian to be meaningful.
    const sData = new Float32Array(numX * n);
    const uData = new Float32Array(numX * n);
    const vData = new Float32Array(numX * n);
    for (let i = 0; i < numX; i++)
      for (let j = 0; j < n; j++) {
        sData[i * n + j] = (i >= 1 && i <= Nc && j >= 1 && j <= Nc) ? 1 : 0;
        const Xu = i * h - h,           Yu = j * h + 0.5 * h - h;
        const Xv = i * h + 0.5 * h - h, Yv = j * h - h;
        uData[i * n + j] =  A * Math.sin(k * Xu) * Math.cos(k * Yu);
        vData[i * n + j] = -A * Math.cos(k * Xv) * Math.sin(k * Yv);
      }

    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    interaction.showObstacle = false;
    const oneStep = async (nu) => {
      solver.resetFlipState();
      solver.setParams({ nu, dt: 1 / 120, omega: 1.9, density: 1000 });
      solver.writeSolidMask(sData);
      solver.writeVelocityU(uData);
      solver.writeVelocityV(vData);
      solver.writeSmoke(new Float32Array(numX * n).fill(1));
      device.queue.writeBuffer(solver.pressureBuffer, 0, new Float32Array(numX * n));
      solver.step(ui.numIters);
      await device.queue.onSubmittedWorkDone();
      const { u, v } = solver.velocityBuffers;
      return {
        u: await readBuf(u), v: await readBuf(v),
        sub: solver.viscSubsteps, clamped: solver.viscClamped,
        nuEff: solver.viscNuEff, nuMax: solver.viscNuMax,
      };
    };

    const base = await oneStep(0);
    const visc = await oneStep(NU);
    const dt = solver.params.dt;

    // Least-squares slope through the origin of (actual increment) against
    // (nu*dt*lap5), over interior faces only.
    //
    // The margin is set by the ghost, not by the stencil. Five cells would be
    // enough for every face and stencil neighbour to be FLUID, but the ghost
    // makes the box no-slip and a wall layer grows in from each side: measured
    // at margin 5 it still contributes 65x the bulk increment at the window
    // edge. The layer's scale is 2*sqrt(nu*dt) = 5.1 cells here, so MARGIN = 20
    // is ~4 of those and puts the wall contribution below 1e-2 of the signal.
    let sae = 0, see = 0, saa = 0, nCell = 0, nOut = 0, mag = 0, nBad = 0;
    const lo = MARGIN, hi = Nc - MARGIN + 1;
    for (const key of ['u', 'v']) {
      const w = base[key], wd = visc[key];
      for (let i = lo; i <= hi; i++)
        for (let j = lo; j <= hi; j++) {
          const idx = i * n + j, c = w[idx];
          const actual = wd[idx] - c;
          if (!Number.isFinite(actual) || !Number.isFinite(c)) { nBad++; continue; }
          const lap = (w[idx + n] + w[idx - n] + w[idx + 1] + w[idx - 1] - 4 * c) / (h * h);
          const expected = NU * dt * lap;
          sae += actual * expected; see += expected * expected; saa += actual * actual;
          nCell++; mag = Math.max(mag, Math.abs(c));
          // Cells where the two disagree by more than 3x. A handful sit on the
          // mode's node lines, where the substepping's second-order term and
          // MacCormack's limiter are both most active; a structural error would
          // instead put most of the window here.
          if (Math.abs(actual - expected) > 3 * Math.abs(expected) + 1e-5) nOut++;
        }
    }

    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);
    return {
      slope: sae / see,
      corr: sae / Math.sqrt(saa * see),
      nCell, nOut, mag, nBad,
      sub: visc.sub, clamped: visc.clamped, nuEff: visc.nuEff, nuMax: visc.nuMax,
      baseSub: base.sub,
    };
  }, { NU, MARGIN });

  console.log(
    `nu delivered / nu requested = ${r.slope.toFixed(5)}  corr = ${r.corr.toFixed(6)}  ` +
    `outliers = ${r.nOut}/${r.nCell}  substeps = ${r.sub}  ` +
    `nu_max = ${r.nuMax.toExponential(3)}`);

  // Guards. A dead device or a collapsed field would make the slope 0/0.
  expect(r.nBad).toBe(0);
  expect(r.nCell).toBeGreaterThan(10000);
  expect(r.mag).toBeGreaterThan(0.1);
  expect(r.baseSub).toBe(0);             // the nu = 0 arm really ran without diffusion
  expect(r.sub).toBeGreaterThan(1);
  expect(r.sub % 2).toBe(1);             // odd, so this also pins the parity bookkeeping
  expect(r.clamped).toBe(false);         // a saturated nu is not the nu we asked for
  expect(r.nuEff).toBeCloseTo(NU, 12);
  // The increment must be SHAPED like the Laplacian, not merely sized like it.
  // A shader that scaled the field, or diffused with the wrong stencil, lands
  // far below this even if its slope happened to come out near 1.
  expect(r.corr).toBeGreaterThan(0.995);
  expect(r.nOut / r.nCell).toBeLessThan(0.01);

  // The payload. A 2x coefficient error lands at 2.0 or 0.5; an h vs h/2 error
  // at 4.0 or 0.25; a per-step rather than per-substep dt at `substeps` = 27x.
  // None of those survive a +-2% window.
  expect(r.slope).toBeGreaterThan(0.98);
  expect(r.slope).toBeLessThan(1.02);
});

test('the viscous substep schedule never lets the explicit coefficient exceed 1/4', async ({ page }) => {
  await boot(page);
  const rows = await page.evaluate(async () => {
    const { solver, ui } = window.__flowlab;
    solver.paused = true;
    const h = solver.h, dt = solver.params.dt;

    // Count the diffuse dispatches actually encoded, keyed off the pipeline.
    // Asserting viscSubsteps alone would not notice a loop that reported a
    // count it never dispatched -- in particular the nu = 0 case, where "0
    // substeps" has to mean "no dispatches", not "one harmless one".
    const proto = GPUComputePassEncoder.prototype;
    const origSetPipeline = proto.setPipeline;
    let seen = 0;
    proto.setPipeline = function (pipeline, ...rest) {
      if (pipeline === solver.diffusePipeline) seen++;
      return origSetPipeline.call(this, pipeline, ...rest);
    };

    const out = [];
    try {
      for (const nu of [0, 1e-5, 1e-3, 1e-1]) {
        solver.setParams({ nu });
        seen = 0;
        solver.step(ui.numIters);
        out.push({
          nu,
          used: solver.viscSubsteps,
          clamped: solver.viscClamped,
          nuEff: solver.viscNuEff,
          nuMax: solver.viscNuMax,
          dispatched: seen,
          want: nu === 0 ? 0 : Math.ceil(nu * dt / (0.25 * h * h)),
          nMax: solver.constructor.N_MAX,
          h, dt,
        });
      }
    } finally {
      proto.setPipeline = origSetPipeline;
    }
    return out;
  });

  for (const r of rows) {
    if (r.nu === 0) {
      expect(r.used).toBe(0);
      expect(r.dispatched).toBe(0);      // "no substeps" must mean no dispatches
      expect(r.clamped).toBe(false);
      expect(r.nuEff).toBe(0);
      continue;
    }

    // Below the ceiling nothing changed: N is whatever the 1/4 limit needs.
    // At and above it, nu SATURATES at nuMax and N pins to N_MAX -- we do not
    // truncate N and leave the coefficient above the limit.
    expect(r.nuMax).toBeCloseTo(r.nMax * 0.25 * r.h * r.h / r.dt, 12);
    expect(r.clamped).toBe(r.nu > r.nuMax);
    expect(r.nuEff).toBeCloseTo(Math.min(r.nu, r.nuMax), 12);
    expect(r.used).toBe(Math.max(1, Math.min(r.nMax, r.want)));
    expect(r.dispatched).toBe(r.used);

    // THE assertion this test exists for. The explicit five-point update
    // amplifies the worst mode by |1 - 8*coeff| per substep, so coeff > 1/4 is
    // divergence, not under-diffusion. Truncating N while keeping dt_sub = dt/N
    // -- the scheme this replaced -- overshoots by want/N_MAX: 1.71 at tier 256
    // with nu = 0.1, i.e. ~12.7^32 growth per frame, Inf then NaN inside one
    // frame. Saturating nu instead pins coeff at exactly 1/4.
    const coeff = r.nuEff * (r.dt / r.used) / (r.h * r.h);
    expect(coeff).toBeLessThanOrEqual(0.25 + 1e-12);
  }

  // Guards: both regimes must actually be exercised, or half the assertions
  // above are vacuous.
  expect(rows.some((r) => r.clamped)).toBe(true);
  expect(rows.some((r) => r.nu > 0 && !r.clamped)).toBe(true);
});

test('a saturated viscosity leaves the field finite and bounded', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);

  // The regression guard for the clamp rewrite. Under the previous scheme --
  // N = min(N_MAX, want) with dt_sub = dt/N -- this exact configuration ran at
  // coeff = 1.71 and blew up to Inf, then NaN, inside a single frame, which the
  // next pressure solve spread over the whole field with no way back short of a
  // preset reload. The Reynolds control Task 9 ships reaches this regime at
  // every tier from 256 up, so "clamped" has to mean bounded.
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const numX = solver.numX, n = solver.numY;
    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    solver.setParams({ nu: 1e-1 });          // ~7x nuMax at this tier
    for (let k = 0; k < 60; k++) solver.step(ui.numIters);
    await device.queue.onSubmittedWorkDone();

    const u = await readBuf(solver.velocityBuffers.u);
    const v = await readBuf(solver.velocityBuffers.v);
    let nBad = 0, peak = 0, moving = 0;
    for (let i = 1; i < numX - 1; i++)
      for (let j = 1; j < n - 1; j++) {
        const a = u[i * n + j], b = v[i * n + j];
        if (!Number.isFinite(a) || !Number.isFinite(b)) { nBad++; continue; }
        peak = Math.max(peak, Math.abs(a), Math.abs(b));
        if (Math.abs(a) > 0.05) moving++;
      }
    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);
    return {
      nBad, peak, moving,
      clamped: solver.viscClamped, used: solver.viscSubsteps,
      nu: solver.params.nu, nuEff: solver.viscNuEff, nuMax: solver.viscNuMax,
    };
  });

  // Guards: the configuration really is the saturated one, or this proves
  // nothing. These are deliberately BEFORE the payload so that a regression in
  // the clamp is reported as a non-finite field, not as a bookkeeping mismatch.
  expect(r.clamped).toBe(true);
  expect(r.used).toBe(32);
  expect(r.nu).toBeGreaterThan(r.nuMax);

  // THE payload. Finite, and bounded by the free stream rather than merely
  // "not NaN". Reverting to N = min(N_MAX, want) with dt_sub = dt/N puts the
  // coefficient at 1.71 here and this reaches Inf within one frame.
  expect(r.nBad).toBe(0);
  expect(r.peak).toBeLessThan(5);

  // ...and the saturation is the documented one: nu pinned to nuMax, which is
  // what makes viscClamped mean "effective Re is higher than requested".
  expect(r.nuEff).toBeCloseTo(r.nuMax, 12);
  // And not a dead field: saturation is UNDER-diffusion, so the flow keeps
  // moving. A field damped to zero would also be finite and bounded.
  expect(r.moving).toBeGreaterThan(1000);
});

/**
 * Both viscous behaviour tests below compare an inviscid run against a viscous
 * one from the SAME developed field. This is a shedding wake, so running one
 * branch after the other and sampling at different physical times would compare
 * vortex phase, not viscosity -- hence the snapshot/restore in each.
 */
test('viscosity grows a no-slip boundary layer without damping the free stream', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, ui, interaction } = window.__flowlab;
    solver.paused = true;

    // A lost device returns zeros from every later readback. A zero field has
    // zero near-wall velocity, which would read as a perfect boundary layer.
    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const n = solver.numY, numX = solver.numX, h = solver.h;
    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    const sMask = await readBuf(solver.solidBuffer);
    const snapU = await readBuf(solver.velocityBuffers.u);
    const snapV = await readBuf(solver.velocityBuffers.v);
    const snapP = await readBuf(solver.pressureBuffer);

    const iC = Math.round(interaction.obstacleX / h);
    const jC = Math.round(interaction.obstacleY / h);
    const rC = Math.ceil(interaction.obstacleRadius / h);
    // Same face classification diffuse.wgsl uses.
    const fluidF = (i, j) => sMask[i * n + j] !== 0 && sMask[(i - 1) * n + j] !== 0;
    const buried = (i, j) => sMask[i * n + j] === 0 && sMask[(i - 1) * n + j] === 0;

    // Mean |u| on the first fluid u-face off a no-slip surface, split by
    // surface, plus the undisturbed free stream well upstream of the cylinder.
    const metrics = (u) => {
      let cyl = 0, cylN = 0, dom = 0, domN = 0, far = 0, farN = 0, nBad = 0;
      for (let i = 1; i < numX - 1; i++)
        for (let j = 1; j < n - 1; j++) {
          if (!fluidF(i, j)) continue;
          const a = Math.abs(u[i * n + j]);
          if (!Number.isFinite(a)) { nBad++; continue; }
          const atWall = buried(i, j - 1) || buried(i, j + 1);
          const nearCyl = Math.abs(i - iC) <= 2 * rC && Math.abs(j - jC) <= 2 * rC;
          if (atWall && nearCyl) { cyl += a; cylN++; }
          else if (atWall) { dom += a; domN++; }
          if (i > 20 && i < iC - 3 * rC && j > 0.3 * n && j < 0.7 * n) { far += a; farN++; }
        }
      return { cyl: cyl / cylN, cylN, dom: dom / domN, domN, far: far / farN, farN, nBad };
    };

    const run = async (nu) => {
      solver.writeVelocityU(snapU);
      solver.writeVelocityV(snapV);
      device.queue.writeBuffer(solver.pressureBuffer, 0, snapP);
      solver.resetFlipState();
      solver.setParams({ nu });
      for (let k = 0; k < 60; k++) solver.step(ui.numIters);
      return metrics(await readBuf(solver.velocityBuffers.u));
    };

    const inviscid = await run(0);
    const viscous = await run(2.5e-3);   // Re = U*D/nu ~ 48
    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);
    return { inviscid, viscous, substeps: solver.viscSubsteps, clamped: solver.viscClamped };
  });

  // Guards: a dead device, a collapsed field, or an empty face set would make
  // every ratio below meaningless.
  expect(r.inviscid.nBad + r.viscous.nBad).toBe(0);
  expect(r.inviscid.cylN).toBeGreaterThan(10);
  expect(r.inviscid.domN).toBeGreaterThan(100);
  expect(r.inviscid.far).toBeGreaterThan(0.5);   // free stream is really flowing
  expect(r.substeps).toBeGreaterThan(1);
  expect(r.clamped).toBe(false);                 // this nu must resolve, not clamp

  // (1) No-slip takes hold: the first fluid face off a wall loses almost all of
  // its velocity. Inviscid, the walls are free-slip and it keeps the free
  // stream. This is the assertion that dies if u_neighbor stops returning
  // -center for a buried face -- skipping those faces leaves the ratio near 1.
  expect(r.viscous.dom).toBeLessThan(r.inviscid.dom * 0.5);
  expect(r.viscous.cyl).toBeLessThan(r.inviscid.cyl * 0.85);

  // (2) It is a boundary LAYER, not global damping: the free stream upstream of
  // the cylinder is untouched. Without this, a shader that simply scaled the
  // whole field down would satisfy (1).
  expect(r.viscous.far).toBeGreaterThan(r.inviscid.far * 0.9);
});

test('an odd substep count leaves the result on the hat pair, and step() follows it', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);

  // Substep parity is the one piece of this task's bookkeeping that no other
  // test pins with a field assertion. The two behavioural tests both run N = 6,
  // and the boundary-layer test asserts substeps > 1, so mutating
  // `velNext = src` to `velNext = dst` passes the entire suite -- and at N = 1,
  // the HIGH-Re end of Task 9's slider, that mutation makes viscosity a total
  // no-op: step() would publish the un-diffused combine output instead.
  //
  // N = 1 is the sharpest case: it is odd, so the result is on the hat pair
  // rather than tilde, and it is the only count at which the wrong slot holds a
  // field that is exactly the inviscid answer.
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const numX = solver.numX, n = solver.numY, h = solver.h;
    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    const snapU = await readBuf(solver.velocityBuffers.u);
    const snapV = await readBuf(solver.velocityBuffers.v);
    const snapP = await readBuf(solver.pressureBuffer);

    // Just under the single-substep ceiling: want = ceil(0.96) = 1, so N = 1
    // with the coefficient at 0.24 -- as large a one-substep kick as the
    // stability limit allows, which keeps the field difference well clear of
    // float32 noise.
    const NU1 = 0.24 * h * h / solver.params.dt;

    const oneStep = async (nu) => {
      solver.writeVelocityU(snapU);
      solver.writeVelocityV(snapV);
      device.queue.writeBuffer(solver.pressureBuffer, 0, snapP);
      solver.resetFlipState();                 // _velCur = 0
      solver.setParams({ nu });
      solver.step(ui.numIters);
      await device.queue.onSubmittedWorkDone();
      return {
        u: await readBuf(solver.velocityBuffers.u),
        cur: solver._velCur, sub: solver.viscSubsteps,
      };
    };

    const inv = await oneStep(0);
    const visc = await oneStep(NU1);

    let maxDiff = 0, mag = 0, nBad = 0, changed = 0;
    for (let i = 1; i < numX - 1; i++)
      for (let j = 1; j < n - 1; j++) {
        const a = inv.u[i * n + j], b = visc.u[i * n + j];
        if (!Number.isFinite(a) || !Number.isFinite(b)) { nBad++; continue; }
        const d = Math.abs(a - b);
        maxDiff = Math.max(maxDiff, d);
        mag = Math.max(mag, Math.abs(a));
        if (d > 1e-6) changed++;
      }

    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);
    return {
      invCur: inv.cur, viscCur: visc.cur, sub: visc.sub, invSub: inv.sub,
      maxDiff, mag, changed, nBad, NU1,
    };
  });

  // The schedule really is the odd one this test is about.
  expect(r.sub).toBe(1);
  expect(r.invSub).toBe(0);

  // Slot bookkeeping. From _velCur = 0 the MacCormack combine lands on tilde =
  // 2, which is where the inviscid step publishes. One diffusion substep reads
  // tilde and writes hat = 1, so the viscous step must publish slot 1.
  expect(r.invCur).toBe(2);
  expect(r.viscCur).toBe(1);

  // The field assertion, which is the half that slot bookkeeping alone does not
  // give: publishing tilde after an odd count would hand back the untouched
  // combine output, so the two runs would agree bit for bit.
  expect(r.nBad).toBe(0);
  expect(r.mag).toBeGreaterThan(0.1);        // guard: not a collapsed field
  expect(r.maxDiff).toBeGreaterThan(1e-3);
  expect(r.changed).toBeGreaterThan(1000);   // and not one lucky cell
});

test('the viscous stencil cannot read the stale i=0 / j=0 ring', async ({ page }) => {
  test.setTimeout(120_000);
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    let deviceLost = false;
    device.lost.then((info) => { deviceLost = info.message || 'lost'; });

    const n = solver.numY, numX = solver.numX;
    const size = numX * n * 4;
    const readBuf = async (src) => {
      const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, st, 0, size);
      device.queue.submit([e.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(st.getMappedRange().slice(0));
      st.unmap(); st.destroy();
      return out;
    };

    const snapU = await readBuf(solver.velocityBuffers.u);
    const snapV = await readBuf(solver.velocityBuffers.v);
    const snapP = await readBuf(solver.pressureBuffer);
    const EPS = 1e-3;

    // advect_velocity and maccormack_velocity both return for i < 1 or j < 1,
    // so the i=0 column and the j=0 row are never written by advection: the slot
    // that goes live after advection carries a ring three steps old. A diffusion
    // stencil reads its neighbours, so it is the first pass that would march
    // that ring inward -- up to N_MAX times per frame.
    //
    // Poison u's i=0 column and v's j=0 row. These two lines are provably dead
    // to the rest of the solver: u_stencil clamps i0 >= 1 and v_stencil clamps
    // j0 >= 1, so neither advection sample ever loads them; u_departure and
    // v_departure reach only v[i-1] and u[j-1], which boundary.wgsl rewrites;
    // and pressure.wgsl weights both by a solid neighbour's s = 0. The
    // inviscid control below asserts exactly that. So any difference this
    // produces is attributable to the viscous pass alone.
    const poison = (u, v) => {
      const pu = new Float32Array(u), pv = new Float32Array(v);
      for (let j = 0; j < n; j++) pu[j] = 1e6;           // u at i = 0
      for (let i = 0; i < numX; i++) pv[i * n] = 1e6;    // v at j = 0
      return { pu, pv };
    };

    // ALL FOUR ring lines, perturbed rather than overwritten. The other two --
    // u along j = 0 and v along i = 0 -- are NOT covered by the FLUID predicate:
    // the u-face at j = 1 has both its flanking cells (i-1, 1) and (i, 1) in the
    // fluid, so that face IS diffused and its stencil does reach u[i][0]. What
    // stops it is the index guard in u_face_buried/v_face_buried, with the solid
    // mask agreeing only because every shipped preset happens to mark those
    // lines solid. So these two lines need their own probe.
    //
    // Additive, and small. A 1e6 overwrite passes through MacCormack's limiter,
    // which clamps the correction to the local min/max of the forward stencil --
    // the probe SATURATES, and the small response it produces says nothing about
    // the gain at realistic ring staleness. EPS = 1e-3 is a perturbation the
    // limiter does not clip, so response/EPS is an actual gain.
    const perturb = (u, v, eps) => {
      const pu = new Float32Array(u), pv = new Float32Array(v);
      for (let j = 0; j < n; j++)    { pu[j] += eps;         pv[j] += eps; }
      for (let i = 0; i < numX; i++) { pu[i * n] += eps;     pv[i * n] += eps; }
      return { pu, pv };
    };

    const run = async (mode, nu) => {
      const { pu, pv } =
        mode === 'poison' ? poison(snapU, snapV) :
        mode === 'perturb' ? perturb(snapU, snapV, EPS) :
        { pu: snapU, pv: snapV };
      solver.writeVelocityU(pu);
      solver.writeVelocityV(pv);
      device.queue.writeBuffer(solver.pressureBuffer, 0, snapP);
      solver.resetFlipState();
      solver.setParams({ nu });
      for (let k = 0; k < 12; k++) solver.step(ui.numIters);
      return { u: await readBuf(solver.velocityBuffers.u), sub: solver.viscSubsteps };
    };

    const diff = (a, b) => {
      let m = 0, mag = 0, nBad = 0;
      for (let i = 1; i < numX - 1; i++)
        for (let j = 1; j < n - 1; j++) {
          const x = a[i * n + j], y = b[i * n + j];
          if (!Number.isFinite(x) || !Number.isFinite(y)) { nBad++; continue; }
          m = Math.max(m, Math.abs(x - y));
          mag = Math.max(mag, Math.abs(x));
        }
      return { m, mag, nBad };
    };

    const cleanVisc = await run('clean', 2.5e-3);
    const dirtyVisc = await run('poison', 2.5e-3);
    const cleanInv = await run('clean', 0);
    const dirtyInv = await run('poison', 0);

    // Phase B: all four lines, non-saturating.
    const pertVisc = await run('perturb', 2.5e-3);
    const pertInv = await run('perturb', 0);

    if (deviceLost) throw new Error('GPU device lost mid-run: ' + deviceLost);
    return {
      visc: diff(cleanVisc.u, dirtyVisc.u),
      inviscid: diff(cleanInv.u, dirtyInv.u),
      gainVisc: diff(cleanVisc.u, pertVisc.u).m / EPS,
      gainInv: diff(cleanInv.u, pertInv.u).m / EPS,
      EPS,
      substeps: cleanVisc.sub,
    };
  });

  expect(r.visc.nBad + r.inviscid.nBad).toBe(0);
  expect(r.substeps).toBeGreaterThan(1);       // guard: one substep barely exercises it
  expect(r.visc.mag).toBeGreaterThan(0.1);     // guard: a collapsed field diffs to 0 trivially

  // Control: without the viscous pass these two lines reach nothing. If this
  // ever becomes non-zero the poison is no longer isolating the viscous pass
  // and the assertion below stops meaning what it says.
  expect(r.inviscid.m).toBe(0);

  // The payload: turning viscosity on must not open a path to them. Bit-exact,
  // because diffuse.wgsl classifies i == 0 and j == 0 as BURIED by index and
  // substitutes a ghost, so the poisoned entries are never loaded.
  expect(r.visc.m).toBe(0);

  // Phase B: all four ring lines, perturbed by EPS rather than overwritten.
  //
  // The inviscid gain here is a PRE-EXISTING defect in the MacCormack velocity
  // chain, not this task's: the forward pass writes the hat pair but not its
  // ring, so hat's ring is stale, and the backward pass then samples fu at
  // j0 = 0 and fv at i0 = 0 and loads it. It is measured, not fixed -- fixing
  // it means making the advect passes write their ring, which is its own task.
  // What is asserted is only that the viscous pass does not make it worse.
  console.log(
    `ring-leak gain: inviscid = ${r.gainInv.toExponential(3)}  ` +
    `viscous = ${r.gainVisc.toExponential(3)}  (eps = ${r.EPS})`);

  // The leak is real but bounded well below 1 -- a perturbation of the ring
  // does not reach the interior at anything like full strength.
  expect(r.gainInv).toBeGreaterThan(0);        // guard: the probe must actually probe
  expect(r.gainInv).toBeLessThan(1);
  // And viscosity does not amplify it. If the diffusion stencil ever started
  // loading the ring, N_MAX substeps of direct injection per frame would put
  // this far above the inviscid path rather than beside it.
  expect(r.gainVisc).toBeLessThan(r.gainInv * 3);
});

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

    // The GPU stores velocities as f32; compare against the exact f32 values
    // rather than the JS f64 literals 0.7 / -0.3.
    const VX = Math.fround(0.7);
    const VY = Math.fround(-0.3);

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
        shape: c.shape, centerX: c.cx, centerY: c.cy, vx: VX, vy: VY,
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
          if (inHere || inLeft) { if (u[idx] !== VX) uMismatch++; }
          // v: cell-owned only — the CPU rasterizer writes no neighbour v face.
          if (inHere) { if (v[idx] !== VY) vMismatch++; }
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

test('the inflow slider respects the backwardStep column-1 mask', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true;

    // Switch to the backward-step preset so the step block makes the lower
    // half of column 1 solid and the original preset inflow is masked there.
    ui._loadAndApplyPreset('backwardStep');

    const { numX, numY, h } = solver;
    const n = numY;
    const domainHeight = numY * h;
    const sg = (await import('/js/presets.js')).PRESETS.backwardStep.stepGeometry;

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

    ui._setInflowVelocity(2.5);

    const after = [];
    for (const pair of solver.velPairs) after.push(await readBuf(pair.u));

    // Rows inside the step block: cell-center y is below the step top, so the
    // preset's loadPreset() never wrote an inflow u-face there. The slider
    // must leave these buried faces at 0 in every slot.
    // Rows above the step: the inflow boundary condition applies, so the
    // slider must write the new value 2.5 in every slot.
    let maskedBad = 0;
    let fluidBad = 0;
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < n; j++) {
        const cy = (j + 0.5) * h / domainHeight;
        const val = after[k][1 * n + j];
        if (cy < sg.y1) {
          if (val !== 0) maskedBad++;
        } else {
          if (val !== 2.5) fluidBad++;
        }
      }
    }

    // The persistent boundaryVelData.uData slice is what main.js re-applies
    // every frame after the pressure solve; it must carry the same mask.
    let bvMaskedBad = 0;
    let bvFluidBad = 0;
    for (let j = 0; j < n; j++) {
      const cy = (j + 0.5) * h / domainHeight;
      const val = ui.boundaryVelData.uData[1 * n + j];
      if (cy < sg.y1) {
        if (val !== 0) bvMaskedBad++;
      } else {
        if (val !== 2.5) bvFluidBad++;
      }
    }

    return { maskedBad, fluidBad, bvMaskedBad, bvFluidBad };
  });

  // Mutation caught: unconditional slider write stuffs inVel into solid/buried
  // faces inside the step block, which advect.wgsl then bilinearly samples.
  expect(r.maskedBad).toBe(0);
  // Mutation caught: the mask accidentally zeros rows that should inflow.
  expect(r.fluidBad).toBe(0);
  expect(r.bvMaskedBad).toBe(0);
  expect(r.bvFluidBad).toBe(0);
});


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
    const oldBBRawIMax = Math.min(numX - 2, Math.ceil((interaction.obstacleX - 0.15 * numX * h + maxExtent) / h + 1));
    const oldBB = {
      iMin: Math.max(1, Math.floor((interaction.obstacleX - 0.15 * numX * h - maxExtent) / h - 1)),
      // The restore zeroes the u-face one column right of the old bbox — the
      // CPU rasterizer did the same (its restore loop wrote uData[(i+1)*n+j]
      // at i=iMax). Include that face column in the union.
      iMax: Math.min(numX - 1, oldBBRawIMax + 1),
      jMin: Math.max(1, Math.floor((interaction.obstacleY + 0.10 * numY * h - maxExtent) / h - 1)),
      jMax: Math.min(numY - 2, Math.ceil((interaction.obstacleY + 0.10 * numY * h + maxExtent) / h + 1)),
    };
    const inUnion = (i, j) =>
      (i >= newBB.iMin && i <= newBB.iMax && j >= newBB.jMin && j <= newBB.jMax) ||
      (i >= oldBB.iMin && i <= oldBB.iMax && j >= oldBB.jMin && j <= oldBB.jMax) ||
      // The GPU rasterizer owns the u face by the cell to its right (the
      // left neighbour's right face), so a one-cell i-shadow can change.
      (i - 1 >= newBB.iMin && i - 1 <= newBB.iMax && j >= newBB.jMin && j <= newBB.jMax);

    const after = { s: await readBuf(solver.solidBuffer), p: await readBuf(solver.pressureBuffer), slots: [] };
    const sBoundaryAfter = await readBuf(solver.sBoundary);
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
    // Three-slot fan-out: the cells the shader actually wrote (non-boundary
    // solid obstacle cells inside the new bbox) must agree across slots.
    // Fluid cells inside the bbox legitimately retain slot-specific advected
    // values from the steps that ran before solver.paused was set, so the
    // whole-bbox assertion cannot hold.
    let slotSkew = 0;
    for (let i = newBB.iMin; i <= newBB.iMax; i++) {
      for (let j = newBB.jMin; j <= newBB.jMax; j++) {
        const idx = i * n + j;
        if (after.s[idx] !== 0.0 || sBoundaryAfter[idx] === 0.0) continue;
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

test('the rasterizer never carves the i=numX-1 outflow column or j-ring cells', async ({ page }) => {
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
    const VX = Math.fround(0.6);
    const VY = Math.fround(-0.4);

    // Center the circle so its footprint spills into the open outflow column
    // i=numX-1, which the pre-PR CPU rasterizer never carved because its bbox
    // was clamped to i <= numX-2. Cell (numX-2, jMid) center is exactly here.
    const centerX = (numX - 1.5) * h;
    const centerY = 0.5 * numY * h;
    const radius = 4.0 * h;

    const uBefore = await readBuf(solver.velPairs[solver._velCur].u);
    const vBefore = await readBuf(solver.velPairs[solver._velCur].v);

    solver.rasterizeObstacle({
      shape: 0, centerX, centerY, vx: VX, vy: VY, radius, angle: 0,
      prevBBox: [1, 0, 0, 0],
    });

    const s = await readBuf(solver.solidBuffer);
    const u = await readBuf(solver.velPairs[solver._velCur].u);
    const v = await readBuf(solver.velPairs[solver._velCur].v);

    // CPU-oracle inside-test for the legitimate left-neighbour u-face write.
    const insideCircle = (i, j) => {
      const dx = (i + 0.5) * h - centerX;
      const dy = (j + 0.5) * h - centerY;
      return dx * dx + dy * dy < radius * radius;
    };

    let sCarved = 0;
    let vChanged = 0;
    let uWrong = 0;
    let ringCarved = 0;

    for (let j = 1; j < n - 1; j++) {
      const idx = (numX - 1) * n + j;
      if (sBoundaryArr[idx] === 0) continue; // belt-and-braces
      if (s[idx] !== 1.0) sCarved++;
      if (v[idx] !== vBefore[idx]) vChanged++;
      const leftInside = insideCircle(numX - 2, j);
      const expectU = leftInside ? VX : uBefore[idx];
      if (u[idx] !== expectU) uWrong++;
    }

    // The top/bottom j-ring is boundary-mask solid and must stay that way.
    for (let i = 0; i < numX; i++) {
      if (s[i * n + 0] !== 0.0) ringCarved++;
      if (s[i * n + (n - 1)] !== 0.0) ringCarved++;
    }

    return { sCarved, vChanged, uWrong, ringCarved };
  });
  expect(r.sCarved).toBe(0);
  expect(r.vChanged).toBe(0);
  expect(r.uWrong).toBe(0);
  expect(r.ringCarved).toBe(0);
});

test('the Interaction.SHAPES name-to-enum mapping matches the WGSL shape cases', async ({ page }) => {
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

    // CPU oracle inside-test (verbatim port of the pre-PR-A code).
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

    const bboxOf = (cx, cy, r) => {
      const maxExtent = Math.max(r, r * 4 * 0.5, r * 3 * 0.5);
      return [
        Math.max(1, Math.floor((cx - maxExtent) / h - 1)),
        Math.min(numX - 2, Math.ceil((cx + maxExtent) / h + 1)),
        Math.max(1, Math.floor((cy - maxExtent) / h - 1)),
        Math.min(numY - 2, Math.ceil((cy + maxExtent) / h + 1)),
      ];
    };

    // Pin the HTML picker list to the enum.
    const SHAPES = interaction.constructor.SHAPES;
    const pickerValues = Array.from(document.querySelectorAll('[data-shape]')).map(b => b.dataset.shape);
    const pickerBad = pickerValues.filter(v => !SHAPES.includes(v));

    const W = numX * h, H = numY * h;
    // Same parameters as the existing CPU-oracle test, chosen off grid lines
    // and away from axis alignment so the f32 WGSL trig and f64 JS oracle agree.
    const paramsByShape = [
      { cx: 0.62 * W, cy: 0.37 * H, r: 0.055, angle: 0.73 },
      { cx: 0.55 * W, cy: 0.61 * H, r: 0.070, angle: 0.50 },
      { cx: 0.48 * W, cy: 0.42 * H, r: 0.045, angle: -0.31 },
      { cx: 0.70 * W, cy: 0.55 * H, r: 0.060, angle: 1.19 },
    ];

    const results = [];
    for (let shapeIdx = 0; shapeIdx < SHAPES.length; shapeIdx++) {
      const name = SHAPES[shapeIdx];
      const p = paramsByShape[shapeIdx];
      interaction.activeShape = name;
      interaction.obstacleRadius = p.r;
      interaction.obstacleAngle = p.angle;
      interaction.rasterizeObstacle(p.cx, p.cy, 0, 0);

      const s = await readBuf(solver.solidBuffer);
      const [iMin, iMax, jMin, jMax] = bboxOf(p.cx, p.cy, p.r);
      let maskMismatch = 0;
      for (let i = iMin; i <= iMax; i++) {
        for (let j = jMin; j <= jMax; j++) {
          const idx = i * n + j;
          const bnd = sBoundaryArr[idx] === 0;
          const expectS = bnd ? 0 : (insideAt(shapeIdx, p.cx, p.cy, p.r, p.angle, i, j) ? 0 : 1);
          if (s[idx] !== expectS) maskMismatch++;
        }
      }
      results.push({ name, shapeIdx, maskMismatch });
    }
    return { pickerValues, pickerBad, results };
  });

  expect(r.pickerBad).toEqual([]);
  for (const res of r.results) {
    expect(res.maskMismatch, `${res.name} (shape ${res.shapeIdx}) mask mismatch`).toBe(0);
  }
});

test('the three-slot rotation and boundary mask survive an applyTier buffer recreation', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, adaptive, ui, device } = window.__flowlab;
    solver.paused = true;

    adaptive.currentTierIndex = 0; // tier 64 — cheap; any tier exercises the path
    adaptive.applyTier(); // resize -> reapplyCurrentPreset -> resetFlipState
    solver.setParams({ nu: 0 }); // inviscid +2 rotation; reapplyCurrentPreset restores nonzero nu from slider

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

/**
 * MEASUREMENT SCAFFOLD for ADR-0011 — NOT the committed gate.
 *
 * Scripted constant-velocity drag through a quieted field (inflow off, all
 * velocity slots zeroed), K solver steps, near-wall metric read back per step.
 * The near-wall set is exactly the faces the ghost change can touch: FLUID
 * u-faces with at least one BURIED stencil neighbour. Run identically on
 * master @ 9e05cf7 and on the branch; the delta is the defect closure.
 *
 * This scaffold asserts the defect's PRESENCE (floor leg below) and therefore
 * FAILS once the fix lands — Task 3 rewrites it into the committed gate.
 */
test('MEASUREMENT SCAFFOLD: scripted constant-velocity drag near-wall metric', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device, interaction, ui } = window.__flowlab;
    solver.paused = true;
    const n = solver.numY, numX = solver.numX, h = solver.h, dt = solver.params.dt;
    const size = numX * n * 4;
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

    const VX = 1.0, K = 40;
    // 4 substeps at coeff 0.2: nu*(dt/4)/h^2 = 0.2 and ceil(nu*dt/(0.25*h^2)) = 4,
    // well under viscNuMax (32 substeps at 0.25), so no saturation.
    const NU = 0.8 * h * h / dt;
    const CX0 = 0.3 * numX * h, CY = 0.5 * n * h;

    const runOnce = async () => {
      // Quiet the field: inflow off, velocity zeroed in every rotation slot.
      ui._setInflowVelocity(0);
      const zeros = new Float32Array(numX * n);
      for (const p of solver.velPairs) {
        device.queue.writeBuffer(p.u, 0, zeros);
        device.queue.writeBuffer(p.v, 0, zeros);
      }
      solver.params.nu = NU;
      // Boot default (karmanVortex) radius 0.06 yields only 28 near-wall faces —
      // below the >50 non-vacuity leg. 0.15 populates the set (measured cnt = 64).
      // Set inside runOnce so both determinism replays use the identical geometry.
      interaction.obstacleRadius = 0.15;
      // Park the obstacle at the start position, stationary, then drag +x.
      interaction.rasterizeObstacle(CX0, CY, 0, 0);
      let peak = -Infinity, trough = Infinity;
      let M = NaN, cnt = 0;
      for (let k = 0; k < K; k++) {
        interaction.rasterizeObstacle(CX0 + (k + 1) * VX * dt, CY, VX, 0);
        solver.step(ui.numIters);
        const s = await readBuf(solver.solidBuffer);
        const u = await readBuf(solver.velocityBuffers.u);
        let sum = 0; cnt = 0;
        for (let i = 2; i < numX - 2; i++) {
          for (let j = 2; j < n - 2; j++) {
            const fluid = s[i * n + j] !== 0 && s[(i - 1) * n + j] !== 0;
            if (!fluid) continue;
            const buried = (a, b) => s[a * n + b] === 0 && s[(a - 1) * n + b] === 0;
            if (buried(i + 1, j) || buried(i - 1, j) || buried(i, j + 1) || buried(i, j - 1)) {
              const uij = u[i * n + j];
              sum += uij; cnt++;
              peak = Math.max(peak, uij);
              trough = Math.min(trough, uij);
            }
          }
        }
        M = sum / cnt;
      }
      return { M, cnt, peak, trough };
    };

    const a = await runOnce();
    const b = await runOnce();
    return { a, b, substeps: solver.viscSubsteps };
  });

  // Determinism pin: two identical replays in one session agree bit-exactly.
  // Without this the master-vs-branch delta would be uninterpretable.
  expect(r.b.M).toBe(r.a.M);
  // Non-vacuity: dozens of ghost-read faces, and the intended 4 substeps ran.
  expect(r.a.cnt).toBeGreaterThan(50);
  expect(r.substeps).toBe(4);
  // Defect-regime floor (master / pre-fix only — Task 3 REMOVES this leg):
  // the near-wall ring measurably lags vx. If this fails on master the
  // harness is not sitting in the regime where the defect bites.
  expect(1.0 - r.a.M).toBeGreaterThan(0.1);
  console.log('NEARWALL_M', r.a.M, 'peak', r.a.peak, 'trough', r.a.trough, 'cnt', r.a.cnt);
});

/**
 * ADR-0011 unit pin: one isolated diffuse dispatch over a crafted field.
 * Face (I, J+1) is BURIED-BY-MASK (both flanking cells solid) and stores the
 * wall velocity W — the rasterizer writes the drag velocity into every inside
 * cell's own face, so a mask-buried face's stored value IS the wall velocity.
 * The ghost must be W + (W - C), placing W on the wall line half a cell away.
 */
test('the viscous ghost places the stored wall velocity on the wall line', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;
    const n = solver.numY, numX = solver.numX, h = solver.h;
    const size = numX * n * 4;
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

    // Crafted field: fluid everywhere except a two-cell solid block above the
    // probe face (I, J), so face (I, J+1) is buried-by-mask and stores W.
    const I = Math.floor(numX / 2), J = Math.floor(n / 2);
    const s = new Float32Array(numX * n).fill(1);
    s[I * n + (J + 1)] = 0;
    s[(I - 1) * n + (J + 1)] = 0;
    const W = 0.7, C = 1.1, A = 0.4, B = 0.2, D = 0.9;
    const u0 = new Float32Array(numX * n);
    u0[I * n + J] = C;
    u0[(I + 1) * n + J] = A;
    u0[(I - 1) * n + J] = B;
    u0[I * n + (J - 1)] = D;
    u0[I * n + (J + 1)] = W; // the buried face's stored wall velocity
    device.queue.writeBuffer(solver.solidBuffer, 0, s);
    device.queue.writeBuffer(solver.velPairs[0].u, 0, u0);
    device.queue.writeBuffer(solver.velPairs[0].v, 0, new Float32Array(numX * n));

    // One isolated substep, slot 0 -> slot 1. NU chosen so coeff = 0.1:
    // coeff = NU*DT/(h*h) with DT the full frame dt (1 substep).
    const DT = solver.params.dt;
    const NU = 0.1 * h * h / DT;
    solver._writeParamsTo(solver.uniformBufVisc, 0, DT, NU);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(solver.diffusePipeline);
    pass.setBindGroup(0, solver.diffuse[0][1]);
    pass.dispatchWorkgroups(Math.ceil(numX / 8), Math.ceil(n / 8), 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    const u1 = await readBuf(solver.velPairs[1].u);

    // f32-replicated expectation in WGSL evaluation order:
    // coeff = nu*dt/(h*h); ghost = w+(w-c); lap = ((ghost+A)+B)+D-4c; out = c+coeff*lap.
    const f = Math.fround;
    const coeff = f(f(NU * DT) / f(h * h));
    const ghost = f(W + f(W - C));
    const lap = f(f(f(f(ghost + A) + B) + D) - f(4 * C));
    const expected = f(C + f(coeff * lap));
    return { actual: u1[I * n + J], expected };
  });

  // 1e-6 ≈ 8 ulp at magnitude ~1 covers f32/f64 transcription; WGSL does not
  // implicitly contract to fma, so this is replication, not fitting.
  // MUTATION: reverting the ghost to `-center` shifts the result by
  // coeff*(ghost-(-C)) = 0.1*(0.3+1.1) = 0.14 — five orders above the band.
  expect(Math.abs(r.actual - r.expected)).toBeLessThan(1e-6);
});

test('a stationary stored velocity reduces the ghost to -center exactly', async ({ page }) => {
  await boot(page);
  const r = await page.evaluate(async () => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;
    const n = solver.numY, numX = solver.numX, h = solver.h;
    const size = numX * n * 4;
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

    // Same scaffold as the test above but with W = 0: the ghost must equal
    // -center, i.e. the pre-ADR-0011 result, so stationary runs are unchanged.
    const I = Math.floor(numX / 2), J = Math.floor(n / 2);
    const s = new Float32Array(numX * n).fill(1);
    s[I * n + (J + 1)] = 0;
    s[(I - 1) * n + (J + 1)] = 0;
    const C = 1.1, A = 0.4, B = 0.2, D = 0.9;
    const u0 = new Float32Array(numX * n);
    u0[I * n + J] = C;
    u0[(I + 1) * n + J] = A;
    u0[(I - 1) * n + J] = B;
    u0[I * n + (J - 1)] = D;
    // buried face (I, J+1) stores 0 — a stationary wall.
    device.queue.writeBuffer(solver.solidBuffer, 0, s);
    device.queue.writeBuffer(solver.velPairs[0].u, 0, u0);
    device.queue.writeBuffer(solver.velPairs[0].v, 0, new Float32Array(numX * n));

    const DT = solver.params.dt;
    const NU = 0.1 * h * h / DT;
    solver._writeParamsTo(solver.uniformBufVisc, 0, DT, NU);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(solver.diffusePipeline);
    pass.setBindGroup(0, solver.diffuse[0][1]);
    pass.dispatchWorkgroups(Math.ceil(numX / 8), Math.ceil(n / 8), 1);
    pass.end();
    device.queue.submit([enc.finish()]);
    const u1 = await readBuf(solver.velPairs[1].u);

    const f = Math.fround;
    const coeff = f(f(NU * DT) / f(h * h));
    const ghost = f(0 + f(0 - C)); // must be exactly -C, including -0 handling
    const lap = f(f(f(f(ghost + A) + B) + D) - f(4 * C));
    const expected = f(C + f(coeff * lap));
    return { actual: u1[I * n + J], expected, ghost, negC: f(-C) };
  });

  // The zero-control leg of the ADR gate, per face: w = 0 is the old rule.
  expect(r.ghost).toBe(r.negC);
  // MUTATION: a formulation like 2*w - center passes numerically here but
  // flips the sign of a zero center — the bit-identity claim rests on
  // w + (w - center); reading a NEIGHBOUR's stored value instead of the
  // face's own would fail the first test, not this one.
  expect(Math.abs(r.actual - r.expected)).toBeLessThan(1e-6);
});
