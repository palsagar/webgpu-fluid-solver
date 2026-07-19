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

    // Column i=1 carries the inflow (presets.js:97) and is preserved only
    // because i=0 is solid (presets.js:94), so the u-face there never advects.
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
  // so phi^ == phi~ == phi^n and `corrected` is exactly phi^. The limiter's
  // phi^ SEED is what makes the clamp the identity there: the combine re-traces
  // unconditionally, and that stencil -- for a trace the forward pass never
  // took -- need not bracket phi^n[idx].
  //
  // Note the revert is FACE-based while cell (1, j) is FLUID, so porting the
  // smoke combine's `if (s[idx] == 0.0)` guard would NOT fire here. Dropping
  // the seed, or clamping to stencil corners alone, fails this test.
  expect(r.inflowDrift).toBe(0);
  expect(r.solidDrift).toBe(0);
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
  const files = [
    'advect.wgsl', 'advect_smoke.wgsl', 'maccormack.wgsl', 'maccormack_velocity.wgsl',
  ];
  const srcs = files.map((f) => [f, readShader(f)]);

  for (const struct of ['Params', 'Stencil']) {
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
