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

test('_velCur advances by 1 and _smokeCur by 2, each visiting all three slots', async ({ page }) => {
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

  // Velocity advection writes one slot ahead, so _velCur advances by 1.
  expect(velSeq).toEqual([0, 1, 2, 0, 1, 2]);

  // Smoke advances by 2: MacCormack's forward pass writes hat = (c+1)%3 and
  // the backward pass + in-place combine land phi^{n+1} in tilde = (c+2)%3.
  // Starting at 0: 0 -> 2 -> 1 -> 0 -> 2 -> 1. Still visits all three slots,
  // since 2 is coprime with 3 -- a solver that advanced smoke by 1 would read
  // phi^ (the uncorrected forward pass) as the next step's phi^n.
  expect(smokeSeq).toEqual([0, 2, 1, 0, 2, 1]);
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

/** Extracts `fn name(...) { ... }` from already-normalized WGSL by brace matching. */
function extractFn(src, name) {
  const start = src.indexOf(`fn ${name}(`);
  if (start === -1) throw new Error(`fn ${name} not found`);
  let depth = 0;
  for (let k = src.indexOf('{', start); k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}' && --depth === 0) return src.slice(start, k + 1);
  }
  throw new Error(`unbalanced braces in fn ${name}`);
}

test('the combine re-traces with the same stencil code advect_smoke traced with', () => {
  // The limiter is correct ONLY because the combine's re-trace reproduces the
  // forward pass's stencil exactly. WGSL has no modules, so scalar_stencil and
  // smoke_departure are duplicated across the two files. Drift between the
  // copies still yields values in [0,1], so the bounds test below cannot catch
  // it -- it would be a silent wrong-answer path. Hence this textual check.
  // Playwright transpiles specs to CJS, so import.meta is unavailable and
  // config.rootDir points at testDir. npm test runs from the repo root.
  const dir = path.join(process.cwd(), 'static', 'shaders');
  const fwd = normalizeWgsl(readFileSync(path.join(dir, 'advect_smoke.wgsl'), 'utf8'));
  const comb = normalizeWgsl(readFileSync(path.join(dir, 'maccormack.wgsl'), 'utf8'));

  for (const fn of ['scalar_stencil', 'smoke_departure']) {
    expect(
      extractFn(comb, fn),
      `${fn} has drifted between advect_smoke.wgsl and maccormack.wgsl`,
    ).toBe(extractFn(fwd, fn));
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
  // And the step advanced velocity by 1, smoke by 2, from (0, 1).
  expect(dispatched.advanced).toEqual({ vc: 1, sc: 0 });
});
