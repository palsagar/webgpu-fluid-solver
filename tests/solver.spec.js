import { test, expect } from '@playwright/test';

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
// rotation than it looks. See the _velCur/_smokeCur and advectSmoke tests
// below for assertions that pin the rotation itself.
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

test('advectSmoke is a 3x3 table indexed [velCur][smokeCur], not a 1-D array', async ({ page }) => {
  await boot(page);
  const shape = await page.evaluate(() => {
    const { solver } = window.__flowlab;
    return {
      outerLength: solver.advectSmoke.length,
      innerLength: solver.advectSmoke[0].length,
      velIndexIndependentOfSmokeIndex: solver.advectSmoke[1][0] !== solver.advectSmoke[0][0],
    };
  });

  expect(shape.outerLength).toBe(3);
  expect(shape.innerLength).toBe(3);
  // Pins that the velocity axis is genuine, not aliased to the smoke axis --
  // the exact property the 3x3 table (vs. a 3-element array indexed by
  // smokeCur alone) exists to provide.
  expect(shape.velIndexIndependentOfSmokeIndex).toBe(true);
});

test('_velCur and _smokeCur rotate through all three slots in lockstep', async ({ page }) => {
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

  expect(velSeq).toEqual([0, 1, 2, 0, 1, 2]);
  expect(smokeSeq).toEqual([0, 1, 2, 0, 1, 2]);
});

test('smoke advection dispatches the bind group for the live velocity slot, not the smoke slot', async ({ page }) => {
  await boot(page);
  const matches = await page.evaluate(() => {
    const { solver, ui } = window.__flowlab;
    solver.paused = true;
    solver.resetFlipState();

    // Force velCur and smokeCur apart. Under normal operation the two
    // counters always advance together from a shared reset, so they are
    // always equal -- meaning a step() that dispatches advectSmoke keyed by
    // [smokeCur][smokeCur] instead of [velCur][smokeCur] would be
    // indistinguishable from correct code under any public API sequence.
    // Forcing a split here is what makes that bug observable; the 3x3 table
    // exists so step() stays correct once something (e.g. viscous
    // substepping) actually causes this split to happen on its own.
    solver._velCur = 0;
    solver._smokeCur = 1;
    const expected = solver.advectSmoke[solver._velCur][solver._smokeCur];

    let captured = null;
    const proto = GPUComputePassEncoder.prototype;
    const origSetBindGroup = proto.setBindGroup;
    proto.setBindGroup = function (index, bindGroup, ...rest) {
      captured = bindGroup; // smoke advection is the last dispatch in step()
      return origSetBindGroup.call(this, index, bindGroup, ...rest);
    };
    try {
      solver.step(ui.numIters);
    } finally {
      proto.setBindGroup = origSetBindGroup;
    }

    return captured === expected;
  });

  expect(matches).toBe(true);
});
