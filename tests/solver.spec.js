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
 */
function readDivergenceSeries(page, steps) {
  return page.evaluate(async (steps) => {
    const { solver, device, ui } = window.__flowlab;
    solver.paused = true; // stop the render loop from stepping too

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
      solver.step(ui.numIters);
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
  }, steps);
}

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
