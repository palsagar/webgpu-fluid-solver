import { test, expect } from '@playwright/test';

/**
 * Smoke tests for the hybrid render path (ADR-0005).
 *
 * The Field View is a WebGPU canvas, which cannot be read with drawImage or a
 * page screenshot once the frame is presented — both return black. Pixels come
 * from FieldRenderer.captureNextFrame(), which copies the rendered texture in
 * the render pass's own command encoder.
 */

const SOLID_RGB = [50, 50, 60];   // SOLID_COLOR in render_field.wgsl
const MAGMA_CLEAR = [251, 252, 191]; // magma at t=1, i.e. smoke m=1 (no dye)

const near = (c, ref, tol = 4) => c.every((v, i) => Math.abs(v - ref[i]) <= tol);

/** Boots the app and runs `frames` frames so the flow develops. */
async function boot(page, frames = 120) {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.renderer?.fieldRenderer, null, { timeout: 20_000 });
  await runFrames(page, frames);
}

function runFrames(page, n) {
  return page.evaluate((n) => new Promise((resolve) => {
    let seen = 0;
    const tick = () => (++seen >= n ? resolve() : requestAnimationFrame(tick));
    requestAnimationFrame(tick);
  }), n);
}

/** Captures a frame and returns an n-by-n grid of [r,g,b] samples. */
function sampleGrid(page, n = 10) {
  return page.evaluate(async (n) => {
    const cap = await window.__flowlab.renderer.fieldRenderer.captureNextFrame();
    const out = [];
    for (let j = 1; j < n; j++) {
      for (let i = 1; i < n; i++) {
        const x = Math.round((i / n) * cap.width);
        const y = Math.round((j / n) * cap.height);
        const o = (y * cap.width + x) * 4;
        out.push([cap.data[o], cap.data[o + 1], cap.data[o + 2]]);
      }
    }
    return out;
  }, n);
}

/** Captures a frame and samples the pixel at the obstacle's center. */
function sampleObstacleCenter(page) {
  return page.evaluate(async () => {
    const { renderer, interaction, solver } = window.__flowlab;
    const cap = await renderer.fieldRenderer.captureNextFrame();
    const x = Math.round((interaction.obstacleX / (solver.numX * solver.h)) * cap.width);
    const y = Math.round((1 - interaction.obstacleY / (solver.numY * solver.h)) * cap.height);
    const o = (y * cap.width + x) * 4;
    return [cap.data[o], cap.data[o + 1], cap.data[o + 2]];
  });
}

async function showPressure(page) {
  await page.evaluate(() => {
    const { renderer } = window.__flowlab;
    renderer.showSmoke = false;
    renderer.showPressure = true;
  });
  await runFrames(page, 40); // let the throttled pressure readback land
}

test('field renders a non-uniform image', async ({ page }) => {
  await boot(page);
  const unique = new Set((await sampleGrid(page)).map((c) => c.join(',')));
  // A black canvas (broken shader or empty LUT) collapses to a single color
  expect(unique.size).toBeGreaterThan(3);
});

test('smoke view maps clear cells through the magma LUT', async ({ page }) => {
  await boot(page);
  const samples = await sampleGrid(page);
  // The Karman preset is mostly clear (m=1) away from the dye streak
  expect(samples.filter((c) => near(c, MAGMA_CLEAR)).length).toBeGreaterThan(samples.length / 2);
});

test('solid cells render as the in-shader solid color', async ({ page }) => {
  await boot(page);
  expect(near(await sampleObstacleCenter(page), SOLID_RGB)).toBe(true);
});

test('pressure view uses a different colormap than smoke', async ({ page }) => {
  await boot(page);
  const smoke = await sampleGrid(page);
  await showPressure(page);
  const pressure = await sampleGrid(page);

  const clearCount = (s) => s.filter((c) => near(c, MAGMA_CLEAR)).length;
  expect(clearCount(smoke)).toBeGreaterThan(0);
  expect(clearCount(pressure)).toBe(0);
});

test('uniform pressure field renders neutral, not saturated', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.renderer?.fieldRenderer, null, { timeout: 20_000 });
  // Freeze the solver and force a genuinely uniform pressure field. Without
  // this the field has real variation and the assertion below is vacuous.
  await page.evaluate(() => {
    const { solver, device } = window.__flowlab;
    solver.paused = true;
    device.queue.writeBuffer(solver.pressureBuffer, 0, new Float32Array(solver.numX * solver.numY));
  });
  await showPressure(page);

  // Guard the guard: the readback must have actually produced a range, or the
  // renderer falls back to [-1,1] and this test proves nothing.
  const range = await page.evaluate(() => window.__flowlab.renderer._pressureRange);
  expect(range).not.toBeNull();
  expect(range[1] - range[0]).toBeGreaterThan(0);

  const fluid = (await sampleGrid(page)).filter((c) => !near(c, SOLID_RGB));
  // A degenerate [min,max] range maps every cell to the t=0 extreme (deep
  // blue). _computePressureRange widens it so they stay near the neutral center.
  expect(fluid.length).toBeGreaterThan(0);
  expect(fluid.filter(([r, , b]) => b - r > 100).length).toBe(0);
});

test('canvases follow the container when the window resizes', async ({ page }) => {
  await boot(page, 30);
  const before = await page.evaluate(() => {
    const { renderer } = window.__flowlab;
    return [renderer.fieldRenderer.canvas.width, renderer.canvas.width];
  });

  await page.setViewportSize({ width: 900, height: 700 });
  await page.waitForFunction(
    (w) => window.__flowlab.renderer.fieldRenderer.canvas.width !== w,
    before[0],
    { timeout: 5000 },
  );

  const after = await page.evaluate(() => {
    const { renderer } = window.__flowlab;
    return [renderer.fieldRenderer.canvas.width, renderer.canvas.width];
  });
  expect(after[0]).not.toBe(before[0]);
  expect(after[1]).not.toBe(before[1]); // overlay must track the field canvas
  expect(after[0]).toBe(after[1]);
});

test('resizing preserves simulation state', async ({ page }) => {
  await boot(page, 30);
  // Move the obstacle off its preset default and place an emitter
  const before = await page.evaluate(() => {
    const { interaction, particles, solver } = window.__flowlab;
    interaction.obstacleX = solver.numX * solver.h * 0.6;
    particles.addEmitter(solver.numX * solver.h * 0.2, solver.numY * solver.h * 0.5);
    return {
      obstacleX: interaction.obstacleX,
      emitters: particles.emitters.length,
      numX: solver.numX,
    };
  });
  expect(before.emitters).toBeGreaterThan(0);

  await page.setViewportSize({ width: 900, height: 700 });
  await page.waitForFunction(
    (w) => window.__flowlab.renderer.fieldRenderer.canvas.width !== w,
    (await page.evaluate(() => window.__flowlab.renderer.fieldRenderer.canvas.width)) + 1,
    { timeout: 5000 },
  );
  await runFrames(page, 30);

  // applyTier() would destroy buffers, reload the preset, and clear emitters
  const after = await page.evaluate(() => {
    const { interaction, particles, solver } = window.__flowlab;
    return { obstacleX: interaction.obstacleX, emitters: particles.emitters.length, numX: solver.numX };
  });
  expect(after.obstacleX).toBeCloseTo(before.obstacleX, 6);
  expect(after.emitters).toBe(before.emitters);
  expect(after.numX).toBe(before.numX);
});

test('invalidating the solid mask mid-readback is not swallowed', async ({ page }) => {
  await boot(page, 30);
  const done = await page.evaluate(async () => {
    const r = window.__flowlab.renderer;
    // Let any in-flight readback settle first
    while (r._solidReadbackPending) await new Promise(requestAnimationFrame);
    // Silence the render loop: otherwise it issues its own (correct) readback
    // while we wait, and we would observe that one instead of the raced one.
    const realDraw = r.draw;
    r.draw = () => {};
    try {
      r._solidReadbackDone = false;
      r.readbackSolid();   // now in flight
      r.invalidateSolid(); // lands before the map resolves
      while (r._solidReadbackPending) await new Promise(requestAnimationFrame);
      return r._solidReadbackDone;
    } finally {
      r.draw = realDraw;
    }
  });
  // The in-flight copy predates the invalidation, so the mask must stay stale
  // and be re-read next frame — otherwise particles use an obsolete obstacle.
  expect(done).toBe(false);
});
