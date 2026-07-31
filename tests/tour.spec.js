import { test, expect } from '@playwright/test';

/**
 * Onboarding tour (spec: docs/superpowers/specs/2026-07-31-onboarding-tour-design.md).
 *
 * The Tour engine is step-agnostic — steps are injected — so most tests drive
 * tiny custom step arrays via `startTour`. The real 12-step STEPS script gets
 * its own walkthrough test in Task 3.
 */

test.beforeEach(async ({ page }) => {
  // Seed 'done' so the Task-4 welcome gate stays out of the way of engine
  // tests. Each Playwright test gets a fresh context, so the seed never
  // leaks between tests; startTour/startRealTour clear it again so flag
  // assertions stay meaningful. Pre-Task-4 builds ignore the flag, hence
  // the click fallback below.
  await page.addInitScript(() => {
    try { localStorage.setItem('flowlab.tour.v1', 'done'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  // Until Task 4 wires the welcome gate, dismiss the current welcome modal so
  // it cannot interfere with pointer probing (its z-index is below the tour's).
  const skip = page.locator('#start-sim-btn');
  if (await skip.isVisible().catch(() => false)) await skip.click();
});

/** Instantiate and start a Tour in the page with a custom step array. */
async function startTour(page, steps) {
  await page.evaluate(async (steps) => {
    const { Tour } = await import('./js/tour.js');
    const { ui, interaction, solver } = window.__flowlab;
    window.__testTour = new Tour({ ui, interaction, solver, steps });
    window.__testTour.start();
  }, steps);
  // Clear the beforeEach seed so the test's own flag writes are what gets asserted.
  await page.evaluate(() => localStorage.removeItem('flowlab.tour.v1'));
}

test('read steps render the overlay, navigate with Next/Back, and tear down on Escape', async ({ page }) => {
  await startTour(page, [
    { target: null, title: 'Step one', body: 'Centered intro.' },
    { target: '#colorbar', title: 'Step two', body: 'Targeted at the colorbar.' },
  ]);

  await expect(page.locator('.tour-dim')).toHaveCount(4);
  await expect(page.locator('.tour-ring')).toHaveCount(1);
  await expect(page.locator('.tour-tooltip')).toHaveCount(1);
  await expect(page.locator('.tour-title')).toHaveText('Step one');
  await expect(page.locator('.tour-counter')).toHaveText('1 / 2');

  await page.locator('.tour-btn-primary').click(); // Next
  await expect(page.locator('.tour-counter')).toHaveText('2 / 2');
  await expect(page.locator('.tour-title')).toHaveText('Step two');

  await page.locator('.tour-btn-secondary').click(); // Back
  await expect(page.locator('.tour-counter')).toHaveText('1 / 2');

  await page.keyboard.press('Escape');
  await expect(page.locator('.tour-dim')).toHaveCount(0);
  await expect(page.locator('.tour-tooltip')).toHaveCount(0);
  const flag = await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'));
  expect(flag).toBe('skipped');
});

test('the dim rects block the page while the hole passes events to the target', async ({ page }) => {
  await startTour(page, [
    { target: '#colorbar', title: 'Only step', body: 'Targeted.' },
  ]);

  const probe = await page.evaluate(() => {
    const dim = document.querySelector('.tour-dim-top');
    const dr = dim.getBoundingClientRect();
    const atDim = document.elementFromPoint(dr.left + dr.width / 2, dr.top + dr.height / 2);
    const cb = document.getElementById('colorbar');
    const cr = cb.getBoundingClientRect();
    const atHole = document.elementFromPoint(cr.left + cr.width / 2, cr.top + cr.height / 2);
    return {
      dimBlocked: atDim?.classList.contains('tour-dim') ?? false,
      holeOpen: atHole === cb || cb.contains(atHole),
    };
  });
  expect(probe.dimBlocked).toBe(true);
  expect(probe.holeOpen).toBe(true);
});

test('a click do-it step has no Next button and advances when the target is clicked', async ({ page }) => {
  await startTour(page, [
    { target: '#btn-play', title: 'Pause it', body: 'Click Pause.', action: { type: 'click' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  await expect(page.locator('.tour-btn-primary')).toHaveCount(0); // do-it: no Next
  await expect(page.locator('.tour-ring')).toHaveClass(/tour-pulse/);

  await page.click('#btn-play'); // through the hole — must reach the app
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
  const paused = await page.evaluate(() => window.__flowlab.solver.paused);
  expect(paused).toBe(true);
});

test('a drag do-it step advances on a canvas drag but not on a click', async ({ page }) => {
  await startTour(page, [
    { target: '#overlay-canvas', title: 'Drag', body: 'Drag the obstacle.', action: { type: 'drag' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  // A bare click (down/up, no travel) must NOT advance.
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.up();
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 60, cy + 30, { steps: 4 });
  await page.mouse.up();
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
});

test('a click-then-canvas do-it step arms on the button and advances in particle mode', async ({ page }) => {
  await startTour(page, [
    { target: '#btn-mode', title: 'Particles', body: 'Click Particles, then the flow.', action: { type: 'click-then-canvas' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  // Canvas click before arming does nothing.
  await page.mouse.click(cx, cy);
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  await page.click('#btn-mode'); // arm — hole retargets to the canvas
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  await page.mouse.click(cx, cy); // particle mode is on: emitter lands, advance
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
  expect(await page.evaluate(() => window.__flowlab.interaction.mode)).toBe('particles');
});

test('click-then-canvas does not arm when Back lands on an already-particle mode step', async ({ page }) => {
  await startTour(page, [
    { target: '#btn-mode', title: 'Particles', body: 'Click Particles, then the flow.', action: { type: 'click-then-canvas' } },
    { target: null, title: 'Read A', body: 'A read step.' },
    { target: null, title: 'Read B', body: 'Another read step.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  // First pass: enable particles, click canvas, advance.
  await page.click('#btn-mode');
  expect(await page.evaluate(() => window.__flowlab.interaction.mode)).toBe('particles');
  await page.mouse.click(cx, cy);
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);

  // Go back: mode is still particles, so the next #btn-mode click toggles it
  // back to obstacle. The tour must stay on this step and keep the hole on
  // the button instead of retargeting to the canvas.
  await page.locator('.tour-btn-secondary').click();
  await page.waitForFunction(() => window.__testTour.stepIndex === 0);

  await page.click('#btn-mode'); // toggles to obstacle
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);
  expect(await page.evaluate(() => window.__testTour._overrideTarget)).toBe(null);

  const probe = await page.evaluate(() => {
    const btn = document.getElementById('btn-mode');
    const r = btn.getBoundingClientRect();
    const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return { atId: at?.id, buttonContains: btn.contains(at) };
  });
  expect(probe.buttonContains).toBe(true);

  // Click #btn-mode again to re-enable particles; this arms the canvas.
  await page.click('#btn-mode'); // toggles to particles
  expect(await page.evaluate(() => window.__testTour._overrideTarget)).toBe('#overlay-canvas');
  expect(await page.evaluate(() => window.__flowlab.interaction.mode)).toBe('particles');
  await page.mouse.click(cx, cy);
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
});

test('drag do-it step advances even if the pointer leaves the canvas mid-drag', async ({ page }) => {
  await startTour(page, [
    { target: '#overlay-canvas', title: 'Drag', body: 'Drag the obstacle.', action: { type: 'drag' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  // Start close to the right edge so a fast flick exits the canvas before
  // enough pointermove events would have accumulated inside the element.
  const startX = box.x + box.width - 3;
  const startY = box.y + box.height / 2;

  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 40, startY + 30, { steps: 2 });
  await page.mouse.up();
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
});
