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
