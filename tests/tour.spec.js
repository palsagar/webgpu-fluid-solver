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

  await page.locator('.tour-footer .tour-btn-primary').click(); // Next
  await expect(page.locator('.tour-counter')).toHaveText('2 / 2');
  await expect(page.locator('.tour-title')).toHaveText('Step two');

  await page.locator('.tour-footer .tour-btn-secondary').click(); // Back
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

  await expect(page.locator('.tour-footer .tour-btn-primary')).toHaveCount(0); // do-it: no Next
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
  await page.locator('.tour-footer .tour-btn-secondary').click();
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

test('click-then-canvas step ignores non-left pointer buttons', async ({ page }) => {
  await startTour(page, [
    { target: '#btn-mode', title: 'Particles', body: 'Click Particles, then the flow.', action: { type: 'click-then-canvas' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  await page.click('#btn-mode'); // arm — hole retargets to the canvas
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  // Right-button canvas press must not advance.
  await page.mouse.move(cx, cy);
  await page.mouse.down({ button: 'right' });
  await page.mouse.up({ button: 'right' });
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  // Normal left-button canvas click should advance.
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

/** Start a tour with the real 12-step STEPS script. */
async function startRealTour(page) {
  await page.evaluate(async () => {
    const { Tour, STEPS } = await import('./js/tour.js');
    const { ui, interaction, solver } = window.__flowlab;
    window.__testTour = new Tour({ ui, interaction, solver, steps: STEPS });
    window.__testTour.start();
  });
  // Clear the beforeEach seed so the test's own flag writes are what gets asserted.
  await page.evaluate(() => localStorage.removeItem('flowlab.tour.v1'));
}

/** Perform the gesture the current do-it step asks for. */
async function performCurrentAction(page) {
  const idx = await page.evaluate(() => window.__testTour.stepIndex);
  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  switch (idx) {
    case 1: // drag obstacle
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      await page.mouse.move(cx + 60, cy + 30, { steps: 4 });
      await page.mouse.up();
      break;
    case 2: // toggle a viz mode
      await page.click('input[data-viz="pressure"]');
      break;
    case 4: // pick a shape
      await page.click('[data-shape="airfoil"]');
      break;
    case 5: // particles mode, then click the flow
      await page.click('#btn-mode');
      await page.mouse.click(cx, cy);
      break;
    case 6: // switch preset
      await page.click('[data-preset="backward-step"]');
      break;
    case 7: // pause
      await page.click('#btn-play');
      break;
    case 8: // open advanced panel
      await page.click('#btn-advanced');
      break;
    default:
      throw new Error(`step ${idx} is not a do-it step`);
  }
  await page.waitForFunction((i) => window.__testTour.stepIndex === i + 1, idx);
}

test('the full STEPS walkthrough completes and resets to a clean Kármán state', async ({ page }) => {
  await startRealTour(page);
  await expect(page.locator('.tour-counter')).toHaveText('1 / 12');

  for (let guard = 0; guard < 12; guard++) {
    const done = await page.evaluate(() => !window.__testTour.active);
    if (done) break;
    const hasNext = await page.locator('.tour-footer .tour-btn-primary').count();
    if (hasNext) await page.locator('.tour-footer .tour-btn-primary').click();
    else await performCurrentAction(page);
    // Do-it steps may trigger GPU work (preset load); give the loop a beat.
    await page.waitForTimeout(100);
  }

  expect(await page.evaluate(() => window.__testTour.active)).toBe(false);
  await expect(page.locator('.tour-tooltip')).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('done');

  // Exit reset: clean Kármán state, driven through the app's own controls.
  const state = await page.evaluate(() => ({
    preset: document.querySelector('.preset-btn.active')?.dataset.preset,
    smoke: document.querySelector('[data-viz="smoke"]').checked,
    pressure: document.querySelector('[data-viz="pressure"]').checked,
    paused: window.__flowlab.solver.paused,
    mode: window.__flowlab.interaction.mode,
    shape: document.querySelector('.shape-btn.active')?.dataset.shape,
    panelOpen: document.getElementById('advanced-panel').classList.contains('visible'),
  }));
  expect(state).toEqual({
    preset: 'karman-vortex', smoke: true, pressure: false,
    paused: false, mode: 'obstacle', shape: 'circle', panelOpen: false,
  });
});

test('skip mid-tour resets state and writes the skipped flag', async ({ page }) => {
  await startRealTour(page);
  await page.locator('.tour-footer .tour-btn-primary').click(); // step 1 → 2
  await performCurrentAction(page);                  // drag → step 3
  await page.click('input[data-viz="pressure"]');    // → step 4 (pressure now ON)
  await page.keyboard.press('Escape');

  expect(await page.evaluate(() => window.__testTour.active)).toBe(false);
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('skipped');
  const clean = await page.evaluate(() => ({
    preset: document.querySelector('.preset-btn.active')?.dataset.preset,
    pressure: document.querySelector('[data-viz="pressure"]').checked,
    smoke: document.querySelector('[data-viz="smoke"]').checked,
  }));
  expect(clean).toEqual({ preset: 'karman-vortex', pressure: false, smoke: true });
});

test('onLeave fires on Escape-triggered tour exit', async ({ page }) => {
  await page.evaluate(async () => {
    const { Tour } = await import('./js/tour.js');
    const { ui, interaction, solver } = window.__flowlab;
    window.__onLeaveCalls = 0;
    window.__onLeaveActionTornDown = null;
    window.__onLeavePressureOn = null;
    const steps = [
      { target: null, title: 'Step 1', body: 'Intro.' },
      {
        target: '#btn-play',
        title: 'Step 2',
        body: 'Exit me.',
        action: { type: 'click' },
        onLeave: (ctx) => {
          window.__onLeaveCalls = (window.__onLeaveCalls || 0) + 1;
          window.__onLeaveActionTornDown = window.__testTour._teardownAction === null;
          window.__onLeavePressureOn = document.querySelector('input[data-viz="pressure"]').checked;
        },
      },
    ];
    window.__testTour = new Tour({ ui, interaction, solver, steps });
    window.__testTour.start();
  });
  await page.evaluate(() => localStorage.removeItem('flowlab.tour.v1'));

  await page.locator('.tour-footer .tour-btn-primary').click(); // step 1 → 2
  // Dirty app state via a real click path while the tour overlay is up.
  await page.evaluate(() => {
    document.querySelector('input[data-viz="pressure"]').click();
  });
  expect(await page.evaluate(() => typeof window.__testTour._teardownAction === 'function')).toBe(true);
  await page.keyboard.press('Escape');

  expect(await page.evaluate(() => window.__onLeaveCalls)).toBe(1);
  expect(await page.evaluate(() => window.__onLeaveActionTornDown)).toBe(true);
  expect(await page.evaluate(() => window.__onLeavePressureOn)).toBe(true);
  expect(await page.evaluate(() => document.querySelector('input[data-viz="pressure"]').checked)).toBe(false);
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('skipped');
});

test('missing-target action step renders a primary button and advances on click', async ({ page }) => {
  await startTour(page, [
    { target: '#no-such-element', title: 'Ghost action', body: 'Target is missing.', action: { type: 'click' } },
    { target: null, title: 'Step 2', body: 'Done.' },
  ]);

  await expect(page.locator('.tour-footer .tour-btn-primary')).toHaveCount(1);
  await page.locator('.tour-footer .tour-btn-primary').click();
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
});

test('a missing target falls back to centered placement and stays advanceable', async ({ page }) => {
  await startTour(page, [
    { target: '#no-such-element', title: 'Ghost', body: 'Nowhere to point.' },
  ]);
  await expect(page.locator('.tour-tooltip')).toBeVisible();
  await expect(page.locator('.tour-ring')).toBeHidden();
  await page.locator('.tour-footer .tour-btn-primary').click(); // Done — single read step ends the tour
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('done');
});


test('first visit shows the slim welcome and Take the Tour starts the tour', async ({ page }) => {
  // Undo the beforeEach seed before navigating, so the gate sees no flag.
  // Init scripts re-run on every navigation, in the order added.
  await page.addInitScript(() => {
    try { localStorage.removeItem('flowlab.tour.v1'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  await expect(page.locator('#welcome-overlay')).toBeVisible();
  await expect(page.locator('#start-tour-btn')).toBeVisible();

  await page.click('#start-tour-btn');
  await page.waitForFunction(() => window.__flowlab.tour?.active === true);
  expect(await page.evaluate(() => window.__flowlab.tour.stepIndex)).toBe(0);
  await expect(page.locator('#welcome-overlay')).toBeHidden();
});

test('a seeded flag suppresses the welcome entirely', async ({ page }) => {
  await page.addInitScript(() => {
    try { localStorage.setItem('flowlab.tour.v1', 'done'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  await expect(page.locator('#welcome-overlay')).toBeHidden();
  expect(await page.evaluate(() => window.__flowlab.tour?.active ?? false)).toBe(false);
});

test('Skip to Simulation writes the flag and dismisses without starting the tour', async ({ page }) => {
  await page.addInitScript(() => {
    try { localStorage.removeItem('flowlab.tour.v1'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  await page.click('#start-sim-btn');
  await expect(page.locator('#welcome-overlay')).toBeHidden();
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('skipped');
  expect(await page.evaluate(() => window.__flowlab.tour?.active ?? false)).toBe(false);
});

test('Replay the Tour from the guide restarts from step 1 on a clean state', async ({ page }) => {
  await page.addInitScript(() => {
    try { localStorage.setItem('flowlab.tour.v1', 'done'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  // Dirty the state: switch preset, then open the guide and replay.
  await page.click('[data-preset="backward-step"]');
  await page.click('#btn-guide');
  await expect(page.locator('#guide-overlay')).toHaveClass(/guide-visible/);
  await page.click('#btn-replay-tour');

  await page.waitForFunction(() => window.__flowlab.tour?.active === true);
  expect(await page.evaluate(() => window.__flowlab.tour.stepIndex)).toBe(0);
  // Guide closed by start(); state reset by start().
  await expect(page.locator('#guide-overlay')).not.toHaveClass(/guide-visible/);
  const preset = await page.evaluate(() => document.querySelector('.preset-btn.active')?.dataset.preset);
  expect(preset).toBe('karman-vortex');
});

test('keyboard shortcuts are gated while the tour is active', async ({ page }) => {
  await page.addInitScript(() => {
    try { localStorage.setItem('flowlab.tour.v1', 'done'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  await page.click('#btn-guide');
  await page.click('#btn-replay-tour');
  await page.waitForFunction(() => window.__flowlab.tour?.active === true);

  await page.keyboard.press('p');
  expect(await page.evaluate(() => window.__flowlab.solver.paused)).toBe(false);

  await page.keyboard.press('Escape'); // end tour (flag: skipped)
  await page.waitForFunction(() => window.__flowlab.tour.active === false);
  await page.keyboard.press('p');
  expect(await page.evaluate(() => window.__flowlab.solver.paused)).toBe(true);
});

// ── Final-review regression tests ──────────────────────────────────────────

test('advanced-step Back trap only advances when the panel opens, not closes', async ({ page }) => {
  await startRealTour(page);
  // Walk to step 9 (0-based index 8), the #btn-advanced do-it step.
  for (let i = 0; i < 8; i++) {
    const hasNext = await page.locator('.tour-footer .tour-btn-primary').count();
    if (hasNext) await page.locator('.tour-footer .tour-btn-primary').click();
    else await performCurrentAction(page);
  }
  await expect(page.locator('.tour-counter')).toHaveText('9 / 12');

  // First click opens the panel and advances.
  await page.click('#btn-advanced');
  await page.waitForFunction(() => window.__testTour.stepIndex === 9);
  await expect(page.locator('#advanced-panel')).toHaveClass(/visible/);

  // Back to the advanced button step; panel remains open.
  await page.locator('.tour-footer .tour-btn-secondary').click();
  await page.waitForFunction(() => window.__testTour.stepIndex === 8);

  // Clicking again closes the panel but must NOT advance.
  await page.click('#btn-advanced');
  await expect(page.locator('#advanced-panel')).not.toHaveClass(/visible/);
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(8);

  // One more click opens the panel and advances.
  await page.click('#btn-advanced');
  await page.waitForFunction(() => window.__testTour.stepIndex === 9);
  await expect(page.locator('#advanced-panel')).toHaveClass(/visible/);
});

test('closing the advanced panel mid-tour hides the ring and keeps the tooltip centered', async ({ page }) => {
  await startRealTour(page);
  // Walk to step 9 (index 8), the do-it step that opens the Advanced panel.
  for (let i = 0; i < 8; i++) {
    const hasNext = await page.locator('.tour-footer .tour-btn-primary').count();
    if (hasNext) await page.locator('.tour-footer .tour-btn-primary').click();
    else await performCurrentAction(page);
  }
  await expect(page.locator('.tour-counter')).toHaveText('9 / 12');

  // Open the panel and advance to step 10 (target #advanced-panel).
  await page.click('#btn-advanced');
  await page.waitForFunction(() => window.__testTour.stepIndex === 9);
  await expect(page.locator('#advanced-panel')).toHaveClass(/visible/);
  await expect(page.locator('.tour-counter')).toHaveText('10 / 12');
  await expect(page.locator('.tour-ring')).toBeVisible();

  // Close the panel through the hole; the deferred relayout detects the
  // now-hidden target and falls back to the centered-card layout.
  await page.click('#btn-close-advanced');
  await expect(page.locator('#advanced-panel')).not.toHaveClass(/visible/);
  await expect(page.locator('.tour-ring')).toBeHidden({ timeout: 1000 });
  await expect(page.locator('.tour-tooltip')).toBeVisible();

  // The remaining read steps are still reachable with Next.
  await page.locator('.tour-footer .tour-btn-primary').click();
  await expect(page.locator('.tour-counter')).toHaveText('11 / 12');
  await page.locator('.tour-footer .tour-btn-primary').click();
  await expect(page.locator('.tour-counter')).toHaveText('12 / 12');
  await page.locator('.tour-footer .tour-btn-primary').click();

  await expect(page.locator('.tour-tooltip')).toHaveCount(0);
  expect(await page.evaluate(() => localStorage.getItem('flowlab.tour.v1'))).toBe('done');
});

test('shape step ignores the mode button inside the shape group', async ({ page }) => {
  await startRealTour(page);
  // Walk to step 5 (0-based index 4), the #shape-group do-it step.
  for (let i = 0; i < 4; i++) {
    const hasNext = await page.locator('.tour-footer .tour-btn-primary').count();
    if (hasNext) await page.locator('.tour-footer .tour-btn-primary').click();
    else await performCurrentAction(page);
  }
  await expect(page.locator('.tour-counter')).toHaveText('5 / 12');
  await expect(page.locator('.tour-title')).toHaveText('Obstacle shapes');

  // #btn-mode lives inside #shape-group but is not a shape button.
  await page.click('#btn-mode');
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(4);

  // An actual shape button should advance.
  await page.click('[data-shape="airfoil"]');
  await page.waitForFunction(() => window.__testTour.stepIndex === 5);
});

test('prefers-reduced-motion disables overlay transitions', async ({ page }) => {
  await page.addInitScript(() => {
    try { localStorage.removeItem('flowlab.tour.v1'); } catch (e) {}
  });
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  await page.emulateMedia({ reducedMotion: 'reduce' });

  await page.click('#start-sim-btn');
  await expect(page.locator('#welcome-overlay')).toBeHidden();

  await startTour(page, [
    { target: '#btn-play', title: 'Step 1', body: 'An action step so the ring pulses.', action: { type: 'click' } },
    { target: null, title: 'Step 2', body: 'Done.' },
  ]);

  const styles = await page.evaluate(() => ({
    welcomeTransition: getComputedStyle(document.getElementById('welcome-overlay')).transitionDuration,
    guideTransition: getComputedStyle(document.getElementById('guide-overlay')).transitionDuration,
    pulseAnimation: getComputedStyle(document.querySelector('.tour-ring.tour-pulse')).animationName,
  }));

  expect(styles.welcomeTransition).toBe('0s');
  expect(styles.guideTransition).toBe('0s');
  expect(styles.pulseAnimation).toBe('none');
});

test('drag step ignores non-left pointer buttons', async ({ page }) => {
  await startTour(page, [
    { target: '#overlay-canvas', title: 'Drag', body: 'Drag the obstacle.', action: { type: 'drag' } },
    { target: null, title: 'Done', body: 'Finished.' },
  ]);

  const box = await page.locator('#overlay-canvas').boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;

  // Right-button drag must not advance.
  await page.mouse.move(cx, cy);
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(cx + 60, cy + 30, { steps: 4 });
  await page.mouse.up({ button: 'right' });
  expect(await page.evaluate(() => window.__testTour.stepIndex)).toBe(0);

  // Normal left-button drag should advance.
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 60, cy + 30, { steps: 4 });
  await page.mouse.up();
  await page.waitForFunction(() => window.__testTour.stepIndex === 1);
});

// ── Challenge-review regression tests ─────────────────────────────────────

test('fatal boot path hides the welcome overlay and shows the reload banner', async ({ page }) => {
  // Real-path stub: force requestDevice to reject so init().catch runs.
  await page.addInitScript(() => {
    try { localStorage.removeItem('flowlab.tour.v1'); } catch (e) {}
  });
  await page.addInitScript(() => {
    GPUAdapter.prototype.requestDevice = async function () {
      throw new Error('injected device failure');
    };
  });
  await page.goto('/');

  await expect(page.locator('#fatal-banner')).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('#welcome-overlay')).toHaveCSS('display', 'none');
});

test('tour reset restores the default resolution tier after a hole click changes it', async ({ page }) => {
  await startRealTour(page);

  // Walk to step 11 (0-based index 10), the #resolution-picker read step.
  for (let i = 0; i < 10; i++) {
    const hasNext = await page.locator('.tour-footer .tour-btn-primary').count();
    if (hasNext) await page.locator('.tour-footer .tour-btn-primary').click();
    else await performCurrentAction(page);
    await page.waitForTimeout(100);
  }
  await expect(page.locator('.tour-counter')).toHaveText('11 / 12');

  // Click a lower tier through the spotlight hole while the overlay is up.
  await page.click('[data-tier="0"]');
  await expect(page.locator('[data-tier="0"]')).toHaveClass(/active/);

  // Finish the tour; reset must restore the default tier and preset.
  await page.locator('.tour-footer .tour-btn-primary').click(); // step 11 → 12
  await page.locator('.tour-footer .tour-btn-primary').click();   // Done

  await expect(page.locator('.tour-tooltip')).toHaveCount(0);
  const state = await page.evaluate(() => ({
    preset: document.querySelector('.preset-btn.active')?.dataset.preset,
    tier: document.querySelector('[data-tier].active')?.dataset.tier,
  }));
  expect(state.preset).toBe('karman-vortex');
  expect(state.tier).toBe('2');
});
