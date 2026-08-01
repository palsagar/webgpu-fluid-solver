import { test, expect } from '@playwright/test';

/**
 * Umami analytics (spec: docs/superpowers/specs/2026-08-01-umami-analytics-design.md).
 *
 * The Playwright webServer runs WITHOUT UMAMI_DOMAIN/UMAMI_ID, so the tag is
 * never injected here: (a) guards the inert-by-construction contract, (b)-(c)
 * pin the trackEvent wrapper contract. Injection itself is verified by the
 * curl smoke in DEPLOYMENT.md.
 */

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
});

test('(a) dev server injects no Umami tag', async ({ page }) => {
  await expect(page.locator('script[data-website-id]')).toHaveCount(0);
});

test('(b) trackEvent without window.umami does not throw', async ({ page }) => {
  const ok = await page.evaluate(async () => {
    const { trackEvent } = await import('./js/analytics.js');
    try {
      trackEvent('test-event');
      trackEvent('test-event', { foo: 1 });
      return true;
    } catch {
      return false;
    }
  });
  expect(ok).toBe(true);
});

test('(c) trackEvent forwards name and props to window.umami.track', async ({ page }) => {
  const calls = await page.evaluate(async () => {
    window.__umamiCalls = [];
    window.umami = { track: (n, p) => window.__umamiCalls.push([n, p]) };
    const { trackEvent } = await import('./js/analytics.js');
    trackEvent('test-event', { foo: 1 });
    return window.__umamiCalls;
  });
  expect(calls).toEqual([['test-event', { foo: 1 }]]);
});

/** Stub window.umami before page scripts run; events land in window.__umamiCalls. */
async function stubUmami(page) {
  await page.addInitScript(() => {
    window.__umamiCalls = [];
    window.umami = { track: (n, p) => window.__umamiCalls.push([n, p]) };
  });
}

/** Dismiss the first-visit welcome modal without starting the tour. */
async function dismissWelcome(page) {
  const skip = page.locator('#start-sim-btn');
  if (await skip.isVisible().catch(() => false)) await skip.click();
}

async function eventNames(page) {
  return page.evaluate(() => window.__umamiCalls.map(c => c[0]));
}

test('(d) tour start and skip fire tour-started / tour-skipped', async ({ page }) => {
  await stubUmami(page);
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.tour, null, { timeout: 20_000 });
  // Fresh context => welcome modal visible and wired (no localStorage seed).
  await page.locator('#start-tour-btn').click();
  await expect.poll(() => eventNames(page)).toContain('tour-started');
  await page.keyboard.press('Escape');
  await expect.poll(() => eventNames(page)).toContain('tour-skipped');
});

test('(e) preset button click fires preset-changed with the preset key', async ({ page }) => {
  await stubUmami(page);
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  await dismissWelcome(page);
  await page.locator('[data-preset="backward-step"]').click();
  await expect.poll(() => page.evaluate(() => window.__umamiCalls))
    .toContainEqual(['preset-changed', { preset: 'backwardStep' }]);
});

test('(f) resolution tier button click fires resolution-changed with a tier index', async ({ page }) => {
  await stubUmami(page);
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  await dismissWelcome(page);
  await page.locator('[data-tier]').first().click();
  await expect.poll(() => eventNames(page)).toContain('resolution-changed');
  const props = await page.evaluate(() => window.__umamiCalls.find(c => c[0] === 'resolution-changed')[1]);
  expect(Number.isInteger(props.tier)).toBe(true);
});

test('(g) insert-on-click in an obstacle-less preset fires obstacle-inserted', async ({ page }) => {
  await stubUmami(page);
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  await dismissWelcome(page);
  await page.locator('[data-preset="backward-step"]').click();
  await page.waitForFunction(() => window.__flowlab.ui.currentPreset === 'backwardStep', null, { timeout: 10_000 });
  // Interaction listens on #overlay-canvas (renderer.canvas), the top layer.
  await page.locator('#overlay-canvas').click();
  await expect.poll(() => eventNames(page)).toContain('obstacle-inserted');
});
