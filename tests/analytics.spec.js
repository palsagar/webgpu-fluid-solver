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
