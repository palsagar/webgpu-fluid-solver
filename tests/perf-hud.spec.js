import { test, expect } from '@playwright/test';

/**
 * The performance HUD and the adaptive controller share one input: the frame
 * time `main.js` measures. It used to be a `performance.now()` bracket around
 * `solver.step()` + `renderer.draw()`, which measures only how long the CPU
 * takes to ENCODE the GPU work — the GPU runs it asynchronously. Measured on
 * the dev machine, that bracket read 0.3–0.5 ms at every tier while the display
 * delivered 8.3 ms at tier 64 and 257 ms at tier 1024:
 *
 *   tier            64      128      256      512     1024
 *   encode HUD     0.3      0.3      0.5      0.3      0.4   ms  ("3618 fps")
 *   true frame    8.33     8.34    17.75    62.01   257.36   ms
 *
 * So the app's most prominent number was wrong by up to 640x, and `adaptive`
 * saw a signal that could never cross either threshold in the useful direction.
 *
 * These tests pin the input, not the numbers: they compare the HUD against
 * frame times measured independently in the page, and check the thresholds
 * against the display's own refresh period, so they hold on any machine.
 */

/** Measures true frame intervals from rAF timestamps, alongside the app's own loop. */
function measureFrames(page, count) {
  return page.evaluate((count) => new Promise((resolve) => {
    const deltas = [];
    let prev = 0;
    function probe(ts) {
      if (prev) deltas.push(ts - prev);
      prev = ts;
      if (deltas.length < count) requestAnimationFrame(probe);
      else resolve({
        avg: deltas.reduce((a, b) => a + b, 0) / deltas.length,
        min: Math.min(...deltas),
        hud: document.getElementById('perf-hud').textContent,
      });
    }
    requestAnimationFrame(probe);
  }), count);
}

test('the perf HUD reports the frame time the display actually delivers', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  // Let the EMA (alpha = 0.1) settle and the first-frame seed wash out.
  await page.waitForTimeout(2000);

  const m = await measureFrames(page, 90);
  const [msText, fpsText] = m.hud.split('\n')[0].split('|');
  const hudMs = parseFloat(msText);
  const hudFps = parseFloat(fpsText);

  expect(Number.isFinite(hudMs)).toBe(true);

  // No frame can beat the display's refresh period, so a HUD reading below the
  // shortest interval the page ever saw is not measuring frames at all. The
  // encode-time bracket read 0.5 ms against an 8.33 ms period; this is the
  // assertion it fails, on any display, without hard-coding either number.
  expect(hudMs).toBeGreaterThanOrEqual(0.9 * m.min);

  // And it must track the real average, not merely clear the floor. 25% covers
  // EMA lag and the jitter of a 90-frame window; the old signal was out by 30x
  // at the startup tier.
  expect(Math.abs(hudMs - m.avg) / m.avg).toBeLessThan(0.25);

  // The fps half of the readout must be consistent with the ms half. Compared
  // in relative terms because the two are rounded independently — ms to one
  // decimal, fps to an integer, both from the same unrounded EMA — so at 8.4 ms
  // the pair reads "8.4 ms | 120 fps" while 1000/8.4 is 119.05. That 0.8% is
  // display rounding; anything larger is two different numbers.
  expect(Math.abs(hudFps - 1000 / hudMs) / hudFps).toBeLessThan(0.02);
});

test('both adaptive thresholds are reachable under wall-clock frame time', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.adaptive, null, { timeout: 20_000 });

  const m = await measureFrames(page, 90);
  // BOTH thresholds are read off the live controller. `downscaleMs` used to be
  // the literal 20 typed here, while `adaptive.js` carried a bare `avg > 20` —
  // so `downscaleMs < 60` was really `20 < 60` and the comparison against
  // `upscaleMs` was against a fabrication. Changing the solver's threshold to
  // 500 failed nothing. It is now a `static DOWNSCALE_MS` the controller
  // actually branches on, so these assertions bind to shipped behaviour.
  const { upscaleMs, downscaleMs } = await page.evaluate(() => ({
    upscaleMs: window.__flowlab.adaptive.constructor.UPSCALE_MS,
    downscaleMs: window.__flowlab.adaptive.constructor.DOWNSCALE_MS,
  }));

  // Guard: `page.evaluate` yields `undefined` for a missing static, and
  // `undefined < 60` is false — but `expect(undefined).toBeLessThan` throws a
  // type error rather than reporting a threshold problem, so say what is wrong.
  expect(typeof downscaleMs, 'AdaptiveController must export DOWNSCALE_MS').toBe('number');

  // `tick` is fed wall-clock frame time, so a tier with headroom does not read
  // "fast" — it reads the display's refresh period, because vsync is what it is
  // waiting on. An upscale threshold at or below that period can therefore
  // never fire. The old value was 8 ms against this machine's 8.33 ms period,
  // and against 16.67 ms on any 60 Hz display.
  expect(upscaleMs).toBeGreaterThan(m.min);

  // The two thresholds must not overlap, or a single frame time would ask for
  // both a promotion and a demotion.
  expect(upscaleMs).toBeLessThan(downscaleMs);

  // Downscale must stay reachable too: tier 512 measures ~61 ms and tier 1024
  // ~257 ms here, both far above 20 ms, so a threshold that drifted above the
  // slow tiers would leave the controller unable to recover from a stall.
  expect(downscaleMs).toBeLessThan(60);
});

test('the controller branches on DOWNSCALE_MS, not on a number beside it', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.adaptive, null, { timeout: 20_000 });

  // The test above pins the VALUE of the constant. This one pins that `tick`
  // actually reads it: exporting a constant the branch ignores would satisfy
  // every assertion up there while the shipped threshold stayed wherever the
  // inline literal left it. Driven with synthetic frame times straddling the
  // constant by ±10%, so it holds for any value the constant is set to and
  // fails the moment the branch stops tracking it.
  const r = await page.evaluate(() => {
    const a = window.__flowlab.adaptive;
    const D = a.constructor.DOWNSCALE_MS;

    const drive = (ms) => {
      a.enabled = true;
      a.manualOverride = false;
      // Tier 512, so there is somewhere to fall to, and pinned as the auto
      // ceiling so a fast frame time cannot promote instead.
      a.currentTierIndex = 3;
      a.maxAutoTierIndex = 3;
      a.frameTimes = [];
      a.lastUpscaleTime = Date.now();
      a.tierStartTime = Date.now() - 3000;  // past the 2 s warmup gate
      const before = a.currentTierIndex;
      for (let k = 0; k < 40; k++) a.tick(ms);
      a.enabled = false;
      return a.currentTierIndex - before;
    };

    return { D, above: drive(D * 1.1), below: drive(D * 0.9) };
  });

  // Just above the threshold the tier must drop exactly once — `downscale()`
  // clears the frame-time window and the warmup gate, so the remaining ticks
  // in the batch cannot compound it.
  expect(r.above).toBe(-1);
  // Just below it, nothing moves. This is the assertion that fails if the
  // branch keeps a literal that has drifted away from the constant: with
  // `avg > 20` inline and DOWNSCALE_MS raised, 0.9x the constant would still
  // clear 20 and demote.
  expect(r.below).toBe(0);
});
