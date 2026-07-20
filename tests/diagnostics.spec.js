import { test, expect } from '@playwright/test';

/**
 * Tests for the honest Reynolds window (static/js/diagnostics.js) and for the
 * badge it drives in the UI.
 *
 * Every expected number here is derived from Task 7's measured table, not from
 * whatever the implementation happens to return:
 *
 *   Floor    Re_min = U*D / (N_MAX * 1/4 * h^2 / dt)      [solver.viscNuMax]
 *   Ceiling  Re_max = U*D / NU_NUM_CONVERGED              [9.823e-4, tier-independent]
 *   Ceiling' Re_max = U*D / NU_NUM_ITERS80[tier]          [under-converged projection]
 *
 * With the Karman reference (U = 1.0, D = 0.12, dt = 1/120, N_MAX = 32) that
 * gives, per tier:
 *
 *   tier      64      128      256      512     1024
 *   floor    0.512   2.048    8.192   32.768  131.072
 *   ceil    122.16  122.16   122.16   122.16   122.16
 *   ceil'   121.69   95.61    32.75    11.93     5.06
 */

// Karman reference, matching Task 7's measurement conditions exactly.
const REF = { dt: 1 / 120, D: 0.12, U: 1.0, nMax: 32 };

/** Floor computed from the substep budget, independently of the module. */
const floorAt = (numY) => (REF.U * REF.D) / (REF.nMax * 0.25 * (1 / numY) ** 2 / REF.dt);

async function loadDiagnostics(page) {
  await page.goto('/');
  return page.evaluate(() => import('/js/diagnostics.js').then((m) => ({
    NU_NUM_CONVERGED: m.NU_NUM_CONVERGED,
    NU_NUM_ITERS80: m.NU_NUM_ITERS80,
    PROJECTION_ITERS_MEASURED: m.PROJECTION_ITERS_MEASURED,
    RE_SLIDER_MIN: m.RE_SLIDER_MIN,
    RE_SLIDER_MAX: m.RE_SLIDER_MAX,
    RE_SLIDER_STEPS: m.RE_SLIDER_STEPS,
  })));
}

/** Evaluates honestWindow/windowState in the page for a given tier. */
function windowFor(page, numY, extra = {}) {
  return page.evaluate(async ({ numY, REF, extra }) => {
    const d = await import('/js/diagnostics.js');
    const w = d.honestWindow({
      h: 1 / numY, dt: REF.dt, D: REF.D, U: REF.U,
      nMax: REF.nMax, nuNum: d.NU_NUM_CONVERGED,
    });
    const reMaxProjection = (REF.U * REF.D) / d.NU_NUM_ITERS80[numY];
    return { ...w, reMaxProjection, ...extra };
  }, { numY, REF, extra });
}

function stateFor(page, args) {
  return page.evaluate(async (args) => {
    const d = await import('/js/diagnostics.js');
    return d.windowState(args);
  }, args);
}

// ── The measured constants themselves ───────────────────────────────────────

test('the shipped numerical viscosity constants are the measured ones', async ({ page }) => {
  const c = await loadDiagnostics(page);

  // Task 7 §2b: converged projection, tier-independent, linear in dt.
  expect(c.NU_NUM_CONVERGED).toBeCloseTo(9.823e-4, 7);

  // Task 7 §2a, MacCormack column, at the shipped numIters = 80.
  expect(c.PROJECTION_ITERS_MEASURED).toBe(80);
  expect(c.NU_NUM_ITERS80[64]).toBeCloseTo(9.8612e-4, 8);
  expect(c.NU_NUM_ITERS80[128]).toBeCloseTo(1.2551e-3, 7);
  expect(c.NU_NUM_ITERS80[256]).toBeCloseTo(3.6638e-3, 7);
  expect(c.NU_NUM_ITERS80[512]).toBeCloseTo(1.0062e-2, 6);
  expect(c.NU_NUM_ITERS80[1024]).toBeCloseTo(2.3718e-2, 6);

  // No estimated fallback may exist: every tier must carry a real measurement.
  for (const tier of [64, 128, 256, 512, 1024]) {
    expect(Number.isFinite(c.NU_NUM_ITERS80[tier])).toBe(true);
    expect(c.NU_NUM_ITERS80[tier]).toBeGreaterThan(0);
  }
});

// ── honestWindow ────────────────────────────────────────────────────────────

test('honest window bounds and badge reasons', async ({ page }) => {
  await page.goto('/');
  const w = await windowFor(page, 256);

  // Floor: N_MAX substeps each at the 1/4 explicit stability limit give
  // nu_max = 32 * 0.25 * h^2/dt = 0.0146484 at tier 256, so Re_min = 8.192.
  // Dropping N_MAX from the formula makes this 262.1 — 32x larger.
  expect(w.reMin).toBeCloseTo(8.192, 3);
  expect(w.reMin).toBeCloseTo(floorAt(256), 6);

  // Ceiling: U*D / 9.823e-4 = 122.162. The brief's placeholder 5e-4 gives 240.
  expect(w.reMax).toBeCloseTo(122.162, 2);

  // Operating point at the shipped 80 iterations: U*D / 3.6638e-3 = 32.753.
  expect(w.reMaxProjection).toBeCloseTo(32.753, 2);

  const inside = await stateFor(page, {
    re: 20, reEff: 20, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(inside.ok).toBe(true);
  expect(inside.code).toBeNull();
  expect(inside.reason).toBeNull();

  // Below the floor: the solver saturates nu, so the effective Re is HIGHER
  // than requested. This is the Task 8 meaning of viscClamped.
  const tooLow = await stateFor(page, {
    re: 2, reEff: w.reMin, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: true,
  });
  expect(tooLow.ok).toBe(false);
  expect(tooLow.code).toBe('clamped');
  expect(tooLow.reason).toMatch(/substep/i);
  expect(tooLow.reason).toMatch(/higher/i);   // must not claim the Re is lower
  expect(tooLow.reason).toContain('8.2');     // the floor, from viscNuMax

  // The prediction alone must fire even when the solver has not stepped yet.
  const tooLowUnstepped = await stateFor(page, {
    re: 2, reEff: 2, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(tooLowUnstepped.code).toBe('clamped');

  // Above both ceilings at tier 256 the projection is what binds (32.8 vs 122).
  const tooHigh = await stateFor(page, {
    re: 1e6, reEff: 1e6, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(tooHigh.ok).toBe(false);
  expect(tooHigh.code).toBe('projection');
  expect(tooHigh.reason).toMatch(/under-converged/i);
  expect(tooHigh.reason).toContain('33');   // the binding ceiling
  expect(tooHigh.reason).toContain('122');  // reconciled with the scheme ceiling
});

test('the binding ceiling is named by which viscosity actually dominates', async ({ page }) => {
  await page.goto('/');

  // Tier 64: the two ceilings agree to 0.4% (121.69 vs 122.16), well inside
  // Task 7's ±10% fit-window uncertainty, so the scheme is what binds.
  const w64 = await windowFor(page, 64);
  expect(w64.reMaxProjection).toBeCloseTo(121.691, 2);
  const s64 = await stateFor(page, {
    re: 400, reEff: 400, reMin: w64.reMin, reMax: w64.reMax,
    reMaxProjection: w64.reMaxProjection, viscClamped: false,
  });
  expect(s64.code).toBe('scheme');
  expect(s64.reason).toMatch(/under-resolved/i);
  expect(s64.reason).toContain('122');

  // Tier 128: 95.6 vs 122.2 — a 22% gap, outside the measurement scatter, so
  // the under-converged projection is the honest ceiling.
  const w128 = await windowFor(page, 128);
  expect(w128.reMaxProjection).toBeCloseTo(95.610, 2);
  const s128 = await stateFor(page, {
    re: 400, reEff: 400, reMin: w128.reMin, reMax: w128.reMax,
    reMaxProjection: w128.reMaxProjection, viscClamped: false,
  });
  expect(s128.code).toBe('projection');

  // Between the two ceilings at tier 128 (95.6 < Re < 122.2) the projection
  // reason must still fire — this is the range the flat ceiling would hide.
  const between = await stateFor(page, {
    re: 110, reEff: 110, reMin: w128.reMin, reMax: w128.reMax,
    reMaxProjection: w128.reMaxProjection, viscClamped: false,
  });
  expect(between.ok).toBe(false);
  expect(between.code).toBe('projection');
});

// ── The empty windows ───────────────────────────────────────────────────────

test('the honest window is empty at tier 1024, and at tier 512 with 80 iterations', async ({ page }) => {
  await page.goto('/');

  // Tier 1024: floor 131.07 sits ABOVE the flat ceiling 122.16. No iteration
  // count can fix this — it is set by h, dt and N_MAX against the scheme's own
  // splitting error. Every slider position must be reported as invalid.
  const w1024 = await windowFor(page, 1024);
  expect(w1024.reMin).toBeCloseTo(131.072, 3);
  expect(w1024.reMax).toBeCloseTo(122.162, 2);
  expect(w1024.reMin).toBeGreaterThan(w1024.reMax);

  for (const re of [0.5, 15, 122, 131, 500]) {
    const st = await stateFor(page, {
      re, reEff: re, reMin: w1024.reMin, reMax: w1024.reMax,
      reMaxProjection: w1024.reMaxProjection, viscClamped: false,
    });
    expect(st.ok, `Re ${re} at tier 1024 must not be reported as honest`).toBe(false);
    expect(st.code).toBe('empty-grid');
    expect(st.reason).toMatch(/no honest Re/i);
    expect(st.reason).toContain('131');
    expect(st.reason).toContain('122');
    expect(st.reason).toMatch(/resolution|dt/i);
  }

  // Tier 512: the grid itself leaves a window (32.77 .. 122.16), but at the
  // shipped 80 iterations the projection ceiling falls to 11.93, below the
  // floor. The advice differs: iterations, not dt.
  const w512 = await windowFor(page, 512);
  expect(w512.reMin).toBeCloseTo(32.768, 3);
  expect(w512.reMaxProjection).toBeCloseTo(11.926, 2);
  expect(w512.reMin).toBeLessThan(w512.reMax);          // grid alone is fine
  expect(w512.reMin).toBeGreaterThan(w512.reMaxProjection); // projection is not

  const st512 = await stateFor(page, {
    re: 50, reEff: 50, reMin: w512.reMin, reMax: w512.reMax,
    reMaxProjection: w512.reMaxProjection, viscClamped: false,
  });
  expect(st512.ok).toBe(false);
  expect(st512.code).toBe('empty-iters');
  expect(st512.reason).toMatch(/iterations/i);

  // Tier 256 is NOT empty — otherwise the emptiness assertions above would be
  // vacuous and the app would badge everywhere.
  const w256 = await windowFor(page, 256);
  expect(w256.reMin).toBeLessThan(w256.reMaxProjection);
});

// ── The slider mapping ──────────────────────────────────────────────────────

test('the Re slider spans the measured window and its midpoint is honest at the startup tier', async ({ page }) => {
  await page.goto('/');
  const c = await loadDiagnostics(page);

  const mapped = await page.evaluate(async () => {
    const d = await import('/js/diagnostics.js');
    return {
      lo:  d.reFromSliderPos(0),
      mid: d.reFromSliderPos(d.RE_SLIDER_STEPS / 2),
      hi:  d.reFromSliderPos(d.RE_SLIDER_STEPS),
      roundTrip: d.sliderPosFromRe(d.reFromSliderPos(37)),
    };
  });

  expect(mapped.lo).toBeCloseTo(c.RE_SLIDER_MIN, 6);
  expect(mapped.hi).toBeCloseTo(c.RE_SLIDER_MAX, 4);
  expect(mapped.roundTrip).toBeCloseTo(37, 6);

  // Bottom must reach the lowest floor in the tier set (tier 64 -> Re 0.512),
  // so the honest low end is representable rather than cropped off.
  expect(c.RE_SLIDER_MIN).toBeLessThanOrEqual(floorAt(64));

  // Top must clear both structural bounds — the flat ceiling (122.16) and the
  // highest floor (tier 1024 -> 131.07) — so every badge regime is reachable.
  expect(c.RE_SLIDER_MAX).toBeGreaterThan(floorAt(1024));
  expect(c.RE_SLIDER_MAX).toBeGreaterThan(122.162);

  // Log spacing: the midpoint must land inside the honest window at the
  // startup tier (256: 8.19 .. 32.75 at 80 iterations), so the default view
  // is badge-free and "absent mid-range" is a real property, not luck.
  const w256 = await windowFor(page, 256);
  expect(mapped.mid).toBeGreaterThan(w256.reMin);
  expect(mapped.mid).toBeLessThan(w256.reMaxProjection);
});

test('the shipped default sits where the Karman wake actually sheds, and says so', async ({ page }) => {
  await page.goto('/');
  // The badge is written by the UI constructor, which runs after WebGPU init.
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });

  // index.html's value attribute must match the documented default, so the
  // rationale in diagnostics.js cannot drift away from what ships.
  const shipped = await page.evaluate(async () => {
    const d = await import('/js/diagnostics.js');
    return {
      attr: Number(document.getElementById('slider-re').getAttribute('value')),
      documented: d.RE_SLIDER_DEFAULT_POS,
      re: d.reFromSliderPos(d.RE_SLIDER_DEFAULT_POS),
    };
  });
  expect(shipped.attr).toBe(shipped.documented);

  // Measured shedding onset for the Karman preset at tier 256 is ~Re 126
  // (unsteadiness 2e-4 at Re 63, 6.2e-3 at 126, 1.1e-1 at 251). Below that the
  // preset shows no vortex street at all.
  expect(shipped.re).toBeGreaterThan(126);

  // And it is knowingly outside the honest window, so the app must open with
  // the badge visible rather than pretending Re 251 is delivered.
  const w = await windowFor(page, 256);
  expect(shipped.re).toBeGreaterThan(w.reMaxProjection);

  const badge = await page.evaluate(() => ({
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
    valRe: document.getElementById('val-re').textContent,
  }));
  expect(badge.visible).toBe(true);
  expect(badge.text).toMatch(/under-converged/i);
  expect(badge.valRe).toBe('251');
});

// ── Live wiring ─────────────────────────────────────────────────────────────

/** Reads the badge and the Re control straight out of the DOM. */
function readBadge(page) {
  return page.evaluate(() => {
    const badge = document.getElementById('re-badge');
    const slider = document.getElementById('slider-re');
    return {
      text: badge?.textContent ?? null,
      visible: badge?.classList.contains('visible') ?? null,
      sliderPos: slider?.value ?? null,
      valRe: document.getElementById('val-re')?.textContent ?? null,
      numY: window.__flowlab.solver.numY,
      nu: window.__flowlab.solver.params.nu,
      viscNuMax: window.__flowlab.solver.viscNuMax,
      viscNuEff: window.__flowlab.solver.viscNuEff,
      viscClamped: window.__flowlab.solver.viscClamped,
    };
  });
}

function setSliderPos(page, pos) {
  return page.evaluate((pos) => {
    const slider = document.getElementById('slider-re');
    slider.value = String(pos);
    slider.dispatchEvent(new Event('input', { bubbles: true }));
  }, pos);
}

test('the Re badge reports the live solver window and updates on tier change without moving the slider', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  const steps = await page.evaluate(async () => (await import('/js/diagnostics.js')).RE_SLIDER_STEPS);

  // Mid-slider at the startup tier (256): inside the window, no badge.
  await setSliderPos(page, steps / 2);
  const mid = await readBadge(page);
  expect(mid.numY).toBe(256);
  expect(mid.visible).toBe(false);
  expect(mid.text).toBe('');

  // The slider must actually be driving the solver's viscosity.
  expect(mid.nu).toBeGreaterThan(0);
  expect(mid.nu).toBeLessThan(mid.viscNuMax);   // representable => not clamped

  // Bottom of the slider: below the floor, solver saturates nu.
  await setSliderPos(page, 0);
  await page.evaluate(() => window.__flowlab.solver.step(window.__flowlab.ui.numIters));
  const low = await readBadge(page);
  expect(low.visible).toBe(true);
  expect(low.text).toMatch(/substep/i);
  expect(low.viscClamped).toBe(true);
  // The solver's own effective viscosity must equal the pure model's floor.
  expect(low.viscNuEff).toBeCloseTo(low.viscNuMax, 12);

  // Top of the slider: above the ceiling. Note the solver has NOT stepped
  // since, so `solver.viscClamped` is still true from the low position — the
  // badge must read saturation from `viscNuMax` (always current) rather than
  // from that flag, or it reports the previous request for a frame.
  await setSliderPos(page, steps);
  const high = await readBadge(page);
  expect(high.viscClamped).toBe(true);          // stale flag still set
  expect(high.nu).toBeLessThan(high.viscNuMax); // but the request is not clamped
  expect(high.visible).toBe(true);
  expect(high.text).toMatch(/under-converged|under-resolved/i);

  // Park mid-range again, then change tier. The badge must change; the slider
  // must not move.
  await setSliderPos(page, steps / 2);
  const before = await readBadge(page);
  expect(before.visible).toBe(false);

  await page.evaluate(() => {
    const { adaptive } = window.__flowlab;
    adaptive.manualOverride = true;
    adaptive.currentTierIndex = adaptive.tiers.indexOf(1024);
    adaptive.applyTier();
  });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  const after = await readBadge(page);
  expect(after.numY).toBe(1024);
  expect(after.sliderPos).toBe(before.sliderPos);   // slider did not move
  expect(after.visible).toBe(true);                 // but the badge appeared
  expect(after.text).toMatch(/no honest Re/i);

  // Back down to 64, where the same slider position is comfortably honest.
  await page.evaluate(() => {
    const { adaptive } = window.__flowlab;
    adaptive.currentTierIndex = adaptive.tiers.indexOf(64);
    adaptive.applyTier();
  });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  const back = await readBadge(page);
  expect(back.numY).toBe(64);
  expect(back.sliderPos).toBe(before.sliderPos);
  expect(back.visible).toBe(false);
});
