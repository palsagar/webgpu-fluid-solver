import { test, expect } from '@playwright/test';

/**
 * Tests for the honest Reynolds window (static/js/diagnostics.js) and for the
 * badge it drives in the UI.
 *
 * Every expected number here is derived from the measured tables, not from
 * whatever the implementation happens to return:
 *
 *   Floor    Re_min = U*D / (N_MAX * 1/4 * h^2 / dt)      [solver.viscNuMax]
 *   Ceiling  Re_max = U*D / nuNumConverged(dt)            [NU_NUM_PER_DT * dt]
 *   Ceiling' Re_max = U*D / NU_NUM_ITERS80[tier]          [under-converged projection]
 *
 * With the Karman reference (U = 1.0, D = 0.12, dt = 1/240, N_MAX = 32) that
 * gives, per tier:
 *
 *   tier      64      128      256      512     1024
 *   floor    0.256   1.024    4.096   16.384   65.536
 *   ceil    237.32  237.32   237.32   237.32   237.32
 *   ceil'   236.12  184.50    59.04    19.15     7.77
 *
 * The floors are exactly half their dt = 1/120 values (floor scales with dt) and
 * the ceiling is 1.94x its old one (nu_num is linear in dt) — the halving of dt
 * opened the window from BOTH sides, which is the whole point of the change.
 */

// Karman reference, matching the measurement conditions exactly.
const REF = { dt: 1 / 240, D: 0.12, U: 1.0, nMax: 32 };

/** Floor computed from the substep budget, independently of the module. */
const floorAt = (numY) => (REF.U * REF.D) / (REF.nMax * 0.25 * (1 / numY) ** 2 / REF.dt);

async function loadDiagnostics(page) {
  await page.goto('/');
  return page.evaluate(() => import('/js/diagnostics.js').then((m) => ({
    NU_NUM_PER_DT: m.NU_NUM_PER_DT,
    NU_NUM_ANCHOR_DT: m.NU_NUM_ANCHOR_DT,
    nuNumAt240: m.nuNumConverged(1 / 240),
    nuNumAt120: m.nuNumConverged(1 / 120),
    nuNumAt60: m.nuNumConverged(1 / 60),
    NU_NUM_ITERS80: m.NU_NUM_ITERS80,
    NU_NUM_ITERS80_DT: m.NU_NUM_ITERS80_DT,
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
      nMax: REF.nMax, nuNum: d.nuNumConverged(REF.dt),
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

  // Converged projection, anchored at dt = 1/240 tier 256: nu_num = 5.0564e-4,
  // so the per-dt coefficient is 5.0564e-4 * 240.
  expect(c.NU_NUM_ANCHOR_DT).toBeCloseTo(1 / 240, 10);
  expect(c.NU_NUM_PER_DT).toBeCloseTo(0.121354, 6);
  expect(c.nuNumAt240).toBeCloseTo(5.0564e-4, 7);

  // The coefficient must actually SCALE — a flat constant passes the line above
  // but fails here, and a flat constant is exactly what this change replaced.
  expect(c.nuNumAt120).toBeCloseTo(2 * c.nuNumAt240, 10);
  expect(c.nuNumAt60).toBeCloseTo(4 * c.nuNumAt240, 10);

  // Anchoring at 1/240 overstates nu_num at 1/120 (measured there: 9.8230e-4)
  // because the halving ratio is 1.943, not 2. Overstating nu_num UNDERSTATES
  // the ceiling, which is the conservative direction — assert the sign, so an
  // anchor flipped to the 1/120 measurement (which would be optimistic at
  // 1/240) fails here rather than silently shipping.
  expect(c.nuNumAt120).toBeGreaterThan(9.8230e-4);
  expect(c.nuNumAt120).toBeLessThan(1.05 * 9.8230e-4);

  // Operating point, measured at the shipped numIters = 80 and dt = 1/240.
  expect(c.PROJECTION_ITERS_MEASURED).toBe(80);
  expect(c.NU_NUM_ITERS80_DT).toBeCloseTo(1 / 240, 10);
  expect(c.NU_NUM_ITERS80[64]).toBeCloseTo(5.0821e-4, 8);
  expect(c.NU_NUM_ITERS80[128]).toBeCloseTo(6.5042e-4, 8);
  expect(c.NU_NUM_ITERS80[256]).toBeCloseTo(2.0324e-3, 7);
  expect(c.NU_NUM_ITERS80[512]).toBeCloseTo(6.2658e-3, 7);
  expect(c.NU_NUM_ITERS80[1024]).toBeCloseTo(1.5446e-2, 6);

  // The table must be MONOTONE in tier: a fixed iteration count converges
  // progressively less well as the grid grows. A table accidentally left at the
  // dt = 1/120 values would still be monotone, hence the exact values above.
  const tiers = [64, 128, 256, 512, 1024];
  for (let i = 1; i < tiers.length; i++) {
    expect(c.NU_NUM_ITERS80[tiers[i]]).toBeGreaterThan(c.NU_NUM_ITERS80[tiers[i - 1]]);
  }

  // Every operating-point value must exceed the converged one at the same dt —
  // an under-converged projection can only ADD dissipation.
  for (const tier of tiers) {
    expect(Number.isFinite(c.NU_NUM_ITERS80[tier])).toBe(true);
    expect(c.NU_NUM_ITERS80[tier]).toBeGreaterThan(0);
    expect(c.NU_NUM_ITERS80[tier]).toBeGreaterThanOrEqual(c.nuNumAt240);
  }
});

// ── honestWindow ────────────────────────────────────────────────────────────

test('honest window bounds and badge reasons', async ({ page }) => {
  await page.goto('/');
  const w = await windowFor(page, 256);

  // Floor: N_MAX substeps each at the 1/4 explicit stability limit give
  // nu_max = 32 * 0.25 * h^2/dt = 0.0292969 at tier 256, so Re_min = 4.096.
  // Dropping N_MAX from the formula makes this 131.1 — 32x larger.
  expect(w.reMin).toBeCloseTo(4.096, 3);
  expect(w.reMin).toBeCloseTo(floorAt(256), 6);

  // Ceiling: U*D / 5.0564e-4 = 237.322. A ceiling still carrying the dt = 1/120
  // value (9.823e-4) gives 122.16, so this fails if the halving was not applied.
  expect(w.reMax).toBeCloseTo(237.322, 2);

  // Operating point at the shipped 80 iterations: U*D / 2.0324e-3 = 59.043.
  expect(w.reMaxProjection).toBeCloseTo(59.043, 2);

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
  expect(tooLow.reason).toContain('4.1');     // the floor, from viscNuMax

  // The prediction alone must fire even when the solver has not stepped yet.
  const tooLowUnstepped = await stateFor(page, {
    re: 2, reEff: 2, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(tooLowUnstepped.code).toBe('clamped');

  // Above both ceilings at tier 256 the projection is what binds (59 vs 237).
  const tooHigh = await stateFor(page, {
    re: 1e6, reEff: 1e6, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(tooHigh.ok).toBe(false);
  expect(tooHigh.code).toBe('projection');
  expect(tooHigh.reason).toMatch(/under-converged/i);
  expect(tooHigh.reason).toContain('59');   // the binding ceiling
  expect(tooHigh.reason).toContain('237');  // reconciled with the scheme ceiling
});

test('the binding ceiling is named by which viscosity actually dominates', async ({ page }) => {
  await page.goto('/');

  // Tier 64: the two ceilings agree to 0.5% (236.12 vs 237.32), well inside
  // Task 7's ±10% fit-window uncertainty, so the scheme is what binds.
  const w64 = await windowFor(page, 64);
  expect(w64.reMaxProjection).toBeCloseTo(236.123, 2);
  const s64 = await stateFor(page, {
    re: 400, reEff: 400, reMin: w64.reMin, reMax: w64.reMax,
    reMaxProjection: w64.reMaxProjection, viscClamped: false,
  });
  expect(s64.code).toBe('scheme');
  expect(s64.reason).toMatch(/under-resolved/i);
  expect(s64.reason).toContain('237');

  // Tier 128: 184.5 vs 237.3 — a 22% gap, outside the measurement scatter, so
  // the under-converged projection is the honest ceiling.
  const w128 = await windowFor(page, 128);
  expect(w128.reMaxProjection).toBeCloseTo(184.496, 2);
  const s128 = await stateFor(page, {
    re: 400, reEff: 400, reMin: w128.reMin, reMax: w128.reMax,
    reMaxProjection: w128.reMaxProjection, viscClamped: false,
  });
  expect(s128.code).toBe('projection');

  // Between the two ceilings at tier 128 (184.5 < Re < 237.3) the projection
  // reason must still fire — this is the range the flat ceiling would hide.
  const between = await stateFor(page, {
    re: 210, reEff: 210, reMin: w128.reMin, reMax: w128.reMax,
    reMaxProjection: w128.reMaxProjection, viscClamped: false,
  });
  expect(between.ok).toBe(false);
  expect(between.code).toBe('projection');
});

// ── The empty windows ───────────────────────────────────────────────────────

test('the honest window is empty only at tier 1024, and for want of iterations', async ({ page }) => {
  await page.goto('/');

  // Tier 1024: floor 65.54 sits above the PROJECTION ceiling 7.77, so no slider
  // position is honest here. But it sits well BELOW the scheme ceiling 237.32,
  // so unlike at dt = 1/120 the grid itself is no longer the obstacle — raising
  // iterations would open a window. The advice must say iterations, not dt.
  const w1024 = await windowFor(page, 1024);
  expect(w1024.reMin).toBeCloseTo(65.536, 3);
  expect(w1024.reMax).toBeCloseTo(237.322, 2);
  expect(w1024.reMaxProjection).toBeCloseTo(7.769, 2);
  expect(w1024.reMin).toBeLessThan(w1024.reMax);             // grid alone is fine
  expect(w1024.reMin).toBeGreaterThan(w1024.reMaxProjection); // iterations are not

  for (const re of [0.5, 15, 66, 237, 500]) {
    const st = await stateFor(page, {
      re, reEff: re, reMin: w1024.reMin, reMax: w1024.reMax,
      reMaxProjection: w1024.reMaxProjection, viscClamped: false,
    });
    expect(st.ok, `Re ${re} at tier 1024 must not be reported as honest`).toBe(false);
    expect(st.code).toBe('empty-iters');
    expect(st.reason).toMatch(/no honest Re/i);
    expect(st.reason).toMatch(/iterations/i);
    expect(st.reason).toContain('66');   // the floor
    expect(st.reason).toContain('7.8');  // the under-converged ceiling
  }

  // Tier 512 is NO LONGER empty. At dt = 1/120 its floor (32.77) sat above its
  // projection ceiling (11.93); halving dt moved the floor to 16.38 and the
  // ceiling to 19.15, so a narrow but real window opened. Asserting this keeps
  // a future dt regression from silently re-emptying it.
  const w512 = await windowFor(page, 512);
  expect(w512.reMin).toBeCloseTo(16.384, 3);
  expect(w512.reMaxProjection).toBeCloseTo(19.152, 2);
  expect(w512.reMin).toBeLessThan(w512.reMaxProjection);

  const st512 = await stateFor(page, {
    re: 18, reEff: 18, reMin: w512.reMin, reMax: w512.reMax,
    reMaxProjection: w512.reMaxProjection, viscClamped: false,
  });
  expect(st512.ok).toBe(true);
  expect(st512.code).toBeNull();

  // Tiers 64 .. 512 are all non-empty — otherwise the tier-1024 assertion above
  // would be vacuous and the app would badge everywhere.
  for (const tier of [64, 128, 256, 512]) {
    const w = await windowFor(page, tier);
    expect(w.reMin, `tier ${tier} must leave a window`).toBeLessThan(w.reMaxProjection);
  }
});

test('the empty-grid reason still fires when the grid itself is the obstacle', async ({ page }) => {
  await page.goto('/');

  // No SHIPPED tier reaches this state at dt = 1/240 — that is the point of the
  // change. The branch is still live code, so it is exercised directly with a
  // floor above the scheme ceiling. Without this the branch would be untested
  // and could rot into the next dt change.
  const st = await stateFor(page, {
    re: 300, reEff: 300, reMin: 400, reMax: 237.322,
    reMaxProjection: 50, viscClamped: false,
  });
  expect(st.ok).toBe(false);
  expect(st.code).toBe('empty-grid');
  expect(st.reason).toMatch(/no honest Re/i);
  expect(st.reason).toMatch(/resolution|dt/i);
  // It must advise dt/resolution, NOT iterations — raising iterations cannot
  // lift a ceiling the scheme itself sets.
  expect(st.reason).not.toMatch(/iterations/i);
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

  // Bottom must reach the lowest floor in the tier set (tier 64 -> Re 0.256),
  // so the honest low end is representable rather than cropped off. This is the
  // assertion that forces MIN down when dt falls: every floor scales with dt.
  expect(c.RE_SLIDER_MIN).toBeLessThanOrEqual(floorAt(64));

  // Top must clear both structural bounds — the flat ceiling (237.32) and the
  // highest floor (tier 1024 -> 65.54) — so every badge regime is reachable.
  expect(c.RE_SLIDER_MAX).toBeGreaterThan(floorAt(1024));
  expect(c.RE_SLIDER_MAX).toBeGreaterThan(237.322);

  // Log spacing: the midpoint must land inside the honest window at the
  // startup tier (256: 4.10 .. 59.04 at 80 iterations), so mid-slider is
  // badge-free and "absent mid-range" is a real property, not luck.
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

  // Measured shedding onset at tier 256, by growth-to-saturation over 30 s:
  // growth(late/early) = 0.638 at Re 57 and 12.2 at Re 60, crossing 1 at 57.5.
  // The default must clear it — below onset the preset shows no street at all.
  const ONSET = 57.46;
  expect(shipped.re).toBeGreaterThan(ONSET);

  // And it must clear it by enough to be VISIBLE, not merely unstable. The
  // saturated fluctuation is 1.0e-3 of the mean flow at Re 57 but 8.8e-1 at
  // Re 72, so a default parked just above onset would technically shed and
  // show nothing. This is the assertion that stops the default drifting down
  // into the honest window at the cost of the picture.
  expect(shipped.re).toBeGreaterThan(72);

  // It is knowingly outside the honest window, so the app opens badged.
  const w = await windowFor(page, 256);
  expect(shipped.re).toBeGreaterThan(w.reMaxProjection);

  // But only just: the whole point of halving dt was to shrink this gap, which
  // went from 7.7x (Re 251 vs ceiling 32.8) to 1.27x. Assert the improvement,
  // so a default that drifts back up fails.
  expect(shipped.re / w.reMaxProjection).toBeLessThan(1.5);

  // The overlap that dt bought is real but unreachable: the window now contains
  // the onset (57.5 < 59.04), yet no slider position lands in that 2.8% band
  // because the log step is 7.9%. Asserting it keeps the rationale honest — if
  // a future change makes a position land there, this fails and gets revisited.
  expect(ONSET).toBeLessThan(w.reMaxProjection);
  const anyBoth = await page.evaluate(async ({ ONSET, ceiling }) => {
    const d = await import('/js/diagnostics.js');
    for (let p = 0; p <= d.RE_SLIDER_STEPS; p++) {
      const re = d.reFromSliderPos(p);
      if (re > ONSET && re <= ceiling) return p;
    }
    return null;
  }, { ONSET, ceiling: w.reMaxProjection });
  expect(anyBoth).toBeNull();

  const badge = await page.evaluate(() => ({
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
    valRe: document.getElementById('val-re').textContent,
  }));
  expect(badge.visible).toBe(true);
  expect(badge.text).toMatch(/under-converged/i);
  expect(badge.valRe).toBe('75');
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
