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
 *   Ceiling' Re_max = U*D / NU_NUM_ITERS256[tier]         [projection at 256 iters]
 *
 * With the Karman reference (U = 1.0, D = 0.12, dt = 1/240, N_MAX = 32) that
 * gives, per tier:
 *
 *   tier      64      128      256      512     1024
 *   floor    0.256   1.024    4.096   16.384   65.536
 *   ceil    237.32  237.32   237.32   237.32   237.32
 *   ceil'   236.22  236.33   154.23    48.91    16.51
 *
 * The floors are exactly half their dt = 1/120 values (floor scales with dt) and
 * the ceiling is 1.94x its old one (nu_num is linear in dt) — the halving of dt
 * opened the window from BOTH sides.
 *
 * `ceil'` is the row that moved when the Karman preset went from 80 to 256
 * pressure iterations. At 80 it read 236.12 / 184.50 / 59.04 / 19.15 / 7.77.
 * The startup tier gained 2.6x, which is what put the shipped default INSIDE
 * the window, and the two coarse tiers converged onto the scheme ceiling.
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
    NU_NUM_ITERS256: m.NU_NUM_ITERS256,
    NU_NUM_ITERS256_DT: m.NU_NUM_ITERS256_DT,
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
    const reMaxProjection = (REF.U * REF.D) / d.NU_NUM_ITERS256[numY];
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

  // Operating point, measured at the shipped numIters = 256 and dt = 1/240.
  // These are the values the badge divides U*D by, so a stale table here is a
  // lying badge. The 80-iteration values they replaced were 5.0821e-4 /
  // 6.5042e-4 / 2.0324e-3 / 6.2658e-3 / 1.5446e-2; every tier below differs
  // from its old value by more than the tolerance, so a table left un-migrated
  // fails rather than passing on a near-miss.
  expect(c.PROJECTION_ITERS_MEASURED).toBe(256);
  expect(c.NU_NUM_ITERS256_DT).toBeCloseTo(1 / 240, 10);
  expect(c.NU_NUM_ITERS256[64]).toBeCloseTo(5.0801e-4, 8);
  expect(c.NU_NUM_ITERS256[128]).toBeCloseTo(5.0777e-4, 8);
  expect(c.NU_NUM_ITERS256[256]).toBeCloseTo(7.7808e-4, 8);
  expect(c.NU_NUM_ITERS256[512]).toBeCloseTo(2.4536e-3, 7);
  expect(c.NU_NUM_ITERS256[1024]).toBeCloseTo(7.2698e-3, 7);

  // The iteration count the table is measured at must match what the Karman
  // preset actually ships, or the badge quotes a number the solver never ran.
  const shippedIters = await page.evaluate(() =>
    import('/js/presets.js').then((m) => m.PRESETS.karmanVortex.numIters));
  expect(shippedIters).toBe(c.PROJECTION_ITERS_MEASURED);

  const tiers = [64, 128, 256, 512, 1024];

  // At 256 iterations the table is NOT monotone from the first tier, unlike at
  // 80. Tiers 64 and 128 are both converged — they sit within 0.5% of the
  // converged constant, and the 0.05% by which 128 falls below 64 is scatter
  // between two converged fits, not a trend. Asserting bare monotonicity here
  // would fail on physically correct values; asserting the two regimes
  // separately is the stronger claim, and still fails on a stale table.
  for (const tier of [64, 128]) {
    expect(c.NU_NUM_ITERS256[tier] / c.nuNumAt240).toBeGreaterThan(0.99);
    expect(c.NU_NUM_ITERS256[tier] / c.nuNumAt240).toBeLessThan(1.01);
  }
  // From tier 256 up the projection binds again and the rise resumes, steeply:
  // each tier is at least 1.5x the one below, so a table that flattened out
  // (or was silently reused from a converged run) fails.
  for (const tier of [256, 512, 1024]) {
    const prev = tiers[tiers.indexOf(tier) - 1];
    expect(c.NU_NUM_ITERS256[tier]).toBeGreaterThan(1.5 * c.NU_NUM_ITERS256[prev]);
  }

  // Every operating-point value must exceed the converged one at the same dt —
  // an under-converged projection can only ADD dissipation.
  for (const tier of tiers) {
    expect(Number.isFinite(c.NU_NUM_ITERS256[tier])).toBe(true);
    expect(c.NU_NUM_ITERS256[tier]).toBeGreaterThan(0);
    expect(c.NU_NUM_ITERS256[tier]).toBeGreaterThanOrEqual(c.nuNumAt240);
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

  // Operating point at the shipped 256 iterations: U*D / 7.7808e-4 = 154.226.
  // At 80 iterations this read 59.043, so a preset left at 80 fails here.
  expect(w.reMaxProjection).toBeCloseTo(154.226, 2);

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

  // Above both ceilings at tier 256 the projection is still what binds, even at
  // 256 iterations: 154 vs 237.
  const tooHigh = await stateFor(page, {
    re: 1e6, reEff: 1e6, reMin: w.reMin, reMax: w.reMax,
    reMaxProjection: w.reMaxProjection, viscClamped: false,
  });
  expect(tooHigh.ok).toBe(false);
  expect(tooHigh.code).toBe('projection');
  expect(tooHigh.reason).toMatch(/under-converged/i);
  expect(tooHigh.reason).toContain('154');  // the binding ceiling
  expect(tooHigh.reason).toContain('237');  // reconciled with the scheme ceiling
  expect(tooHigh.reason).toContain('256');  // the iteration count it is quoted at
});

test('the binding ceiling is named by which viscosity actually dominates', async ({ page }) => {
  await page.goto('/');

  // Tiers 64 AND 128 are now scheme-limited: at 256 iterations both ceilings
  // land at 236.2 / 236.3 against the scheme's 237.32, agreeing to 0.5% — well
  // inside Task 7's ±10% fit-window uncertainty. At 80 iterations tier 128 read
  // 184.5 and was attributed to the projection, so this pair of assertions is
  // what records that raising iterations MOVED the attribution boundary.
  for (const [tier, ceil] of [[64, 236.216], [128, 236.327]]) {
    const w = await windowFor(page, tier);
    expect(w.reMaxProjection).toBeCloseTo(ceil, 2);
    const s = await stateFor(page, {
      re: 400, reEff: 400, reMin: w.reMin, reMax: w.reMax,
      reMaxProjection: w.reMaxProjection, viscClamped: false,
    });
    expect(s.code, `tier ${tier} must blame the scheme, not the projection`).toBe('scheme');
    expect(s.reason).toMatch(/under-resolved/i);
    expect(s.reason).toContain('237');
  }

  // Tier 256: 154.2 vs 237.3 — a 35% gap, outside the measurement scatter, so
  // the under-converged projection is the honest ceiling. This is the startup
  // tier, so it is the attribution the user actually sees.
  const w256 = await windowFor(page, 256);
  expect(w256.reMaxProjection).toBeCloseTo(154.226, 2);
  const s256 = await stateFor(page, {
    re: 400, reEff: 400, reMin: w256.reMin, reMax: w256.reMax,
    reMaxProjection: w256.reMaxProjection, viscClamped: false,
  });
  expect(s256.code).toBe('projection');

  // Between the two ceilings at tier 256 (154.2 < Re < 237.3) the projection
  // reason must still fire — this is the range the flat ceiling would hide.
  const between = await stateFor(page, {
    re: 200, reEff: 200, reMin: w256.reMin, reMax: w256.reMax,
    reMaxProjection: w256.reMaxProjection, viscClamped: false,
  });
  expect(between.ok).toBe(false);
  expect(between.code).toBe('projection');
});

// ── The empty windows ───────────────────────────────────────────────────────

test('the honest window is empty only at tier 1024, and for want of iterations', async ({ page }) => {
  await page.goto('/');

  // Tier 1024: floor 65.54 sits above the PROJECTION ceiling 16.51, so no
  // slider position is honest here — even at 256 iterations, which lifted that
  // ceiling from 7.77 but nowhere near far enough. It sits well BELOW the
  // scheme ceiling 237.32, so the grid itself is not the obstacle: more
  // iterations would still open a window. The advice must say iterations, not dt.
  const w1024 = await windowFor(page, 1024);
  expect(w1024.reMin).toBeCloseTo(65.536, 3);
  expect(w1024.reMax).toBeCloseTo(237.322, 2);
  expect(w1024.reMaxProjection).toBeCloseTo(16.507, 2);
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
    expect(st.reason).toContain('17');   // the under-converged ceiling, rounded

    // The advice must be one the app can actually honour. Closing 65.5 vs 16.5
    // is a factor of 4 in nu_num; the 80 -> 256 escalation was 3.2x in
    // iterations and bought 2.1x, and the control stops at 320 (a further
    // 1.25x). So "raise iterations" is not a fix here — and it would be
    // invisible anyway, since NU_NUM_ITERS256 is pinned at 256. Resolution and
    // dt both move the floor directly and are both reachable from the UI.
    expect(st.reason).not.toMatch(/raise\s+(the\s+)?iterations/i);
    expect(st.reason).toMatch(/lower the resolution/i);
    expect(st.reason).toMatch(/\bdt\b/);
  }

  // Tier 512 is NO LONGER empty, and the margin is no longer marginal. At
  // dt = 1/120 its floor (32.77) sat above its projection ceiling (11.93);
  // halving dt opened a 16.38 .. 19.15 sliver, and 256 iterations widened that
  // to 16.38 .. 48.91. Asserting both bounds keeps a dt OR an iteration
  // regression from silently re-emptying it.
  const w512 = await windowFor(page, 512);
  expect(w512.reMin).toBeCloseTo(16.384, 3);
  expect(w512.reMaxProjection).toBeCloseTo(48.908, 2);
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

// ── The projection ceiling's own validity ───────────────────────────────────

test('a projection ceiling above the converged one is refused, not min()-ed away', async ({ page }) => {
  await page.goto('/');

  // A since-removed preset's real operating point (ADR-0009): U = 2.0,
  // D = 2*0.15, dt = 1/60, 40 pressure iterations. The preset is gone; its
  // parameter set stays as the adversarial case — nothing else exercises both
  // off-slice axes at once. NU_NUM_ITERS256 is measured at dt =
  // 1/240 AND 256 iterations, so carrying it here is carrying it off both
  // axes at once — and it produces a physically impossible pair.
  const wt = await page.evaluate(async () => {
    const d = await import('/js/diagnostics.js');
    const U = 2.0, D = 0.30, dt = 1 / 60;
    const w = d.honestWindow({ h: 1 / 256, dt, D, U, nMax: 32, nuNum: d.nuNumConverged(dt) });
    return { ...w, carried: (U * D) / d.NU_NUM_ITERS256[256] };
  });

  // The impossible pair, stated in numbers: an under-converged solve cannot
  // dissipate LESS than a converged one, so a projection ceiling of 771
  // against a converged ceiling of 297 is not a measurement of this point.
  expect(wt.reMax).toBeCloseTo(296.65, 1);
  expect(wt.carried).toBeCloseTo(771.13, 1);
  expect(wt.carried).toBeGreaterThan(wt.reMax);

  // Inside the grid's own window but with no valid projection ceiling: the
  // honest answer is that the ceiling here is unmeasured. Taking min() of the
  // two — what shipped before — reported `ok` and quoted 297 against a true
  // ceiling lower by an unmeasured multiple.
  const st = await stateFor(page, {
    re: 100, reEff: 100, reMin: wt.reMin, reMax: wt.reMax,
    reMaxProjection: wt.carried, viscClamped: false,
  });
  expect(st.ok).toBe(false);
  expect(st.code).toBe('unmeasured');
  expect(st.reason).toMatch(/unmeasured/i);
  expect(st.reason).toContain('256');    // the iteration count it WAS measured at
  expect(st.reason).toContain('240');    // and the dt
  expect(st.reason).toContain('297');    // the bound that IS known
  expect(st.reason).not.toContain('771'); // the number it must never quote

  // An omitted projection ceiling means the same thing: no number is not a
  // number to reason from, so it must not silently pass as honest.
  const omitted = await stateFor(page, {
    re: 100, reEff: 100, reMin: wt.reMin, reMax: wt.reMax, viscClamped: false,
  });
  expect(omitted.code).toBe('unmeasured');

  // Above the scheme ceiling the attribution must be the SCHEME, never the
  // projection — a ceiling that violated the invariant cannot be blamed for
  // anything. This is the branch `min()` would have gotten backwards.
  const above = await stateFor(page, {
    re: 400, reEff: 400, reMin: wt.reMin, reMax: wt.reMax,
    reMaxProjection: wt.carried, viscClamped: false,
  });
  expect(above.code).toBe('scheme');
  expect(above.reason).toContain('297');
  expect(above.reason).not.toContain('771');

  // Below the floor the solver's own saturation is measured ground truth and
  // still outranks the unmeasured ceiling.
  const below = await stateFor(page, {
    re: 1, reEff: wt.reMin, reMin: wt.reMin, reMax: wt.reMax,
    reMaxProjection: wt.carried, viscClamped: true,
  });
  expect(below.code).toBe('clamped');

  // And the invariant must hold at the point the table IS measured at, or the
  // guard above would be firing on the Karman preset too.
  const karman = await windowFor(page, 256);
  expect(karman.reMaxProjection).toBeLessThanOrEqual(karman.reMax);
});

test('honestWindow floors on the nuMax it is handed, not a private copy', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.solver, null, { timeout: 20_000 });

  const r = await page.evaluate(async () => {
    const d = await import('/js/diagnostics.js');
    const { solver } = window.__flowlab;
    const args = { h: solver.h, dt: solver.params.dt, D: 0.12, U: 1.0, nMax: 32 };
    const nuNum = d.nuNumConverged(args.dt);
    return {
      solverNuMax: solver.viscNuMax,
      derived: d.honestWindow({ ...args, nuNum }),
      passed: d.honestWindow({ ...args, nuNum, nuMax: solver.viscNuMax }),
      doubled: d.honestWindow({ ...args, nuNum, nuMax: 2 * solver.viscNuMax }),
    };
  });

  // The module's default and the solver's getter are the same formula today —
  // that is the duplication, and this is the assertion that catches it drifting.
  expect(r.derived.nuMax).toBeCloseTo(r.solverNuMax, 12);
  expect(r.passed.reMin).toBeCloseTo(r.derived.reMin, 9);

  // And the parameter must actually be USED, not accepted and ignored: double
  // the saturation limit and the floor must halve.
  expect(r.doubled.nuMax).toBeCloseTo(2 * r.solverNuMax, 12);
  expect(r.doubled.reMin).toBeCloseTo(r.derived.reMin / 2, 9);
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
  // startup tier (256: 4.10 .. 154.23 at 256 iterations), so mid-slider is
  // badge-free and "absent mid-range" is a real property, not luck.
  const w256 = await windowFor(page, 256);
  expect(mapped.mid).toBeGreaterThan(w256.reMin);
  expect(mapped.mid).toBeLessThan(w256.reMaxProjection);
});

test('the shipped default sits inside the honest window and opens un-badged', async ({ page }) => {
  await page.goto('/');
  // The badge is written by the UI constructor, which runs after WebGPU init.
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });

  // WHAT THIS TEST COVERS, AND WHAT IT DOES NOT.
  //
  // Everything below is ARITHMETIC over the shipped constants: the slider
  // default, the measured tables, and the window they imply. It is worth
  // asserting — it is what stops the default drifting outside the window, and
  // what pins index.html's `value` attribute to the documented position — but
  // none of it touches the solver.
  //
  // In particular `ONSET` below is a literal transcribed from an offline
  // growth-rate measurement that this suite does not reproduce. It is used here
  // only to check the SLIDER MAPPING against a number, so `expect(shipped.re >
  // ONSET)` is a statement about the mapping, not about the flow. That the
  // solver still actually sheds at this position is a separate claim, and it is
  // measured on the live solver by the test immediately after this one.

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

  // Measured shedding onset at tier 256, dt = 1/240, 256 iterations, by the
  // zero crossing of the LINEAR GROWTH RATE — a window-independent criterion.
  // sigma = -0.0975 / -0.0036 / +0.0771 at Re 50 / 52 / 54, linear in Re, so
  // the crossing is 52.2; the +-0.3 is the spread over 53 combinations of fit
  // subset and analysis window. The superseded 57.5 came from a 30 s amplitude
  // RATIO, which near onset measures the window: at Re 52 the e-folding time is
  // 278 s and 30 s cannot tell it from a saturated limit cycle.
  //
  // The UPPER end of the interval is used here, so the assertion holds for any
  // onset the measurement is consistent with rather than only the best estimate.
  const ONSET = 52.5;
  expect(shipped.re).toBeGreaterThan(ONSET);

  // And it must clear it by enough to be VISIBLE, not merely unstable. A Hopf
  // bifurcation saturates at A_sat ~ sqrt(Re - Re_c), so just above onset the
  // street is real but far too weak to see. Saturated wake RMS as a fraction of
  // the free stream, verified flat over the final two 20 s windows:
  //
  //   Re      55      57.5     60      65     74.8     100     140
  //   A_sat  0.082   0.116   0.144   0.191   0.264   0.402   0.546
  //
  // A default parked just above onset would technically shed and show nothing;
  // 74.8 lands at 0.264, thirteen times the shedding gate. It is also slow:
  // near onset the growth rate vanishes, so Re 55 needs ~80 s of simulation
  // time to reach its plateau against ~12 s at Re 60. This is the assertion
  // that stops the default drifting down into the honest window at the cost of
  // the picture.
  expect(shipped.re).toBeGreaterThan(72);

  // It is INSIDE the honest window, so the app opens un-badged. This is the
  // assertion the numIters escalation exists to satisfy: at 80 iterations the
  // ceiling was 59.04 and this same position was 1.27x outside it.
  const w = await windowFor(page, 256);
  expect(shipped.re).toBeLessThanOrEqual(w.reMaxProjection);
  expect(shipped.re).toBeLessThanOrEqual(w.reMax);
  expect(shipped.re).toBeGreaterThan(w.reMin);

  // And inside with MARGIN, not on the cliff edge. The ceiling must sit at
  // least 1.5x above the default, so a ceiling regression that merely grazes
  // the default (as 59.04 did) fails here rather than shipping.
  expect(w.reMaxProjection / shipped.re).toBeGreaterThan(1.5);

  // The band that is both honest and shedding is now wide enough to CHOOSE
  // within rather than merely land in: at 80 iterations it was Re 52.2 .. 59.0,
  // a 13% band against a 7.9% log step, so exactly one slider position lived
  // there (pos 71, Re 55.2) on a wake 6% above onset. It must now hold several,
  // and the shipped default must be one of them.
  expect(ONSET).toBeLessThan(w.reMaxProjection);
  const both = await page.evaluate(async ({ ONSET, ceiling }) => {
    const d = await import('/js/diagnostics.js');
    const hits = [];
    for (let p = 0; p <= d.RE_SLIDER_STEPS; p++) {
      const re = d.reFromSliderPos(p);
      if (re > ONSET && re <= ceiling) hits.push(p);
    }
    return hits;
  }, { ONSET, ceiling: w.reMaxProjection });
  expect(both.length).toBeGreaterThan(5);
  expect(both).toContain(shipped.documented);

  const badge = await page.evaluate(() => ({
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
    valRe: document.getElementById('val-re').textContent,
  }));
  expect(badge.visible).toBe(false);
  expect(badge.text).toBe('');
  expect(badge.valRe).toBe('75');
});

/**
 * Steps the LIVE solver at whatever the UI currently has configured, sampling
 * the transverse velocity at the app's own probe cell.
 *
 * Replicates `main.js`'s per-frame sequence exactly — smoke inlet before the
 * step, inflow column re-applied after it — but off the rAF loop, so the run
 * is not paced by the display. `solver.paused` is set for the duration so the
 * frame loop does not step the field underneath the sampler.
 *
 * Only the probe cell is copied back (4 bytes), not the whole velocity buffer:
 * at one readback per 10 steps a full-field copy would dominate the run.
 */
async function stepAndSampleWake(page, { settleSteps, sampleSteps, sampleEvery }) {
  return page.evaluate(async ({ settleSteps, sampleSteps, sampleEvery }) => {
    const { solver, ui, interaction } = window.__flowlab;
    const d = await import('/js/diagnostics.js');

    const cell = d.probeCell({
      obstacleX: interaction.obstacleX,
      obstacleY: interaction.obstacleY,
      D: 2 * interaction.obstacleRadius,
      h: solver.h,
      numX: solver.numX,
      numY: solver.numY,
    });
    if (!cell) return { error: 'no probe cell for this geometry' };
    const byteOffset = (cell.i * solver.numY + cell.j) * 4;

    const wasPaused = solver.paused;
    solver.paused = true;
    // Let any frame already in flight finish before taking the field over.
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

    const readProbeV = async () => {
      const stage = solver.device.createBuffer({
        size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = solver.device.createCommandEncoder();
      enc.copyBufferToBuffer(solver.velocityBuffers.v, byteOffset, stage, 0, 4);
      solver.device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const val = new Float32Array(stage.getMappedRange().slice(0))[0];
      stage.unmap();
      stage.destroy();
      return val;
    };

    const oneStep = () => {
      if (ui.smokeInletData) {
        solver.device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
      }
      solver.step(ui.numIters);
      if (ui.boundaryVelData) {
        solver.writeInflowColumn(
          1, ui.boundaryVelData.uData, 1 * solver.numY, solver.numY);
      }
    };

    const t0 = performance.now();
    for (let k = 0; k < settleSteps; k++) oneStep();

    const t = [], v = [];
    for (let k = 0; k < sampleSteps; k++) {
      oneStep();
      if (k % sampleEvery === 0) {
        t.push(solver.simTime);
        v.push(await readProbeV());
      }
    }

    solver.paused = wasPaused;

    const n = v.length;
    const mean = v.reduce((a, b) => a + b, 0) / n;
    const rms = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n);

    // Feed the app's OWN detector, so what this test calls "shedding" is
    // exactly what the readout calls shedding — not a second opinion.
    const probe = new d.StrouhalProbe();
    for (let k = 0; k < n; k++) probe.push(v[k], t[k]);
    const verdict = probe.read({ D: 2 * interaction.obstacleRadius, U: ui._inflowVelocity() });

    return {
      cell, n, mean, rms, verdict,
      U: ui._inflowVelocity(),
      D: 2 * interaction.obstacleRadius,
      numIters: ui.numIters,
      dt: solver.params.dt,
      numY: solver.numY,
      simTimeSpan: t[n - 1] - t[0],
      wallMs: performance.now() - t0,
      allFinite: v.every(Number.isFinite),
      vSpread: Math.max(...v) - Math.min(...v),
    };
  }, { settleSteps, sampleSteps, sampleEvery });
}

test('the shipped default actually sheds, observed on the live solver', async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });

  // WHY THIS TEST EXISTS.
  //
  // The test above says the shipped Re clears a transcribed onset literal. That
  // literal came from an offline growth-rate sweep this suite does not run, so
  // if the SOLVER's onset moved — a MacCormack regression, a projection change,
  // a boundary change altering the blockage — nothing would notice, and the
  // onset is the number this branch re-measured twice. This one steps the real
  // solver at the shipped default and asks the app's own detector what it sees.
  //
  // WHAT IT COVERS: that at the shipped operating point (Karman, tier 256,
  // dt = 1/240, 256 iterations, Re 74.8) the wake is unsteady, and that its
  // frequency and amplitude land where they were measured. It fails if the
  // onset rises above 74.8 (the wake would read `steady`), and it fails if the
  // wake's character changes enough to move St or A_sat out of the measured band.
  //
  // WHAT IT DOES NOT COVER: the onset VALUE. Locating the crossing needs a
  // sweep in Re, each point from identical initial conditions with a 15 s
  // spin-up and 50 s of logging — minutes of wall clock per point. That stays
  // offline. This test bounds the onset from above (it is below 74.8) and
  // nothing more.

  // Pin the tier: an adaptive switch mid-run would resize the grid and change
  // the operating point the measured numbers below belong to.
  await page.evaluate(() => { window.__flowlab.adaptive.manualOverride = true; });

  const cfg = await page.evaluate(() => {
    const { ui, solver } = window.__flowlab;
    // Identical initial conditions, so the settle window below starts from the
    // impulsive start rather than from however long the page happened to sit.
    ui.reapplyCurrentPreset();
    return { preset: ui.currentPreset, numIters: ui.numIters, dt: solver.params.dt,
             numY: solver.numY };
  });
  expect(cfg.preset).toBe('karmanVortex');
  expect(cfg.numIters).toBe(256);
  expect(cfg.dt).toBeCloseTo(1 / 240, 10);
  expect(cfg.numY).toBe(256);

  // 2400 settle steps = 10.0 s of simulation time, then 1280 sampled steps at
  // one sample per 10 = 128 samples, which is exactly the minimum
  // `StrouhalProbe.read()` accepts (PROBE_CAPACITY / 2) and spans 5.3 s — about
  // 7.9 shedding periods at St 0.18.
  //
  // The settle length is set by measurement, not by guessing. At 1200 steps
  // (5 s) the run reads St = 0.169 at rms/U = 0.064 — shedding, but still deep
  // in the growth phase, only a quarter of the way to the measured plateau. At
  // 2400 it reads St = 0.178 at rms/U = 0.230, within 1% and 13% of the offline
  // saturated values (0.180 and 0.2636). The extra 1200 steps cost ~17 s of
  // wall clock and are what makes the amplitude assertion below able to have a
  // real margin instead of grazing the shedding gate.
  const r = await stepAndSampleWake(page, {
    settleSteps: 2400, sampleSteps: 1280, sampleEvery: 10,
  });
  expect(r.error).toBeUndefined();
  console.log(
    `live wake @ Re 74.8: state = ${r.verdict.state}  St = ${r.verdict.st}  ` +
    `rms/U = ${(r.rms / r.U).toFixed(4)}  span = ${r.simTimeSpan.toFixed(2)} s  ` +
    `(${(r.wallMs / 1000).toFixed(1)} s wall)`);

  // The series is a real one: finite, and not a constant field. Without this a
  // dead device would read rms = 0 and land in `steady`, which is a verdict
  // about physics delivered from a buffer carrying none.
  expect(r.allFinite).toBe(true);
  expect(r.vSpread).toBeGreaterThan(0);
  expect(r.n).toBe(128);
  expect(r.U).toBeCloseTo(1.0, 6);
  expect(r.D).toBeCloseTo(0.12, 6);

  // THE PAYLOAD. The app's own detector, on the app's own default, must say the
  // wake sheds. If the solver's onset rises past Re 74.8 this reads `steady`.
  expect(r.verdict.state).toBe('shedding');

  // And the amplitude must be the measured one, not merely over the 0.02 gate.
  // A_sat = 0.2636 at Re 74.8 offline (saturation-verified, drift 0.01% over
  // the final two 20 s windows); this run measures 0.230, 13% short because it
  // settles for 10 s rather than 115 s. The lower bound is 7.5x the shedding
  // gate, so a wake that merely trips the detector fails here, and the upper
  // bound sits below the Re 100 plateau (0.402).
  expect(r.rms / r.U).toBeGreaterThan(0.15);
  expect(r.rms / r.U).toBeLessThan(0.35);

  // And the frequency must be the measured one: St = 0.180 at Re 74.8, from the
  // offline sweep that read 0.166 at Re 55 up to 0.200 at Re 140.
  //
  // The band is +-1 zero crossing, not +-1%, and that is what a 5.3 s window
  // buys. `read()` estimates f as (crossings - 1) / (t_last - t_first) over
  // ~7.9 periods, so gaining or losing a single crossing moves St to 0.204 or
  // 0.152 with the physics unchanged. Asserting tighter than the window can
  // resolve would be a flaky test, not a stronger one. It still excludes both
  // ways the detector has actually been seen to fail: 2f (0.36, from feeding it
  // the streamwise component) and f/2 (0.09, from a wake that stops alternating).
  expect(r.verdict.st).toBeGreaterThan(0.15);
  expect(r.verdict.st).toBeLessThan(0.21);
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

/** Drives one of the advanced-panel sliders the way a user would. */
function setSlider(page, id, value) {
  return page.evaluate(({ id, value }) => {
    const el = document.getElementById(id);
    el.value = String(value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return parseFloat(el.value);   // what the range input actually snapped to
  }, { id, value });
}

test('the badge follows the dt slider, which both of its bounds depend on', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  // The shipped default at the shipped dt: inside the window, no badge.
  const start = await readBadge(page);
  expect(start.visible).toBe(false);

  // dt sets BOTH bounds — the ceiling through nuNumConverged(dt), the floor
  // through viscNuMax, which divides by dt — so a dt move that leaves the
  // badge untouched is a badge describing the previous timestep.
  const midDt = await setSlider(page, 'slider-dt', 0.0102);
  const mid = await page.evaluate(() => ({
    ...(() => {
      const b = document.getElementById('re-badge');
      return { text: b.textContent, visible: b.classList.contains('visible') };
    })(),
    dt: window.__flowlab.solver.params.dt,
  }));
  expect(mid.dt).toBeCloseTo(midDt, 12);          // the slider reached the solver
  expect(mid.dt).toBeGreaterThan(1 / 240);
  // 2.4x the anchor dt, so the measured projection table no longer applies:
  // the floor rose 4.10 -> 10.0 and the scheme ceiling fell 237 -> 97, and the
  // badge must now say the ceiling here is unmeasured rather than keep quoting
  // 154 from a run at dt = 1/240.
  expect(mid.visible).toBe(true);
  expect(mid.text).toMatch(/unmeasured/i);
  expect(mid.text).toContain('97');

  // Far enough up and the window closes entirely: at dt ~ 0.033 the floor
  // (32.4) passes the scheme ceiling (30.0), which is empty-grid, not
  // empty-iters — no iteration count can lift a ceiling the scheme sets.
  await setSlider(page, 'slider-dt', 0.033);
  const high = await page.evaluate(() => ({
    text: document.getElementById('re-badge').textContent,
    empty: document.getElementById('re-badge').classList.contains('empty'),
  }));
  expect(high.text).toMatch(/no honest Re/i);
  expect(high.text).not.toMatch(/iterations/i);
  expect(high.empty).toBe(true);

  // Back to the bottom of the slider. Two things must hold: the badge clears,
  // and the solver returns to the preset's dt rather than a snapped neighbour.
  // The range input's min IS 1/240 now — with min 0.004 / step 1e-4 the nearest
  // position was 0.0042, so this round trip used to leave the solver running a
  // dt 0.8% away from the preset while the readout claimed otherwise.
  const bottom = await setSlider(page, 'slider-dt', 0.001);   // clamps to min
  const restored = await page.evaluate(() => ({
    dt: window.__flowlab.solver.params.dt,
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
  }));
  expect(bottom).toBeCloseTo(1 / 240, 6);
  expect(Math.abs(restored.dt - 1 / 240) / (1 / 240)).toBeLessThan(1e-4);
  expect(restored.visible).toBe(false);
  expect(restored.text).toBe('');
});

test('the badge stops quoting the projection ceiling when iterations leave its measured point', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  expect((await readBadge(page)).visible).toBe(false);

  // NU_NUM_ITERS256 is one slice: 256 iterations. At any other count the true
  // ceiling has moved and nothing has measured where to — so the badge must
  // stop quoting 154 rather than keep it while the solver runs 200.
  await setSlider(page, 'slider-iters', 200);
  const off = await page.evaluate(() => ({
    numIters: window.__flowlab.ui.numIters,
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
  }));
  expect(off.numIters).toBe(200);
  expect(off.visible).toBe(true);
  expect(off.text).toMatch(/unmeasured/i);
  expect(off.text).toContain('237');    // the scheme ceiling, which still holds
  expect(off.text).not.toContain('154'); // the ceiling it may no longer claim

  // Back at the measured count the claim returns.
  await setSlider(page, 'slider-iters', 256);
  const on = await readBadge(page);
  expect(on.visible).toBe(false);
  expect(on.text).toBe('');
});

test('the badge stops quoting the projection ceiling when the inflow leaves U = 1', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  // The AMPLITUDE axis, and the one the gate covered last. Every Taylor-Green
  // fit behind NU_NUM_PER_DT and NU_NUM_ITERS256 ran at A = 1.0, which is why
  // diagnostics.js says the ceiling is "approximate — and optimistic — for any
  // preset whose U differs from 1.0". Karman ships inVel = 1.0, so the app
  // opens ON the measured slice — but the inflow slider spans 0.5 .. 5.0, and
  // one drag leaves it while the badge used to keep quoting a measured ceiling.
  // The direction is not even known: a plausible nu_num ~ A^2 scaling would
  // make the true ceiling FALL as U rises, i.e. the quoted number would be
  // optimistic exactly where a user reaches by turning the flow up.
  const start = await page.evaluate(() => ({
    U: window.__flowlab.ui._inflowVelocity(),
    visible: document.getElementById('re-badge').classList.contains('visible'),
  }));
  expect(start.U).toBeCloseTo(1.0, 6);
  expect(start.visible).toBe(false);

  // Up: the regime where the unmeasured correction is optimistic.
  const up = await setSlider(page, 'slider-invel', 2.0);
  expect(up).toBeCloseTo(2.0, 6);
  const offUp = await page.evaluate(() => ({
    U: window.__flowlab.ui._inflowVelocity(),
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
  }));
  expect(offUp.U).toBeCloseTo(2.0, 6);
  expect(offUp.visible).toBe(true);
  expect(offUp.text).toMatch(/unmeasured/i);
  // The reason must NAME the axis that actually moved. Before this gate the
  // `unmeasured` text attributed the whole residual unknown to the pressure
  // solve, so the amplitude gap was invisible in the UI even once it opened.
  expect(offUp.text).toMatch(/U = 1/);
  expect(offUp.text).toMatch(/amplitude|inflow/i);
  // The scheme ceiling at U = 2 is U*D/nu_num = 0.24/5.0564e-4 = 475, and it is
  // still honest to quote it — nu_num's h- and dt-scaling is measured, only its
  // amplitude dependence is not. What must be gone is the projection ceiling
  // (0.24/7.7808e-4 = 308), which is quoted nowhere once the gate opens.
  expect(offUp.text).toContain('475');
  expect(offUp.text).not.toContain('308');

  // Down as well as up: the gate is about leaving the measured point, not about
  // exceeding it. A `U > 1` test alone would pass on a one-sided gate.
  await setSlider(page, 'slider-invel', 1.0);
  expect((await readBadge(page)).visible).toBe(false);

  const down = await setSlider(page, 'slider-invel', 0.5);
  expect(down).toBeCloseTo(0.5, 6);
  const offDown = await page.evaluate(() => ({
    visible: document.getElementById('re-badge').classList.contains('visible'),
    text: document.getElementById('re-badge').textContent,
  }));
  expect(offDown.visible).toBe(true);
  expect(offDown.text).toMatch(/unmeasured/i);

  // And back at U = 1.0 the claim returns — the gate must not be a one-way trip
  // that silently disables the measured ceiling for the rest of the session.
  await setSlider(page, 'slider-invel', 1.0);
  const back = await readBadge(page);
  expect(back.visible).toBe(false);
  expect(back.text).toBe('');
});

test('an obstacle-less preset applies zero viscosity, whatever preset was viewed first', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.solver.paused = true; });

  // Load Karman first: it carries an obstacle of radius 0.06, so the interaction
  // handler is left holding a non-zero obstacleRadius. backwardStep has
  // obstacle: null, and loadPreset only reassigns obstacleRadius inside
  // `if (preset.obstacle)` — so switching to it INHERITS Karman's 0.06.
  const karman = await page.evaluate(() => {
    const { ui, interaction, solver } = window.__flowlab;
    ui._loadAndApplyPreset('karmanVortex');
    return {
      showObstacle: interaction.showObstacle,
      radius: interaction.obstacleRadius,
      nu: solver.params.nu,
    };
  });
  expect(karman.showObstacle).toBe(true);
  expect(karman.radius).toBeCloseTo(0.06, 6);
  // Karman DOES force a viscosity — the obstacle is real — so the zero below is
  // a real transition, not a field that was already zero.
  expect(karman.nu).toBeGreaterThan(0);

  const step = await page.evaluate(() => {
    const { ui, interaction, solver } = window.__flowlab;
    ui._loadAndApplyPreset('backwardStep');
    return {
      showObstacle: interaction.showObstacle,
      inheritedRadius: interaction.obstacleRadius,   // still 0.06 — the phantom
      nu: solver.params.nu,
      valRe: document.getElementById('val-re').textContent,
      badgeVisible: document.getElementById('re-badge').classList.contains('visible'),
    };
  });

  // The obstacle is gone from the flow...
  expect(step.showObstacle).toBe(false);
  // ...but its radius is still on the handler. That is exactly the phantom the
  // showObstacle gate must ignore; if this were 0 the bug could not manifest and
  // the assertions below would pass vacuously.
  expect(step.inheritedRadius).toBeCloseTo(0.06, 6);

  // THE PAYLOAD. No body => no Reynolds number => zero forced viscosity, and the
  // Re readout says nothing. Remove the `!showObstacle` clause from
  // _updateReBadge and nu becomes U*D/re = 1.5 * 0.12 / re — a non-zero
  // viscosity inherited from Karman's geometry — and val-re shows that Re
  // instead of '--'.
  expect(step.nu).toBe(0);
  expect(step.valRe).toBe('--');
  expect(step.badgeVisible).toBe(false);
});

// ── The Strouhal probe ──────────────────────────────────────────────────────
//
// Expected values below come from the SIGNAL, not from the detector. The
// synthetic wake is built at a known frequency f = St*U/D with St = 0.2, so
// St must come back out; the noise case carries an amplitude four orders below
// the shedding gate, so it must NOT come back as a frequency at all.

test('Strouhal detector recovers a known frequency and reports steady flow', async ({ page }) => {
  await page.goto('/');
  const r = await page.evaluate(async () => {
    const { StrouhalProbe } = await import('/js/diagnostics.js');
    const D = 0.12, U = 1.0;

    // Synthetic wake: St = 0.2 => f = St*U/D = 1.667 Hz in simulation time.
    // Sampled every 10 steps at dt = 1/240, i.e. every 0.04167 s — the app's
    // own cadence, 14.4 samples per shedding period.
    const f = 0.2 * U / D;
    const dtS = 10 / 240;
    const shedding = new StrouhalProbe();
    for (let k = 0; k < 400; k++) {
      const t = k * dtS;
      shedding.push(0.35 * Math.sin(2 * Math.PI * f * t), t);
    }

    // Below onset: numerical noise only, no coherent oscillation. Amplitude
    // 5e-4 of U, which is the saturated wake fluctuation measured 0.5 below
    // the shedding onset — the regime the gate exists to call 'steady'.
    const steady = new StrouhalProbe();
    let seed = 1;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
    for (let k = 0; k < 400; k++) steady.push(0.001 * rnd(), k * dtS);

    const fresh = new StrouhalProbe();
    fresh.push(0.1, 0);

    // A different known frequency, to prove the detector reports the SIGNAL
    // rather than a constant near 0.2 that the test would not distinguish.
    const half = new StrouhalProbe();
    for (let k = 0; k < 400; k++) {
      const t = k * dtS;
      half.push(0.35 * Math.sin(2 * Math.PI * (0.5 * f) * t), t);
    }

    return {
      shedding: shedding.read({ D, U }),
      steady: steady.read({ D, U }),
      fresh: fresh.read({ D, U }),
      half: half.read({ D, U }),
    };
  });

  expect(r.shedding.state).toBe('shedding');
  expect(r.shedding.st).toBeGreaterThan(0.19);
  expect(r.shedding.st).toBeLessThan(0.21);

  expect(r.steady.state).toBe('steady');   // must NOT report a confident St from noise
  expect(r.steady.st).toBe(null);
  expect(r.fresh.state).toBe('measuring');

  // Halving the frequency must halve St. A detector that returned a fixed
  // number, or one keyed off D/U alone, passes the case above and fails here.
  expect(r.half.state).toBe('shedding');
  expect(r.half.st).toBeGreaterThan(0.095);
  expect(r.half.st).toBeLessThan(0.105);
});

test('the detector refuses an under-sampled series, a dead field, and reversed time', async ({ page }) => {
  await page.goto('/');
  const r = await page.evaluate(async () => {
    const { StrouhalProbe } = await import('/js/diagnostics.js');
    const D = 0.12, U = 1.0;
    const ST_TRUE = 0.18;
    const f = ST_TRUE * U / D;          // 1.5 Hz, period 0.667 s

    // Build the SAME wake at two sample intervals, both reachable from the dt
    // slider: 10*dt at the bottom (the preset, 1/240) and at the top (0.033).
    // A wake carries a harmonic and noise; a pure tone flatters a zero-crossing
    // detector in exactly the regime this guard exists to police.
    let seed = 1;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
    const wake = (dtS, n = 300) => {
      seed = 1;
      const p = new StrouhalProbe();
      for (let k = 0; k < n; k++) {
        const t = k * dtS;
        p.push(0.35 * (Math.sin(2 * Math.PI * f * t)
                     + 0.2 * Math.sin(4 * Math.PI * f * t + 0.6)
                     + 0.08 * rnd()), t);
      }
      return p;
    };

    // Dead field: a readback of all zeros, or of any constant. rms = 0 falls
    // through the shedding gate and used to print a verdict about the wake.
    const dead = new StrouhalProbe();
    for (let k = 0; k < 300; k++) dead.push(0, k * (10 / 240));
    const frozen = new StrouhalProbe();
    for (let k = 0; k < 300; k++) frozen.push(0.4213, k * (10 / 240));

    // Reversed timestamps: same shedding signal, clock running backwards.
    const back = new StrouhalProbe();
    const src = wake(10 / 240);
    for (let k = 0; k < src.v.length; k++) back.push(src.v[k], -src.t[k]);

    return {
      shipped:  { r: wake(10 / 240).read({ D, U }), spp: 1 / (f * (10 / 240)) },
      sliderTop:{ r: wake(10 * 0.033).read({ D, U }), spp: 1 / (f * 10 * 0.033) },
      // 10*dt = 0.025*10: 2.67 samples/period. Also refused — the measured
      // error cliff for a harmonic-rich, still-growing wake is at 3.1.
      near:     { r: wake(10 * 0.025).read({ D, U }), spp: 1 / (f * 10 * 0.025) },
      dead: dead.read({ D, U }),
      frozen: frozen.read({ D, U }),
      back: back.read({ D, U }),
      ST_TRUE,
    };
  });

  // The shipped sample rate resolves this wake 16x over and must report it.
  expect(r.shipped.spp).toBeGreaterThan(15);
  expect(r.shipped.r.state).toBe('shedding');
  expect(r.shipped.r.st).toBeGreaterThan(0.97 * r.ST_TRUE);
  expect(r.shipped.r.st).toBeLessThan(1.03 * r.ST_TRUE);

  // The top of the dt slider samples this same wake barely twice per period.
  // The detector still finds crossings there and, before the guard, returned a
  // confident two-decimal number 12-22% below the truth. It must now decline.
  expect(r.sliderTop.spp).toBeLessThan(2.1);
  expect(r.sliderTop.r.state).toBe('unresolved');
  expect(r.sliderTop.r.st).toBe(null);
  expect(r.near.r.state).toBe('unresolved');
  expect(r.near.r.st).toBe(null);

  // A constant series is not a steady wake — it is no measurement at all.
  // 'steady — no shedding' is a claim about the physics; these carry none.
  expect(r.dead.state).toBe('no-signal');
  expect(r.dead.st).toBe(null);
  expect(r.frozen.state).toBe('no-signal');

  // Reversed timestamps must not render as a negative Strouhal number.
  expect(r.back.st === null || r.back.st > 0).toBe(true);
  expect(r.back.state).not.toBe('shedding');
});

test('probeCell tracks the obstacle and refuses the frozen outflow columns', async ({ page }) => {
  await page.goto('/');
  const r = await page.evaluate(async () => {
    const { probeCell } = await import('/js/diagnostics.js');
    const h = 1 / 256, numX = 600, numY = 256, D = 0.12;
    const at = (x, y = 0.5) => probeCell({ obstacleX: x, obstacleY: y, D, h, numX, numY });
    const lastFluid = (numX - 3) * h;            // deepest column the probe may sit in
    return {
      mid:      at(0.7),
      moved:    at(1.2),
      // 2D upstream of the deepest legal column: exactly on the boundary.
      onEdge:   at(lastFluid - 2 * D),
      // One cell further downstream: i = numX-2, which diffuses against the
      // frozen ring and must be refused.
      pastEdge: at(lastFluid - 2 * D + h),
      atRing:   at((numX - 1) * h - 2 * D),
      offLeft:  at(-1.0),
      noD:      probeCell({ obstacleX: 0.7, obstacleY: 0.5, D: 0, h, numX, numY }),
      lowJ:     at(0.7, 0),                       // j = 0 is BURIED in diffuse.wgsl
      highJ:    at(0.7, (numY - 1) * h),          // j = numY-1 is the frozen ring
    };
  });

  // 2 diameters downstream, on the obstacle's own centreline. The 2 is written
  // out here rather than imported, so PROBE_DOWNSTREAM_DIAMETERS moving breaks
  // this instead of silently redefining what the readout means.
  expect(r.mid).toEqual({ i: Math.round((0.7 + 2 * 0.12) * 256), j: 128, x: 0.7 + 0.24, y: 0.5 });
  expect(r.mid.i).toBe(241);

  // Moving the obstacle 0.5 downstream moves the probe 0.5 downstream: 128 cells.
  expect(r.moved.i - r.mid.i).toBe(128);
  expect(r.moved.j).toBe(r.mid.j);

  // The guard, at the exact cell where it must engage.
  expect(r.onEdge.i).toBe(600 - 3);
  expect(r.pastEdge).toBe(null);
  expect(r.atRing).toBe(null);
  expect(r.offLeft).toBe(null);
  expect(r.noD).toBe(null);
  expect(r.lowJ).toBe(null);
  expect(r.highJ).toBe(null);
});

test('the probe samples the live wake in simulation time and clears on a drag', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui?.probe, null, { timeout: 20_000 });
  // A tier switch resizes the grid and clears the probe, which would make the
  // sample count below a measurement of the adaptive controller.
  await page.evaluate(() => { window.__flowlab.adaptive.manualOverride = true; });

  const before = await page.evaluate(() => window.__flowlab.ui.probe.length);
  await page.waitForTimeout(3000);

  const run = await page.evaluate(() => {
    const { ui, solver, renderer, interaction } = window.__flowlab;
    const p = ui.probe;
    // Timestamp spacing, in units of dt. The readback fires every 10 frames and
    // the loop steps once per frame, so consecutive samples must be 10 steps
    // apart — the signature of simulation time. Wall-clock stamps would show
    // frame-rate jitter instead, and would not be a multiple of dt at all.
    const gaps = [];
    for (let k = 1; k < p.t.length; k++) gaps.push((p.t[k] - p.t[k - 1]) / solver.params.dt);
    return {
      n: p.length,
      simTime: solver.simTime,
      dt: solver.params.dt,
      gapMin: Math.min(...gaps),
      gapMax: Math.max(...gaps),
      // Every gap must be a whole number of 10-step readback slots. Asserting
      // gap == 10 exactly is machine-dependent: `readbackVelocity` self-blocks
      // on `_velReadbackPending`, so a readback that outlives its 10 frames on
      // a slow GPU (or at tier 1024) legitimately produces a gap of 20 and
      // would fail a healthy app. What must hold is that the stamps are
      // simulation time — integer multiples of 10*dt — not wall clock.
      gapsAreWholeSlots: gaps.every((g) => Math.abs(g / 10 - Math.round(g / 10)) < 1e-6),
      monotonic: gaps.every((g) => g > 0),
      allFinite: p.v.every((x) => Number.isFinite(x)),
      // Non-zero field: a lost device or a collapsed solve reads back all
      // zeros, which would satisfy every check above and none of the physics.
      vSpread: Math.max(...p.v) - Math.min(...p.v),
      showProbe: renderer.showProbe,
      probeCellNonNull: !!renderer.probe && interaction.showObstacle,
      lost: false,
    };
  });

  expect(run.showProbe).toBe(true);
  expect(run.n).toBeGreaterThan(before);
  expect(run.n).toBeGreaterThan(5);
  expect(run.allFinite).toBe(true);
  expect(run.monotonic).toBe(true);
  expect(run.simTime).toBeGreaterThan(0);
  // Whole readback slots, and at least one. Wall-clock stamps would be neither.
  expect(run.gapsAreWholeSlots).toBe(true);
  expect(run.gapMin).toBeGreaterThanOrEqual(10 - 1e-6);
  // The wake is not identically zero — the readback carried real flow.
  expect(run.vSpread).toBeGreaterThan(0);

  // A drag changes the geometry, so the series before it describes a different
  // flow. Same clearing path as the particles.
  const afterDrag = await page.evaluate(() => {
    const { ui, interaction, solver } = window.__flowlab;
    interaction.rasterizeObstacle(solver.numX * solver.h * 0.45, interaction.obstacleY, 0, 0);
    return ui.probe.length;
  });
  expect(afterDrag).toBe(0);
});

test('changing Re or inflow clears the probe rather than averaging across two flows', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui?.probe, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.adaptive.manualOverride = true; });

  // These sliders do NOT go through the renderer's invalidation path, which is
  // what clears the probe on a drag / preset / tier change. Before this fix
  // moving Re left a window straddling two Reynolds numbers for up to ~43 s of
  // wall clock and reported the average as a measurement; moving inflow
  // instantly rescaled every stored sample by the NEW U, and since U also sits
  // in the shedding gate (rms < THRESHOLD * U), raising it could flip a
  // shedding wake to 'steady' with no change in the physics.
  const move = async (id, value) => {
    await page.waitForFunction(() => window.__flowlab.ui.probe.length > 3, null, { timeout: 30_000 });
    return page.evaluate(({ id, value }) => {
      const before = window.__flowlab.ui.probe.length;
      const el = document.getElementById(id);
      el.value = String(value);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return { before, after: window.__flowlab.ui.probe.length };
    }, { id, value });
  };

  const re = await move('slider-re', 60);
  expect(re.before).toBeGreaterThan(3);
  expect(re.after).toBe(0);

  const invel = await move('slider-invel', 1.5);
  expect(invel.before).toBeGreaterThan(3);
  expect(invel.after).toBe(0);

  const dt = await move('slider-dt', 0.01);
  expect(dt.before).toBeGreaterThan(3);
  expect(dt.after).toBe(0);
});

test('the probe freezes while paused and advances on single-step', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__flowlab?.ui?.probe, null, { timeout: 20_000 });
  await page.evaluate(() => { window.__flowlab.adaptive.manualOverride = true; });
  await page.waitForFunction(() => window.__flowlab.ui.probe.length > 3, null, { timeout: 30_000 });

  // Paused, the readback keeps firing every 10 frames on a field that is not
  // moving. Those duplicate samples are not 10 steps of flow — they would pad
  // the window with a flat line and drag a shedding wake under the gate.
  const frozen = await page.evaluate(() => {
    window.__flowlab.solver.paused = true;
    return { n: window.__flowlab.ui.probe.length, t: window.__flowlab.solver.simTime };
  });
  await page.waitForTimeout(1500);
  const stillFrozen = await page.evaluate(() => ({
    n: window.__flowlab.ui.probe.length, t: window.__flowlab.solver.simTime,
  }));
  expect(stillFrozen.t).toBe(frozen.t);      // the field really did not move
  expect(stillFrozen.n).toBe(frozen.n);      // and no samples were invented

  // Single-stepping DOES advance the field, 10 steps between readbacks exactly
  // as the running loop does, so it must feed the probe and repaint the
  // readout — not leave both stuck at whatever the last running frame left.
  await page.evaluate(async () => {
    for (let k = 0; k < 60; k++) {
      window.__flowlab.ui._stepOnce();
      await new Promise((r) => setTimeout(r, 8));
    }
  });
  await page.waitForTimeout(500);
  const stepped = await page.evaluate(() => ({
    n: window.__flowlab.ui.probe.length, t: window.__flowlab.solver.simTime,
    paused: window.__flowlab.solver.paused,
  }));
  expect(stepped.paused).toBe(true);
  expect(stepped.t).toBeGreaterThan(frozen.t);
  expect(stepped.n).toBeGreaterThan(stillFrozen.n);
});
