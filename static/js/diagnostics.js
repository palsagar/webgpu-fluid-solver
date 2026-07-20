/**
 * Flow diagnostics: the Reynolds range this solver can honestly represent.
 *
 * Pure functions over numbers — no DOM, no GPU. Everything here is testable
 * without booting the renderer, which is why it does not live in ui.js.
 *
 * ─── Where the constants come from ──────────────────────────────────────────
 *
 * Task 7 measured the scheme's own numerical viscosity by Taylor-Green decay
 * (`tests/solver.spec.js`, "Taylor-Green decay yields a numerical viscosity").
 * The measurement overturned the model the plan and ADR-0008 assumed:
 *
 *   - With the pressure projection converged, `nu_num` is INDEPENDENT of h
 *     (flat to 0.5% across an 8x refinement, at both timesteps measured)
 *     and LINEAR in dt. It is an operator-splitting error in TIME, not grid
 *     diffusion.
 *   - So the analytic ceiling `Re <= (D/(k*h))^2`, which RISES with resolution,
 *     is the wrong model. Refining the grid cannot lower `nu_num`. Only
 *     reducing dt can.
 *
 * Hence ONE measured coefficient and a FLAT ceiling across all tiers. There is
 * no per-tier table for the converged value and no estimated fallback anywhere
 * in this module: an unmeasured number driving a real ceiling is exactly the
 * failure this branch exists to prevent.
 */

/**
 * Numerical viscosity of the MacCormack advection + splitting scheme with the
 * projection converged, PER UNIT dt, in domain-units^2/s^2.
 *
 * Stated as a coefficient rather than a bare constant because `nu_num` is
 * LINEAR in dt, so a flat number is only correct at the one timestep it was
 * measured at. The three shipped presets do not share a timestep — Karman runs
 * dt = 1/240, windTunnel and backwardStep run dt = 1/60 — and a 1/120-derived
 * constant made the ceiling optimistic by ~2x on the latter two.
 *
 * ─── The measurement ────────────────────────────────────────────────────────
 *
 * Taylor-Green decay, amplitude A = 1.0, float32, projection escalated until
 * the fit stops moving (2048 and 4096 iterations agree to 5 significant
 * figures). Anchored at dt = 1/240, tier 256:
 *
 *   nu_num(1/240) = 5.0564e-4   r^2 = 0.99994
 *
 * so NU_NUM_PER_DT = 5.0564e-4 * 240 = 0.121354.
 *
 * h-independence re-confirmed at this dt — tiers 64 / 128 / 256 give
 * 5.0802e-4 / 5.0675e-4 / 5.0564e-4, a 0.5% spread across a 4x refinement.
 *
 * ─── Linearity is close but not exact, and the error is conservative ────────
 *
 * Measured at the same tier, dt = 1/120 gives 9.8230e-4 (r^2 = 0.99979) — a
 * ratio of 1.943 under a halving of dt, not 2.000. So a single linear
 * coefficient cannot fit both timesteps exactly. Anchoring at 1/240 (the
 * flagship preset's dt, and the one where the escalation was carried furthest)
 * makes the model OVERSTATE `nu_num` at larger dt by ~3% at 1/120 — which
 * UNDERSTATES the ceiling. That is the safe direction for a claim about
 * honesty, and it is why this anchor was chosen over the 1/120 one.
 *
 * ─── What is NOT measured ───────────────────────────────────────────────────
 *
 * The AMPLITUDE dependence. Every fit above used A = 1.0, matching Karman's
 * U = 1.0. `windTunnel` runs U = 2.0 and `backwardStep` U = 1.5, and a
 * plausible `nu_num ~ A^2 dt` scaling would move the ceiling by up to 4x on
 * those presets. NOTHING here corrects for that, because nothing measured it.
 * The ceiling is therefore approximate — and optimistic — for any preset whose
 * U differs from 1.0. Measuring `nu_num` across A is the next thing to do.
 */
export const NU_NUM_PER_DT = 0.121354;

/** The timestep NU_NUM_PER_DT was anchored at, and the Karman preset's dt. */
export const NU_NUM_ANCHOR_DT = 1 / 240;

/**
 * Converged numerical viscosity at a given timestep, in domain-units^2/s.
 * @param {number} dt
 * @returns {number}
 */
export function nuNumConverged(dt) {
  return NU_NUM_PER_DT * dt;
}

/** Pressure iteration count at which NU_NUM_ITERS256 was measured. */
export const PROJECTION_ITERS_MEASURED = 256;

/** The timestep NU_NUM_ITERS256 was measured at. See the caveat below. */
export const NU_NUM_ITERS256_DT = 1 / 240;

/**
 * Numerical viscosity at the OPERATING POINT the app actually ships:
 * `numIters = 256`, where the red-black SOR projection is converged at the two
 * coarse tiers and still under-converged at the three finer ones.
 *
 * Keyed by tier (numY), measured at dt = 1/240 by the same Taylor-Green decay
 * fit as the converged constant above:
 *
 *   tier       64       128      256       512      1024
 *   nu_num  5.080e-4 5.078e-4 7.781e-4  2.454e-3  7.270e-3
 *   Re_max    236.2    236.3    154.2      48.9      16.5
 *
 * ─── Why this is no longer monotonic from the first tier ────────────────────
 *
 * At the 80 iterations this preset shipped before Task 9 it rose at every tier,
 * because the projection residual dominated everywhere. At 256 it does not:
 * tiers 64 and 128 both land
 * at 5.08e-4, which IS the converged value (5.0564e-4) to within 0.5% — the
 * projection has stopped binding there and the advection scheme's own splitting
 * error is all that is left. The 0.05% by which 128 sits below 64 is fit
 * scatter between two converged measurements, not a trend. The rise resumes
 * from tier 256 up, where a fixed iteration count again converges progressively
 * less well as the grid grows.
 *
 * `windowState` reads this correctly without special-casing: at the converged
 * tiers `reMaxProjection ~ reMax`, so `CEILING_AGREEMENT_TOL` attributes the
 * ceiling to the scheme rather than blaming the projection for it.
 *
 * ─── Fit quality ───────────────────────────────────────────────────────────
 *
 * r^2 = 0.99994 / 0.99994 / 0.99986 / 0.99860 / 0.98926 across the tiers. The
 * 1024 fit is the weakest of the set: the decay is fast enough there (0.71 in
 * log-KE over the window, against 0.05 at tier 64) that the departure from a
 * pure exponential is visible. The value is reproducible to all six digits
 * across fresh page loads, so the scatter is not noise — but 1024's ceiling
 * carries more model error than the rest, and it is the one tier where the
 * window is empty anyway.
 *
 * ─── Why this table is NOT scaled by dt, unlike the converged constant ──────
 *
 * Because it is not linear in dt, and the departure grows with the tier. At the
 * 80 iterations that preceded this table the ratio under a halving of dt ran
 * 1.94 / 1.93 / 1.80 / 1.61 / 1.54 across the tiers — 2x only where the
 * splitting error still
 * dominates, falling away as the projection residual takes over, which is
 * exactly the part that does not care about dt. A single `per-dt` coefficient
 * would therefore be a fiction here.
 *
 * ─── Where this table is VALID, and what happens outside it ────────────────
 *
 * At dt = 1/240 AND numIters = 256 — which is the Karman preset and only the
 * Karman preset. `windTunnel` runs dt = 1/60 at 40 iterations and
 * `backwardStep` dt = 1/60 at 60 iterations, so on both counts the true nu_num
 * there is HIGHER and the real projection ceiling LOWER than this table.
 *
 * These numbers used to be quoted on those presets anyway, with only a prose
 * caveat. They no longer are. The carry produced a physically impossible pair:
 * on `windTunnel` it gave a projection ceiling of Re 771 against a converged
 * scheme ceiling of Re 297 — an under-converged solve dissipating LESS than a
 * converged one. `windowState` now checks exactly that (`reMaxProjection <=
 * reMax`) and refuses to claim a ceiling it has not measured, and `ui.js` only
 * supplies this table at the operating point it was measured at. Measuring the
 * table at each preset's own operating point is the way to close it; until
 * then the badge says "unmeasured" rather than quoting the wrong grid.
 */
export const NU_NUM_ITERS256 = {
  64:   5.0801e-4,
  128:  5.0777e-4,
  256:  7.7808e-4,
  512:  2.4536e-3,
  1024: 7.2698e-3,
};

/**
 * How close the two ceilings must be before we stop blaming the projection.
 *
 * Task 7 reports a ~10% systematic uncertainty on `nu_num` from the choice of
 * fit window (1.02e-3 / 9.87e-4 / 9.36e-4 at 120 / 300 / 600 steps). Below that
 * separation the two measurements are the same number and the honest attribution
 * is the scheme; above it the projection genuinely dominates.
 *
 * At the shipped 256 iterations the ratio `reMaxProjection / reMax` runs
 *
 *   tier    64      128     256     512    1024
 *   ratio  0.995   0.996   0.650   0.206   0.070
 *
 * so the two coarse tiers are attributed to the scheme and the three fine ones
 * to the projection. The split moved with the iteration count — at 80 only tier
 * 64 was scheme-limited — which is the point: raising iterations converts a
 * projection ceiling into a scheme ceiling, and the badge now says so.
 */
export const CEILING_AGREEMENT_TOL = 0.9;

// ── The Re control's range ──────────────────────────────────────────────────
//
// Log-spaced, because Re spans three decades across the tier set and the
// honest window at any one tier is a narrow slice of it.
//
// All bounds below are at the Karman preset's dt = 1/240. Halving dt from
// 1/120 halved every floor (they scale with dt) and roughly doubled the
// ceiling, so the whole structure moved and these are re-derived, not inherited.
//
//   RE_SLIDER_MIN = 0.25 — the lowest floor in the tier set (tier 64: 0.256),
//                          so the honest low end is reachable, not cropped.
//   RE_SLIDER_MAX = 500  — clears both structural bounds with headroom: the
//                          flat ceiling (237.3) and the highest floor (tier
//                          1024: 65.5). Every badge regime is therefore
//                          reachable at every tier, including tier 1024 where
//                          the window is empty and both ends must badge.
//
// Their geometric mean is 11.2, which lands inside the startup tier's honest
// window (256 at 256 iterations: 4.10 .. 154.23) — so mid-slider is badge-free
// by construction rather than by luck.

export const RE_SLIDER_MIN = 0.25;
export const RE_SLIDER_MAX = 500;
export const RE_SLIDER_STEPS = 100;

/**
 * Where the control ships (mirrored by index.html's `value` attribute) — Re 74.8.
 *
 * INSIDE the honest window, so the app opens un-badged on a visible vortex
 * street. That is what raising `numIters` 80 -> 256 bought: the position did not
 * move, the window grew out past it. At the startup tier the window is now
 * 4.10 .. 154.23, and Re 74.8 sits 1.43x above the measured shedding onset
 * (52.2) and 2.06x below the ceiling.
 *
 * ─── THE ONSET IS 52.2, NOT 57.5. Two superseded measurements ──────────────
 *
 * Both earlier numbers came from a FIXED-WINDOW amplitude criterion, and both
 * were too high for the same reason. Near a Hopf bifurcation the growth rate
 * vanishes, so the amplitude is still moving when any fixed window closes: the
 * window's length, not the flow, sets where the criterion trips.
 *
 *   Re ~126   Task 9. Settled 400 steps (3.33 s ~ 5 shedding periods) and read
 *             the wake's unsteadiness. At dt = 1/240 that gives a SMOOTH ramp
 *             over Re 30..210 with no bifurcation in it at all.
 *   Re 57.5   Its replacement, and the table this block used to carry: run 30 s
 *             from an identical impulsive start and compare the fluctuation at
 *             t = 5 s against t = 30 s, calling onset where the RATIO crosses 1.
 *             The same error, one order of magnitude smaller. That table read
 *
 *               Re      50      53      55      57      60      72
 *               growth  0.001   0.005   0.173   0.638  12.2   158.8
 *
 *             and is SUPERSEDED — it is reproduced here only so the number it
 *             produced can be traced. Do not reason from it.
 *
 * ─── The criterion that replaced them, and why it cannot drift ─────────────
 *
 * Fit the perturbation's exponential growth rate and find where it crosses
 * zero. `sigma` is a property of the flow; unlike an amplitude ratio it does
 * not depend on how long the point ran, so no choice of window can move it.
 *
 * Per point: identical initial conditions (full field reload + resetFlipState),
 * 15 s spin-up so the base flow forms, then ONE deterministic transverse kick
 * (a Gaussian blob 1D behind the cylinder, amplitude 1e-3 U — identical at
 * every Re, so only the RATE differs between points), then 50 s logged at the
 * app's own 10-step cadence. `sigma` is the slope of ln(RMS) against t, fitted
 * only where the signal is above the float32 round-off floor (~3e-7) and below
 * the onset of nonlinear saturation. Tier 256, dt = 1/240, 256 iterations.
 *
 *   Re        44        47        50        52        54        56
 *   sigma  -0.3772   -0.2159   -0.0975   -0.0036   +0.0771   +0.1526
 *   r^2     0.9989    0.9950    0.9733    (see below) 0.9993   0.9982
 *
 * Re 58 and 60 were run too and are NOT in the table: above Re 56 the wake
 * crosses from the round-off floor to nonlinear saturation in under 4 s, which
 * leaves too few windows for a fit worth quoting. They are not needed — the
 * crossing is bracketed by 50/52/54.
 *
 * `sigma` is linear in Re across the whole set (dsigma/dRe ~ 0.042, r^2 > 0.996)
 * — the textbook Hopf shape — and crosses zero at **Re 52.2**.
 *
 * Uncertainty **+-0.3**: the full spread of the crossing over 15 bracketing fit
 * subsets x 7 analysis-window variants (window length, settle time, floor and
 * ceiling of the fit range) — 53 combinations, all landing in 52.02 .. 52.39.
 * That systematic spread dwarfs the statistical error on any single `sigma`
 * (+-0.003 or better), so it is the honest bar.
 *
 * Re 52's r^2 of 0.81 is not a bad fit — it is a nearly FLAT line, which has
 * almost no variance for a fit to explain. Its standard error is +-0.00018.
 * That point is also the whole argument: at sigma = -0.0036 the e-folding time
 * is 278 s, so across any 30 s window Re 52 is indistinguishable from a
 * saturated limit cycle. The superseded criterion could not have found this.
 *
 * The decaying and growing signals are both the SHEDDING mode, not the kick
 * washing downstream: detrended by the fitted exponential, the residual
 * oscillation reads St = 0.163 .. 0.173 at every point, decaying ones included.
 *
 * ─── The onset barely moves when nu_num doubles ────────────────────────────
 *
 * This was previously argued from 56.8 (dt = 1/120) against 57.5 (dt = 1/240),
 * but BOTH of those came from the superseded fixed-window method, so the
 * comparison inherited its bias and could not support the conclusion. Repeated
 * with the growth-rate criterion, same geometry, same kick, same 15 s + 50 s:
 *
 *   dt = 1/120   Re      46        49        52        55        58
 *                sigma  -0.2551   -0.1367   -0.0163   +0.0974   +0.2020
 *
 * crossing zero at Re 52.5 (range 52.41 .. 52.57 over the bracketing subsets),
 * against 52.2 +- 0.3 at dt = 1/240. Doubling the scheme's numerical viscosity
 * moved the onset by 0.6%, and DOWNWARD as dt falls — the direction less
 * dissipation predicts, and far too small to be what sets the threshold.
 *
 * So the conclusion the old comparison reached is right after all, and now
 * rests on evidence of the same kind on both sides: the onset is NOT set by
 * nu_num. It is set by the geometry — the channel blockage (D/H = 0.12) and
 * the staircased cylinder — which is why it sits above the textbook unconfined
 * 47 rather than below it.
 *
 * ─── Why pos 75 specifically ───────────────────────────────────────────────
 *
 * Halving dt made the overlap exist at all: the tier-256 ceiling rose
 * 32.8 -> 59.0, past the onset. But 52.2 .. 59.0 is a 13% band against a 7.9%
 * log step, so exactly ONE position lived in it — pos 71, Re 55.2, a wake 6%
 * above onset whose saturated fluctuation is small and which takes tens of
 * seconds of simulated time to grow. (The superseded onset made this band
 * 2.8% wide and the old text claimed no position landed in it at all; with the
 * onset corrected downward, one does. It was a poor default either way.)
 *
 * Raising numIters 80 -> 256 lifted that ceiling to 154.23, which is what made
 * the band wide enough to choose within rather than merely land in: 14 slider
 * positions are now both honest and above onset. pos 75 is kept because the
 * street there is already unmistakable and because moving higher would spend
 * the new margin for no visual gain. Neighbouring positions are both honest
 * too (pos 74 -> Re 69.2, pos 76 -> Re 80.8), so the default no longer sits on
 * a cliff edge the way it did when the ceiling was 59.
 *
 * The badge is now silent from pos 0 up to pos 84 (Re 148) at the startup tier;
 * pos 85 (Re 160) is the first that trips it.
 */
export const RE_SLIDER_DEFAULT_POS = 75;

/** Slider position (0 .. RE_SLIDER_STEPS) to Reynolds number, log-spaced. */
export function reFromSliderPos(pos) {
  const t = pos / RE_SLIDER_STEPS;
  return RE_SLIDER_MIN * (RE_SLIDER_MAX / RE_SLIDER_MIN) ** t;
}

/** Inverse of reFromSliderPos. */
export function sliderPosFromRe(re) {
  return RE_SLIDER_STEPS * Math.log(re / RE_SLIDER_MIN) / Math.log(RE_SLIDER_MAX / RE_SLIDER_MIN);
}

/** Re values are shown to one decimal below 10 and rounded above it. */
export function fmtRe(re) {
  if (!Number.isFinite(re)) return '--';
  return re < 10 ? re.toFixed(1) : String(Math.round(re));
}

/**
 * Bounds of the Reynolds range this grid can honestly represent.
 *
 * Floor: the explicit viscous pass needs `nu*dt_sub/h^2 <= 1/4` per substep, so
 * with `nMax` substeps the largest representable viscosity is
 * `nMax*0.25*h^2/dt` — the solver's own `viscNuMax`. Requests above it are
 * SATURATED (Task 8), so the flow runs at a higher effective Re than asked for.
 *
 * Ceiling: above the Re where the physical viscosity falls below the scheme's
 * own numerical viscosity, the label would be a lie. `nuNum` is measured, never
 * estimated — pass `nuNumConverged(dt)`, which scales the measured coefficient
 * by the caller's own timestep rather than assuming the anchor's.
 *
 * `nuMax` is optional and defaults to that same formula. Pass the solver's own
 * `viscNuMax` getter when one is at hand: the two expressions are identical
 * today (`fluid-solver.js`), and a duplicated formula is a formula that can
 * drift. Keeping the default means this module stays pure and testable with no
 * solver instance, while the app runs off the number the solver actually
 * saturates at rather than a copy of it.
 *
 * @param {{h:number, dt:number, D:number, U:number, nMax:number, nuNum:number,
 *          nuMax?:number}} args
 * @returns {{reMin:number, reMax:number, nuMax:number}}
 */
export function honestWindow({ h, dt, D, U, nMax, nuNum, nuMax = nMax * 0.25 * h * h / dt }) {
  return {
    reMin: (U * D) / nuMax,
    reMax: (U * D) / nuNum,
    nuMax,
  };
}

/**
 * Whether the requested Re sits inside the window, and why not if it does not.
 *
 * Two ceilings go in and ONE comes out. `reMax` is what the advection scheme
 * alone would allow with the projection converged; `reMaxProjection` is what
 * the shipped iteration count actually delivers. The binding ceiling is the
 * lower of the two, and `code` names which mechanism set it — so the badge
 * carries a single number with an attribution, not two contradictory ones.
 *
 * `reMaxProjection` must satisfy `reMaxProjection <= reMax`: an under-converged
 * projection can only ADD dissipation, so its ceiling can never sit ABOVE the
 * converged one. A pair that violates the invariant is not a measurement of
 * this operating point — it is a measurement of a different one, carried here.
 * Rather than take `min()` and quote a plausible-looking number, this reports
 * `'unmeasured'`: the ceiling here is known only to be at or below `reMax`.
 * That is also what an omitted `reMaxProjection` means, hence the `Infinity`
 * default — no number is not a number to reason from.
 *
 * Codes:
 *   null            inside the window
 *   'clamped'       below the floor — nu saturated, effective Re is HIGHER
 *   'projection'    above the ceiling, and the under-converged pressure solve
 *                   is what set that ceiling (fixable: raise iterations)
 *   'scheme'        above the ceiling set by the scheme's own splitting error
 *                   (not fixable by iterations or by refining the grid)
 *   'unmeasured'    the projection ceiling was not measured at this operating
 *                   point, so no ceiling below `reMax` is claimed
 *   'empty-grid'    floor > scheme ceiling: NO honest Re exists at this grid,
 *                   at any iteration count. Only a coarser grid or a smaller
 *                   dt opens it.
 *   'empty-iters'   floor > projection ceiling, but the grid alone would leave
 *                   a window. The iteration count is what emptied it — but see
 *                   the reason text: at the one tier that reaches this state
 *                   the gap is far too wide for the iterations control to close.
 *
 * @param {{re:number, reEff?:number, reMin:number, reMax:number,
 *          reMaxProjection?:number, viscClamped?:boolean}} args
 * @returns {{ok:boolean, code:string|null, reason:string|null}}
 */
export function windowState({
  re, reEff = re, reMin, reMax, reMaxProjection = Infinity, viscClamped = false,
}) {
  const projectionKnown = reMaxProjection <= reMax;
  const ceiling = projectionKnown ? Math.min(reMax, reMaxProjection) : reMax;

  // Empty window first: no slider position is honest, so reporting anything
  // the user could "fix" by moving the control would be a lie.
  if (reMin > ceiling) {
    if (reMin > reMax) {
      return {
        ok: false,
        code: 'empty-grid',
        reason: `No honest Re at this grid — the viscous floor (Re ${fmtRe(reMin)}) sits above the `
              + `ceiling (Re ${fmtRe(reMax)}). Lower the resolution, or reduce dt.`,
      };
    }
    // The advice is deliberately NOT "raise iterations", even though the
    // iteration count is what emptied this window. Tier 1024 is the only
    // shipped tier that reaches this state, and there the gap is 65.5 vs 16.5 —
    // a factor of 4 in nu_num. The 80 -> 256 escalation was 3.2x in iterations
    // and bought 2.1x (1.5446e-2 -> 7.2698e-3), and the iterations control
    // stops at 320, a further 1.25x. So no setting the control offers closes
    // it, and telling the user to raise iterations would be advice the app can
    // neither reflect on screen (the table is pinned at 256) nor honour.
    // Resolution and dt both move the floor directly, and both are available.
    return {
      ok: false,
      code: 'empty-iters',
      reason: `No honest Re at this grid with ${PROJECTION_ITERS_MEASURED} pressure iterations — the `
            + `viscous floor (Re ${fmtRe(reMin)}) sits above the under-converged ceiling `
            + `(Re ${fmtRe(reMaxProjection)}), by more than the iterations control can close. `
            + `Lower the resolution, or reduce dt.`,
    };
  }

  // `viscClamped` is the solver's ground truth from the last step; `re < reMin`
  // is the prediction, and covers the case where it has not stepped yet.
  if (viscClamped || re < reMin) {
    return {
      ok: false,
      code: 'clamped',
      reason: `Below Re ${fmtRe(reMin)} the viscous substep budget saturates, so the flow runs at a `
            + `higher effective Re (${fmtRe(reEff)}) than the one shown.`,
    };
  }

  if (re > ceiling) {
    if (reMaxProjection < CEILING_AGREEMENT_TOL * reMax) {
      return {
        ok: false,
        code: 'projection',
        reason: `Pressure solve under-converged at this grid — the projection residual, not the `
              + `advection scheme, sets the ceiling here: Re ${fmtRe(reMaxProjection)} instead of `
              + `${fmtRe(reMax)} (measured at ${PROJECTION_ITERS_MEASURED} iterations). `
              + `Raise iterations, or lower the resolution.`,
      };
    }
    return {
      ok: false,
      code: 'scheme',
      reason: `Under-resolved above Re ${fmtRe(reMax)} — the scheme's own numerical viscosity `
            + `exceeds the physical one here, so the true Re stays near ${fmtRe(reMax)}.`,
    };
  }

  // Below the scheme ceiling, but with no projection ceiling measured at this
  // operating point there is nothing to certify the request against. Say that,
  // rather than pass it as honest or invent a bound for it.
  if (!projectionKnown) {
    const anchor = Math.round(1 / NU_NUM_ITERS256_DT);
    return {
      ok: false,
      code: 'unmeasured',
      reason: `Ceiling unmeasured at this configuration — the projection ceiling is only measured `
            + `at ${PROJECTION_ITERS_MEASURED} pressure iterations and dt = 1/${anchor}, and does `
            + `not carry to this timestep and iteration count. The advection scheme alone allows `
            + `Re ${fmtRe(reMax)}; an under-converged pressure solve can only lower that, by an `
            + `amount nothing here has measured.`,
    };
  }

  return { ok: true, code: null, reason: null };
}

// ── The downstream probe and the Strouhal detector ──────────────────────────
//
// Everything below stays pure: numbers in, numbers out. `probeCell` decides
// WHERE to sample without touching a buffer, so the placement guard is testable
// without a GPU; `StrouhalProbe` turns a timestamped series into a frequency
// without knowing where the series came from.

/**
 * How far downstream the probe sits, in obstacle diameters.
 *
 * 2D is inside the formation region's downstream edge for a circular cylinder,
 * where the shed vortices have rolled up and the transverse velocity signal is
 * strongest. Further out the signal survives but the wake has spread and the
 * amplitude falls; closer in the near-wake recirculation contaminates it.
 */
export const PROBE_DOWNSTREAM_DIAMETERS = 2;

/**
 * Grid cell the probe samples, or `null` when no valid cell exists.
 *
 * ─── Why the last two columns are excluded ──────────────────────────────────
 *
 * `diffuse.wgsl` copies the domain ring through unchanged (`i >= numX-1`,
 * `j >= numY-1`) so the two ping-ponged velocity slots stay in agreement across
 * substeps. That line therefore carries no viscous update at all, and the line
 * INSIDE it diffuses against a neighbour that never moves — a one-cell layer of
 * frozen boundary storage plus a one-cell layer contaminated by it. A probe
 * sitting in either would read a boundary artifact and report its cadence as a
 * shedding frequency. Both ends of both axes are excluded for the same reason;
 * the low ends (`i = 0`, `j = 0`) are BURIED by index in the same shader.
 *
 * This is not hypothetical at the shipped geometry: the obstacle is draggable,
 * and dragging it to within 2D of the outflow walks the probe straight into
 * that layer. Returning `null` there — rather than clamping the probe back
 * inside — is deliberate: a clamped probe silently stops being "2 diameters
 * downstream", and the St it produced would no longer mean what the readout
 * says it means. No cell, no sample, no number.
 *
 * @param {{obstacleX:number, obstacleY:number, D:number, h:number,
 *          numX:number, numY:number}} args
 * @returns {{i:number, j:number, x:number, y:number}|null}
 */
export function probeCell({ obstacleX, obstacleY, D, h, numX, numY }) {
  if (!(D > 0) || !(h > 0) || !Number.isFinite(obstacleX) || !Number.isFinite(obstacleY)) {
    return null;
  }
  const x = obstacleX + PROBE_DOWNSTREAM_DIAMETERS * D;
  const i = Math.round(x / h);
  const j = Math.round(obstacleY / h);
  // Upper bounds stop two cells short: numX-1 is the frozen ring, numX-2
  // diffuses against it. Same on the j axis.
  if (!(i >= 1 && i <= numX - 3)) return null;
  if (!(j >= 1 && j <= numY - 3)) return null;
  return { i, j, x, y: obstacleY };
}

/**
 * Samples retained. At 10 solver steps per sample and the Karman preset's
 * dt = 1/240 that is 0.04167 s of simulation time each, so 256 spans 10.7 s —
 * about 18 shedding periods at St = 0.2, U = 1, D = 0.12.
 *
 * The window is stated in PERIODS because that is what sets the frequency
 * resolution; the sample RATE (14.4 per period) is far above what is needed and
 * is fixed by the renderer's 10-frame readback throttle, not chosen here.
 *
 * The cost is latency, and it is real: the loop takes one step per displayed
 * frame, so at 60 fps and dt = 1/240 simulation time runs at a quarter of wall
 * time. Half a window — the minimum `read()` accepts — is ~21 s of wall clock,
 * a full one ~43 s. The readout says `measuring...` for that whole time rather
 * than fitting a frequency to three periods and calling it a measurement.
 */
const PROBE_CAPACITY = 256;

/**
 * Below this RMS transverse velocity, relative to U, the wake is steady and any
 * frequency fitted to it is a property of the noise, not of the flow.
 *
 * ─── There is NO amplitude gap, and the old note claiming one was wrong ─────
 *
 * This constant used to be justified by "nothing lands between 1.6e-5 and
 * 8.1e-2 once settled, and 0.02 is inside that gap". That is false as physics.
 * A Hopf bifurcation saturates at A_sat ~ sqrt(Re - Re_c), which is a CONTINUUM
 * through every amplitude as Re approaches onset from above. The apparent gap
 * was an artifact of the Re values that happened to be sampled, and the "once
 * settled" numbers in it were not settled: they came from 30 s windows on
 * wakes still growing by up to 271x across the window that measured them.
 *
 * ─── What the saturated amplitude actually does ────────────────────────────
 *
 * Measured at this probe (2D downstream, tier 256, dt = 1/240, 256 iterations),
 * identical initial conditions per point, 115 s of simulation time each. Every
 * value below is SATURATION-VERIFIED — the RMS over the final two 20 s windows
 * agrees to the drift shown, rather than being read off a single window and
 * assumed to have settled:
 *
 *   Re         55      57.5      60       65      74.8     100      140
 *   A_sat   8.19e-2  1.159e-1 1.437e-1 1.910e-1 2.636e-1 4.024e-1 5.460e-1
 *   drift     0.83%    0.02%    0.24%    0.02%    0.01%    0.26%    0.21%
 *
 * A^2 is linear in Re, as the Hopf form requires: fitted over the three points
 * nearest onset it gives A^2 = 2.788e-3 (Re - 52.62), r^2 = 0.9996 — an
 * INDEPENDENT estimate of the onset, agreeing with the growth-rate crossing
 * (52.2 +- 0.3) to within 1%. The fit drifts up to 53.3 as points further from
 * onset enter, which is the expected direction for an asymptotic law, so the
 * growth-rate crossing remains the quoted number and this is the cross-check.
 *
 * ─── What the gate therefore costs, in Re ──────────────────────────────────
 *
 * Almost nothing, and that is the real justification. Because the square-root
 * rise is so steep just above onset, a threshold in AMPLITUDE is very nearly a
 * threshold in Re: A = 0.02 sits at Re 52.8 by the fit above, about 0.3% above
 * the onset of 52.2. So the gate does not silence a meaningful band of shedding
 * flow — it silences a sliver next to onset, and everything below it.
 *
 * ─── It still costs TIME near onset, which is unavoidable ──────────────────
 *
 * The amplitude is small near onset AND slow to get there: the growth rate
 * vanishes at the bifurcation, so at Re 55 the wake needs ~80 s of simulation
 * time (~5 minutes of wall clock at this preset) to reach its 8.19e-2 plateau.
 * Until it does, the readout says `steady`. That is the right direction to err:
 * the failure this gate exists to prevent is a confident St for a wake that is
 * not shedding, not a late verdict for one that is. Near onset the readout
 * therefore says `steady` first and switches to a number once the wake has
 * actually grown — a measurement in progress, not a wrong answer.
 */
const SHEDDING_RMS_THRESHOLD = 0.02;

/** Zero-crossing hysteresis, as a fraction of signal RMS. */
const HYSTERESIS = 0.25;

/** Crossings needed before a frequency is claimed — i.e. at least two periods. */
const MIN_CROSSINGS = 3;

/**
 * Samples per detected period below which the frequency is NOT resolved by the
 * sample interval, and no St is reported.
 *
 * The sample interval is `10 * dt` and `dt` is a user-facing slider, so the app
 * ships a reachable path to a badly under-sampled series. Nothing used to check
 * that the interval resolved the frequency the detector had just claimed: at
 * the top of the dt slider the readout kept printing a confident two-decimal
 * number that was up to 22% low.
 *
 * ─── Where the cliff actually is ────────────────────────────────────────────
 *
 * Measured by feeding this detector synthetic wakes at a KNOWN St (0.16, 0.18,
 * 0.20), five phases and three noise seeds each, swept over samples-per-period
 * in steps of 0.05. Worst relative error over the whole set, by signal shape:
 *
 *   signal shape                              stays <= 3% for spp >=
 *   pure tone                                        2.25
 *   + 20% second harmonic                            2.50
 *   + 20% harmonic, 8% broadband noise               2.50
 *   + 30% harmonic, 10% noise                        2.70
 *   + 20% harmonic, 8% noise, still growing          2.95
 *   + 35% harmonic, 15% noise, still growing         3.10
 *
 * Below each boundary the error is not a graceful degradation — it jumps to
 * 10-20%, because an aliased sample lands on the wrong side of the hysteresis
 * band and a whole crossing is lost or invented.
 *
 * A pure tone flatters a zero-crossing detector and this probe never sees one:
 * a real wake carries a harmonic, carries noise, and near onset is still
 * growing. So the binding boundary is 3.1, not the textbook 2. 4 is a 1.3x
 * margin on it, and leaves the shipped configuration (16.0 samples per period
 * at dt = 1/240) a factor of 4 clear.
 *
 * What it costs: at a St ~ 0.18 wake the guard binds above dt ~ 0.0167, i.e.
 * the top ~half of the dt slider now reports `under-sampled` instead of a
 * number. That is the intended trade — the alternative is the confident wrong
 * number the guard exists to remove.
 */
const MIN_SAMPLES_PER_PERIOD = 4;

/**
 * Detects vortex-shedding frequency from a transverse-velocity time series and
 * reports it as a Strouhal number, St = f*D/U.
 *
 * ─── Simulation time, never wall time ───────────────────────────────────────
 *
 * Samples MUST be timestamped with `solver.simTime` (accumulated steps * dt).
 * The app deliberately varies its frame rate — `adaptive` switches grid tiers
 * under load, and the loop takes exactly one step per displayed frame — so a
 * wall-clock series would report the frame rate rather than the physics, and
 * would jump by a factor of several the moment a tier switch landed.
 *
 * ─── Why the transverse component ───────────────────────────────────────────
 *
 * On the wake centreline the streamwise velocity dips once per shed vortex
 * REGARDLESS of which side it came from, so `u` oscillates at 2f and a detector
 * fed `u` would report twice the true Strouhal number. `v` is antisymmetric
 * about the centreline and alternates with the shedding side, so it carries f.
 *
 * ─── What the number is, and is not ─────────────────────────────────────────
 *
 * It is the Strouhal number of THIS geometry: a staircased cylinder spanning
 * D/H = 0.12 of a channel, whose walls are free-slip while nu = 0 and no-slip
 * the moment the Re control puts a real viscosity in (see `diffuse.wgsl`: the
 * viscous stencil reads the j = 0 and j = numY-1 lines as ghost cells, placing
 * a zero-velocity wall line half a cell outside the domain). Blockage raises St
 * above the unconfined value, and growing wall boundary layers raise the
 * effective blockage further as Re falls. The same confinement is why shedding
 * here starts above the textbook unconfined Re 47.
 *
 * So this is not an unconfined-cylinder St and should not be compared to one
 * without that caveat. It is a measurement of the flow the app is actually
 * solving, which is the only thing it can honestly claim.
 *
 * ─── Measured at this probe, ON SATURATED WAKES ─────────────────────────────
 *
 * The previous version of this table was quoted to three significant figures
 * but measured on signals still growing by 4.7x to 271x across the very window
 * that measured them. It has been re-measured: 115 s of simulation time per
 * point from identical initial conditions, with saturation VERIFIED (the RMS
 * over the final two 20 s windows agreeing to better than 0.3%, except at
 * Re 55 — see below) before any frequency is quoted.
 *
 * `+-` is the scatter of St across FOUR disjoint 256-sample windows at the end
 * of each run — the app's own window length, so it is the spread a user would
 * actually see between successive readouts, not a fit residual:
 *
 *   Re      55*     57.5     60       65      74.8     100      140
 *   St     0.166   0.168   0.170    0.173    0.180    0.190    0.200
 *   +-     0.001   0.001   0.0000   0.001    0.001    0.0000   0.001
 *
 * Three decimals are earned: the window-to-window scatter is <= 0.0008 (0.5%)
 * at every point. A fourth would not be.
 *
 * * Re 55 is the ONE point that did not fully saturate in 115 s. Its final two
 *   20 s windows agree to 0.83%, but the window before them sits 16% lower, so
 *   the amplitude was still creeping. Near onset the growth rate vanishes and
 *   saturation takes proportionally longer; running it out was impractical at
 *   ~400 s of wall clock per point. Its St is quoted at the same precision
 *   because the FREQUENCY settles well before the amplitude does — it moved by
 *   0.0007 across the final four windows while the amplitude was still moving.
 *   Treat it as good to +-0.001 with that caveat, not as a saturated value.
 *
 * Against the previous table the changes are small but real, and they are
 * concentrated exactly where the old measurement was most transient: Re 60
 * moves 0.168 -> 0.170 (it was growing 271x across its old window) and Re 65
 * moves 0.174 -> 0.173. The saturated points confirm the SHAPE the old table
 * reported.
 *
 * St RISES with Re across this range and reaches 0.200 at Re 140. That is the
 * expected shape, not a defect: the familiar "St ~ 0.2" is the high-Re plateau,
 * and in the Re 50..150 band the unconfined correlation St = 0.212(1 - 21.2/Re)
 * (Roshko) gives 0.134 .. 0.180. These values run 10-25% ABOVE that curve, in
 * the direction blockage predicts, and converge toward it as Re grows and the
 * wall layers thin. A detector tuned to return 0.2 everywhere would have hidden
 * exactly this structure.
 */
export class StrouhalProbe {
  constructor() {
    this.v = [];
    this.t = [];
  }

  /**
   * Appends one transverse-velocity sample at simulation time `simTime`.
   *
   * Non-finite samples are DROPPED rather than stored. A lost device or a
   * collapsed field surfaces as NaN in the readback, and a single NaN in the
   * series would poison the mean, the RMS, and every comparison in `read()` —
   * turning a dead simulation into a permanent `measuring...` instead of an
   * obviously stalled readout.
   */
  push(vSample, simTime) {
    if (!Number.isFinite(vSample) || !Number.isFinite(simTime)) return;
    this.v.push(vSample);
    this.t.push(simTime);
    if (this.v.length > PROBE_CAPACITY) { this.v.shift(); this.t.shift(); }
  }

  clear() { this.v.length = 0; this.t.length = 0; }

  /** Samples currently held. */
  get length() { return this.v.length; }

  /**
   * @param {{D: number, U: number}} geom - current diameter and free-stream
   *   speed, read live at call time so dragging the obstacle (which changes
   *   nothing here but D is read from it) or moving the inflow slider cannot
   *   leave St scaled by a geometry the flow no longer has.
   * @returns {{state: 'measuring'|'no-signal'|'steady'|'unresolved'|'shedding',
   *            st: number|null}}
   */
  read({ D, U }) {
    const n = this.v.length;
    if (n < PROBE_CAPACITY / 2 || !(U > 0) || !(D > 0)) {
      return { state: 'measuring', st: null };
    }

    // A dead field, not a steady one. An all-zeros readback — a lost device, a
    // collapsed solve, or a probe reading a cell the solver never writes —
    // gives rms = 0, which falls straight through the gate below and prints
    // `steady — no shedding`: a verdict about the physics of a wake, delivered
    // from a buffer that carries no physics at all. A perfectly constant series
    // is not evidence of a steady flow; it is evidence of no measurement.
    let vMin = this.v[0], vMax = this.v[0];
    for (let k = 1; k < n; k++) {
      if (this.v[k] < vMin) vMin = this.v[k];
      if (this.v[k] > vMax) vMax = this.v[k];
    }
    if (!(vMax > vMin)) return { state: 'no-signal', st: null };

    const mean = this.v.reduce((a, b) => a + b, 0) / n;
    const dev = this.v.map((x) => x - mean);
    const rms = Math.sqrt(dev.reduce((a, b) => a + b * b, 0) / n);

    // The gate. Without it a steady wake's numerical noise crosses the
    // hysteresis band tens of times per window and the detector returns a
    // confident-looking St from it — the prescribed-not-measured failure this
    // whole readout exists to eliminate.
    if (rms < SHEDDING_RMS_THRESHOLD * U) return { state: 'steady', st: null };

    // Count upward crossings with hysteresis, timing first to last so the
    // estimate averages over every period in the window rather than resolving
    // one. Single-period timing would be quantised by the sample interval
    // (7% of a period); across ~18 periods the same quantisation is 0.4%.
    const hi = HYSTERESIS * rms;
    let armed = false, crossings = 0, tFirst = null, tLast = null;
    for (let k = 0; k < n; k++) {
      if (dev[k] < -hi) armed = true;
      else if (armed && dev[k] > hi) {
        armed = false;
        crossings++;
        if (tFirst === null) tFirst = this.t[k];
        tLast = this.t[k];
      }
    }
    // `tLast > tFirst`, not `!==`: the strict form also rejects a series whose
    // timestamps run backwards, which `!==` accepts and turns into a NEGATIVE
    // frequency — the readout would render a tidy `-0.17`. It also rejects NaN,
    // which `!==` admits.
    //
    // This is defense in depth, not the only line: a reversed series also has a
    // negative mean sample interval, so the resolution check below rejects it
    // anyway. Mutating this line back to `===` does NOT fail the suite for that
    // reason. It stays because it is the correct predicate for what it guards,
    // and because it must not depend on a check further down that a later edit
    // could reorder or remove.
    if (crossings < MIN_CROSSINGS || !(tLast > tFirst)) return { state: 'measuring', st: null };

    const f = (crossings - 1) / (tLast - tFirst);

    // Does the sample interval actually resolve the period just claimed? The
    // mean interval over the whole series, not the nominal 10*dt, so a readback
    // that outlived its 10-frame slot widens the interval here rather than
    // being assumed away. See MIN_SAMPLES_PER_PERIOD.
    const interval = (this.t[n - 1] - this.t[0]) / (n - 1);
    if (!(interval > 0) || 1 / (f * interval) < MIN_SAMPLES_PER_PERIOD) {
      return { state: 'unresolved', st: null };
    }

    return { state: 'shedding', st: (f * D) / U };
  }
}
