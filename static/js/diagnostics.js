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
 * 4.10 .. 154.23, and Re 74.8 sits 1.30x above the measured shedding onset
 * (57.5) and 2.06x below the ceiling.
 *
 * The reasoning below is entirely re-measured, because Task 9's onset number
 * turned out to be an artifact.
 *
 * ─── Task 9's Re ~126 onset was measuring a transient ───────────────────────
 *
 * It settled 400 steps (3.33 s) and read the wake's unsteadiness. Repeating
 * that at dt = 1/240 gives a SMOOTH ramp over Re 30..210 with no bifurcation
 * anywhere in it — because 3.33 s is only ~5 shedding periods (St ~ 0.2,
 * D = 0.12, U = 1 -> period ~0.6 s) and the growth rate vanishes near onset, so
 * the amplitude is still climbing when the window opens. A threshold laid
 * across that ramp reports the settle time, not the physics.
 *
 * ─── What replaced it ──────────────────────────────────────────────────────
 *
 * A bifurcation test: run 30 s (~50 shedding periods) from an identical
 * impulsive start, and compare the wake fluctuation at t = 5 s against t = 30 s.
 * Below onset a perturbation decays; above it grows to a limit cycle. The
 * separation is 4-5 orders of magnitude, so the threshold is unambiguous:
 *
 *   dt = 1/240   Re      50      53      55      57      60      72
 *                growth  0.001   0.005   0.173   0.638  12.2   158.8
 *                sat.    7e-7    5e-6    2.5e-4  1.0e-3  2.8e-2  8.8e-1
 *
 * Onset (growth = 1) is Re 57.5. The same sweep at dt = 1/120 gives Re 56.8.
 *
 * ─── The onset did NOT move when nu_num halved ─────────────────────────────
 *
 * 56.8 -> 57.5 is a 1.1% shift, and UPWARD. Halving the scheme's numerical
 * viscosity changed the shedding threshold by nothing. So the onset is NOT set
 * by nu_num — it is set by the geometry: the channel blockage (D/H = 0.12
 * raises the critical Re above the textbook unconfined 47) and the staircased
 * cylinder. Any story in which the onset is "inflated by numerical viscosity"
 * is wrong, and Task 9's 126 was method, not physics.
 *
 * ─── Why pos 75 specifically ───────────────────────────────────────────────
 *
 * Halving dt first made the overlap exist at all: the tier-256 ceiling rose
 * 32.8 -> 59.0 while the onset stayed at 57.5. But 57.5 .. 59.0 is a 2.8% band
 * and the slider's log step is 7.9%, so no position landed in it, and widening
 * the slider would not have helped — the limit-cycle amplitude 2% above onset
 * is ~1e-3 of the mean flow and takes tens of seconds of simulated time to
 * appear. It is a vortex street only in the sense that a thermometer reads a
 * fever at 37.1 C.
 *
 * Raising numIters 80 -> 256 lifted that ceiling to 154.23, which is what made
 * the band wide enough to choose within rather than merely land in. pos 75 is
 * kept because the street there is already unmistakable — the saturated wake
 * fluctuation is ~0.9 of the mean flow — and because moving higher would spend
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
