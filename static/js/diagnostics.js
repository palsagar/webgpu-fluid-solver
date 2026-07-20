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

/** Pressure iteration count at which NU_NUM_ITERS80 was measured. */
export const PROJECTION_ITERS_MEASURED = 80;

/** The timestep NU_NUM_ITERS80 was measured at. See the caveat below. */
export const NU_NUM_ITERS80_DT = 1 / 240;

/**
 * Numerical viscosity at the OPERATING POINT the app actually ships:
 * `numIters = 80`, where the red-black SOR projection is badly under-converged
 * and its residual, not the advection scheme, dominates the error.
 *
 * This is why a flat 237 is not the whole truth — at 80 iterations it
 * understates the error by up to 31x at tier 1024. Keyed by tier (numY),
 * measured at dt = 1/240:
 *
 *   tier      64      128      256      512     1024
 *   nu_num  5.08e-4  6.50e-4  2.03e-3  6.27e-3  1.54e-2
 *   Re_max   236.1    184.5     59.1     19.2      7.8
 *
 * Unlike the converged constant this one DOES rise with resolution, because a
 * fixed iteration count converges progressively less well as the grid grows.
 *
 * ─── Why this table is NOT scaled by dt, unlike the converged constant ──────
 *
 * Because it is not linear in dt, and the departure grows with the tier. The
 * same table at dt = 1/120 measured 9.8612e-4 / 1.2551e-3 / 3.6638e-3 /
 * 1.0062e-2 / 2.3718e-2, so the ratio under a halving of dt runs
 *
 *   tier    64     128     256     512    1024
 *   ratio  1.94    1.93    1.80    1.61    1.54
 *
 * — 2x only where the splitting error still dominates, falling away as the
 * projection residual takes over, which is exactly the part that does not care
 * about dt. A single `per-dt` coefficient would therefore be a fiction here.
 *
 * CONSEQUENCE, stated plainly: this table is only valid at dt = 1/240. The
 * `windTunnel` and `backwardStep` presets run dt = 1/60 and are given the same
 * numbers, where the true values are HIGHER and so the real projection ceiling
 * is LOWER than the badge claims. On those two presets the projection ceiling
 * is optimistic by an unmeasured factor. The converged ceiling above is scaled
 * correctly for them; this one is not.
 */
export const NU_NUM_ITERS80 = {
  64:   5.0821e-4,
  128:  6.5042e-4,
  256:  2.0324e-3,
  512:  6.2658e-3,
  1024: 1.5446e-2,
};

/**
 * How close the two ceilings must be before we stop blaming the projection.
 *
 * Task 7 reports a ~10% systematic uncertainty on `nu_num` from the choice of
 * fit window (1.02e-3 / 9.87e-4 / 9.36e-4 at 120 / 300 / 600 steps). Below that
 * separation the two measurements are the same number and the honest attribution
 * is the scheme; above it the projection genuinely dominates. At tier 64 the
 * ratio is 0.995 (scheme); at tier 128 it is 0.78 (projection).
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
// window (256 at 80 iterations: 4.10 .. 59.05) — so mid-slider is badge-free by
// construction rather than by luck.

export const RE_SLIDER_MIN = 0.25;
export const RE_SLIDER_MAX = 500;
export const RE_SLIDER_STEPS = 100;

/**
 * Where the control ships (mirrored by index.html's `value` attribute) — Re 75.
 *
 * Still OUTSIDE the honest window, so the app still opens badged — but at 1.27x
 * the ceiling instead of Task 9's 7.7x. The reasoning is entirely re-measured,
 * because Task 9's onset number turned out to be an artifact.
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
 * ─── Why the default is still outside the window ───────────────────────────
 *
 * The window DOES now contain the onset — the tier-256 ceiling rose 32.8 -> 59.0
 * while the onset stayed at 57.5, so Re 57.5 .. 59.0 is both honest AND
 * shedding, where at dt = 1/120 no such Re existed. But that overlap is 2.8%
 * wide and the slider's log step is 7.9%, so NO slider position lands in it:
 * pos 71 is Re 55.2 (honest, below onset, no street) and pos 72 is Re 59.5
 * (sheds, just over the ceiling). Widening the slider to reach it would not
 * help either — the limit-cycle amplitude 2% above onset is ~1e-3 of the mean
 * flow and takes tens of seconds of simulated time to appear. It is a vortex
 * street only in the sense that a thermometer reads a fever at 37.1 C.
 *
 * So the choice remains Task 9's, on much better numbers: pos 75 -> Re 74.8,
 * where the saturated fluctuation is ~0.9 of the mean flow — an unmistakable
 * street — and the badge states the honest ceiling is 59. What the dt change
 * bought is that the gap between "advertised" and "honest" fell from a factor
 * of 7.7 to a factor of 1.27.
 *
 * The badge remains absent mid-slider: positions up to 71 (Re 55) are silent.
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
 * @param {{h:number, dt:number, D:number, U:number, nMax:number, nuNum:number}} args
 * @returns {{reMin:number, reMax:number, nuMax:number}}
 */
export function honestWindow({ h, dt, D, U, nMax, nuNum }) {
  const nuMax = nMax * 0.25 * h * h / dt;
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
 * Codes:
 *   null            inside the window
 *   'clamped'       below the floor — nu saturated, effective Re is HIGHER
 *   'projection'    above the ceiling, and the under-converged pressure solve
 *                   is what set that ceiling (fixable: raise iterations)
 *   'scheme'        above the ceiling set by the scheme's own splitting error
 *                   (not fixable by iterations or by refining the grid)
 *   'empty-grid'    floor > scheme ceiling: NO honest Re exists at this grid,
 *                   at any iteration count. Only a coarser grid or a smaller
 *                   dt opens it.
 *   'empty-iters'   floor > projection ceiling, but the grid alone would leave
 *                   a window. Raising iterations opens it.
 *
 * @param {{re:number, reEff?:number, reMin:number, reMax:number,
 *          reMaxProjection?:number, viscClamped?:boolean}} args
 * @returns {{ok:boolean, code:string|null, reason:string|null}}
 */
export function windowState({
  re, reEff = re, reMin, reMax, reMaxProjection = Infinity, viscClamped = false,
}) {
  const ceiling = Math.min(reMax, reMaxProjection);

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
    return {
      ok: false,
      code: 'empty-iters',
      reason: `No honest Re at this grid with ${PROJECTION_ITERS_MEASURED} pressure iterations — the `
            + `viscous floor (Re ${fmtRe(reMin)}) sits above the under-converged ceiling `
            + `(Re ${fmtRe(reMaxProjection)}). Raise iterations, or lower the resolution.`,
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

  return { ok: true, code: null, reason: null };
}
