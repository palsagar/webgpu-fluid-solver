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
 *     (9.82e-4 at tiers 64 through 512, flat to 0.5% across an 8x refinement)
 *     and LINEAR in dt (ratios 1.72, 1.91, 1.98 -> 2 over a 8x dt sweep).
 *     It is an operator-splitting error in TIME, not grid diffusion.
 *   - So the analytic ceiling `Re <= (D/(k*h))^2`, which RISES with resolution,
 *     is the wrong model. Refining the grid cannot lower `nu_num`. Only
 *     reducing dt can.
 *
 * Hence ONE measured constant and a FLAT ceiling across all tiers. There is no
 * per-tier table for the converged value and no estimated fallback anywhere in
 * this module: an unmeasured number driving a real ceiling is exactly the
 * failure this branch exists to prevent.
 */

/**
 * Numerical viscosity of the MacCormack advection + splitting scheme with the
 * projection converged, in domain-units^2/s. Measured at dt = 1/120, amplitude
 * A = 1.0, float32. Tier-independent (see above).
 *
 * Re_max = U*D/NU_NUM_CONVERGED = 0.12/9.823e-4 = 122.2 for the Karman
 * reference (U = 1.0, D = 2 x 0.06).
 */
export const NU_NUM_CONVERGED = 9.823e-4;

/** Pressure iteration count at which NU_NUM_ITERS80 was measured. */
export const PROJECTION_ITERS_MEASURED = 80;

/**
 * Numerical viscosity at the OPERATING POINT the app actually ships:
 * `numIters = 80`, where the red-black SOR projection is badly under-converged
 * and its residual, not the advection scheme, dominates the error.
 *
 * This is why a flat 122.2 is not the whole truth — at 80 iterations it
 * understates the error by up to 24x at tier 1024. Keyed by tier (numY),
 * MacCormack column of Task 7 §2a.
 *
 * Unlike the converged constant this one DOES rise with resolution, because a
 * fixed iteration count converges progressively less well as the grid grows.
 *
 *   tier      64      128      256      512     1024
 *   nu_num  9.86e-4  1.26e-3  3.66e-3  1.01e-2  2.37e-2
 *   Re_max   121.7     95.6     32.8     11.9      5.1
 */
export const NU_NUM_ITERS80 = {
  64:   9.8612e-4,
  128:  1.2551e-3,
  256:  3.6638e-3,
  512:  1.0062e-2,
  1024: 2.3718e-2,
};

/**
 * How close the two ceilings must be before we stop blaming the projection.
 *
 * Task 7 reports a ~10% systematic uncertainty on `nu_num` from the choice of
 * fit window (1.02e-3 / 9.87e-4 / 9.36e-4 at 120 / 300 / 600 steps). Below that
 * separation the two measurements are the same number and the honest attribution
 * is the scheme; above it the projection genuinely dominates. At tier 64 the
 * ratio is 0.996 (scheme); at tier 128 it is 0.78 (projection).
 */
export const CEILING_AGREEMENT_TOL = 0.9;

// ── The Re control's range ──────────────────────────────────────────────────
//
// Log-spaced, because Re spans three decades across the tier set and the
// honest window at any one tier is a narrow slice of it.
//
//   RE_SLIDER_MIN = 0.5  — the lowest floor in the tier set (tier 64: 0.512),
//                          so the honest low end is reachable, not cropped.
//   RE_SLIDER_MAX = 500  — clears both structural bounds with headroom: the
//                          flat ceiling (122.2) and the highest floor (tier
//                          1024: 131.1). Every badge regime is therefore
//                          reachable at every tier, including tier 1024 where
//                          the window is empty and both ends must badge.
//
// Their geometric mean is 15.8, which lands inside the startup tier's honest
// window (256 at 80 iterations: 8.2 .. 32.8) — so mid-slider is badge-free by
// construction rather than by luck.

export const RE_SLIDER_MIN = 0.5;
export const RE_SLIDER_MAX = 500;
export const RE_SLIDER_STEPS = 100;

/**
 * Where the control ships (mirrored by index.html's `value` attribute) — Re 251.
 *
 * This is deliberately OUTSIDE the honest window, and the app therefore opens
 * with the badge showing. A slider sweep of the Karman preset at tier 256 (400
 * steps per point) measured the wake's unsteadiness — the RMS change in v down
 * a column at 55% of the channel over 30 frames:
 *
 *   Re        0.5      16       32       63      126      251      500
 *   unsteady  2.3e-4   3.2e-4   2.0e-4   3.0e-4  6.2e-3   1.1e-1   4.1e-1
 *
 * The wake is dead steady up to Re ~63 and only sheds above ~126. The honest
 * window at this tier is 8.2 .. 32.8, so there is NO overlap between "the
 * Karman preset shows a vortex street" and "the badge is silent".
 *
 * Defaulting inside the window would open a preset named "Karman Vortex" on a
 * flow with no vortices — a lie told by the picture instead of by a label.
 * Defaulting here shows the advertised flow and states plainly, in the badge,
 * that the honest ceiling is Re 33 rather than the 251 requested. The badge
 * being absent mid-slider remains a real property: positions 50-60 are silent.
 */
export const RE_SLIDER_DEFAULT_POS = 90;

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
 * estimated — pass NU_NUM_CONVERGED.
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
