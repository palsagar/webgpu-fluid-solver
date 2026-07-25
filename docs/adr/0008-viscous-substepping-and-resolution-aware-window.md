---
status: accepted
---

# Viscous substepping and a resolution-aware honest window

The Reynolds slider keeps one fixed range (0.25 – 500, log-spaced, 101 positions) across all resolution tiers and never moves. When the requested Re leaves the range this grid and this pressure solve can actually deliver, a badge names the reason rather than the control clamping itself. The explicit viscous pass runs `N = ceil(nu*dt / (0.25 * h^2))` five-point substeps per frame, capped at `N_MAX = 32`; past the cap the *coefficient* saturates (`nu` is clamped to `viscNuMax = N_MAX * 0.25 * h^2 / dt`) rather than `N` being truncated.

This supersedes ADR-0007's fixed ~10–5000 cap and its position that "the cap is the credibility feature". The cap is not the credibility feature. The *measurement* is; the badge is how it reaches the user.

## The derivation this ADR was drafted with — and why it is wrong

The plan that produced this branch proposed the window as a pair of analytic bounds.

**The floor was right.** An explicit five-point Laplacian is stable only for `nu*dt/h^2 <= 1/4`. With `nu = U*D/Re` that is a *lower* bound on Re, and with `N` substeps each carrying `dt/N`:

```
Re_min = U*D / viscNuMax = 4*U*D*dt / (N_MAX * h^2)
```

It rises as `h^-2` — **four times per tier**. At the Kármán operating point (`U = 1.0`, `D = 0.12`, `dt = 1/240`, `N_MAX = 32`):

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `Re_min` | 0.256 | 1.024 | 4.096 | 16.384 | 65.536 |

**The ceiling was falsified.** The draft argued that the upper bound comes from resolving the cylinder's boundary layer, `delta ~ D/sqrt(Re)`; demanding `k` cells across `delta` gives `Re <= (D/(k*h))^2`, which *rises* with resolution — 14.7 / 59.0 / 235.9 / 943.7 / 3774.9 at `k = 2` across tiers 64–1024. Their ratio,

```
floor/ceiling = 4*U*dt*k^2 / D
```

has the `h` terms cancel, so the draft concluded the window's *shape* is resolution-independent (the ratio is 1.11 at `k = 2` and the Kármán parameters) and that a single-substep explicit pass therefore has an empty window at every tier, with no choice of cap able to fix it.

The algebra is fine. **The premise is not.** Task 7 measured the real ceiling by Taylor–Green decay and it disagrees with `(D/(k*h))^2` by **8.3x at tier 64, 7.2x at 256, 79x at 512 and 746x at 1024** — and, more damning than the magnitude, the two **scale in opposite directions**. The analytic model rises with resolution (14.7 → 3775); the measurement at the then-shipped operating point falls with it (121.7 → 5.1).

**What the measurement found instead.** With the projection converged, the scheme's own numerical viscosity is:

- **Independent of `h`.** 9.8575e-4 / 9.8308e-4 / 9.8232e-4 / 9.8680e-4 at tiers 64 / 128 / 256 / 512 (`dt = 1/120`) — flat to 0.5% across an 8x grid refinement. Re-confirmed at `dt = 1/240`: 5.0802e-4 / 5.0675e-4 / 5.0564e-4 at tiers 64 / 128 / 256.
- **Linear in `dt`.** At tier 128, `dt = 1/60 … 1/480` gives 1.6951e-3 / 9.8309e-4 / 5.1419e-4 / 2.6009e-4 — successive ratios 1.72, 1.91, 1.98 → 2. Across the shipped halving, 9.8232e-4 → 5.0564e-4, ratio **1.943**.

So `nu_num` is an **operator-splitting error in time, not grid-scale diffusion**. Refining the grid cannot lower it; only reducing `dt` can. The boundary-layer model assumes precisely the opposite, and it is not what governs this solver.

**Both of the draft's premises were wrong in the same direction.** It expected the ceiling to rise with resolution and the floor/ceiling ratio to be flat. Measurement says the scheme ceiling is *flat* in `h`, the delivered ceiling *falls* with `h` (below), and the ratio therefore rises as `h^-2`. The window is genuinely resolution-aware — just not in the direction the derivation predicted, and for a different reason.

**Substepping is what makes a window exist at all.** With a single substep (`N = 1`) the floor is 32x higher — 8.19 / 32.77 / 131.07 / 524.29 / 2097.15 — against a flat scheme ceiling of 237.32. That leaves tiers 512 and 1024 empty outright and tier 256 with a 1.18x band (131.07 … 154.23). The draft's conclusion that one substep is unusable is broadly right; its stated reason is not.

## Where the ceiling comes from

Measured, not asserted. A Taylor–Green vortex decays as `exp(-2*nu*k^2*t)`; run with physical viscosity off, the fitted decay rate **is** the scheme's own numerical viscosity, and the ceiling is the Re at which the requested physical viscosity falls below it: `Re_max = U*D/nu_num`. The harness is `measureNuNum` in `tests/solver.spec.js`, on a closed square sub-box of `numY - 2` cells with a phase-swapped mode that vanishes wall-normal on all four walls (the naive field drives fluid into the walls at full amplitude and measures the solver fighting an impossible BC). The discrete divergence of the seeded field cancels identically on the MAC grid — measured `max|div| = 5.96e-8`, machine zero in float32.

There are **two** ceilings, and the app quotes the binding one:

**Scheme ceiling** — the advection scheme's own floor, with the projection converged. Since `nu_num` is linear in `dt` and flat in `h`, one constant covers every tier:

```
NU_NUM_PER_DT = 0.121354        nu_num(dt) = NU_NUM_PER_DT * dt
```

anchored at the converged `5.0564e-4` at `dt = 1/240` (r² = 0.99994; 2048 and 4096 iterations agree to five significant figures). That gives **Re = 237.32**. The anchor is deliberate: linearity is 1.943, not 2.000, so one coefficient cannot fit both timesteps, and anchoring at 1/240 *overstates* `nu_num` by 2.9% at 1/120 — which *understates* the ceiling, the conservative direction for an honesty claim. A test asserts that sign.

**Projection ceiling** — what the shipped iteration count actually delivers. A fixed iteration count converges progressively less well as the grid grows, so the delivered `nu_num` rises with resolution and the delivered ceiling falls. At the shipped `numIters = 256`, `dt = 1/240`:

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `nu_num` | 5.0801e-4 | 5.0777e-4 | 7.7808e-4 | 2.4536e-3 | 7.2698e-3 |
| `Re_max` | 236.22 | 236.33 | **154.23** | 48.91 | 16.51 |
| r² | 0.99994 | 0.99994 | 0.99986 | 0.99860 | 0.98926 |

Tiers 64 and 128 land within 0.5% of the converged constant — the projection has stopped binding there and the badge attributes the ceiling to the scheme. From tier 256 up the projection binds and the badge says so.

**The resulting window**, floor from the substep budget and ceiling from the table above:

| tier | floor | delivered ceiling | window |
|---|---|---|---|
| 64 | 0.256 | 236.22 | 0.26 … 236 |
| 128 | 1.024 | 236.33 | 1.02 … 236 |
| 256 | 4.096 | 154.23 | 4.10 … 154 (startup tier) |
| 512 | 16.384 | 48.91 | 16.4 … 48.9 |
| 1024 | 65.536 | 16.51 | **empty** |

Tier 1024's window is empty because the floor sits above the *projection* ceiling — but well below the *scheme* ceiling of 237, so the grid itself is not the obstacle. The badge reports `empty-iters`, not `empty-grid`, and advises the two levers that exist (lower the tier, or reduce `dt`) rather than "raise iterations", which cannot close a factor of 4 in `nu_num` with a control that stops at 320.

## Considered options

- **Fixed 10–5000 cap per ADR-0007** — rejected. 5000 is above the measured scheme ceiling by more than 20x at every tier, and 10 is below the floor at tiers 64–256, so the cap's ends bear no relation to either real bound.
- **Truncating `N` at `N_MAX` instead of saturating `nu`** — rejected by measurement. Truncating leaves the coefficient above 1/4 and the explicit update divergent: at `nu = 0.1`, tier 256, it produces **141 811 non-finite interior cells after 60 steps**. Saturating `nu` pins the coefficient at exactly 1/4 — under-diffusive but bounded — and `viscClamped` reports that the *effective* Re is higher than the number on the control.
- **Implicit diffusion** — unconditionally stable at fixed cost, but a second iterative solver to write and tune, and an under-converged implicit solve silently contributes its own error to the very quantity this branch exists to measure.
- **Smaller `dt`** — partially taken. Halving the Kármán preset to `dt = 1/240` halved every floor and roughly doubled the ceiling (122.16 → 237.32), because `nu_num` is linear in `dt` and the floor scales with it. It costs no frame time (one `step()` per rAF regardless of `dt`) but halves simulated time per wall second, so time-to-vortex-street roughly doubled (~10–15 s → ~20–30 s of wall clock).
- **Hard clamp with per-tier bounds** — rejected. `adaptive.js` retunes the tier at runtime, so a per-tier clamp would silently rewrite the user's Re mid-run, and a control whose range jumps on its own reads as a bug.
- **Quoting the measured table on every preset** — rejected, and it had already shipped wrong. `NU_NUM_ITERS256` is measured on one slice (`dt = 1/240`, 256 iterations). Carried onto a since-removed preset (ADR-0009) it produced a projection ceiling of Re 771 against a scheme ceiling of Re 297 — an under-converged solve dissipating *less* than a converged one, which is impossible. `windowState` now enforces `reMaxProjection <= reMax` and reports `unmeasured` rather than quoting a number from a different configuration.

## Does the viscous pass deliver the viscosity it is asked for

**`nu_delivered / nu_requested = 0.99751`**, correlation 0.99927, over 93 312 faces at `nu = 1.2e-2` (27 substeps); at `nu = 1e-4` (`N = 1`, where the operator expansion is exact) the slope is 1.00000.

Measured directly rather than by decay fit, and the difference matters: the closed Taylor–Green box is analytically free-slip but the viscous pass's ghost makes it no-slip, so wall layers dominate the KE budget and a decay fit returns anywhere from 0.25x to 4.2x the true value depending on window margin and run length. Any configuration landing near 1.0 would do so by luck. Instead, two single steps from a byte-identical field — one at `nu = 0`, one at `nu = NU` — share their pressure solve, extrapolation and advection exactly and differ by precisely the viscous increment, so a least-squares slope of actual against expected is per-cell and analytic.

## The error mode that produced four wrong numbers

Recorded here because it is more durable than any single value in this ADR, and it is what makes the rest of them trustworthy.

**Four separate times on this branch a number was produced by settling too briefly near a bifurcation or inside a transient, and twice those numbers were recorded as results before being caught:**

1. **Task 3's `numIters` table** measured five iteration counts sequentially on one solver instance, mistaking time evolution for the effect under study. Reproducibility did *not* rule it out — a deterministic simulation reproduces an artifact byte-identically. Reset-to-identical-IC per block halved the apparent effect (a 71.7% reduction over 20→120 iterations became 38.8%) and reversed the max-norm trend outright.
2. **Shedding onset Re ≈ 126** — a fixed 3.33 s settle, read as a threshold. It was measuring the settle time.
3. **Its replacement, Re ≈ 57.5** — a 30 s growth *ratio*. Same defect, one order smaller.
4. **The Strouhal table** was quoted to three significant figures on signals still growing by up to 271x across their own measurement window.

The fix in every case was to stop reading an amplitude over a fixed window and measure something that is a property of the flow. For the onset: fit the perturbation's exponential growth rate `sigma` from identical initial conditions plus one identical deterministic kick, and find where `sigma(Re)` crosses zero. `sigma` does not depend on how long the point ran, which is the whole failure mode.

**Onset: Re_c = 52.2 ± 0.3.** `sigma` = −0.3772 / −0.2159 / −0.0975 / −0.0036 / +0.0771 / +0.1526 at Re 44 / 47 / 50 / 52 / 54 / 56, linear in Re (`dsigma/dRe ≈ 0.042`, r² > 0.996) — a textbook Hopf bifurcation. The uncertainty is the full spread across 53 combinations of fit subset and analysis-window variant (52.02 … 52.39), which dwarfs any single standard error.

**Why 57.5 was produced, quantitatively: at Re 52 the e-folding time is 278 s.** Across any 30 s window Re 52 is indistinguishable from a saturated limit cycle. No fixed-window amplitude criterion could have found this, however carefully it was applied.

Two independent cross-checks agree: the saturated-amplitude law `A_sat^2 = 2.788e-3 (Re - 52.62)` (r² = 0.9996 near onset, within 1%), and the same growth-rate criterion at `dt = 1/120` where `nu_num` doubles, which crosses at 52.5 — a 0.6% shift.

## Consequences and limitations

Stated here because a decision record that omits them is the failure this branch exists to prevent.

- **The `nu_num` constant is measured at `dt = 1/240`, `A = 1.0`, on the Kármán preset. The amplitude (`U`) dependence was never measured.** `backwardStep` runs `U = 1.5`; a plausible `A^2` scaling would move its ceiling by ~2x. It shows an `unmeasured` badge rather than a number.
- **Tier 512 and 1024 converged values are extrapolated, not measured.** The escalation killed the browser at 4096 iterations (GPU watchdog). The flat 237.32 there is inferred from the tier 64–256 flatness. It decides nothing structural — tier 1024's verdict rests on the projection ceiling, which *is* measured.
- **The domain walls silently change from free-slip to no-slip whenever `nu > 0`.** `diffuse.wgsl` classifies the domain ring as BURIED and reads it as a ghost, placing a zero-velocity wall half a cell outside the domain. Channel wall layers then add to the effective blockage, and they thicken as Re falls — so the confound is Re-dependent, not a constant offset. Disclosed in the `#val-st` tooltip; not corrected, because correcting it needs a blockage calibration nothing here has measured.
- **Onset 52.2 against the textbook unconfined ~47 is a property of this geometry, not an error** — channel blockage `D/H = 0.12` and a staircased cylinder both push it up.
- **St sits 11–25% above Roshko's unconfined correlation** across Re 57.5–140, converging toward it as Re rises and the wall layers thin. That is the direction blockage predicts.
- **Task 3's controlled `numIters` measurement was taken inside the startup transient** (35 steps). That caveat travels with its numbers; the max-norm column in particular must not be read as evidence of a convergence problem.
- **`adaptive.js`'s `UPSCALE_MS = 12` was calibrated on one machine and one display.** No frame can beat vsync, so on a 60 Hz display nothing auto-promotes at all. Conservative, documented on the constant, and the manual tier buttons are unaffected.
- **The app's `nu_num` numbers are from one Apple M-series GPU in float32.** The SOR residual floor (~6.8e-4) is hardware-dependent, so operating-point values may shift elsewhere; the committed assertion bands are deliberately loose for this reason.
