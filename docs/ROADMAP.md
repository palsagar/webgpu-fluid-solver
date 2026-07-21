# Roadmap

North star: **visual wow + physics credibility** — gasp in 10 seconds, survive a CFD expert's 2 minutes.

Decided 2026-07-06 (see ADRs 0005–0008 for the load-bearing decisions).

0. ~~**Pressure projection applied every step**~~ ✅ done (2026-07-19) — a latent bug meant projection landed on alternate steps only. Caught by splitting the per-step `Σ|div|` series into even- and odd-indexed subsequences: the relative imbalance between their means measured **0.384**, against a 0.15 threshold. Re-baselining the Kármán iteration count afterwards produced a table that a review then rejected as **phase-confounded** — all five counts were sampled sequentially on one solver instance, so time evolution was being read as the effect of iterations, and reproducibility could not rule it out because a deterministic sim reproduces an artifact byte-identically. Re-measured with a reset to identical initial conditions per block: mean `Σ|div|` **1472 / 1245 / 1114 / 1024 / 901** at 20 / 40 / 60 / 80 / 120 iterations. The direction survives; the magnitude does not (a 71.7% reduction over 20→120 became 38.8%), and the max-norm trend **reverses** — `max|div|` *rises* 2.73e-1 → 3.36e-1 with more iterations. Caveat that travels with these numbers: the 35-step window sits inside the startup transient, so they must not be quoted as evidence of a convergence problem. `numIters` was left at 80 here and raised later, in step 3, on frame-cost grounds.

1. ~~**GPU field rendering + 1024 tier**~~ ✅ done (2026-07-06; tier 1024 measured ~58 ms/frame (~17 fps) on the dev machine **at the 80 pressure iterations shipping then** — at the 256 iterations step 3 raised it to, the same tier measures **257 ms/frame (3.9 fps)**) — WebGPU render pass for the Field View (bilinear, colormap LUT texture, solids in-shader); overlays stay Canvas 2D on top. Add 1024 to adaptive tiers; keep red-black Gauss-Seidel, tune iterations by measurement. No multigrid unless 1024 can't hold 60 fps. [ADR-0005]

2. ~~**MacCormack advection**~~ ✅ done (2026-07-20) — second-order, min/max limited, three dispatches per field (forward, backward, limited combine), for smoke and velocity. Payoff measured by Taylor–Green decay against semi-Lagrangian on the identical field, projection converged so the advection scheme is what is being compared:

   | tier | 64 | 128 | 256 | 512 | 1024 |
   |---|---|---|---|---|---|
   | `nu_num` semi-Lagrangian | 2.9972e-3 | 1.4578e-3 | 1.1263e-3 | 1.0372e-3 | 1.5373e-3 |
   | `nu_num` MacCormack | 9.8575e-4 | 9.8308e-4 | 9.8232e-4 | 9.8680e-4 | 1.5133e-3 |
   | ratio | 0.329 | 0.674 | 0.872 | 0.951 | 0.984 |

   Strictly lower at every tier. The margin is largest where advection dominates and shrinks where the projection residual takes over — which is also the finding that redirected step 3: at the 80 iterations shipped at the time, MacCormack bought only 2–5% above tier 128 because projection error swamped it. Tiers 512 and 1024 are **not converged** (the GPU watchdog cut the escalation off at 4096 iterations); their values are the largest reachable, not the limit. [ADR-0006]

3. ~~**Viscosity + Re slider + Strouhal readout**~~ ✅ done (2026-07-20) — explicit five-point diffusion pass with `N = ceil(nu·dt/(0.25 h²))` substeps capped at 32 (past the cap `nu` saturates rather than `N` truncating — truncating leaves the scheme divergent, measured 141 811 non-finite cells). Operator fidelity `ν_delivered/ν_requested = **0.99751**` (corr 0.99927, 93 312 faces). Re slider spans a fixed **0.25 – 500** at every tier; a badge names the reason when the requested Re leaves the honest window rather than the control clamping. Live St from a downstream Probe on the Kármán preset — **shipped**, sampling 2 diameters downstream in simulation time.

   **The honest window, measured** (Kármán, `dt = 1/240`, 256 pressure iterations). Floor = viscous substep budget; ceiling = `U·D/ν_num` from Taylor–Green decay:

   | tier | 64 | 128 | 256 | 512 | 1024 |
   |---|---|---|---|---|---|
   | floor | 0.256 | 1.024 | 4.096 | 16.384 | 65.536 |
   | ceiling | 236.22 | 236.33 | **154.23** | 48.91 | 16.51 |
   | window | 0.26–236 | 1.02–236 | **4.10–154** | 16.4–48.9 | **empty** |

   The scheme's own ceiling is flat at **237.32** — `ν_num` is independent of `h` and linear in `dt`, so it is a time-splitting error, not grid diffusion. Refining the grid cannot lower it. Tier 1024's window is empty for want of iterations, not grid.

   **Flagship demo:** the vortex street dies below the measured onset **Re_c = 52.2 ± 0.3**, found by fitting the perturbation's exponential growth rate σ from identical initial conditions and locating where σ(Re) crosses zero. Two earlier onset numbers — **126** and **57.5** — were both produced by fixed-window amplitude criteria and are both wrong; near a Hopf bifurcation the growth rate vanishes, so a fixed window measures its own length rather than the flow. At Re 52 the e-folding time is **278 s**, so over any 30 s window it is indistinguishable from a saturated limit cycle. Onset sits above the textbook unconfined 47 in the direction channel blockage (`D/H = 0.12`) and the staircased cylinder both predict.

   **Measured St** on saturation-verified wakes: **0.166 / 0.168 / 0.170 / 0.173 / 0.180 / 0.190 / 0.200** at Re 55 / 57.5 / 60 / 65 / 74.8 / 100 / 140 (± ≤ 0.001). St is *not* ≈ 0.2 in this range and should not be — 0.2 is the high-Re plateau. These sit 11–25% above Roshko's unconfined correlation and converge toward it as Re rises, which is what blockage predicts. [ADR-0007, ADR-0008]

4. **Blow mode** — default mouse mode: drag injects momentum + Smoke at the cursor (write all three rotation slots).

5. **Freehand Draw mode + ε slider** — rasterize drawn solids into the Solid Mask (needs new invalidation path, eraser); Confinement exposed as labeled-artificial, default-off. [ADR-0006]

Deferred / rejected: GPU compute particles (revisit at ~100× particle counts, ADR-0003), multigrid pressure, 2048 tier, multiple parametric obstacles (subsumed by Draw mode), nominal-Re readout (rejected permanently, ADR-0007), smoke diffusion, a GPU-side probe ring buffer.

## Known gaps

Recorded here so nothing above reads as more settled than it is, and so the list survives the merge — it previously lived only in untracked working-tree notes.

### Defects that are measured and disclosed, not fixed

- **Dragging the obstacle resets the velocity field to the preset's initial conditions.** `interaction.js` allocates `_uData` / `_vData` as CPU mirrors, `presets.js` seeds them at preset load, and **nothing ever refreshes them from the GPU**. Every `rasterizeObstacle()` — i.e. every drag, shape change, and rotation — ends with `writeVelocityU(this._uData)` / `writeVelocityV(this._vData)`, pushing the whole stale array back over the live field. So a drag does not perturb the flow, it restarts it. Pre-existing (predates this branch), user-visible on the flagship demo, and until now disclosed in no tracked file. Magnitude: total — the entire interior velocity field, not a neighbourhood of the obstacle. The fix is a readback (or a GPU-side rasterizer) and is a follow-up.
- **The MacCormack velocity chain leaks its stale `i = 0` / `j = 0` ring into the interior.** `advect.wgsl`'s entry point returns for `i < 1 || j < 1`, so neither the forward nor the backward pass writes those lines; the forward pass leaves `phi^`'s ring stale and the backward pass samples it. Measured gain from an `EPS = 1e-3` perturbation of all four ring lines: **9.894e-3 inviscid**, **8.345e-4 with viscosity on** — diffusion damps it ~12×, because `diffuse.wgsl` classifies those lines as buried *by index* and substitutes a ghost rather than loading them. Bounded well below 1 and asserted in `tests/solver.spec.js`; fixing it means making the advect passes write their ring, which changes what every downstream stencil reads. Noted in `advect.wgsl` and `maccormack_velocity.wgsl` where it originates, and in `diffuse.wgsl` where it is defended against.
- **The domain walls silently change from free-slip to no-slip whenever `ν > 0`.** `diffuse.wgsl` reads the `j = 0` and `j = numY-1` lines as ghost cells, placing a zero-velocity wall half a cell outside the domain. This alters the effective blockage and shifts measured St the moment the Re control applies a viscosity — i.e. always, in the shipped configuration. Disclosed in the St tooltip; not corrected.
- **The viscous pass imposes a *stationary* no-slip wall on a *moving* obstacle.** The MacCormack advection path preserves the moving-wall BC during a drag (`interaction.js` writes the drag velocity `vx` into solid cells and the face to their right; the advection reverts those faces to `vx`, pinned by a test). The viscous pass throws it away: `diffuse.wgsl` classifies a face flanked by two solid cells as buried and ghosts it to `-center`, pinning the wall line at **zero** regardless of `vx`. So while an obstacle is dragged with `ν > 0`, diffusion drags the near-wall fluid toward zero instead of toward `vx`, partially cancelling the shear the moving wall imparts. Bounded (coeff ≤ 1/4), drag-only, and today **masked** by the field-reset defect above — you cannot observe a clean moving-wall boundary layer during a drag when the whole field is being restarted every drag frame. Fixing it means ghosting against the stored wall velocity rather than zero, which changes the viscous stencil; deferred with the field-reset fix.
- **The 3-slot rotation raised drag-time upload cost ~50%, unthrottled.** `writeVelocityU`/`writeVelocityV` now fan out to three velocity pairs (was two), and they run on every `mousemove` during a drag plus every inflow-slider `input`. At tier 1024 that is 3 × 4 MB per component per call. The solid *readback* has a re-entry guard for exactly this; the velocity *upload* does not. Expect stalls when dragging at 1024 with the Re control on. Same code path as the field-reset defect above, so its fix subsumes this.

### Gaps in the measured numbers

- **Amplitude (`U`) dependence of `ν_num` was never measured.** Every Taylor–Green fit ran at `A = 1.0`, and not even the *sign* of the correction is established — a plausible `ν_num ~ A²` scaling would make the true ceiling *fall* as `U` rises, i.e. the quoted ceiling is optimistic exactly where a user reaches by turning the flow up. Kármán ships `inVel = 1.0` so the app opens on the measured slice, but the inflow slider spans 0.5–5.0. `U` is now part of the measured-point gate (`ui.js`), so leaving 1.0 badges `unmeasured` rather than quoting a ceiling; `windTunnel` (`U = 2.0`) and `backwardStep` (`U = 1.5`) are off-slice on the `dt`/iterations axes too. Measuring `ν_num` across `A` is the way to close it — no correction is applied in the meantime, by design.
- **Tier 512 and 1024 converged values are extrapolated**, not measured — the browser died at 4096 iterations.
- **St is measured at one probe position, one tier, and one preset.** Sensitivity to any of the three is unmeasured.
- **`adaptive.js`'s `UPSCALE_MS = 12` is calibrated on one machine and one display.** It sits between the 120 Hz dev machine's vsync period (8.33 ms) and its first GPU-bound tier (17.75 ms). On a 60 Hz display no tier can beat 16.67 ms, so nothing auto-promotes and adaptive resolution silently does nothing in the upward direction. Conservative, and deliberate, but uncalibrated anywhere but here.
- **Step 0's controlled iteration table was taken inside the startup transient.**

### Test gaps

The suite is green and the mutations behind each assertion are recorded, but these are uncovered:

- **No descriptor test for pressure/boundary bind-group slot indexing.** The `diffuse` pipeline has one (`tests/solver.spec.js` captures the descriptor and checks which buffer lands in which slot); `pressure.wgsl` and `boundary.wgsl` do not. Slot indexing is precisely the class of bug that opened this branch, and nothing currently covers it for the two oldest pipelines.
- **No test for the `diffuse` bind group's `[src][dst]` transposition.** Swapping the two would still produce a plausible-looking field.
- **No rotation invariant after a resize or tier change.** The three-slot velocity rotation is checked in steady state, not across the buffer recreation that `applyTier()` performs.
- **The shedding ONSET value is not measured in-suite.** `tests/diagnostics.spec.js` steps the live solver at the shipped default and confirms the wake sheds at the measured St and amplitude, which bounds the onset from *above* (it is below Re 74.8). Locating the crossing needs a sweep in Re at minutes of wall clock per point, and stays offline.
- **Several assertions pin the live solver against offline-transcribed reference values** the suite never recomputes — the onset `52.2`, the operating-point `ν_num = 2.0324e-3`, the St band. The tolerances are justified in-file, but the reference values come from multi-minute offline sweeps not in the repo, so a failure cannot be told from a stale transcription without re-running them.
- **The `resetFlipState` → clear-the-probe coupling is enforced only by convention.** `resetFlipState` cannot clear the probe (owned by the UI), so it warns in a comment that any caller must; today every caller happens to route through `invalidateSolid`/`_updateReBadge`, which do. A new path that calls `resetFlipState` without going through one of those would silently mix two runs' samples into one Strouhal window. Unenforced by any test.

### Deployment hardening (not security-critical today)

A three-persona adversarial review found no live vulnerability — the app has no auth, no data store, and every dynamic value reaches the DOM via `textContent`, so there is no injection sink. These are defense-in-depth notes for the deployed static server:

- **CSP is `frame-ancestors 'none'` only.** No `script-src` / `default-src` / `connect-src`. Unreachable as XSS today (no HTML-injection sink), but the moment any future change introduces an `innerHTML`/template sink fed by the GitHub star response or a URL param, the missing directives remove the backstop.
- **An unauthenticated `api.github.com` fetch fires on every page load** (`index.html`), leaking each visitor's IP/`Referer` to GitHub. Failure is handled and the response reaches the DOM only as `textContent`. Render the star count at build time, or drop it, if that matters to the deployer.
- **No URL / `localStorage` state deserializer exists, and that is load-bearing.** Client inputs (tier, iterations, Re) can drive heavy GPU work, but because no link can encode that state, a visitor can only load-test their own GPU. If shareable state is ever added, it must re-clamp resolution and iteration count on the restore path, not only in the button handlers.
