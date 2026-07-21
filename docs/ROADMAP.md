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

## Known gaps in the measured numbers

Recorded here so nothing above reads as more settled than it is.

- **Amplitude (`U`) dependence of `ν_num` was never measured.** Every fit used `A = 1.0`. `windTunnel` runs `U = 2.0` and `backwardStep` `U = 1.5`, so their ceiling is approximate and both presets show an `unmeasured` badge rather than a number.
- **Tier 512 and 1024 converged values are extrapolated**, not measured — the browser died at 4096 iterations.
- **The domain walls silently change from free-slip to no-slip whenever `ν > 0`**, altering effective blockage and shifting measured St. Disclosed, not corrected.
- **St is measured at one probe position, one tier, and one preset.** Sensitivity to any of the three is unmeasured.
- **`adaptive.js`'s `UPSCALE_MS = 12` is calibrated on one machine and one display.** On a 60 Hz display nothing auto-promotes.
- **Step 0's controlled iteration table was taken inside the startup transient.**
