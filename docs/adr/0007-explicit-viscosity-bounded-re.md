---
status: accepted, partially superseded by 0008; implemented, with several of its numbers corrected by measurement (see Current state)
---

# Explicit viscosity with a bounded Reynolds slider; Strouhal measured, never prescribed

The solver gains an explicit viscous diffusion pass, making Reynolds number physically real and user-controllable. The Re slider is bounded to the regime this solver can actually deliver: above that, physical viscosity drops below the scheme's numerical diffusion and the label would be a lie (see ADR-0006). The Kármán preset displays a live Strouhal number computed from a downstream velocity probe — measured from the simulation, never hard-coded — so shedding onset and shedding frequency are emergent results rather than captions.

The decision text originally named three figures as expected outcomes: a **~10–5000** cap, **St ≈ 0.2**, and onset near the textbook unconfined **Re ≈ 47**. All three were falsified by measurement and none ships. They are struck here rather than quietly edited, and what replaced them is in *Current state* below and in [ADR-0008](0008-viscous-substepping-and-resolution-aware-window.md), which supersedes this ADR's treatment of the cap.

## Considered options

- **"Nominal Re" readout without a viscous term** — rejected: the solver was inviscid (Euler + numerical diffusion), so any displayed Re would be fabricated.
- **Uncapped Re slider** — rejected: honest only where viscosity is resolved; the cap is the credibility feature.

## Current state

The nominal-Re readout is **gone**. `ui.js`'s `_updateRe()` (`Re = U·D/h`) was deleted and replaced by a real Re control: the slider sets `nu = U·D/Re`, the explicit viscous pass integrates it, and `diagnostics.js` reports when the requested Re leaves the window this grid can deliver.

Several numbers in the decision above were **falsified by measurement** and do not ship as written. They are stated below at the values measured at the shipped operating point — Kármán at `dt = 1/240` and `numIters = 256`, `U = 1.0`, `D = 0.12`. The mechanism behind the window, the substepping scheme that sets its floor, and the derivation that was proposed and then falsified are all in [ADR-0008](0008-viscous-substepping-and-resolution-aware-window.md).

- **The cap is 0.25–500, not ~10–5000.** Task 7's Taylor–Green calibration found `nu_num` *independent of h* and *linear in dt* — a time-splitting error, not grid diffusion. So the ceiling is flat across every tier and refining the grid cannot raise it; only reducing `dt` can. At the shipped `dt = 1/240` the converged value is `nu_num = 5.0564e-4` (r² = 0.99994), giving a scheme ceiling of `Re = U·D/nu_num ≈ 237`. The floor is the viscous substep budget, `U·D/viscNuMax`, which scales with `dt` and ranges **0.256** (tier 64) to **65.5** (tier 1024). Below the scheme ceiling sits a second, lower one: the projection ceiling actually delivered by the shipped iteration count (tier 256: **154**). The slider spans 0.25–500 so that every regime — floor, both ceilings, and the empty window — is reachable at every tier.
- **At tier 1024 the window is empty for want of iterations, not grid.** The floor (65.5) sits above the *projection* ceiling (16.5) but well below the *scheme* ceiling (237), so the grid itself is not the obstacle — the badge reports `empty-iters`, not `empty-grid`. The gap is a factor of 4 in `nu_num`, more than the iterations control can close (80 → 256 iterations bought 2.1×, and the control stops at 320), so the badge advises the two levers that do work: a lower tier, or a smaller `dt`.
- **Shedding onset is Re ≈ 52.2 ± 0.3 here, not the physical 47 — and neither the Re ≈ 126 nor the Re ≈ 57.5 earlier measurements reported.** Both of those were fixed-window amplitude criteria, and near a Hopf bifurcation the growth rate vanishes, so a fixed window measures its own length rather than the flow. The 126 came from a 3.33 s settle; the 57.5 from a 30 s growth *ratio*, which is the same error one order smaller. The criterion is now **window-independent**: fit the perturbation's exponential growth rate `sigma` from an identical impulsive start plus an identical deterministic kick, and locate where `sigma(Re)` crosses zero. Measured at tier 256, `dt = 1/240`, 256 iterations, 50 s logged per point — `sigma` = −0.377, −0.216, −0.0975, −0.0036, +0.0771, +0.1526 at Re 44, 47, 50, 52, 54, 56, linear in Re (`dsigma/dRe ≈ 0.042`, r² > 0.996), crossing zero at **52.2**. The uncertainty is the full spread over every fit subset × analysis-window variant tried — 53 combinations, all landing in **52.02 … 52.39**. (An earlier draft of this line quoted a narrower 52.07 … 52.28 from a partial sweep; the wider spread is the one the completed sweep produced, and it is the one that sets the ± 0.3.) Re 52 is the point that shows why the old method failed: its e-folding time is 278 s, so over any 30 s window it is indistinguishable from a flat limit cycle. Onset still sits above the textbook unconfined 47, in the direction the channel blockage (`D/H = 0.12`) and the staircased cylinder both predict.
- **The app opens un-badged.** The default sits at Re 74.8 — 1.43× above onset, 2.06× below the tier-256 projection ceiling of 154 — inside the honest window, on a fully developed vortex street. Raising `numIters` 80 → 256 is what bought this: the slider position did not move, the ceiling rose past it (59.0 → 154.2).

Where a number has not been measured, the UI declines to quote one. `NU_NUM_ITERS256` is measured on a single slice (`dt = 1/240`, 256 iterations, across tiers), so off that slice — the other two presets, or any move of the `dt`/iterations sliders — the badge reports the ceiling as **unmeasured** rather than carrying the table across. `windowState` enforces the invariant that made this visible: an under-converged projection cannot dissipate *less* than a converged one, so `reMaxProjection ≤ reMax` must hold, and on `windTunnel` the carried table violated it (Re 771 against Re 297).

## The Strouhal readout, as shipped

The live readout **is implemented**: `#val-st` in the Flow Info panel, fed by a `StrouhalProbe` sampling a single velocity cell **2 diameters downstream** of the obstacle, in *simulation* time. It reports one of three states — `measuring…` while the window fills, `steady — no shedding` when the wake RMS is below the amplitude gate, or a number — and it refuses to fit a frequency to a steady flow, to an under-sampled signal, or to a collapsed field.

`probeCell()` returns `null` rather than clamping when the target cell would land in the two outflow columns `diffuse.wgsl` copies through unchanged. A clamped probe silently stops being "2 diameters downstream", so the number it produced would no longer mean what the readout says. At the shipped geometry the guard only engages if the obstacle is dragged into the last ~11% of the channel — reachable by hand, which is why it exists.

**Measured St, on saturation-verified wakes** (115 s of simulation time per point, saturation verified by the final two 20 s windows agreeing to the drift shown, rather than read off one window and assumed settled). Kármán preset, tier 256, `dt = 1/240`, 256 iterations:

| Re | 55* | 57.5 | 60 | 65 | 74.8 (default) | 100 | 140 |
|---|---|---|---|---|---|---|---|
| `A_sat` | 8.19e-2 | 1.159e-1 | 1.437e-1 | 1.910e-1 | 2.636e-1 | 4.024e-1 | 5.460e-1 |
| drift | 0.83% | 0.02% | 0.24% | 0.02% | 0.01% | 0.26% | 0.21% |
| **St** | 0.166 | 0.168 | 0.170 | 0.173 | **0.180** | 0.190 | 0.200 |
| ± | 0.001 | 0.001 | 0.0000 | 0.001 | 0.001 | 0.0000 | 0.001 |

\* Re 55 did not fully saturate and is labelled so in the code; its frequency settles well before its amplitude does, which is why an St is still quoted.

**St is not ≈ 0.2 in this range, and should not be.** 0.2 is the high-Re plateau (Re ≳ 300). Across Re 57.5 → 140 the measured value rises monotonically 0.168 → 0.200, sitting **11–25% above** Roshko's unconfined `St = 0.212(1 − 21.2/Re)` and converging toward it as Re rises and the channel wall layers thin — the direction and the trend that blockage predicts for a cylinder occupying 12% of the channel with no-slip walls. The wall confound is disclosed in the `#val-st` tooltip, not corrected: correcting it needs a blockage calibration nothing here has measured.

Two honest costs of the readout, both documented in the code:

- **Latency.** `dt = 1/240` at one step per rAF frame means simulation time runs at ~0.25x wall clock, so the first number takes ~21 s of wall clock and a full window ~43 s. Shortening the window would cost frequency resolution.
- **Near onset the verdict is run-length dependent.** At Re 55 the readout says `steady` for the first ~40 s of simulation time and then switches to a number. That is honest — it declines to claim a frequency it cannot yet see — but a user sweeping the slider quickly will read `steady` at Re values that do shed.
