---
status: accepted; Re control implemented, two of its numbers corrected by measurement (see Current state)
---

# Explicit viscosity with a bounded Reynolds slider; Strouhal measured, never prescribed

The solver gains an explicit viscous diffusion pass, making Reynolds number physically real and user-controllable. The Re slider is capped to the regime the grid actually resolves (~10–5000): above that, physical viscosity drops below the scheme's numerical diffusion and the label would be a lie (see ADR-0006). The Kármán preset displays a live Strouhal number computed from a downstream velocity probe — measured from the simulation, never hard-coded — so the app demonstrates St ≈ 0.2 and shedding onset near Re ≈ 47 as emergent results.

## Considered options

- **"Nominal Re" readout without a viscous term** — rejected: the solver was inviscid (Euler + numerical diffusion), so any displayed Re would be fabricated.
- **Uncapped Re slider** — rejected: honest only where viscosity is resolved; the cap is the credibility feature.

## Current state

The nominal-Re readout is **gone**. `ui.js`'s `_updateRe()` (`Re = U·D/h`) was deleted and replaced by a real Re control: the slider sets `nu = U·D/Re`, the explicit viscous pass integrates it, and `diagnostics.js` reports when the requested Re leaves the window this grid can deliver.

Two numbers in the decision above were **falsified by measurement** and do not ship as written. Both are stated below at the values measured at the shipped operating point — Kármán at `dt = 1/240` and `numIters = 256`, `U = 1.0`, `D = 0.12`.

- **The cap is 0.25–500, not ~10–5000.** Task 7's Taylor–Green calibration found `nu_num` *independent of h* and *linear in dt* — a time-splitting error, not grid diffusion. So the ceiling is flat across every tier and refining the grid cannot raise it; only reducing `dt` can. At the shipped `dt = 1/240` the converged value is `nu_num = 5.0564e-4` (r² = 0.99994), giving a scheme ceiling of `Re = U·D/nu_num ≈ 237`. The floor is the viscous substep budget, `U·D/viscNuMax`, which scales with `dt` and ranges **0.256** (tier 64) to **65.5** (tier 1024). Below the scheme ceiling sits a second, lower one: the projection ceiling actually delivered by the shipped iteration count (tier 256: **154**). The slider spans 0.25–500 so that every regime — floor, both ceilings, and the empty window — is reachable at every tier.
- **At tier 1024 the window is empty for want of iterations, not grid.** The floor (65.5) sits above the *projection* ceiling (16.5) but well below the *scheme* ceiling (237), so the grid itself is not the obstacle — the badge reports `empty-iters`, not `empty-grid`. The gap is a factor of 4 in `nu_num`, more than the iterations control can close (80 → 256 iterations bought 2.1×, and the control stops at 320), so the badge advises the two levers that do work: a lower tier, or a smaller `dt`.
- **Shedding onset is Re ≈ 57.5 here, not the physical 47 — and not the Re ≈ 126 an earlier measurement reported.** That 126 was a 3.33 s settle reading a still-growing transient, not a bifurcation. Replaced by a 30 s growth-to-saturation test (~50 shedding periods) at tier 256: growth crosses 1 between Re 57 (0.638) and Re 60 (12.2), i.e. **onset 57.5**. Halving `dt` — which halves `nu_num` — moved it only 56.8 → 57.5, so the onset is **not** set by the solver's numerical viscosity. It is set by the geometry: channel blockage (`D/H = 0.12`) raises the critical Re above the textbook unconfined 47, and the cylinder is staircased on the grid.
- **The app opens un-badged.** The default sits at Re 74.8 — 1.30× above onset, 2.06× below the tier-256 projection ceiling of 154 — inside the honest window, on a fully developed vortex street. Raising `numIters` 80 → 256 is what bought this: the slider position did not move, the ceiling rose past it (59.0 → 154.2).

Where a number has not been measured, the UI declines to quote one. `NU_NUM_ITERS256` is measured on a single slice (`dt = 1/240`, 256 iterations, across tiers), so off that slice — the other two presets, or any move of the `dt`/iterations sliders — the badge reports the ceiling as **unmeasured** rather than carrying the table across. `windowState` enforces the invariant that made this visible: an under-converged projection cannot dissipate *less* than a converged one, so `reMaxProjection ≤ reMax` must hold, and on `windTunnel` the carried table violated it (Re 771 against Re 297).

The live Strouhal readout is **not implemented**; there is no `St` element and no probe. It remains open work.
