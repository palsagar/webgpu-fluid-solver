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

Two numbers in the decision above were **falsified by measurement** and do not ship as written:

- **The cap is 0.5–500, not ~10–5000.** Task 7's Taylor–Green calibration found `nu_num = 9.823e-4` *independent of h* and *linear in dt* — a time-splitting error, not grid diffusion. The ceiling is therefore flat at `Re = U·D/nu_num ≈ 122` across every tier, and refining the grid cannot raise it. The floor is the viscous substep budget, `U·D/viscNuMax`, which ranges 0.5 (tier 64) to 131 (tier 1024). At tier 1024 the floor exceeds the ceiling and the window is **empty**; the badge says so rather than offering a setting that does not exist.
- **Shedding onset is near Re ≈ 126 here, not the physical 47.** Measured by wake unsteadiness at tier 256 (steady up to Re ≈ 63, shedding above ≈ 126). The gap is the solver's own numerical viscosity, and it means the Kármán preset only shows a vortex street *outside* the honest window — which is why the app opens badged rather than opening on a flow with no vortices.

The live Strouhal readout is **not implemented**; there is no `St` element and no probe. It remains open work.
