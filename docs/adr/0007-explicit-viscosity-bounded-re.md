---
status: accepted (not yet implemented — see docs/ROADMAP.md)
---

# Explicit viscosity with a bounded Reynolds slider; Strouhal measured, never prescribed

The solver gains an explicit viscous diffusion pass, making Reynolds number physically real and user-controllable. The Re slider is capped to the regime the grid actually resolves (~10–5000): above that, physical viscosity drops below the scheme's numerical diffusion and the label would be a lie (see ADR-0006). The Kármán preset displays a live Strouhal number computed from a downstream velocity probe — measured from the simulation, never hard-coded — so the app demonstrates St ≈ 0.2 and shedding onset near Re ≈ 47 as emergent results.

## Considered options

- **"Nominal Re" readout without a viscous term** — rejected: the solver was inviscid (Euler + numerical diffusion), so any displayed Re would be fabricated.
- **Uncapped Re slider** — rejected: honest only where viscosity is resolved; the cap is the credibility feature.

## Current state

The rejected nominal-Re readout still ships: `ui.js`'s `_updateRe()` computes `Re = U·D/h` into the "Flow Info" panel, and `index.html` documents the caveat in a tooltip. It is scheduled for removal when the viscous pass lands and replaces it with a real Re control — not a defect to delete on sight.
