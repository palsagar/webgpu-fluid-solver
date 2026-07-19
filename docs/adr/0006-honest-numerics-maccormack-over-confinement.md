---
status: accepted (not yet implemented — see docs/ROADMAP.md)
---

# Honest numerics: MacCormack advection; artificial terms labeled and default-off

Advection diffusion is fixed with MacCormack (predictor-corrector, second-order, min/max limited) rather than by turning on vorticity confinement. Confinement is exposed as a user-facing control explicitly labeled artificial (ε), default off. The site presents itself as a flow solver by a CFD practitioner; energy injected by an unlabeled artificial term would undermine exactly the audience the project targets. This is a standing policy, not a one-off: any non-physical term added for looks must be labeled as such in the UI and default to off.

## Considered options

- **Vorticity confinement default-on** (the standard fluid-demo look) — rejected: reads as "screensaver" to viewers who can tell, which is the portfolio's core audience.
- **BFECC** — comparable accuracy to MacCormack at 3 advection passes vs 2 + correction; MacCormack chosen for fewer dispatches.
