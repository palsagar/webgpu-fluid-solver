---
status: accepted; MacCormack shipped, Confinement still unbuilt (ROADMAP step 5)
---

# Honest numerics: MacCormack advection; artificial terms labeled and default-off

Advection diffusion is fixed with MacCormack (predictor-corrector, second-order, min/max limited) rather than by turning on vorticity confinement. Confinement is exposed as a user-facing control explicitly labeled artificial (ε), default off. The site presents itself as a flow solver by a CFD practitioner; energy injected by an unlabeled artificial term would undermine exactly the audience the project targets. This is a standing policy, not a one-off: any non-physical term added for looks must be labeled as such in the UI and default to off.

## Considered options

- **Vorticity confinement default-on** (the standard fluid-demo look) — rejected: reads as "screensaver" to viewers who can tell, which is the portfolio's core audience.
- **BFECC** — comparable accuracy to MacCormack at 3 advection passes vs 2 + correction; MacCormack chosen for fewer dispatches.

## Current state

**MacCormack shipped**, for smoke (`maccormack.wgsl`) and for velocity (`maccormack_velocity.wgsl`), as three dispatches per field: forward advect, backward advect, limited combine.

The payoff was measured, not assumed. Against semi-Lagrangian on the identical field, with the projection converged so the advection scheme is what is being compared, the ratio `nu_num(MacCormack) / nu_num(semi-Lagrangian)` is:

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| ratio | 0.329 | 0.674 | 0.872 | 0.951 | 0.984 |

Strictly lower at every tier, and the margin is largest where advection dominates (3.0x at tier 64) and shrinks where the projection residual takes over — the expected ordering, and itself evidence that the measurement resolves the right thing. Numbers and method: [ADR-0008](0008-viscous-substepping-and-resolution-aware-window.md).

Two design outcomes differ from the decision text and are worth recording, because both were forced by the dye inlet passing *through* a solid cell:

- **The limiter's stencil-corner solid test runs on the backward pass only.** Applied to the forward pass it walls the dye out of the domain entirely and the smoke field goes uniformly clear, silently. The same mechanism protects the inflow velocity BC at column `i = 1`, which survives only because `i = 0` is solid.
- **The limiter bounds are seeded with `phi^` and widened by fluid corners only.** The seed is what makes a reverted face collapse to exactly first-order semi-Lagrangian rather than picking up a spurious correction.

**Confinement (ε) is not implemented.** It remains ROADMAP step 5. The standing policy in the decision above — any non-physical term added for looks must be labeled artificial in the UI and default to off — is unchanged and binds whenever it lands.
