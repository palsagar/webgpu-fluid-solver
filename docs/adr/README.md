# Architecture Decision Records

Each ADR records one load-bearing decision and why the alternatives lost. "Implemented" tracks whether the decision is in the shipped code — a decision can be settled without being built yet.

| # | Title | Status | Implemented |
|---|-------|--------|-------------|
| [0001](0001-cpu-side-rendering-via-readback.md) | CPU-side rendering via GPU readback and 2D canvas | Superseded by 0005 (Field View only) | Partly — overlay readbacks remain; the CPU field path is gone |
| [0002](0002-explicit-bind-group-layouts.md) | Explicit bind group layouts instead of `layout: 'auto'` | Accepted | Yes |
| [0003](0003-cpu-lagrangian-particles.md) | CPU-side Lagrangian particles instead of a GPU particle pass | Accepted | Yes |
| [0004](0004-no-lid-driven-cavity-preset.md) | No lid-driven cavity preset | Accepted | Yes |
| [0005](0005-hybrid-gpu-field-rendering.md) | Hybrid GPU field rendering, overlays stay on Canvas 2D | Accepted | Yes |
| [0006](0006-honest-numerics-maccormack-over-confinement.md) | Honest numerics: MacCormack advection; artificial terms labeled and default-off | Accepted | Partly — MacCormack shipped; Confinement (ε) is [ROADMAP](../ROADMAP.md) step 5 |
| [0007](0007-explicit-viscosity-bounded-re.md) | Explicit viscosity with a bounded Reynolds slider; Strouhal measured, never prescribed | Accepted, partially superseded by 0008 | Yes — several of its numbers corrected by measurement |
| [0008](0008-viscous-substepping-and-resolution-aware-window.md) | Viscous substepping and a resolution-aware honest window | Accepted | Yes |
| [0009](0009-no-wind-tunnel-preset.md) | Removed third preset | Accepted | Yes |
| [0010](0010-gpu-side-obstacle-rasterization.md) | GPU-side obstacle rasterization; no CPU field mirrors | Accepted | No — spec'd, [ROADMAP](../ROADMAP.md) PR A |

The vocabulary in [CONTEXT.md](../../CONTEXT.md) covers both shipped and target-state features. What remains target-state is Confinement (ε) from ADR-0006, plus the Blow and Draw mouse modes — [ROADMAP](../ROADMAP.md) steps 4–5. Everything in ADRs 0007 and 0008 has shipped.
