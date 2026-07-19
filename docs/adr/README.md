# Architecture Decision Records

Each ADR records one load-bearing decision and why the alternatives lost. "Implemented" tracks whether the decision is in the shipped code — a decision can be settled without being built yet.

| # | Title | Status | Implemented |
|---|-------|--------|-------------|
| [0001](0001-cpu-side-rendering-via-readback.md) | CPU-side rendering via GPU readback and 2D canvas | Superseded by 0005 (Field View only) | Partly — overlay readbacks remain; the CPU field path is gone |
| [0002](0002-explicit-bind-group-layouts.md) | Explicit bind group layouts instead of `layout: 'auto'` | Accepted | Yes |
| [0003](0003-cpu-lagrangian-particles.md) | CPU-side Lagrangian particles instead of a GPU particle pass | Accepted | Yes |
| [0004](0004-no-lid-driven-cavity-preset.md) | No lid-driven cavity preset | Accepted | Yes |
| [0005](0005-hybrid-gpu-field-rendering.md) | Hybrid GPU field rendering, overlays stay on Canvas 2D | Accepted | Yes |
| [0006](0006-honest-numerics-maccormack-over-confinement.md) | Honest numerics: MacCormack advection; artificial terms labeled and default-off | Accepted (not yet implemented) | No — [ROADMAP](../ROADMAP.md) step 2 |
| [0007](0007-explicit-viscosity-bounded-re.md) | Explicit viscosity with a bounded Reynolds slider; Strouhal measured, never prescribed | Accepted (not yet implemented) | No — [ROADMAP](../ROADMAP.md) step 3 |

The vocabulary in [CONTEXT.md](../../CONTEXT.md) covers both shipped and target-state features; ADRs 0006 and 0007 are the target-state half.
