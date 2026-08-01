# Roadmap

North star: **visual wow + physics credibility** — gasp in 10 seconds, survive a CFD expert's 2 minutes.

Hub: [../README.md](../README.md) · Index: [README.md](README.md)

## Shipped milestones

0. **Pressure projection every step** ✅ (2026-07-19) — fixed alternate-step bug. Measurement details in [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md).
1. **GPU field rendering + 1024 tier** ✅ (2026-07-06) — WebGPU field pass, Canvas 2D overlays, 1024 added but manual-only. [ADR-0005](adr/0005-hybrid-gpu-field-rendering.md).
2. **MacCormack advection** ✅ (2026-07-20) — second-order, min/max limited, three dispatches per field. [ADR-0006](adr/0006-honest-numerics-maccormack-over-confinement.md).
3. **Viscosity + Re slider + Strouhal readout** ✅ (2026-07-20) — explicit diffusion with `N = ceil(nu·dt/(0.25 h²))` substeps capped at 32 (saturation, not truncation). Fixed Re slider 0.25–500; badge names bound. Live St from downstream Probe. Window, onset, and St measurements in [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md) and [numerical-methods.md](numerical-methods.md).
4. **GPU-side obstacle rasterization** ✅ (2026-07-27, PR A) — `rasterize_obstacle.wgsl`, boundary mask on GPU, no CPU mirrors. [ADR-0010](adr/0010-gpu-side-obstacle-rasterization.md).
5. **Moving-wall viscous BC** ✅ (2026-07-29, PR B) — `diffuse.wgsl` ghosts against stored wall velocity instead of pinning dragged wall at zero. [ADR-0011](adr/0011-moving-wall-viscous-bc.md).
6. **Onboarding tour** ✅ (this branch, PR #8) — first-visit welcome + 12-step spotlight tour, Replay link, `flowlab.tour.v1` gating. `static/js/tour.js`; `tests/tour.spec.js` (25 tests).
7. **Insert-on-Click** ✅ (this branch, PR #8) — first canvas pointerdown in obstacle-less preset inserts obstacle and sets `showObstacle = true`. `static/js/interaction.js:163-166`.

## Planned

- **Blow mode** — default mouse mode: dragging injects momentum + Smoke at cursor; current Obstacle drag becomes non-default, switched by explicit toggle. Build notes: reuse `writeVelocityU/V/Smoke` (all three rotation slots); no solver change; avoid CPU whole-field uploads per `mousemove`.
- **Freehand Draw mode + ε slider** — rasterize drawn solids into Solid Mask; eraser counterpart. Confinement exposed as labeled-artificial, default-off per [ADR-0006](adr/0006-honest-numerics-maccormack-over-confinement.md). Build notes: lands on PR A's rasterizer; eraser semantics should match PR A's vacated-cell rule (zero velocity/pressure, Smoke = 1.0) unless a later PR argues otherwise.

## Sequencing for the next PRs

- ~~PR A — GPU-side obstacle rasterization~~ ✅ shipped.
- ~~PR B — moving-wall viscous BC~~ ✅ shipped.
- **PR C — Blow mode**.
- **PR D — Draw mode + eraser + ε slider**. If shareable/serialized state is added, re-clamp resolution and iterations on restore.

## Known gaps

Numerical detail is in [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md) and [numerical-methods.md](numerical-methods.md); the gaps below are the register of record. Highlights:
- MacCormack velocity chain leaks stale `i=0`/`j=0` ring into interior (asserted, bounded).
- Domain walls switch from free-slip to no-slip whenever `ν > 0`, shifting measured St.
- Amplitude dependence of `ν_num` unmeasured; off-slice operating points badge `unmeasured`.
- Tier 512 and 1024 converged values extrapolated; tier 1024's honest window is empty for want of iterations, not grid.
- Adaptive upscale threshold calibrated on one machine/display.
- Test gaps: pressure/boundary bind-group slot indexing, `diffuse` `[src][dst]` transposition, several rasterizer hardening assertions, shedding onset not measured in-suite.
- St sensitivity unmeasured across probe position, tier, and preset.
- Several assertions pin the live solver against offline-transcribed reference values (onset 52.2, operating-point `ν_num`, St band) that the suite never recomputes in-suite.
- `resetFlipState` relies on an unenforced probe-clear convention; a caller bypassing `invalidateSolid`/`_updateReBadge` would mix two runs' samples into one Strouhal window.
- Deployment constraints: CSP is `frame-ancestors 'none'` only (no script-src/default-src/connect-src); an unauthenticated `api.github.com` fetch fires on every page load, leaking the visitor IP to GitHub (Referrer-Policy: no-referrer means no Referer is sent).

## Deferred / rejected

GPU compute particles (revisit at ~100× counts, [ADR-0003](adr/0003-cpu-lagrangian-particles.md)), multigrid pressure, 2048 tier, multiple parametric obstacles (subsumed by Draw), nominal-Re readout (rejected, [ADR-0007](adr/0007-explicit-viscosity-bounded-re.md)), smoke diffusion, GPU-side probe ring buffer, uniform-flow bluff-body preset (removed, [ADR-0009](adr/0009-no-wind-tunnel-preset.md)).
