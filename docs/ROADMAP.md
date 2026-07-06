# Roadmap

North star: **visual wow + physics credibility** — gasp in 10 seconds, survive a CFD expert's 2 minutes.

Decided 2026-07-06 (see ADRs 0005–0007 for the load-bearing decisions).

1. **GPU field rendering + 1024 tier** — WebGPU render pass for the Field View (bilinear, colormap LUT texture, solids in-shader); overlays stay Canvas 2D on top. Add 1024 to adaptive tiers; keep red-black Gauss-Seidel, tune iterations by measurement. No multigrid unless 1024 can't hold 60 fps. [ADR-0005]
2. **MacCormack advection** — second-order, min/max limited. Visible payoff requires step 1. [ADR-0006]
3. **Viscosity + Re slider + Strouhal readout** — explicit diffusion pass; Re capped to resolvable regime (~10–5000); live St from a downstream Probe on the Kármán preset. Flagship demo: vortex street dies near Re ≈ 47. [ADR-0007]
4. **Blow mode** — default mouse mode: drag injects momentum + Smoke at the cursor (write both ping-pong buffers).
5. **Freehand Draw mode + ε slider** — rasterize drawn solids into the Solid Mask (needs new invalidation path, eraser); Confinement exposed as labeled-artificial, default-off. [ADR-0006]

Deferred / rejected: GPU compute particles (revisit at ~100× particle counts, ADR-0003), multigrid pressure, 2048 tier, multiple parametric obstacles (subsumed by Draw mode), nominal-Re readout (rejected permanently, ADR-0007).
