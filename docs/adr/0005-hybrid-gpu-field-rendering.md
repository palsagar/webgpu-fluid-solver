---
status: accepted
---

# Hybrid GPU field rendering, overlays stay on Canvas 2D

The Field View moves to a WebGPU render pass: a fragment shader samples the field buffer with bilinear filtering, applies the colormap LUT as a texture, and draws solid cells in-shader. Overlays (streamlines, arrows, particles, obstacle outline) stay on a transparent Canvas 2D element layered on top, fed by the existing velocity readback. This supersedes ADR-0001 for the field only — the per-frame CPU pixel loop was the binding constraint on grid size, and removing it unlocks 1024+ grids, while the readback (needed by overlays regardless) shrinks to overlay cadence.

## Considered options

- **Full GPU pipeline (field + GPU particles)** — rejected for now: bundles two rewrites, and CPU particles (ADR-0003) are not a bottleneck at ~5000 particles. Revisit if particle counts need to grow ~100×.
- **Keep CPU rendering** — rejected: caps the Field View at 512-cell blockiness regardless of numerics quality.

## Consequences

- Two stacked canvases (WebGPU + 2D); pointer events belong to the top layer.
- Solid-mask rendering moves into WGSL — `invalidateSolid()`/solid readback disappears.
- `putImageData`, the CPU colormap LUT loop, and the field staging buffer go away.
