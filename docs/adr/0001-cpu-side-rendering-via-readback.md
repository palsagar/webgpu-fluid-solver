---
status: superseded by ADR-0005 (for the Field View; readback-driven overlays remain)
---

# CPU-side rendering via GPU readback and 2D canvas

*Historical record — present tense below describes the pre-ADR-0005 code. The Field View is now a WebGPU render pass; only the overlay readbacks survive.*

The simulation runs entirely in WebGPU compute, but rendering does **not** use a WebGPU render pipeline. Field data is read back to the CPU via staging buffers and drawn with `putImageData`; overlays (streamlines, arrows, particles, obstacles) use the Canvas 2D API. The overlays need CPU-side velocity data regardless (streamline integration, arrow geometry, particle advection), so the readback exists either way; at ≤512-cell grids the extra cost of also drawing the field from CPU is small, and Canvas 2D keeps the renderer simple and debuggable.

## Consequences

- Field image resolution equals grid resolution (browser upscales the canvas).
- Overlay freshness is bounded by the readback cadence (every 10 frames).
- Rendering cost grows with grid size on the CPU side — this is the binding constraint against much larger grids, not compute.
