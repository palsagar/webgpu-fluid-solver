# FlowLab — Documentation Index

Real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders. Eulerian grid-based solver: MAC staggered grid, iterative pressure projection, MacCormack advection, explicit viscous diffusion with automatic substepping. Hybrid rendering: WebGPU field pass + Canvas 2D overlays.

Every user-visible number is measured; see [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md). Requires Chrome 113+ with WebGPU.

Hub: [../README.md](../README.md)

## Documentation

| Document | What it covers |
|---|---|
| [System Architecture](architecture.md) | Stack, module graph, frame loop, presets, adaptive resolution, particles |
| [GPU Pipeline](gpu-pipeline.md) | Buffer layout, three-slot rotation, compute dispatches, bind groups, rendering |
| [Numerical Methods](numerical-methods.md) | Equations, MAC grid, pressure solver, MacCormack, diffusion, measured numerical viscosity, Strouhal |
| [Roadmap](ROADMAP.md) | Shipped milestones, planned Blow/Draw/Confinement, known gaps |
| [Decision Records](adr/README.md) | 11 ADRs covering shipped and rejected decisions |
| [Project Vocabulary](../CONTEXT.md) | Canonical terms and what to call them |

## Tour & Tests

The onboarding tour (`static/js/tour.js`) is covered by `tests/tour.spec.js` (25 tests). The full Playwright suite is 104/104: 78 solver/diagnostics/render/perf-hud + 25 tour + 1 insert-on-click.
