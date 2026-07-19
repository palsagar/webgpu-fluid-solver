# 🌀 FlowLab — Technical Documentation

Real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders. The solver uses an Eulerian (grid-based) approach with a MAC staggered grid, iterative pressure projection, and semi-Lagrangian advection. Rendering is hybrid: a WebGPU render pass draws the colormapped field straight from the simulation buffers, and a transparent 2D canvas above it carries the overlays — streamlines, velocity arrows, tracer particles, and the obstacle outline.

## System Overview

```mermaid
graph TD
    subgraph Server["FastAPI Server"]
        S1[uvicorn / server.py]
        S2[static/ directory]
        S1 -->|serves| S2
    end

    subgraph Browser["Browser Client"]
        UI[UI Controls & Presets]
        Orch[JS Orchestrator — main.js]
        Sim[FluidSolver — fluid-solver.js]
        Ren[Renderer — overlay canvas + readbacks]
        FRen[FieldRenderer — WebGPU render pass]
        UI --> Orch
        Orch --> Sim
        Orch --> Ren
        Ren --> FRen
    end

    subgraph GPU["WebGPU Device"]
        B[Storage Buffers — u, v, p, s, m + ping-pong pairs]
        C2[pressure.wgsl]
        C3[boundary.wgsl]
        C4[advect.wgsl]
        C5[render_field.wgsl]
        C2 --> B
        C3 --> B
        C4 --> B
    end

    S2 -->|HTTP| Browser
    Sim -->|dispatch compute| GPU
    FRen -->|render pass, reads| B
    FRen --> C5
    Ren -->|readback velocity / solid / pressure| B
```

## Documentation

| Document | Description |
|----------|-------------|
| [System Architecture](architecture.md) | Tech stack, module graph, frame loop, presets, particle tracer |
| [Numerical Methods](numerical-methods.md) | Governing equations, MAC grid, pressure solver, advection |
| [GPU Pipeline](gpu-pipeline.md) | Buffer layout, compute dispatch, bind groups, rendering |
| [Decision Records](adr/README.md) | Index of ADRs — what was decided, and what has actually shipped |

## Quick Start

```bash
uv run uvicorn server:app --port 8000
```

Open `http://localhost:8000` in Chrome 113+ (WebGPU required).
