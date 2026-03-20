# WebGPU Eulerian Fluid Solver

Real-time 2D incompressible flow simulation running entirely on the GPU via WebGPU compute shaders. The solver uses an Eulerian (grid-based) approach with a MAC staggered grid, iterative pressure projection, and semi-Lagrangian advection. A 2D canvas renders the output using colormap visualization, streamlines, and velocity arrows. A minimal FastAPI backend serves the static files.

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
        Ren[Renderer — 2D Canvas]
        UI --> Orch
        Orch --> Sim
        Orch --> Ren
    end

    subgraph GPU["WebGPU Device"]
        B[Storage Buffers — u, v, p, s, m + ping-pong pairs]
        C1[integrate.wgsl]
        C2[pressure.wgsl]
        C3[boundary.wgsl]
        C4[advect.wgsl]
        C1 --> B
        C2 --> B
        C3 --> B
        C4 --> B
    end

    S2 -->|HTTP| Browser
    Sim -->|dispatch compute| GPU
    Ren -->|readback buffers| B
```

## Documentation

| Document | Description |
|----------|-------------|
| [System Architecture](architecture.md) | Tech stack, module graph, frame loop, presets |
| [Numerical Methods](numerical-methods.md) | Governing equations, MAC grid, pressure solver, advection |
| [GPU Pipeline](gpu-pipeline.md) | Buffer layout, compute dispatch, bind groups, rendering |

## Quick Start

```bash
uv run uvicorn server:app --port 8000
```

Open `http://localhost:8000` in Chrome 113+ (WebGPU required).
