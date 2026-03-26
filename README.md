<div align="center">

# 🌀 FlowLab

**Real-time fluid dynamics in your browser, powered by WebGPU**

Drag obstacles through the flow. Watch vortices form. Explore pressure fields, streamlines, and smoke visualization — all running at 800+ fps on the GPU.

[![WebGPU](https://img.shields.io/badge/WebGPU-Compute_Shaders-blue)](https://www.w3.org/TR/webgpu/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

</div>

## Features

- **GPU-accelerated solver** — Red-Black Gauss-Seidel pressure projection + semi-Lagrangian advection, all in WGSL compute shaders
- **Interactive obstacles** — Drag circles, squares, airfoils, or wedges through the fluid with velocity coupling
- **Multiple visualizations** — Smoke dye (magma colormap), pressure field (viridis), streamlines, velocity arrows
- **Curated presets** — Wind tunnel, Karman vortex street, backward-facing step
- **Advanced controls** — Adjust timestep, relaxation, iterations, inflow velocity, grid resolution
- **Adaptive resolution** — Auto-scales grid from 64 to 512 based on frame rate
- **800+ fps** at 256x256 on modern GPUs

## Quick Start

**With Docker** (no Python installation required):

```bash
git clone https://github.com/palsagar/webgpu-fluid-solver.git
cd webgpu-fluid-solver
docker build -t flowlab .
docker run -p 8000:8000 flowlab
```

**With Python**:

```bash
git clone https://github.com/palsagar/webgpu-fluid-solver.git
cd webgpu-fluid-solver
uv run uvicorn server:app --port 8000
```

Open `http://localhost:8000` in Chrome 113+ (WebGPU required).

## How It Works

The solver implements a staggered MAC grid with:

1. **Gravity integration** — applies body forces to the velocity field
2. **Pressure projection** — Red-Black Gauss-Seidel with SOR enforces incompressibility
3. **Boundary extrapolation** — copies interior velocities to boundary cells
4. **Semi-Lagrangian advection** — backtraces particles through the velocity field with bilinear interpolation

All four steps run as WebGPU compute shaders dispatched ~84 times per frame (4 + 2 x 40 iterations). Data stays on the GPU — the CPU only reads back field values for visualization via async staging buffers.

## Project Structure

```
server.py                  # FastAPI server (~10 lines)
static/
  index.html               # UI shell
  css/style.css             # Dark theme
  js/
    main.js                 # Entry point, animation loop
    fluid-solver.js         # GPU buffer management, compute dispatch
    renderer.js             # 2D canvas rendering, colormaps, overlays
    interaction.js           # Mouse/touch drag, shape rasterization
    presets.js               # Preset configurations
    ui.js                    # DOM bindings, sliders, keyboard shortcuts
    adaptive.js              # Frame-rate-based resolution scaling
  shaders/
    integrate.wgsl           # Gravity/force integration
    pressure.wgsl            # Red-Black Gauss-Seidel pressure solver
    boundary.wgsl            # Boundary extrapolation
    advect.wgsl              # Semi-Lagrangian advection (velocity + smoke)
  colormaps/
    viridis.png              # Scientific colormaps (256x1 LUT textures)
    coolwarm.png
    magma.png
```

## Browser Requirements

WebGPU support required: Chrome 113+, Edge 113+, or Firefox Nightly with `dom.webgpu.enabled` flag.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `P` | Play / Pause |
| `M` | Step one frame |
| `1-3` | Load preset |

## Documentation

For detailed technical documentation, see the **[Documentation Hub](docs/README.md)** — covering system architecture, numerical methods, and the GPU compute pipeline.

## Background

During my early PhD days (~2018) I wrote a 2D incompressible Navier-Stokes solver in Fortran 90 + CUDA — roughly 7k lines of code that sat on a dusty hard drive for years. The core numerics are standard CFD: staggered MAC grid, Gauss-Seidel pressure solve, semi-Lagrangian advection with operator splitting.

In 2026 I decided to see how far [Claude Code](https://claude.ai) (Anthropic's Opus 4.6) could take it. First pass: port the whole thing from Fortran 90 / CUDA V8 to modern C++20 / CUDA 12. The result was surprisingly solid — it handled the staggered grid data structures, pressure projection, and CUDA kernel modernization with minimal hand-holding.

That went well enough that I wanted to push further: remap the entire compute and rendering pipeline from C++/CUDA onto WebGPU, so the solver runs entirely in the browser using the client-side GPU for all the heavy lifting. No server-side compute, no WASM — just vanilla JS orchestrating WGSL compute shaders. What made it fun was figuring out the WebGPU-specific patterns — explicit bind group layouts (auto-layout silently drops unused bindings), ping-pong buffer management, and getting velocity readback performant enough for CPU-side particle tracing.

The key insight was that WebGPU's compute shader model maps naturally onto the same data-parallel patterns — workgroup dispatches over a uniform grid, storage buffer ping-pong for advection, red-black coloring for the pressure solve — that made the CUDA implementation effective. What changes is not the mathematics but the deployment model: instead of batch runs on a cluster, the simulation runs interactively at hundreds of frames per second on commodity hardware, with the user as an active participant — dragging obstacles, injecting tracer particles, and observing flow phenomena like vortex shedding and recirculation in real time.
