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

```bash
# Clone and run
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

## Credits

Based on an original CPU fluid solver by Sagar Pal (2024). Rewritten for WebGPU with compute shaders.
