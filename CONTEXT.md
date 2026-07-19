# FlowLab

Real-time 2D incompressible flow simulation in the browser. Users pick a scenario, watch the flow evolve, and interact with it (drag obstacles, toggle visualizations).

## Language

Some entries below are **target-state vocabulary** — decided in [ADR-0006](docs/adr/0006-honest-numerics-maccormack-over-confinement.md) / [ADR-0007](docs/adr/0007-explicit-viscosity-bounded-re.md) but not yet built. They are marked _Not yet implemented_. Use the terms in design discussion; do not assume the feature exists in the code.

### Simulation

**Grid**:
The uniform staggered (MAC) discretization of the rectangular flow domain. Its size is set by a Resolution Tier.

**Cell**:
One unit of the Grid. Every cell is either fluid or solid, per the Solid Mask.

**Solid Mask**:
The per-cell classification of solid vs. fluid. Walls, the step, and Obstacles all exist only as entries in the Solid Mask.
_Avoid_: obstacle mask, boundary mask

**Smoke**:
The passive dye carried by the flow, used purely for visualization. Full concentration is dark; absence is clear.
_Avoid_: dye, marker, density (— "density" means the fluid's physical density, a solver parameter)

**Preset**:
A named, self-contained scenario: solver parameters, Boundary Type, Inflow, optional Obstacle, and default visualization toggles. Current presets: Wind Tunnel, Kármán Vortex, Backward Step.
_Avoid_: scene, demo, example

**Boundary Type**:
The wall/inflow/outflow topology of a Preset — `windTunnel` (open right edge) or `backwardStep` (step geometry on the left).

**Inflow**:
The fixed horizontal velocity injected just inside the left wall, re-applied every frame so the pressure solve cannot drift it.

**Smoke Inlet**:
The band of cells at the left edge where Smoke is re-injected each frame.

**Obstacle**:
A user-draggable solid shape (circle, etc.) rasterized into the Solid Mask. Moving it re-rasterizes the mask and clears stale Smoke in its old footprint.

**Pressure Iteration**:
One red-black Gauss-Seidel (SOR) sweep of the pressure projection. Presets choose how many run per step.
_Avoid_: Jacobi iteration (— the solver is Gauss-Seidel, not Jacobi; a stale comment in `presets.js` says otherwise)

### Visualization

**Field View**:
The colormapped image of one scalar field — Smoke or pressure — filling the canvas.

**Overlay**:
A vector visualization drawn on top of the Field View: Streamlines, velocity arrows, Particles, or the Obstacle outline.

**Streamline**:
A curve everywhere tangent to the instantaneous velocity field. Recomputed when fresh velocity data arrives, drawn from cache in between.

**Particle**:
A massless Lagrangian tracer advected by the flow, leaving a fading trail. Spawned continuously by Emitters.
_Avoid_: sprite, tracer particle

**Emitter**:
A fixed location that continuously spawns Particles (a few per frame), producing a steady visible stream.

**Colormap**:
A scientific color lookup table (magma, viridis, coolwarm) mapping scalar values to color.

**Confinement (ε)**:
An explicitly-labeled, default-off control that injects artificial vorticity for visual effect. Always presented as artificial — never silently on.
_Not yet implemented_ (ADR-0006).
_Avoid_: swirl boost, turbulence (— it is neither)

**Resolution Tier**:
One of the discrete Grid sizes (64 / 128 / 256 / 512 / 1024 cells tall). Switched manually or by Adaptive Resolution.

**Adaptive Resolution**:
Automatic Resolution Tier switching driven by measured frame times: downscale fast when slow, upscale cautiously with a cooldown.

### Interaction

**Blow**:
The default mouse mode: dragging injects momentum and Smoke at the cursor — a moving momentum source, not a special effect.
_Not yet implemented_ — the current default mouse mode drags the Obstacle.
_Avoid_: splat, stir, force brush

**Draw**:
A mouse mode that rasterizes freehand solid shapes into the Solid Mask (with an eraser counterpart). Mouse modes are always switched by explicit toggle, never by implicit gestures.
_Not yet implemented_.

### Diagnostics

**Reynolds Number (Re)**:
A user-controllable physical parameter (via explicit viscosity), valid only within the grid-resolvable range. Never displayed as a nominal/fake value.
_Not yet implemented_ (ADR-0007) — a nominal `Re = U·D/h` readout still ships in the Flow Info panel, to be replaced when the viscous pass lands.

**Probe**:
A fixed sampling point in the flow whose velocity time-series feeds Diagnostics.
_Not yet implemented_ (ADR-0007).

**Strouhal Number (St)**:
The dimensionless vortex-shedding frequency, measured live from a Probe — an emergent result, never prescribed.
_Not yet implemented_ (ADR-0007).

## Example dialogue

> **Dev**: When the user drags the Obstacle, do we move a mesh?
> **Expert**: No — there is no mesh. Dragging re-rasterizes the Obstacle into the Solid Mask and clears the Smoke left in its old footprint.
> **Dev**: And the Smoke is the thing being simulated?
> **Expert**: No, Smoke is passive — it just rides the velocity field so you can see it. The simulation state is velocity and pressure. Turning Smoke off changes nothing physically.
> **Dev**: So Particles are the same as Smoke?
> **Expert**: Same idea, different representation. Smoke is a field advected per Cell; Particles are individual tracers advected point-by-point from Emitters. Both are Overlay-level visualization, not physics.
