# FlowLab

Real-time 2D incompressible flow simulation in the browser. Users pick a scenario, watch the flow evolve, and interact with it (drag obstacles, toggle visualizations).

## Language

Some entries below are **target-state vocabulary** — decided but not yet built. They are marked _Not yet implemented_. Use the terms in design discussion; do not assume the feature exists in the code. Everything in [ADR-0007](docs/adr/0007-explicit-viscosity-bounded-re.md) and [ADR-0008](docs/adr/0008-viscous-substepping-and-resolution-aware-window.md) has shipped; what remains unbuilt is Confinement (ε) from [ADR-0006](docs/adr/0006-honest-numerics-maccormack-over-confinement.md) plus the Blow and Draw mouse modes.

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
One red-black Gauss-Seidel (SOR) sweep of the pressure projection. Presets choose how many run per step. Not a free knob: the count sets the delivered Numerical Viscosity, and therefore the top of the honest Reynolds window.
_Avoid_: Jacobi iteration (— the solver is red-black Gauss-Seidel with over-relaxation, not Jacobi)

**MacCormack**:
The advection scheme: a semi-Lagrangian forward trace, a reversed retrace, and a limited combine `phi^{n+1} = phi^ + (phi^n − phi~)/2` clamped to the fluid corners of the departure stencil. Three GPU dispatches per field. It replaced plain semi-Lagrangian advection, whose numerical diffusion it cuts by up to 3x.
_Avoid_: BFECC (— a different scheme, considered and rejected in ADR-0006), "second-order advection" unqualified

**Viscous Substep**:
One application of the explicit five-point diffusion pass. A frame runs `N = ceil(nu·dt/(0.25 h²))` of them, capped at 32, because a single pass at the frame's timestep would violate the explicit stability limit. Past the cap the viscosity saturates rather than the count truncating.
_Avoid_: viscous iteration (— it is a time substep, not an iterative solve)

**Numerical Viscosity**:
The unrequested diffusion the advection scheme and the under-converged pressure solve add on top of the requested viscosity. **Measured, never estimated** — by fitting the decay of a Taylor–Green vortex with physical viscosity off. It is what sets the honest Reynolds ceiling. It is independent of grid spacing and linear in the timestep, i.e. a time-splitting error, not grid diffusion.
_Avoid_: artificial viscosity (— that is a term deliberately added; this one is a defect of the scheme), numerical dissipation

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
A scientific color lookup table mapping scalar values to color. Two are loaded as GPU LUT textures: magma (Smoke) and coolwarm (pressure). `static/colormaps/viridis.png` ships in the repo but nothing loads it.

**Confinement (ε)**:
An explicitly-labeled, default-off control that injects artificial vorticity for visual effect. Always presented as artificial — never silently on.
_Not yet implemented_ (ADR-0006).
_Avoid_: swirl boost, turbulence (— it is neither)

**Resolution Tier**:
One of the discrete Grid sizes (64 / 128 / 256 / 512 / 1024 cells tall). Switched manually or by Adaptive Resolution — except 1024, which is manual-only.

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
A user-controllable physical parameter: the slider sets `nu = U·D/Re` and the viscous pass integrates it. The slider's range is fixed at 0.25–500 and **never moves** — instead a badge names the bound when the requested Re leaves the Honest Window. Never displayed as a nominal/fake value.
_Avoid_: nominal Re (— rejected permanently, ADR-0007)

**Honest Window**:
The Reynolds range a given Resolution Tier and Pressure Iteration count can actually deliver. Floor = the largest viscosity the Viscous Substep budget can integrate; ceiling = where the requested viscosity falls below the measured Numerical Viscosity. Both bounds are measured. When the app has not measured the ceiling at a given operating point it says `unmeasured` rather than quoting a number from a different one.
_Avoid_: Re cap, clamp (— the control is never clamped; the badge is the mechanism)

**Probe**:
A fixed sampling point in the flow whose velocity time-series feeds Diagnostics. Placed 2 diameters downstream of the Obstacle, sampled in simulation time. Returns nothing rather than clamping if that cell would fall outside the valid interior.

**Strouhal Number (St)**:
The dimensionless vortex-shedding frequency `St = f·D/U`, measured live from a Probe — an emergent result, never prescribed. The readout refuses to produce a number for a steady flow, an under-sampled signal, or a collapsed field, and says so.
_Avoid_: "St ≈ 0.2" as a claim about this app (— 0.2 is the high-Re plateau; the measured values here run 0.166–0.200 over Re 55–140)

## Example dialogue

> **Dev**: When the user drags the Obstacle, do we move a mesh?
> **Expert**: No — there is no mesh. Dragging re-rasterizes the Obstacle into the Solid Mask and clears the Smoke left in its old footprint.
> **Dev**: And the Smoke is the thing being simulated?
> **Expert**: No, Smoke is passive — it just rides the velocity field so you can see it. The simulation state is velocity and pressure. Turning Smoke off changes nothing physically.
> **Dev**: So Particles are the same as Smoke?
> **Expert**: Same idea, different representation. Smoke is a field advected per Cell; Particles are individual tracers advected point-by-point from Emitters. Both are Overlay-level visualization, not physics.
