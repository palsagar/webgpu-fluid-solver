# FlowLab

Real-time 2D incompressible flow simulation in the browser. Users pick a scenario, watch the flow evolve, and interact with it (drag obstacles, toggle visualizations).

Hub: [README.md](README.md) · Index: [docs/README.md](docs/README.md)

## Language

Some entries are **target-state vocabulary** — decided but not yet built, marked _Not yet implemented_. Use them in design discussion; do not assume the feature exists. Everything in [ADR-0007](docs/adr/0007-explicit-viscosity-bounded-re.md) and [ADR-0008](docs/adr/0008-viscous-substepping-and-resolution-aware-window.md) has shipped; what remains unbuilt is Confinement (ε) from [ADR-0006](docs/adr/0006-honest-numerics-maccormack-over-confinement.md) plus Blow and Draw mouse modes.

### Simulation

**Grid**: uniform staggered (MAC) discretization. Size set by Resolution Tier.

**Cell**: one unit of the Grid; fluid or solid per Solid Mask.

**Solid Mask**: per-cell solid vs fluid classification. Walls, step, and Obstacles exist only here. _Avoid_ "obstacle mask". Solid Mask = Boundary Mask + Obstacle footprint.

**Boundary Mask**: permanent solids of a Preset (walls, step), i.e. Solid Mask minus Obstacle footprint. Uploaded to GPU once per preset load; obstacle rasterizer restores vacated cells to it and never carves it ([ADR-0010](docs/adr/0010-gpu-side-obstacle-rasterization.md)).

**Smoke**: passive dye for visualization. _Avoid_ dye, marker, density.

**Preset**: named scenario with solver params, Boundary Type, Inflow, optional Obstacle, default viz toggles. Current presets: Kármán Vortex (obstacle visible) and Backward Step (ships obstacle-less; first click inserts the obstacle). _Avoid_ scene, demo, example.

**Boundary Type**: wall/inflow/outflow topology — `windTunnel` (open right edge) or `backwardStep` (step on left).

**Inflow**: fixed horizontal velocity injected just inside left wall, re-applied each frame.

**Smoke Inlet**: left-edge band where Smoke is re-injected each frame.

**Obstacle**: user-draggable solid shape rasterized into Solid Mask. Dragging re-rasterizes on GPU ([ADR-0010](docs/adr/0010-gpu-side-obstacle-rasterization.md)): new footprint carved with drag velocity, old footprint restored from Boundary Mask, stale Smoke cleared; outside the union bounding box the live field is untouched. A drag ending re-rasterizes with zero velocity so stored wall velocity does not outlive motion ([ADR-0011](docs/adr/0011-moving-wall-viscous-bc.md)). In obstacle-less presets the first canvas pointerdown inserts the obstacle at the click point and reveals it (`static/js/interaction.js:163-166`).

**Pressure Iteration**: one red-black Gauss-Seidel (SOR) sweep of pressure projection. Preset count sets delivered Numerical Viscosity and honest Reynolds ceiling. _Avoid_ Jacobi iteration.

**MacCormack**: advection scheme: semi-Lagrangian forward trace, reversed retrace, limited combine `phi^{n+1} = phi^ + (phi^n − phi~)/2` clamped to fluid corners of departure stencil. Three GPU dispatches per field. _Avoid_ BFECC (different scheme, rejected in [ADR-0006](docs/adr/0006-honest-numerics-maccormack-over-confinement.md)); avoid unqualified "second-order advection".

**Viscous Substep**: one explicit five-point diffusion pass. Frame runs `N = ceil(nu·dt/(0.25 h²))` capped at 32; past the cap viscosity saturates rather than count truncating. _Avoid_ viscous iteration.

**Wall Ghost**: value substituted for a buried face in Viscous Substep. Buried by Solid Mask ghosts to `w + (w − center)`; buried by index on `i=0`/`j=0` ring ghosts to `−center` because stored value is stale ([ADR-0011](docs/adr/0011-moving-wall-viscous-bc.md)). Stationary wall (`w=0`) reduces to `−center`.

**Numerical Viscosity**: unrequested diffusion from advection and under-converged pressure solve. **Measured, never estimated**, by fitting Taylor–Green decay with physical viscosity off. Sets honest Reynolds ceiling. Independent of grid spacing and linear in timestep — time-splitting error, not grid diffusion. _Avoid_ artificial viscosity, numerical dissipation.

### Visualization

**Field View**: colormapped scalar image (Smoke or pressure) filling canvas.

**Overlay**: vector viz on top — Streamlines, velocity arrows, Particles, Obstacle outline.

**Streamline**: curve tangent to instantaneous velocity field; recomputed when fresh velocity arrives.

**Particle**: massless Lagrangian tracer advected by flow, leaving fading trail. Spawned by Emitters. _Avoid_ sprite, tracer particle.

**Emitter**: fixed location that continuously spawns Particles.

**Colormap**: scientific color LUT. Loaded as GPU textures: magma (Smoke), coolwarm (pressure). `static/colormaps/viridis.png` ships but is unused.

**Confinement (ε)**: labeled, default-off artificial vorticity control. _Not yet implemented_ ([ADR-0006](docs/adr/0006-honest-numerics-maccormack-over-confinement.md)). _Avoid_ swirl boost, turbulence.

**Resolution Tier**: discrete Grid height (64/128/256/512/1024). Switched manually or by Adaptive Resolution — 1024 is manual-only.

**Adaptive Resolution**: automatic tier switching driven by measured frame times.

### Interaction

**Blow**: default mouse mode (target): dragging injects momentum and Smoke at cursor — moving momentum source, not special effect. _Not yet implemented_; current default drags Obstacle. _Avoid_ splat, stir, force brush.

**Draw**: mouse mode that rasterizes freehand solids into Solid Mask (with eraser). _Not yet implemented_. Mouse modes switch by explicit toggle, never implicit gesture.

**Onboarding Tour**: first-visit welcome modal + 12-step spotlight tour. Highlights live controls via a four-dim-rect spotlight hole; do-it steps detected from DOM events. "Replay the Tour" link in the guide modal footer. Gated by localStorage flag `flowlab.tour.v1`; reduced-motion aware. Implementation: `static/js/tour.js`; tests: `tests/tour.spec.js` (25 tests).

**Insert-on-Click**: in an obstacle-less preset (Backward Step), the first canvas pointerdown inserts the obstacle at the click point and sets `interaction.showObstacle = true`, so the overlay ring, shape buttons, and Re/St badges reflect the body now in the flow (`static/js/interaction.js:163-166`). Shift+mousemove rotation remains guarded while `showObstacle` is false.

### Diagnostics

**Reynolds Number (Re)**: user control; slider sets `nu = U·D/Re` and viscous pass integrates it. Range fixed 0.25–500, never moves; badge names the bound when requested Re leaves Honest Window. Never displayed as nominal/fake value. _Avoid_ nominal Re (rejected permanently, [ADR-0007](docs/adr/0007-explicit-viscosity-bounded-re.md)).

**Honest Window**: Reynolds range a given tier and Pressure Iteration count can deliver. Floor = largest viscosity Viscous Substep budget can integrate; ceiling = `U·D/ν_num`. Both measured; absent measurement says `unmeasured`. _Avoid_ Re cap, clamp — control is never clamped; badge is the mechanism.

**Probe**: fixed downstream sampling point whose velocity time-series feeds Diagnostics. Placed 2 diameters downstream of Obstacle in simulation time. Returns nothing rather than clamping if outside valid interior.

**Strouhal Number (St)**: dimensionless vortex-shedding frequency `St = f·D/U`, measured live from Probe. Refuses a number for steady flow, under-sampled signal, or collapsed field. _Avoid_ "St ≈ 0.2" as claim about this app (0.2 is high-Re plateau; measured values run 0.166–0.200 over Re 55–140).
