# Numerical Methods & Scientific Assumptions

Hub: [../README.md](../README.md) · Index: [README.md](README.md) · Details: [ADR-0006](adr/0006-honest-numerics-maccormack-over-confinement.md), [ADR-0007](adr/0007-explicit-viscosity-bounded-re.md), [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md) · Pipeline: [gpu-pipeline.md](gpu-pipeline.md) · Architecture: [architecture.md](architecture.md)

## 1. Governing Equations

The solver integrates the 2D incompressible Navier–Stokes equations:

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
```

```math
\nabla \cdot \mathbf{u} = 0
```

`ν` is physical viscosity, set by the Reynolds control (`ν = U·D/Re`), and integrated explicitly by the viscous pass. The advection scheme contributes its own numerical viscosity, which is measured and used to bound the honest Reynolds window — see [§9](#9-numerical-viscosity-measured) and [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md).

## 2. Staggered MAC Grid

Pressure `p`, smoke `m`, and solid flag `s` live at cell centers. Horizontal velocity `u` lives on vertical (left) faces; vertical velocity `v` lives on horizontal (bottom) faces.

```
     i=0        i=1        i=2
   +---v---+---v---+---v---+
   |       |       |       |
j=2  u p,m,s u p,m,s u p,m,s
   |       |       |       |
   +---v---+---v---+---v---+
```

- `u_{i,j}` at `(i·h, j·h + h/2)`
- `v_{i,j}` at `(i·h + h/2, j·h)`
- `p, m, s_{i,j}` at `(i·h + h/2, j·h + h/2)`

**Indexing:** column-major `idx = i * numY + j`, domain `numX × numY`, cell size `h = 1 / numY`.

## 3. Operator Splitting

`step()` order:

```
project (numIters × red/black)
extrapolate (boundary H, then V)
advect velocity (MacCormack, 3 passes)
advect smoke    (MacCormack, 3 passes)
diffuse × N
```

Across successive frames the effective splitting is **advect → diffuse → project**. Diffusion runs after smoke so the smoke passes still read the time-`n` velocity field in the same command buffer.

**Projection runs on every step.** A latent bug once made it land on alternate steps; splitting `Σ|div|` into even/odd subsequences showed a relative imbalance of **0.384** against a 0.15 threshold. A regression test pins it.

### Pressure Projection (`pressure.wgsl`)

Discrete divergence:

```math
D_{i,j} = u_{i+1,j} - u_{i,j} + v_{i,j+1} - v_{i,j}
```

Red-Black SOR pressure correction:

```math
p_{\text{corr}} = -\frac{\omega \cdot D_{i,j}}{s_{\text{total}}}
```

```math
p_{i,j} \mathrel{+}= \frac{\rho \cdot h}{\Delta t} \cdot p_{\text{corr}}
```

Velocity correction (multiplied by neighbor solid flags):

```math
\begin{aligned}
u_{i,j}   &\mathrel{-}= s_{i-1,j} \cdot p_{\text{corr}} \\
u_{i+1,j} &\mathrel{+}= s_{i+1,j} \cdot p_{\text{corr}} \\
v_{i,j}   &\mathrel{-}= s_{i,j-1} \cdot p_{\text{corr}} \\
v_{i,j+1} &\mathrel{+}= s_{i,j+1} \cdot p_{\text{corr}}
\end{aligned}
```

Over-relaxation: `ω = 1.9`. The Kármán preset runs **numIters = 256**.

| numIters | 80 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|
| `nu_num` | 2.0324e-3 | 1.3448e-3 | 1.1133e-3 | 9.6107e-4 | **7.7808e-4** |
| `Re_max` | 59.04 | 89.23 | 107.79 | 124.86 | **154.23** |

## 4. Boundary Extrapolation

Two 1D passes copy interior velocities to domain boundaries, enforcing zero-gradient (free-slip):

- Horizontal pass: `u[i,0] = u[i,1]`, `u[i,numY-1] = u[i,numY-2]`
- Vertical pass: `v[0,j] = v[1,j]`, `v[numX-1,j] = v[numX-2,j]`

This overwrites any forced wall velocity, so **lid-driven cavity is not supported**.

Once `ν > 0`, the viscous pass reads the domain ring as a ghost (`-center`), placing a zero-velocity wall half a cell outside the domain — i.e. the walls become **no-slip**. The solver therefore switches from free-slip to no-slip silently when viscosity is active.

## 5. MacCormack Advection

MacCormack advection is predictor–corrector, second-order, min/max limited, built on semi-Lagrangian backtraces. Three passes per field:

| Pass | Direction | Reads | Writes |
|---|---|---|---|
| Forward | `+dt` | `φ^n` | `φ^` (hat) |
| Backward | `-dt` | `φ^` | `φ~` (tilde) |
| Combine | — | `φ^n`, `φ^`, `φ~` | `φ^{n+1}` |

```math
\phi^{n+1} = \hat{\phi} + \tfrac{1}{2}\left(\phi^n - \tilde{\phi}\right)
```

The combine clamps `φ^{n+1}` to `[lo, hi]`, seeded with `φ^` and widened only by **fluid** stencil corners. Seeding with `φ^` makes reverted faces degrade to first-order semi-Lagrangian exactly. The stencil-corner solid test runs on the **backward pass only**, gated on `params.dt < 0.0` (`advect_smoke.wgsl:133`); the dye inlet at solid column `i = 0` survives only because the forward pass does not test it. Velocity keeps `u^n, v^n` bound as the origin field on the backward pass so reverted faces do not inject `φ^` into the corrector.

Measured `nu_num(MacCormack) / nu_num(semi-Lagrangian)`: 0.329 / 0.674 / 0.872 / 0.951 / 0.984 at tiers 64 / 128 / 256 / 512 / 1024 (strictly lower at every tier). Details and derivation: [ADR-0006](adr/0006-honest-numerics-maccormack-over-confinement.md), [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md).

## 6. Explicit Viscous Diffusion

`diffuse.wgsl` integrates `∂u/∂t = ν∇²u` with a five-point Laplacian:

```math
u^{k+1}_{i,j} = u^{k}_{i,j} + \frac{\nu \, \Delta t_{\text{sub}}}{h^2}\left(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}\right)
```

Stability requires `ν·dt_sub/h² ≤ 1/4`. With `N` substeps per frame:

```math
N = \left\lceil \frac{\nu \, \Delta t}{0.25\, h^2} \right\rceil, \qquad N \le N_{\max} = 32
```

Past the cap, `ν` saturates — `N` is **not** truncated. Truncating leaves the coefficient above 1/4 and diverges: at `ν = 0.1`, tier 256, truncation produced **141 811 non-finite interior cells after 60 steps**. Saturation clamps `ν` to

```math
\nu_{\max} = N_{\max} \cdot \tfrac{1}{4} \cdot \frac{h^2}{\Delta t}
```

The honest Re floor is `Re_min = U·D/ν_max = 4·U·D·dt/(N_MAX·h²)`, rising as `h⁻²`: 0.256 / 1.024 / 4.096 / 16.384 / 65.536 across tiers 64–1024 at the Kármán operating point.

### Face classification

| Case | Test | Diffused? | Neighbour |
|---|---|---|---|
| FLUID | both flanking cells fluid | yes | stored value |
| WALL | exactly one flanking cell solid | no, copied through | stored value |
| BURIED (mask) | both flanking cells solid | no | `w + (w - center)` from wall velocity (ADR-0011) |
| BURIED (index / top-row `u`) | `i == 0` / `j == 0`, or top-row `u` (`j == numY - 1`) | no | ghost `-center` |

Diffusing wall-normal faces against a ghost collapses the free stream from 0.93 to 0.10.

### Effect and fidelity

The ghost places a no-slip wall half a cell outside the surface:

| | inviscid | viscous | ratio |
|---|---|---|---|
| mean \|u\|, first fluid face off domain wall | 0.8437 | 0.0358 | 0.042 |
| mean \|u\|, first fluid face off cylinder | 0.1499 | 0.0930 | 0.620 |
| mean \|u\|, free stream upstream | 0.9283 | 0.9315 | 1.003 |

**Operator fidelity: `ν_delivered / ν_requested = 0.99751`**, correlation 0.99927, over 93 312 faces at `ν = 1.2e-2` (27 substeps). Measured by differencing two single steps from a byte-identical field (`ν = 0` vs `ν = NU`), which share pressure, extrapolation, and advection exactly. A Taylor–Green decay fit is unusable here: the closed box is analytically free-slip but the ghost makes it no-slip, so wall layers dominate the KE budget and return 0.25×–4.2× the true value.

### Substep parity

The pass ping-pongs between two rotation slots; `step()` publishes the final source slot so velocity and smoke rotation indices can diverge (hence the `[velCur][smokeCur]` 3×3 bind-group table; see [gpu-pipeline.md](gpu-pipeline.md)). An obstacle dragged against the top row has its buried u-faces ghosted to `-center` (stationary), matching ADR-0011's disclosed limitation.

## 7. Smoke Transport

Smoke density `m` is a passive scalar advected by the same MacCormack scheme. Cell-center backtrace velocity is the average of adjacent face velocities:

```math
\begin{aligned}
u_{\text{center}} &= (u_{i,j} + u_{i+1,j}) / 2 \\
v_{\text{center}} &= (v_{i,j} + v_{i,j+1}) / 2
\end{aligned}
```

**Convention:** `m = 1.0` is clear, `m = 0.0` is fully dark. The renderer uses the magma colormap over a fixed `[0, 1]` range. Dye is injected at the inflow column (`m = 0.0`) from JavaScript every frame. Smoke is **not** diffused.

## 8. Boundary Conditions

- **Solids.** `s[i,j] = 0`. Pressure skips them; advection skips faces/cells adjacent to solids and reverts on backward pass if the departure stencil touches a solid (`params.dt < 0.0`). Diffusion handles them via the FLUID/WALL/BURIED face split (§6).
- **Inflow.** Fixed velocity at column `i = 1`, re-applied from JavaScript after each `step()`. It survives advection because `i = 0` is solid, and survives diffusion because the face is a WALL face copied through. See [gpu-pipeline.md](gpu-pipeline.md#4-the-three-slot-rotation) for rotation-slot handling.
- **Outflow.** Right boundary uses the boundary extrapolation pass plus advection carrying flow out. `diffuse.wgsl` copies the domain ring through unchanged, so the last columns are frozen.

The domain walls are free-slip at `ν = 0` and no-slip at `ν > 0` (§4). This is disclosed, not corrected; correcting it requires a blockage calibration that has not been measured.

## 9. Numerical Viscosity, Measured

| Property | Value |
|---|---|
| Temporal order | First-order (backtrace + operator splitting) |
| Spatial order | Second-order (MacCormack predictor–corrector, min/max limited) |
| Advection stability | Unconditionally stable |
| Viscous stability | Conditional: `ν·dt_sub/h² ≤ 1/4` via substepping |

### How it is measured

A phase-swapped Taylor–Green mode is seeded in a closed square sub-box of `numY - 2` cells, with physical viscosity off. Kinetic energy decays as `exp(-2·ν·k²·t)`; the fitted decay rate is the scheme's own numerical viscosity. The harness is `measureNuNum` in `tests/solver.spec.js`. Discrete divergence cancels to machine zero on the MAC grid (`max|div| = 5.96e-8`).

### Measured values

At the shipped operating point (`dt = 1/240`):

```
NU_NUM_PER_DT = 0.121354        nu_num(dt) = NU_NUM_PER_DT * dt
```

anchored at `nu_num = 5.0564e-4` (r² = 0.99994), giving a flat **scheme ceiling Re = 237.32** at every tier.

The **projection ceiling** actually delivered by `numIters = 256`:

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `nu_num` | 5.0801e-4 | 5.0777e-4 | 7.7808e-4 | 2.4536e-3 | 7.2698e-3 |
| `Re_max` | 236.22 | 236.33 | **154.23** | 48.91 | 16.51 |
| r² | 0.99994 | 0.99994 | 0.99986 | 0.99860 | 0.98926 |

The app quotes the **binding** of the two ceilings. The full honest window is 0.26–236 / 1.02–236 / **4.10–154** / 16.4–48.9 / empty across tiers 64–1024.

### What is not measured

- **Amplitude dependence.** Fits used `A = 1.0`; `backwardStep` (`U = 1.5`) shows `unmeasured`.
- **Tier 512/1024 converged values** are extrapolated from tiers 64–256; 4096 iterations killed the browser.
- **Hardware dependence.** Values are from one Apple M-series GPU in float32; the SOR residual floor shifts elsewhere.

Caveats. The fit window is fixed at 300 steps for every quoted `nu_num`. Window sensitivity is about ±10% systematic: tier-256 converged fits give 1.02e-3 / 9.87e-4 / 9.36e-4 at 120 / 300 / 600 steps. Dead-device safeguards (`device.lost`, analytic initial-KE check, positive-finite-sample check) stop a lost GPU from returning zeros that would masquerade as a flat fit.

## 10. Strouhal Measurement

The Kármán preset reports a live Strouhal number `St = f·D/U` from a downstream velocity probe.

### Probe and refusals

The probe samples one velocity cell **2 diameters downstream** of the obstacle every 10 steps in *simulation* time. `probeCell()` returns `null` if the target falls in frozen outflow columns/rows, rather than clamping silently.

Three mutation-tested refusals:

- **Amplitude gate.** RMS below `0.02·U` returns `steady — no shedding`. Without it, noise yields `St = 0.645` (3.2× truth).
- **Resolution gate.** Fewer than 4 samples/period returns `under-sampled`; the threshold is a 1.3× margin over the measured 3.10 error cliff for harmonic-rich, still-growing wakes.
- **Collapsed-field gate.** A constant series returns `no-signal`.

### Measured St

On saturation-verified wakes (tier 256, `dt = 1/240`, 256 iterations):

| Re | 55* | 57.5 | 60 | 65 | 74.8 (default) | 100 | 140 |
|---|---|---|---|---|---|---|---|
| **St** | 0.166 | 0.168 | 0.170 | 0.173 | **0.180** | 0.190 | 0.200 |
| ± | 0.001 | 0.001 | 0.0000 | 0.001 | 0.001 | 0.0000 | 0.001 |

\* Re 55's amplitude had not fully saturated; its frequency had.

St is **11–25% above** Roshko's unconfined correlation and converges toward it as Re rises — the direction blockage predicts for a cylinder occupying 12% of a no-slip channel.

### Shedding onset

**Re_c = 52.2 ± 0.3.** Perturbation growth rate `σ` from identical impulsive starts plus an identical kick:

| Re | 44 | 47 | 50 | 52 | 54 | 56 |
|---|---|---|---|---|---|---|
| `σ` (1/s) | −0.3772 | −0.2159 | −0.0975 | −0.0036 | +0.0771 | +0.1526 |

`σ` is linear in Re (`dσ/dRe ≈ 0.042`, r² > 0.996); the crossing is the Hopf bifurcation point. Uncertainty is the full spread across 53 fit-subset/window variants (52.02 … 52.39).

### The four-error lesson

Four times a number was produced by settling too briefly near a bifurcation or inside a transient, and twice it was recorded before being caught:

1. The `numIters` table (§3) was measured sequentially on one solver instance, confounding time evolution with the effect. Resetting to identical ICs per block halved the apparent effect and reversed the max-norm trend.
2. Onset **Re ≈ 126** from a fixed 3.33 s settle.
3. Onset **Re ≈ 57.5** from a 30 s growth ratio — same defect, one order smaller.
4. A Strouhal table quoted to three significant figures on signals still growing up to 271× across their window.

The fix: stop reading amplitude over a fixed window and measure a property of the flow. At Re 52 the e-folding time is **278 s**, so any fixed window treats the transient as a limit cycle. Growth-rate fitting removes the run-length dependence.
