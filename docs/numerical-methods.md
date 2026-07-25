# Numerical Methods & Scientific Assumptions

## 1. Governing Equations

The solver implements the incompressible Navier-Stokes equations in two dimensions:

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
```

```math
\nabla \cdot \mathbf{u} = 0
```

where **u** = (u, v) is the velocity field, p is pressure, rho is density, and nu is the kinematic viscosity.

**The viscous term is explicit and real.** `nu` is a solver parameter driven by the Reynolds control (`nu = U·D/Re`), integrated by a five-point Laplacian pass that runs `N = ceil(nu·dt/(0.25 h²))` substeps per frame, capped at 32 — see [§5, Explicit Viscous Diffusion](#5-explicit-viscous-diffusion). Delivered accuracy is measured, not assumed: `nu_delivered / nu_requested = 0.99751` (correlation 0.99927, over 93 312 faces).

The advection scheme also contributes an unrequested numerical viscosity on top of `nu`. That quantity is **measured rather than estimated**, and it is what bounds the range of Reynolds numbers the app will claim — see [§8, Numerical Viscosity](#8-numerical-viscosity-measured). Both bounds and the reasoning behind them are recorded in [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md).

## 2. Staggered MAC Grid

The solver uses a Marker-and-Cell (MAC) staggered grid. Pressure (p), smoke density (m), and the solid flag (s) are stored at cell centers. Horizontal velocity (u) is stored on vertical (left) cell faces, and vertical velocity (v) is stored on horizontal (bottom) cell faces.

```
     i=0        i=1        i=2
   +---v---+---v---+---v---+
   |       |       |       |
j=2  u  p,m,s  u  p,m,s  u  p,m,s
   |       |       |       |
   +---v---+---v---+---v---+
   |       |       |       |
j=1  u  p,m,s  u  p,m,s  u  p,m,s
   |       |       |       |
   +---v---+---v---+---v---+
   |       |       |       |
j=0  u  p,m,s  u  p,m,s  u  p,m,s
   |       |       |       |
   +---v---+---v---+---v---+
```

- `u` sits on the left face of each cell: u_{i,j} is at position (i*h, j*h + h/2)
- `v` sits on the bottom face of each cell: v_{i,j} is at position (i*h + h/2, j*h)
- `p, m, s` sit at cell centers: (i*h + h/2, j*h + h/2)

**Indexing:** Column-major with `idx = i * numY + j`, where i is the column index and j is the row index. The domain has numX x numY cells with cell size `h = 1 / numY`.

**Why staggered grids?** On a collocated grid, the pressure Laplacian stencil couples only every-other cell, creating a null space that permits checkerboard pressure oscillations. On a MAC grid, velocity components live on cell faces and the discrete divergence naturally couples all neighboring pressures, eliminating this mode.

## 3. Operator Splitting

Each time step is split into sequential sub-steps, each implemented as one or more WGSL compute shaders. `step()` encodes them in this order:

```
project (numIters x red/black)      in place on the live velocity slot
extrapolate (boundary H, then V)    in place on the same slot
advect velocity (MacCormack, 3 passes)
advect smoke    (MacCormack, 3 passes)
diffuse x N                          ping-pong between rotation slots
```

Composed across successive calls, the velocity field sees

```
… advect -> diffuse | project -> extrapolate -> advect -> diffuse | project …
```

i.e. **advect → diffuse → project**, the standard splitting. Diffusion acts on the advected, not-yet-projected field, and the next frame's projection removes the divergence it introduced. Diffusion is encoded *after* the smoke passes so that, within the shared command buffer, the smoke passes still read the time-`n` velocity field.

**Projection runs on every step.** It did not always: a latent bug meant it landed on alternate steps only, caught by splitting the per-step `Σ|div|` series into even- and odd-indexed subsequences — the relative imbalance between their means measured **0.384** against a 0.15 threshold. A regression test pins it.

### Step 1: Pressure Projection (`pressure.wgsl`)

The pressure solve enforces the divergence-free constraint. The discrete divergence at cell (i,j) is:

```math
D_{i,j} = u_{i+1,j} - u_{i,j} + v_{i,j+1} - v_{i,j}
```

The solver uses Red-Black Gauss-Seidel with Successive Over-Relaxation (SOR). The pressure correction for cell (i,j) is:

```math
p_{\text{corr}} = -\frac{\omega \cdot D_{i,j}}{s_{\text{total}}}
```

where s_total = s_{i-1,j} + s_{i+1,j} + s_{i,j-1} + s_{i,j+1} counts the number of fluid neighbors (solid cells contribute 0). Pressure is updated as:

```math
p_{i,j} \mathrel{+}= \frac{\rho \cdot h}{\Delta t} \cdot p_{\text{corr}}
```

The four neighboring face velocities are then corrected:

```math
\begin{aligned}
u_{i,j}   &\mathrel{-}= s_{i-1,j} \cdot p_{\text{corr}} \\
u_{i+1,j} &\mathrel{+}= s_{i+1,j} \cdot p_{\text{corr}} \\
v_{i,j}   &\mathrel{-}= s_{i,j-1} \cdot p_{\text{corr}} \\
v_{i,j+1} &\mathrel{+}= s_{i,j+1} \cdot p_{\text{corr}}
\end{aligned}
```

Multiplying by the neighbor's solid flag ensures velocities on solid-wall faces are not modified.

**Red-Black coloring:** Cells where `(i + j) % 2 == 0` are "red"; the rest are "black". Same-color cells never share an edge, so all red cells can be updated in parallel without data races, then all black cells. Each pressure iteration requires two GPU dispatches (one per color). The `params.color` uniform selects which color to process.

**Over-relaxation:** omega = 1.9 (set from JavaScript). Values in (1, 2) accelerate convergence of Gauss-Seidel.

**How many iterations, and why it matters more than expected.** The Kármán preset runs **256**. This is not a cosmetic choice: the iteration count sets the *delivered* numerical viscosity, and therefore the honest Reynolds ceiling (§8). At tier 256, `dt = 1/240`:

| numIters | 80 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|
| `nu_num` | 2.0324e-3 | 1.3448e-3 | 1.1133e-3 | 9.6107e-4 | **7.7808e-4** |
| `Re_max` | 59.04 | 89.23 | 107.79 | 124.86 | **154.23** |

256 was chosen against a measured frame-time rule: at the startup tier it holds median 16.60 ms (60.2 fps), spending ~86% of the 60 fps budget on the dev machine. 128 is the conservative fallback — 7.33 ms, zero dropped frames at 120 Hz, ceiling 89.2 — and is a one-line preset change plus a re-measured table.

Convergence in `Σ|div|` against iteration count, measured with a reset to identical initial conditions per block (**not** sequentially — see §9): mean `Σ|div|` = 1472 / 1245 / 1114 / 1024 / 901 at 20 / 40 / 60 / 80 / 120 iterations. The `max|div|` norm moves the *other way* over the same range (2.73e-1 → 3.36e-1). This measurement sits inside the startup transient and must not be quoted as evidence of a convergence problem; the mechanism is discussed in §9.

### Step 2: Boundary Extrapolation (`boundary.wgsl`)

Two 1D passes copy interior velocities to domain boundaries, enforcing free-slip conditions:

**Horizontal pass** (one thread per column i):
- `u[i, 0] = u[i, 1]` -- bottom boundary
- `u[i, numY-1] = u[i, numY-2]` -- top boundary

**Vertical pass** (one thread per row j):
- `v[0, j] = v[1, j]` -- left boundary
- `v[numX-1, j] = v[numX-2, j]` -- right boundary

This extrapolation copies the nearest interior velocity to the boundary, which enforces zero normal derivative (free-slip / zero-shear). Note: this is also why lid-driven cavity is infeasible -- any forced velocity at a wall boundary gets overwritten by this extrapolation step.

**Caveat: this is not the whole wall boundary condition once `nu > 0`.** The viscous pass reads the domain ring as a ghost (`-center`), which places a zero-velocity wall half a cell outside the domain — a **no-slip** condition. So the domain walls are free-slip while `nu = 0` and no-slip the moment the Re control applies a viscosity. This is disclosed rather than corrected; see §7 and §9.

### Step 3: MacCormack Advection

Advection is MacCormack (predictor–corrector, second-order, min/max limited), built on the same unconditionally stable semi-Lagrangian backtrace. See [§4](#4-maccormack-advection-and-the-limiter).

## 4. MacCormack Advection and the Limiter

Semi-Lagrangian advection is unconditionally stable but strongly diffusive: for each grid point **x** the departure point is `x_d = x - dt·u(x)`, and the bilinear interpolation at `x_d` smears the field by an amount that compounds every step. MacCormack cancels the leading error term with a second, reversed trace.

**Three passes per field**, each a separate dispatch:

| Pass | Direction | Reads | Writes |
|---|---|---|---|
| Forward | `+dt` | `phi^n`, advected by `u^n` | `phi^` (hat) |
| Backward | `-dt` | `phi^` (hat), advected by `u^n` | `phi~` (tilde) |
| Combine | — | `phi^n`, `phi^`, `phi~` | `phi^{n+1}`, in place into tilde |

The sign flip is carried by a second uniform buffer (`uniformBufNegDt`) rather than a second shader. The correction is

```math
\phi^{n+1} = \hat{\phi} + \tfrac{1}{2}\left(\phi^n - \tilde{\phi}\right)
```

`phi~` is where `phi^n` lands after a round trip, so `phi^n - phi~` estimates the round-trip error, and half of it corrects the forward pass.

**The limiter.** The raw correction is not bounded — it can overshoot into new extrema, which for smoke means negative dye and for velocity means spurious energy. The combine clamps `phi^{n+1}` to `[lo, hi]`, where the interval is **seeded with `phi^`** and then widened by the values at the departure stencil's corners. Three details are load-bearing, and each was arrived at by a failure:

- **Seeded with `phi^`, not with the corner range.** The seed guarantees `lo <= phi^ <= hi`, so wherever a face or cell reverted (advection is skipped next to solids) the clamp collapses to the identity and the scheme degrades to exactly first-order semi-Lagrangian rather than picking up a spurious correction. Without the seed the reverted case is inconsistent.
- **Only *fluid* corners widen the interval.** Including solid corners lets stale in-solid values into the bounds; excluding them *without* the seed gives `lo = hi = 1.0` at the inlet, pinning it to clear forever.
- **A separate stencil-corner solid test runs in the advection pass, on the backward pass only**, gated on `params.dt < 0.0` (`advect_smoke.wgsl:133`) — distinct from the combine's fluid-only corner drop above, which is unconditional. The dye inlet is written into column `i = 0`, which is *solid*; cell `i = 1` picks the dye up only because its forward departure point clamps back into that column. A corner test on the forward pass walls the dye out entirely and the smoke field goes uniformly clear, silently. The inflow velocity BC at `i = 1` survives by exactly the same mechanism.

**Velocity uses a separate backward binding for `phi^n`.** Reusing the advect shader for the backward pass would copy reverted faces from its *input* (which is `phi^`), so a reverted face would write `phi~ = phi^` and the correction would be non-zero at precisely the inflow faces. The velocity backward pass therefore keeps `u^n, v^n` bound as the origin field while advecting `phi^`.

**Measured payoff**, against semi-Lagrangian on the identical field with the projection converged:

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `nu_num` semi-Lagrangian | 2.9972e-3 | 1.4578e-3 | 1.1263e-3 | 1.0372e-3 | 1.5373e-3 |
| `nu_num` MacCormack | 9.8575e-4 | 9.8308e-4 | 9.8232e-4 | 9.8680e-4 | 1.5133e-3 |
| ratio | 0.329 | 0.674 | 0.872 | 0.951 | 0.984 |

Strictly lower at every tier. The 3.0x margin at tier 64 is the scheme's real gain; it shrinks at finer tiers because the projection residual, not advection, becomes the dominant error there. Tiers 512 and 1024 are **not converged** — the GPU watchdog cut the iteration escalation off at 4096 — so those two columns are the largest reachable values, not the limit.

## 5. Explicit Viscous Diffusion

`diffuse.wgsl` integrates `du/dt = nu*lap(u)` with a five-point Laplacian:

```math
u^{k+1}_{i,j} = u^{k}_{i,j} + \frac{\nu \, \Delta t_{\text{sub}}}{h^2}\left(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}\right)
```

### Stability and the substep derivation

An explicit five-point Laplacian is stable only for

```math
\frac{\nu \, \Delta t}{h^2} \le \frac{1}{4}
```

A single pass at the frame's `dt` violates this over most of the Re slider's range. Splitting the frame into `N` substeps of `dt/N` each divides the coefficient by `N`, so the smallest `N` that satisfies the limit is

```math
N = \left\lceil \frac{\nu \, \Delta t}{0.25\, h^2} \right\rceil, \qquad N \le N_{\max} = 32
```

**Past the cap, `nu` saturates — `N` is not truncated.** Truncating `N` would leave the coefficient *above* 1/4 and the update divergent: measured at `nu = 0.1`, tier 256, truncation produces **141 811 non-finite interior cells after 60 steps**. Instead `nu` is clamped to

```math
\nu_{\max} = N_{\max} \cdot \tfrac{1}{4} \cdot \frac{h^2}{\Delta t}
```

so the coefficient sits at exactly 1/4 at the ceiling and below it everywhere else — under-diffusive but bounded. `solver.viscClamped` reports this, and it means the **effective Re is higher than the number on the control**, not that the solver is unstable.

`nu_max` is also what sets the **floor** of the honest Reynolds window: `Re_min = U·D/nu_max = 4·U·D·dt/(N_MAX·h²)`, which rises as `h^-2` — four times per tier. At the Kármán operating point that is 0.256 / 1.024 / 4.096 / 16.384 / 65.536 across tiers 64–1024.

### Face classification

The pass classifies each velocity face three ways, matching the face-based convention the advection passes already use (`s[idx] != 0 && s[(i-1)*n+j] != 0`):

| Case | Test | Diffused? | Read as a neighbour |
|---|---|---|---|
| FLUID | both flanking cells fluid | yes | stored value |
| WALL | exactly one flanking cell solid | no, copied through | stored value (the no-penetration BC) |
| BURIED | both flanking cells solid, **or** `i == 0` / `j == 0` | no | ghost `-center` |

The two-case split "solid iff both cells are solid" is *not* the complement of the FLUID test — the gap between them is the wall-normal face, and it matters. Diffusing wall-normal faces against a ghost destroys the inflow BC: measured, it collapses the free stream from 0.93 to 0.10.

The index-based BURIED classification for `i == 0` / `j == 0` is load-bearing, not defence in depth: for `u[.,j=0]` and `v[i=0,.]` both flanking cells *are* fluid, so the FLUID predicate does not block the read, and the mask test itself would underflow at `i-1 == -1`.

### Effect, measured

The ghost places a zero-velocity wall half a cell outside the surface, i.e. **no-slip**. A boundary layer forms and the free stream is untouched:

| | inviscid | viscous | ratio |
|---|---|---|---|
| mean \|u\|, first fluid face off the domain wall | 0.8437 | 0.0358 | 0.042 |
| mean \|u\|, first fluid face off the cylinder | 0.1499 | 0.0930 | 0.620 |
| mean \|u\|, free stream upstream | 0.9283 | 0.9315 | 1.003 |

**Operator fidelity: `nu_delivered / nu_requested = 0.99751`**, correlation 0.99927, over 93 312 faces at `nu = 1.2e-2` (27 substeps). Measured by differencing two single steps from a byte-identical field — one at `nu = 0`, one at `nu = NU` — which share their pressure solve, extrapolation and advection exactly and so differ by precisely the viscous increment. A Taylor–Green decay fit is *not* usable for this: the closed box is analytically free-slip but the ghost makes it no-slip, and the resulting wall layers dominate the KE budget, returning anywhere from 0.25x to 4.2x the true value depending on window margin and run length.

### Substep parity

The pass ping-pongs between two rotation slots, so after `N` substeps the result lands on a different slot depending on `N`'s parity. `step()` therefore publishes the final source slot rather than advancing by a fixed offset — **the first thing in the codebase that makes the velocity and smoke rotation indices diverge**, which is why the smoke bind groups are a 3x3 `[velCur][smokeCur]` table.

## 6. Smoke Transport

Smoke density m is a passive scalar advected by the same MacCormack scheme (`advect_smoke.wgsl` for the two traces, `maccormack.wgsl` for the limited combine). The backtrace velocity at cell center (i*h + h/2, j*h + h/2) is the average of the two adjacent face velocities:

```math
\begin{aligned}
u_{\text{center}} &= (u_{i,j} + u_{i+1,j}) / 2 \\
v_{\text{center}} &= (v_{i,j} + v_{i,j+1}) / 2
\end{aligned}
```

**Convention:** `m = 1.0` means clear (no dye); `m = 0.0` means fully dark dye. The renderer maps m through the magma colormap over a fixed [0, 1] range (no auto-ranging).

**Smoke inlet:** Certain cells at the inflow column are written to `m = 0.0` every frame from JavaScript, continuously injecting dye into the flow. Smoke is **not** diffused — there is no scalar diffusion pass.

## 7. Boundary Conditions

### Solid Walls

Solid cells have `s[i,j] = 0`. The compute passes handle them as follows:

- **Pressure:** Skips solid cells entirely. The s-flag terms in velocity correction prevent modifying velocities on solid faces.
- **Boundary:** Does not check solids (operates on domain edges only).
- **Advection:** Skips faces/cells where an adjacent cell is solid, preserving zero-flux conditions. Its backward pass additionally reverts any cell whose departure stencil touches a solid, gated on `params.dt < 0.0` (`advect_smoke.wgsl:133`). The MacCormack combine's limiter *separately* drops solid corners from its bounds — unconditionally, since the combine only ever runs forward (§4).
- **Diffusion:** Face-classified FLUID / WALL / BURIED (§5). Solid cells are never written, so an obstacle's drag velocity — which *is* the moving-wall BC — survives the viscous pass untouched.

### Inflow

Fixed velocity is imposed at column i=1, re-applied from JavaScript after each `solver.step()` call. Inflow values survive advection because the left wall (i=0) is solid -- the advection condition `s[(i-1)*n + j] != 0` fails at i=1, so the velocity is not overwritten. In the viscous pass the same face is a WALL face and is copied through. Re-application after the solver step prevents drift from the pressure solve.

See [Three-Slot Rotation](gpu-pipeline.md#4-the-three-slot-rotation) for why inflow velocities must be written to every rotation slot.

### Open Outflow

The right boundary uses the boundary extrapolation step (v copied from interior) combined with advection naturally carrying flow out of the domain. `diffuse.wgsl` copies the domain ring (first and last rows and columns) through unchanged, so those boundary lines carry no viscous update and the cells just inside them diffuse against frozen neighbors — which is why the Strouhal probe refuses to sit in the frozen outflow columns or rows (§9).

### Domain walls: free-slip or no-slip, depending on `nu`

The two conditions coexist and the solver switches between them silently:

- **`nu = 0`** — the walls are free-slip, set by `boundary.wgsl`'s zero-gradient extrapolation.
- **`nu > 0`** — `diffuse.wgsl` reads the ring as a ghost, imposing no-slip half a cell outside the domain. Channel wall boundary layers form, they add to the effective blockage, and they thicken as Re falls.

**This is disclosed, not corrected.** It shifts the measured Strouhal number, and because the layer thickness is Re-dependent the shift is not a constant offset. Correcting it would require a blockage calibration nothing in this project has measured, and inventing one is exactly the failure the branch exists to prevent.

### Lid-Driven Cavity (Not Supported)

The boundary extrapolation step copies interior velocities to wall cells, overwriting any forced wall velocity. This makes lid-driven cavity -- which requires a fixed tangential velocity along the top wall -- infeasible without modifying the boundary shader.

## 8. Numerical Viscosity, Measured

| Property | Value |
|---|---|
| Temporal order | First-order (first-order backtrace, first-order operator splitting) |
| Spatial order | Second-order (MacCormack predictor–corrector, min/max limited) |
| Advection stability | Unconditionally stable (no CFL restriction on the backtrace) |
| Viscous stability | Conditional: `nu·dt_sub/h² <= 1/4`, enforced by substepping (§5) |

### How it is measured

A single Taylor–Green mode is an exact **steady** solution of the 2D Euler equations, so with physical viscosity off the correct answer is "nothing happens" and *all* observed decay is numerical. Kinetic energy decays as `exp(-2*nu*k²*t)`, so a least-squares fit of `log(KE)` against `t` returns the scheme's own numerical viscosity directly. The harness is `measureNuNum` in `tests/solver.spec.js`.

Three details make the measurement trustworthy rather than merely reproducible:

- **The seeded field is a closed-box mode.** The obvious phase (`u = A cos(kx) sin(ky)`) is divergence-free but drives fluid into all four walls at full amplitude — the decay it produces is the solver fighting an impossible BC. The phase-swapped field vanishes wall-normal on all four walls, and is fitted to a square sub-box of `numY - 2` cells with a shifted origin so the mode closes on a ~2.2:1 domain.
- **The discrete divergence cancels identically on the MAC grid**, not merely to truncation order. Measured `max|div| = 5.96e-8` — float32 machine zero.
- **Device loss is guarded.** A lost GPU returns zeros from every readback, and zeros filtered out of a fit produce a spectacular-looking result from a dead device. `device.lost` is latched, the initial KE is checked against the analytic value, and any non-finite or non-positive sample throws.

### The measured values

**Converged** — the advection scheme's own floor, with the projection iterated until `nu_num` stops moving. Here `nu_num` is **independent of `h`** and **linear in `dt`**:

| dt | 1/60 | 1/120 | 1/240 | 1/480 |
|---|---|---|---|---|
| `nu_num` (tier 128) | 1.6951e-3 | 9.8309e-4 | 5.1419e-4 | 2.6009e-4 |
| `nu_num` (tier 256) | 1.7019e-3 | 9.8650e-4 | 5.1518e-4 | 2.6075e-4 |

Successive ratios 1.72 / 1.91 / 1.98 → 2, and the two tiers agree to three digits at every `dt`. So the app carries a single constant:

```
NU_NUM_PER_DT = 0.121354        nu_num(dt) = NU_NUM_PER_DT * dt
```

anchored at the converged **5.0564e-4 at dt = 1/240** (r² = 0.99994), giving a flat scheme ceiling of **Re = 237.32** at every tier.

> **On the ~2% gap with the table above.** The direct dt-sweep reads `nu_num(1/240)` as 5.14–5.15e-4 (→ Re ≈ 233), while the shipped `NU_NUM_PER_DT` comes from the linear `nu_num`-vs-`dt` regression anchored at dt = 1/240, which lands at **5.0564e-4** (→ **Re 237.32**). The two are separate measurements and the ~2% difference is inter-sweep scatter, not a discrepancy to resolve. **The fitted 5.0564e-4 / Re 237.32 is the number of record** — it is what `diagnostics.js` ships, what drives the badge, and what ADR-0008 and the ROADMAP quote — so every downstream reference uses it rather than the table's directly-read column.

**This is an operator-splitting error in time, not grid diffusion.** Refining the grid cannot lower it; only reducing `dt` can. An earlier design assumed the opposite — that the ceiling is set by resolving the cylinder boundary layer and therefore *rises* with resolution — and that model disagrees with measurement by 8.3x to 746x and in the opposite direction. [ADR-0008](adr/0008-viscous-substepping-and-resolution-aware-window.md) records the reversal.

**At the shipped operating point** (`numIters = 256`, `dt = 1/240`), the projection is converged only at the two coarse tiers, so the *delivered* `nu_num` rises with resolution and the delivered ceiling falls:

| tier | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `nu_num` | 5.0801e-4 | 5.0777e-4 | 7.7808e-4 | 2.4536e-3 | 7.2698e-3 |
| `Re_max` | 236.22 | 236.33 | **154.23** | 48.91 | 16.51 |
| r² | 0.99994 | 0.99994 | 0.99986 | 0.99860 | 0.98926 |

Combined with the substep floor from §5, the honest window at each tier is 0.26–236 / 1.02–236 / **4.10–154** / 16.4–48.9 / empty. The UI never clamps the Re slider to these; it shows a badge naming which bound was crossed and why.

### What is not measured

- **Amplitude dependence.** Every fit used `A = 1.0`. `windTunnel` (`U = 2.0`) and `backwardStep` (`U = 1.5`) run outside that, so their ceiling is approximate and both show an `unmeasured` badge rather than a number.
- **Tier 512 and 1024 converged values** are extrapolated from the flatness at 64–256; the browser died at 4096 iterations.
- **Fit-window sensitivity** is roughly ±10% systematic (at tier 256 converged: 1.02e-3 / 9.87e-4 / 9.36e-4 at 120 / 300 / 600 steps). The window is fixed at 300 steps for every number above.
- All values are from **one Apple M-series GPU in float32**; the SOR residual floor is hardware-dependent.

## 9. Strouhal Measurement, and the Error Mode That Made It Hard

The Kármán preset reports a live Strouhal number `St = f·D/U`, recovered from the solver's own wake.

**The probe** samples one velocity cell **2 diameters downstream** of the obstacle, every 10 steps, in *simulation* time (a wall-clock stamp would jitter and would not be a whole multiple of `dt`). `probeCell()` returns `null` rather than clamping when the target cell falls in the frozen outflow columns or rows, because a clamped probe silently stops being "2 diameters downstream" and the number it produced would no longer mean what the readout says.

**Three refusals** stand between the raw series and a displayed number, each mutation-tested:

- **Amplitude gate.** Below an RMS of `0.02·U` the readout says `steady — no shedding`. Without it, pure noise yields a confident-looking St of 0.645 — 3.2x the true value.
- **Resolution gate.** Fewer than 4 samples per period returns `under-sampled`. The threshold is measured, not textbook: feeding the shipped detector synthetic wakes at known St, the error stays ≤3% above 2.25 samples/period for a pure tone but only above **3.10** for a harmonic-rich, still-growing wake — which is what the probe actually sees. Below the cliff the error jumps to 10–20%, not gracefully.
- **Collapsed-field gate.** A constant series returns `no-signal`, not a physics verdict read off a dead buffer.

### The measured results

**Shedding onset: Re_c = 52.2 ± 0.3.** **Measured St** on saturation-verified wakes (115 s of simulation time per point, tier 256, `dt = 1/240`, 256 iterations):

| Re | 55* | 57.5 | 60 | 65 | 74.8 (default) | 100 | 140 |
|---|---|---|---|---|---|---|---|
| **St** | 0.166 | 0.168 | 0.170 | 0.173 | **0.180** | 0.190 | 0.200 |
| ± | 0.001 | 0.001 | 0.0000 | 0.001 | 0.001 | 0.0000 | 0.001 |

\* Re 55's amplitude had not fully saturated; its frequency had.

**St is not ≈ 0.2 in this range, and should not be.** 0.2 is the high-Re plateau (Re ≳ 300). These values sit **11–25% above** Roshko's unconfined `St = 0.212(1 − 21.2/Re)` and converge toward it as Re rises and the wall layers thin — the direction blockage predicts for a cylinder occupying 12% of a no-slip channel. Onset likewise sits above the textbook unconfined ~47 for the same reason, plus the staircased cylinder.

### The error mode

This is the most transferable thing in this document.

**Four times on this project a number was produced by settling too briefly near a bifurcation or inside a transient, and twice those numbers were recorded as results before being caught:**

1. The pressure-iteration table (§3) was **phase-confounded** — five configurations measured sequentially on one solver instance, so time evolution was read as the effect under study. Reproducibility did *not* rule this out: a deterministic simulation reproduces the artifact byte-identically. Only a reset to identical initial conditions per block does.
2. A shedding onset of **Re ≈ 126**, from a fixed 3.33 s settle.
3. Its replacement, **Re ≈ 57.5**, from a 30 s growth ratio — the same defect, one order smaller.
4. A Strouhal table quoted to three significant figures on signals still growing by up to 271x across their own measurement window.

**The fix is to stop reading an amplitude over a fixed window and measure a property of the flow instead.** For the onset: fit the perturbation's exponential growth rate `sigma` from identical initial conditions plus one identical deterministic kick, and find where `sigma(Re)` crosses zero. `sigma` does not depend on how long the point ran.

| Re | 44 | 47 | 50 | 52 | 54 | 56 |
|---|---|---|---|---|---|---|
| `sigma` (1/s) | −0.3772 | −0.2159 | −0.0975 | −0.0036 | +0.0771 | +0.1526 |
| SE | 0.0026 | 0.0024 | 0.0017 | 0.00018 | 0.00022 | 0.0013 |

`sigma` is linear in Re (`dsigma/dRe ≈ 0.042`, r² > 0.996) — a textbook Hopf bifurcation — crossing zero at **52.2**. The quoted uncertainty is the full spread across 53 combinations of fit subset and analysis-window variant (52.02 … 52.39), which dwarfs any single standard error.

**Why 57.5 was produced, quantitatively: at Re 52 the e-folding time is 278 s.** Across any 30 s window Re 52 is indistinguishable from a saturated limit cycle. No fixed-window amplitude criterion, however carefully applied, could have found this.
