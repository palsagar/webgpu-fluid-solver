# Numerical Methods & Scientific Assumptions

## 1. Governing Equations

The solver implements the incompressible Navier-Stokes equations in two dimensions:

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \mathbf{g}
```

```math
\nabla \cdot \mathbf{u} = 0
```

where **u** = (u, v) is the velocity field, p is pressure, rho is density, and **g** is gravitational acceleration (applied in the vertical direction only).

**Key simplification:** There is no explicit viscosity (nu) term. The semi-Lagrangian advection scheme introduces numerical diffusion that acts as an implicit viscosity proportional to h^2 / dt, where h is the cell size and dt is the time step. The effective Reynolds number therefore depends on grid resolution -- finer grids produce less numerical diffusion and higher effective Re.

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

Each time step is split into four sequential sub-steps, each implemented as a separate WGSL compute shader. This operator-splitting approach decouples the physics into independently solvable stages.

### Step 1: Force Integration (`integrate.wgsl`)

Gravity is applied as a forward Euler update to the vertical velocity component:

```math
v_{i,j}^* = v_{i,j}^n + g \cdot \Delta t
```

The update is applied only where both adjacent cells are fluid (`s[i,j] != 0` and `s[i, j-1] != 0`), since v_{i,j} sits on the face between cells (i,j) and (i, j-1). Interior cells only -- the shader skips the boundary ring (`i < 1`, `i >= numX`, `j < 1`, `j >= numY - 1`).

Horizontal velocity u is unchanged (no horizontal body forces).

### Step 2: Pressure Projection (`pressure.wgsl`)

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

**Over-relaxation:** omega = 1.9 (set from JavaScript). Values in (1, 2) accelerate convergence of Gauss-Seidel; 1.9 is near-optimal for typical grid sizes.

### Step 3: Boundary Extrapolation (`boundary.wgsl`)

Two 1D passes copy interior velocities to domain boundaries, enforcing free-slip conditions:

**Horizontal pass** (one thread per column i):
- `u[i, 0] = u[i, 1]` -- bottom boundary
- `u[i, numY-1] = u[i, numY-2]` -- top boundary

**Vertical pass** (one thread per row j):
- `v[0, j] = v[1, j]` -- left boundary
- `v[numX-1, j] = v[numX-2, j]` -- right boundary

This extrapolation copies the nearest interior velocity to the boundary, which enforces zero normal derivative (free-slip / zero-shear). Note: this is also why lid-driven cavity is infeasible -- any forced velocity at a wall boundary gets overwritten by this extrapolation step.

### Step 4: Semi-Lagrangian Advection (`advect.wgsl`)

Advection uses the unconditionally stable semi-Lagrangian method. For each grid point **x**, the departure point is found by backtracing along the velocity:

```math
\mathbf{x}_d = \mathbf{x} - \Delta t \cdot \mathbf{u}(\mathbf{x})
```

The new field value is then sampled at x_d via bilinear interpolation.

**Velocity advection** (`advect_velocity`): Two separate advections for u and v. The backtrace velocity at each face center is computed by averaging neighboring components of the other velocity:

- For u at face (i*h, j*h + h/2): the backtrace u-velocity is u[i,j] directly; the backtrace v-velocity is the average of the four surrounding v-values: `(v[i-1,j] + v[i,j] + v[i-1,j+1] + v[i,j+1]) * 0.25`.
- For v at face (i*h + h/2, j*h): the backtrace v-velocity is v[i,j] directly; the backtrace u-velocity is the average of the four surrounding u-values: `(u[i,j-1] + u[i,j] + u[i+1,j-1] + u[i+1,j]) * 0.25`.

Advection is skipped when either adjacent cell is solid (`s` check), preserving wall boundary conditions.

**Bilinear sampling** accounts for the stagger offset: `sample_u` offsets by h/2 in y (since u lives on vertical faces), and `sample_v` offsets by h/2 in x (since v lives on horizontal faces). Coordinates are clamped to the domain interior.

## 4. Smoke Transport

Smoke density m is a passive scalar advected by the same semi-Lagrangian method (`advect_smoke` entry point). The backtrace velocity at cell center (i*h + h/2, j*h + h/2) is the average of the two adjacent face velocities:

```math
\begin{aligned}
u_{\text{center}} &= (u_{i,j} + u_{i+1,j}) / 2 \\
v_{\text{center}} &= (v_{i,j} + v_{i,j+1}) / 2
\end{aligned}
```

**Convention:** `m = 1.0` means clear (no dye); `m = 0.0` means fully dark dye. The renderer maps m through the magma colormap over a fixed [0, 1] range (no auto-ranging).

**Smoke inlet:** Certain cells at the inflow column are written to `m = 0.0` every frame from JavaScript, continuously injecting dye into the flow.

## 5. Boundary Conditions

### Solid Walls

Solid cells have `s[i,j] = 0`. All four compute shaders handle them:

- **Integrate:** Skips faces where either adjacent cell is solid.
- **Pressure:** Skips solid cells entirely. The s-flag terms in velocity correction prevent modifying velocities on solid faces.
- **Boundary:** Does not check solids (operates on domain edges only).
- **Advection:** Skips faces/cells where an adjacent cell is solid, preserving zero-flux conditions.

### Inflow

Fixed velocity is imposed at column i=1, re-applied from JavaScript after each `solver.step()` call. Inflow values survive advection because the left wall (i=0) is solid -- the advection condition `s[(i-1)*n + j] != 0` fails at i=1, so the velocity is not overwritten. Re-application after the solver step prevents drift from the pressure solve.

See [Ping-Pong Buffers](gpu-pipeline.md#4-ping-pong-buffers) for why inflow velocities must be written to both GPU buffers.

### Open Outflow

The right boundary uses the boundary extrapolation step (v copied from interior) combined with semi-Lagrangian advection naturally carrying flow out of the domain.

### Lid-Driven Cavity (Not Supported)

The boundary extrapolation step copies interior velocities to wall cells, overwriting any forced wall velocity. This makes lid-driven cavity -- which requires a fixed tangential velocity along the top wall -- infeasible without modifying the boundary shader.

## 6. Stability & Accuracy

| Property | Value |
|---|---|
| Temporal order | First-order (Forward Euler integration + first-order backtrace) |
| Spatial order | Second-order (bilinear interpolation at departure points) |
| Stability | Unconditionally stable (semi-Lagrangian advection has no CFL restriction) |

**Numerical viscosity:** The semi-Lagrangian backtrace with bilinear interpolation introduces artificial diffusion:

```math
\nu_{\text{eff}} \sim O\!\left(\frac{h^2}{\Delta t}\right)
```

This means the effective Reynolds number increases with grid resolution (smaller h) and larger time steps. At typical simulation parameters, the numerical viscosity dominates any physical viscosity that might be desired.

**Reynolds number display:** The UI shows an approximate Reynolds number computed as:

```math
\text{Re} \approx \frac{U \cdot D}{h}
```

where U is the inflow velocity, D is a characteristic length (obstacle diameter), and h is the cell size (standing in for the effective numerical viscosity, since nu_eff is O(h) in the relevant regime). This is an order-of-magnitude estimate, not a precise calculation.
