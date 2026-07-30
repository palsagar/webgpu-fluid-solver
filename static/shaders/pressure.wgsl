// ============================================================================
// Pressure Solver — Gauss-Seidel with SOR (Successive Over-Relaxation)
//
// Enforces the incompressibility constraint (div(u) = 0) by iteratively
// adjusting velocities and pressure. Uses a red-black coloring scheme so
// that even/odd cells can be solved in parallel without data races. The
// host dispatches this shader twice per iteration: once for "red" cells
// (color=0) and once for "black" cells (color=1).
//
// Grid layout: MAC (Marker-and-Cell) staggered grid where u lives on
// vertical cell faces and v lives on horizontal cell faces. Pressure and
// the solid mask (s) are cell-centered. s=0 means solid, s=1 means fluid.
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,          // cell size (spacing)
    dt: f32,         // time step
    omega: f32,      // SOR relaxation factor (1 < omega < 2 for over-relaxation)
    density: f32,    // fluid density
    color: u32,      // red-black coloring: 0 or 1
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;  // horizontal velocity (staggered, on vertical faces)
@group(0) @binding(2) var<storage, read_write> v: array<f32>;  // vertical velocity (staggered, on horizontal faces)
@group(0) @binding(3) var<storage, read> s: array<f32>;        // solid mask: 0 = solid, 1 = fluid
@group(0) @binding(4) var<storage, read_write> p: array<f32>;  // pressure (cell-centered)

// normalize: Subtract a fixed interior reference from every pressure cell.
// Pressure is only defined up to an additive constant; this pins the gauge
// so the SOR residual does not drift into the constant mode over many steps.
//
// The reference cell is skipped here and zeroed by `normalize_ref` in a
// separate dispatch. Reading p[refIdx] while another invocation writes it
// in the same pass is a data race; keeping the reference read-only in this
// pass means every invocation sees the pre-pass value consistently.
@compute @workgroup_size(8, 8)
fn normalize(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i >= params.numX || j >= n) { return; }
    let idx = i * n + j;
    let refIdx = (params.numX / 2u) * n + (n / 2u);
    if (idx == refIdx) { return; }
    p[idx] -= p[refIdx];
}

// normalize_ref: Zero the reference cell. Runs as a single-thread dispatch
// after `normalize` so the reference is not read and written in the same pass.
@compute @workgroup_size(1, 1)
fn normalize_ref(@builtin(global_invocation_id) id: vec3u) {
    let n = params.numY;
    let refIdx = (params.numX / 2u) * n + (n / 2u);
    p[refIdx] = 0.0;
}

// clear: Zero the entire pressure field. Used when the obstacle has been
// teleported far enough that the previous pressure initial guess is more
// corrupting than helpful, giving the projection solve a fresh start.
@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i >= params.numX || j >= n) { return; }
    let idx = i * n + j;
    p[idx] = 0.0;
}

// main: One Gauss-Seidel SOR iteration for a single cell (i, j).
// Each thread handles one cell. Skips boundary cells, solid cells,
// and cells whose color doesn't match the current pass.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;

    // Skip boundary layer (outermost ring of cells)
    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }
    // Red-black ordering: only process cells matching the current color pass
    if ((i + j) % 2u != params.color) { return; }

    let idx = i * n + j;
    // Skip solid cells
    if (s[idx] == 0.0) { return; }

    // Neighbor solid flags — used both as weights and to skip fully-enclosed cells
    let sx0 = s[(i - 1u) * n + j];
    let sx1 = s[(i + 1u) * n + j];
    let sy0 = s[i * n + j - 1u];
    let sy1 = s[i * n + j + 1u];
    let sTotal = sx0 + sx1 + sy0 + sy1;
    if (sTotal == 0.0) { return; }  // cell surrounded by solids — nothing to solve

    // Velocity divergence across the cell: (u_right - u_left) + (v_top - v_bottom)
    let div = u[(i + 1u) * n + j] - u[idx] + v[i * n + j + 1u] - v[idx];
    // SOR pressure correction: scale divergence by relaxation factor
    let pCorr = -div / sTotal * params.omega;

    // Update pressure from velocity correction (p = rho * h / dt * correction)
    p[idx] += params.density * params.h / params.dt * pCorr;

    // Push velocities on fluid-facing faces to reduce divergence
    // Solid neighbors (s=0) zero out their contribution automatically
    u[idx]               -= sx0 * pCorr;
    u[(i + 1u) * n + j] += sx1 * pCorr;
    v[idx]               -= sy0 * pCorr;
    v[i * n + j + 1u]   += sy1 * pCorr;
}
