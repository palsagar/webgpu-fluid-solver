// ============================================================================
// Boundary Extrapolation — Copies interior velocities to domain walls
//
// After the pressure solve and advection steps, boundary cells may hold
// stale or zero velocities. This shader extrapolates the nearest interior
// velocity values to the domain edges (Neumann-style zero-gradient BC).
//
// Two entry points handle the two wall orientations:
//   - extrapolate_horizontal: copies u along the bottom (j=0) and top
//     (j=n-1) walls for each column i
//   - extrapolate_vertical: copies v along the left (i=0) and right
//     (i=numX-1) walls for each row j
//
// NOTE: This overwrites any forced wall velocity (e.g., lid-driven cavity
// top wall), which is why that preset is not currently supported.
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,
    dt: f32,
    omega: f32,
    density: f32,
    color: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;  // horizontal velocity
@group(0) @binding(2) var<storage, read_write> v: array<f32>;  // vertical velocity

// extrapolate_horizontal: For each column i, copy u from the first/last
// interior row to the bottom/top boundary row (zero-gradient in y).
// One thread per column.
@compute @workgroup_size(64)
fn extrapolate_horizontal(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = params.numY;
    if (i >= params.numX) { return; }
    // Bottom wall: copy from j=1 to j=0
    u[i * n + 0u] = u[i * n + 1u];
    // Top wall: copy from j=n-2 to j=n-1
    u[i * n + n - 1u] = u[i * n + n - 2u];
}

// extrapolate_vertical: For each row j, copy v from the first/last
// interior column to the left/right boundary column (zero-gradient in x).
// One thread per row.
@compute @workgroup_size(64)
fn extrapolate_vertical(@builtin(global_invocation_id) id: vec3u) {
    let j = id.x;
    let n = params.numY;
    if (j >= n) { return; }
    // Left wall: copy from i=1 to i=0
    v[0u * n + j] = v[1u * n + j];
    // Right wall: copy from i=numX-2 to i=numX-1
    v[(params.numX - 1u) * n + j] = v[(params.numX - 2u) * n + j];
}
