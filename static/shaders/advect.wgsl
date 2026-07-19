// ============================================================================
// Semi-Lagrangian Advection (velocity) — transports the u and v fields forward
//
// Uses the semi-Lagrangian (backward-trace) method: for each grid point,
// trace a virtual particle backward in time by -dt using the current
// velocity, then bilinearly interpolate the field value at the departure
// point. This is unconditionally stable for any dt.
//
// One entry point:
//   - advect_velocity: advects the u and v fields (writes to buf4=u_new,
//     buf5=v_new). The host rotates which slot is read and which is written.
//
// Smoke lives in advect_smoke.wgsl / maccormack.wgsl: it advects by MacCormack,
// which needs a third buffer binding and so cannot share this module's binding
// block (WGSL forbids two module-scope bindings at the same @group/@binding).
//
// The bilinear sampling functions account for the MAC staggered grid:
//   - u lives on vertical faces → no x offset, h/2 y offset
//   - v lives on horizontal faces → h/2 x offset, no y offset
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,          // cell size
    dt: f32,         // time step
    omega: f32,      // (unused here, shared uniform struct with pressure shader)
    density: f32,    // (unused here)
    color: u32,      // (unused here)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> u: array<f32>;    // horizontal velocity (current)
@group(0) @binding(2) var<storage, read> v: array<f32>;    // vertical velocity (current)
@group(0) @binding(3) var<storage, read> s: array<f32>;    // solid mask
@group(0) @binding(4) var<storage, read_write> buf4: array<f32>;  // u_new (output)
@group(0) @binding(5) var<storage, read_write> buf5: array<f32>;  // v_new (output)

// sample_u: Bilinearly interpolate the u velocity field at an arbitrary
// world-space position (x_in, y_in).
//
// On a MAC grid, u is stored at vertical cell faces: position (i*h, j*h + h/2).
// So there is no x offset (u index aligns with x directly) but y is offset
// by h/2 to reach cell-face centers.
//
// Returns the interpolated horizontal velocity at the given point.
fn sample_u(x_in: f32, y_in: f32) -> f32 {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;
    let n = ny;

    // Clamp to domain interior to avoid out-of-bounds reads
    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    // x index: no offset because u lives on vertical faces at x = i*h
    let x0f = floor(x * h1);
    let x0 = min(u32(x0f), nx - 1u);
    let tx = (x - x0f * h) * h1;        // fractional distance in x
    let x1 = min(x0 + 1u, nx - 1u);

    // y index: subtract h/2 because u is centered at y = j*h + h/2
    let y0f = floor((y - h2) * h1);
    let y0 = min(u32(y0f), ny - 1u);
    let ty = ((y - h2) - y0f * h) * h1;  // fractional distance in y
    let y1 = min(y0 + 1u, ny - 1u);

    // Bilinear interpolation weights
    let sx = 1.0 - tx;
    let sy = 1.0 - ty;

    return sx * sy * u[x0 * n + y0] +
           tx * sy * u[x1 * n + y0] +
           tx * ty * u[x1 * n + y1] +
           sx * ty * u[x0 * n + y1];
}

// sample_v: Bilinearly interpolate the v velocity field at an arbitrary
// world-space position (x_in, y_in).
//
// On a MAC grid, v is stored at horizontal cell faces: position (i*h + h/2, j*h).
// So x is offset by h/2 but y has no offset.
//
// Returns the interpolated vertical velocity at the given point.
fn sample_v(x_in: f32, y_in: f32) -> f32 {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;
    let n = ny;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    // x index: subtract h/2 because v is centered at x = i*h + h/2
    let x0f = floor((x - h2) * h1);
    let x0 = min(u32(x0f), nx - 1u);
    let tx = ((x - h2) - x0f * h) * h1;
    let x1 = min(x0 + 1u, nx - 1u);

    // y index: no offset because v lives on horizontal faces at y = j*h
    let y0f = floor(y * h1);
    let y0 = min(u32(y0f), ny - 1u);
    let ty = (y - y0f * h) * h1;
    let y1 = min(y0 + 1u, ny - 1u);

    let sx = 1.0 - tx;
    let sy = 1.0 - ty;

    return sx * sy * v[x0 * n + y0] +
           tx * sy * v[x1 * n + y0] +
           tx * ty * v[x1 * n + y1] +
           sx * ty * v[x0 * n + y1];
}

// advect_velocity: Semi-Lagrangian advection of u and v fields.
//   buf4 = u_new (output), buf5 = v_new (output)
//
// For each velocity face, traces backward by -dt*(local velocity) and
// samples the current field at the departure point. Skips faces adjacent
// to solid cells (those retain their pre-set values, e.g., inflow BC).
@compute @workgroup_size(8, 8)
fn advect_velocity(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    let h = params.h;
    let h2 = 0.5 * h;
    let dt = params.dt;

    if (i < 1u || i >= params.numX || j < 1u || j >= n) { return; }

    let idx = i * n + j;

    // Default: copy current values (overwritten below for fluid faces)
    buf4[idx] = u[idx];
    buf5[idx] = v[idx];

    // Advect u component at face position (i*h, j*h + h/2)
    // Only advect if both cells sharing this face are fluid
    if (s[idx] != 0.0 && s[(i - 1u) * n + j] != 0.0 && j < n - 1u) {
        let x = f32(i) * h;
        let y = f32(j) * h + h2;
        let cu = u[idx];
        // Interpolate v to the u-face location by averaging four surrounding v-faces
        let cv = (v[(i - 1u) * n + j] + v[idx] +
                  v[(i - 1u) * n + j + 1u] + v[i * n + j + 1u]) * 0.25;
        // Trace backward and sample u at the departure point
        buf4[idx] = sample_u(x - dt * cu, y - dt * cv);
    }

    // Advect v component at face position (i*h + h/2, j*h)
    // Only advect if both cells sharing this face are fluid
    if (s[idx] != 0.0 && s[i * n + j - 1u] != 0.0 && i < params.numX - 1u) {
        let x = f32(i) * h + h2;
        let y = f32(j) * h;
        // Interpolate u to the v-face location by averaging four surrounding u-faces
        let cu = (u[i * n + j - 1u] + u[idx] +
                  u[(i + 1u) * n + j - 1u] + u[(i + 1u) * n + j]) * 0.25;
        let cv = v[idx];
        // Trace backward and sample v at the departure point
        buf5[idx] = sample_v(x - dt * cu, y - dt * cv);
    }
}
