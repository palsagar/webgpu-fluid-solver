// ============================================================================
// Semi-Lagrangian advection of the staggered velocity field.
//
// This is one MacCormack sub-pass. It serves as BOTH the forward pass
// (positive dt, fu/fv == u/v == phi^n) and the backward pass (negative dt via
// a separate uniform buffer, fu/fv == phi^, u/v == phi^n). The combine lives
// in maccormack_velocity.wgsl -- it needs a different meaning for bindings
// 3..6, and WGSL forbids two module-scope bindings at the same @group/@binding,
// so it cannot share this module.
//
// The role split is what makes the backward pass possible: bindings 1/2 are the
// ADVECTING velocity and simultaneously phi^n, while 4/5 are the field being
// ADVECTED. On the forward pass the same buffers are bound to both (aliasing
// two read-only bindings is legal); on the backward pass 4/5 carry phi^.
// Because 1/2 stay at phi^n on both passes, the unreliable-trace revert below
// writes phi^n on both passes -- no `mOrig` binding of the kind advect_smoke.wgsl
// needs. A revert therefore makes the correction term (phi^n - phi~)/2 vanish
// exactly, which is what preserves the inflow BC at i=1 (see the face guards).
//
// The bilinear stencils account for the MAC staggered grid:
//   - u lives on vertical faces   -> no x offset, h/2 y offset
//   - v lives on horizontal faces -> h/2 x offset, no y offset
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,          // cell size
    dt: f32,         // time step (negated for the MacCormack backward pass)
    omega: f32,      // (unused here)
    density: f32,    // (unused here)
    color: u32,      // (unused here)
    nu: f32,         // (unused here)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> u: array<f32>;    // u^n: advecting velocity AND phi^n
@group(0) @binding(2) var<storage, read> v: array<f32>;    // v^n: advecting velocity AND phi^n
@group(0) @binding(3) var<storage, read> s: array<f32>;    // solid mask
@group(0) @binding(4) var<storage, read> fu: array<f32>;   // u field being advected (== u on the forward pass)
@group(0) @binding(5) var<storage, read> fv: array<f32>;   // v field being advected (== v on the forward pass)
@group(0) @binding(6) var<storage, read_write> outU: array<f32>;
@group(0) @binding(7) var<storage, read_write> outV: array<f32>;

// Grid indices and weights for a bilinear stencil at (x_in, y_in).
struct Stencil {
    i0: u32, i1: u32, j0: u32, j1: u32,
    tx: f32, ty: f32,
}

// Stencil for the u field, which is stored at vertical cell faces:
// position (i*h, j*h + h/2). No x offset; y offset of h/2.
fn u_stencil(x_in: f32, y_in: f32) -> Stencil {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    let x0f = floor(x * h1);
    let y0f = floor((y - h2) * h1);

    var st: Stencil;
    st.i0 = min(u32(x0f), nx - 1u);
    st.i1 = min(st.i0 + 1u, nx - 1u);
    st.j0 = min(u32(y0f), ny - 1u);
    st.j1 = min(st.j0 + 1u, ny - 1u);
    st.tx = (x - x0f * h) * h1;
    st.ty = ((y - h2) - y0f * h) * h1;
    return st;
}

// Stencil for the v field, which is stored at horizontal cell faces:
// position (i*h + h/2, j*h). x offset of h/2; no y offset.
fn v_stencil(x_in: f32, y_in: f32) -> Stencil {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    let x0f = floor((x - h2) * h1);
    let y0f = floor(y * h1);

    var st: Stencil;
    st.i0 = min(u32(x0f), nx - 1u);
    st.i1 = min(st.i0 + 1u, nx - 1u);
    st.j0 = min(u32(y0f), ny - 1u);
    st.j1 = min(st.j0 + 1u, ny - 1u);
    st.tx = ((x - h2) - x0f * h) * h1;
    st.ty = (y - y0f * h) * h1;
    return st;
}

// Departure point of the u-face at (i*h, j*h + h/2), traced by -params.dt.
// A negative params.dt turns this into the forward trace the backward pass needs.
// v is averaged from the four v-faces surrounding the u-face.
// Reads v at j+1 and i-1, so callers must hold 1 <= i and 1 <= j < numY-1.
fn u_departure(i: u32, j: u32) -> vec2f {
    let n = params.numY;
    let h = params.h;
    let h2 = 0.5 * h;
    let idx = i * n + j;
    let cu = u[idx];
    let cv = (v[(i - 1u) * n + j] + v[idx] +
              v[(i - 1u) * n + j + 1u] + v[i * n + j + 1u]) * 0.25;
    return vec2f(f32(i) * h - params.dt * cu,
                 f32(j) * h + h2 - params.dt * cv);
}

// Departure point of the v-face at (i*h + h/2, j*h), traced by -params.dt.
// u is averaged from the four u-faces surrounding the v-face.
// Reads u at i+1 and j-1, so callers must hold 1 <= j and 1 <= i < numX-1.
fn v_departure(i: u32, j: u32) -> vec2f {
    let n = params.numY;
    let h = params.h;
    let h2 = 0.5 * h;
    let idx = i * n + j;
    let cu = (u[i * n + j - 1u] + u[idx] +
              u[(i + 1u) * n + j - 1u] + u[(i + 1u) * n + j]) * 0.25;
    let cv = v[idx];
    return vec2f(f32(i) * h + h2 - params.dt * cu,
                 f32(j) * h - params.dt * cv);
}

fn sample_fu(st: Stencil) -> f32 {
    let n = params.numY;
    let sx = 1.0 - st.tx;
    let sy = 1.0 - st.ty;
    return sx * sy * fu[st.i0 * n + st.j0] +
           st.tx * sy * fu[st.i1 * n + st.j0] +
           st.tx * st.ty * fu[st.i1 * n + st.j1] +
           sx * st.ty * fu[st.i0 * n + st.j1];
}

fn sample_fv(st: Stencil) -> f32 {
    let n = params.numY;
    let sx = 1.0 - st.tx;
    let sy = 1.0 - st.ty;
    return sx * sy * fv[st.i0 * n + st.j0] +
           st.tx * sy * fv[st.i1 * n + st.j0] +
           st.tx * st.ty * fv[st.i1 * n + st.j1] +
           sx * st.ty * fv[st.i0 * n + st.j1];
}

// advect_velocity: one semi-Lagrangian pass over both velocity components.
// Serves as the MacCormack forward pass (dt > 0, fu/fv == u/v) and the
// backward pass (dt < 0, fu/fv == phi^).
//
// The default write is phi^n, not the input field -- see the header. A face
// keeps phi^n whenever either cell sharing it is solid, which is what carries
// the inflow BC: presets.js puts the inflow in column i=1 and makes i=0 solid,
// so the u-face at i=1 never advects, on either pass.
//
// NOTE: deliberately no stencil-corner solid test, unlike advect_smoke.wgsl.
// Solid cells hold a genuine wall BC for velocity (zero, or the obstacle's own
// velocity during a drag, rewritten in full by interaction.js every frame), so
// the forward interpolation is entitled to blend them -- that is how no-slip
// enters the advected field. There is no velocity analogue of the stale-dye
// problem that motivates the smoke path's corner test.
@compute @workgroup_size(8, 8)
fn advect_velocity(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;

    if (i < 1u || i >= params.numX || j < 1u || j >= n) { return; }

    let idx = i * n + j;

    // Default: revert to phi^n. Overwritten below for advectable faces.
    outU[idx] = u[idx];
    outV[idx] = v[idx];

    // u-face at (i*h, j*h + h/2). Advect only if both cells sharing it are fluid.
    if (s[idx] != 0.0 && s[(i - 1u) * n + j] != 0.0 && j < n - 1u) {
        let d = u_departure(i, j);
        outU[idx] = sample_fu(u_stencil(d.x, d.y));
    }

    // v-face at (i*h + h/2, j*h). Advect only if both cells sharing it are fluid.
    if (s[idx] != 0.0 && s[i * n + j - 1u] != 0.0 && i < params.numX - 1u) {
        let d = v_departure(i, j);
        outV[idx] = sample_fv(v_stencil(d.x, d.y));
    }
}
