// ============================================================================
// Semi-Lagrangian advection of the scalar smoke/dye field.
//
// This is one MacCormack sub-pass. It serves as BOTH the forward pass
// (positive dt, mIn == mOrig == phi^n) and the backward pass (negative dt via
// a separate uniform buffer, mIn == phi^, mOrig == phi^n). The combine lives
// in maccormack.wgsl -- it needs a different meaning for bindings 3/4/5, and
// WGSL forbids two module-scope bindings at the same @group/@binding, so it
// cannot share this module.
//
// Smoke is cell-centred (position i*h + h/2, j*h + h/2), so the advecting
// velocity at each cell centre is averaged from the two flanking faces.
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
@group(0) @binding(1) var<storage, read> u: array<f32>;    // advecting velocity
@group(0) @binding(2) var<storage, read> v: array<f32>;    // advecting velocity
@group(0) @binding(3) var<storage, read> s: array<f32>;    // solid mask
@group(0) @binding(4) var<storage, read> mIn: array<f32>;  // field being advected
@group(0) @binding(5) var<storage, read> mOrig: array<f32>; // phi^n (== mIn on the forward pass)
@group(0) @binding(6) var<storage, read_write> mOut: array<f32>;

// Grid indices and weights for a cell-centred bilinear stencil at (x_in, y_in).
struct Stencil {
    i0: u32, i1: u32, j0: u32, j1: u32,
    tx: f32, ty: f32,
}

fn scalar_stencil(x_in: f32, y_in: f32) -> Stencil {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    let x0f = floor((x - h2) * h1);
    let y0f = floor((y - h2) * h1);

    var st: Stencil;
    st.i0 = min(u32(x0f), nx - 1u);
    st.i1 = min(st.i0 + 1u, nx - 1u);
    st.j0 = min(u32(y0f), ny - 1u);
    st.j1 = min(st.j0 + 1u, ny - 1u);
    st.tx = ((x - h2) - x0f * h) * h1;
    st.ty = ((y - h2) - y0f * h) * h1;
    return st;
}

// Departure point for the cell centre (i, j) traced by -params.dt.
// A negative params.dt turns this into the forward trace the backward pass needs.
fn smoke_departure(i: u32, j: u32) -> vec2f {
    let n = params.numY;
    let h = params.h;
    let h2 = 0.5 * h;
    let idx = i * n + j;
    let cu = (u[idx] + u[(i + 1u) * n + j]) * 0.5;
    let cv = (v[idx] + v[i * n + j + 1u]) * 0.5;
    return vec2f(f32(i) * h + h2 - params.dt * cu,
                 f32(j) * h + h2 - params.dt * cv);
}

// True when any corner of the bilinear stencil sits in a solid cell, so the
// interpolated value is contaminated by whatever the obstacle fill happens to
// hold rather than by transported dye.
//
// Applied on the BACKWARD pass ONLY -- see the dt < 0.0 guard in advect_smoke.
// The two passes need opposite answers here:
//
//   Backward pass: reverting means "fall back to first order". The backward
//   trace at i=1 runs DOWNSTREAM (to ~3.63h at Karman settings) into fluid, so
//   the inlet is untouched; near obstacles the correction term zeroes and the
//   combine collapses to plain semi-Lagrangian, which is exactly what we want.
//
//   Forward pass: reverting means "do not advect at all". Worse, this solver
//   injects both dye and inflow THROUGH solid boundary cells -- the preset
//   writes the inlet band into column i=0, part of the solid left wall, and
//   cells at i=1 pick it up only because their departure point clamps back
//   into that column. Rejecting solid stencil corners on the forward pass
//   therefore walls dye out of the domain entirely (the field stays uniformly
//   1.0 forever) and freezes dye in a one-cell halo around every obstacle.
fn stencil_touches_solid(st: Stencil) -> bool {
    let n = params.numY;
    return s[st.i0 * n + st.j0] == 0.0 || s[st.i1 * n + st.j0] == 0.0 ||
           s[st.i0 * n + st.j1] == 0.0 || s[st.i1 * n + st.j1] == 0.0;
}

// advect_smoke: one semi-Lagrangian pass. Serves as both the MacCormack
// forward pass (dt > 0, mIn == mOrig) and the backward pass (dt < 0).
//
// At solid cells mOut = mOrig on both passes: solids hold no transported dye
// and must keep whatever was written into them (obstacle fill, or the inlet
// band in the left wall). On the backward pass that also makes the correction
// term (phi^n - phi~)/2 vanish; on the forward pass it reproduces the old
// semi-Lagrangian kernel's "copy through" behaviour.
//
// The sign of params.dt is the free discriminator between the two passes -- it
// is already negated for the backward pass via a separate uniform buffer -- so
// the pass-specific stencil-corner test below costs neither a binding nor a
// second shader module.
@compute @workgroup_size(8, 8)
fn advect_smoke(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }

    let idx = i * n + j;
    if (s[idx] == 0.0) {
        mOut[idx] = mOrig[idx];
        return;
    }

    let d = smoke_departure(i, j);
    let st = scalar_stencil(d.x, d.y);

    // Backward pass only. Reverting here zeroes the correction term, so the
    // combine degrades to first-order SL near obstacles -- bounded by
    // construction, and the intended behaviour where the trace is unreliable.
    if (params.dt < 0.0 && stencil_touches_solid(st)) {
        mOut[idx] = mOrig[idx];
        return;
    }

    let sx = 1.0 - st.tx;
    let sy = 1.0 - st.ty;
    mOut[idx] = sx * sy * mIn[st.i0 * n + st.j0] +
                st.tx * sy * mIn[st.i1 * n + st.j0] +
                st.tx * st.ty * mIn[st.i1 * n + st.j1] +
                sx * st.ty * mIn[st.i0 * n + st.j1];
}
