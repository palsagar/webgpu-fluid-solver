// ============================================================================
// MacCormack combine (smoke)
//
// phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped to the min/max of the forward
// trace's bilinear stencil so the correction cannot create new extrema.
//
// Separate file from advect_smoke.wgsl on purpose: bindings 3/4/5 mean
// different buffers here, and WGSL forbids two module-scope bindings at the
// same @group/@binding. `Params`, `Stencil`, `scalar_stencil` and
// `smoke_departure` are re-declared verbatim below -- WGSL has no modules, so
// duplication is the only way to share them. Keep the two copies in sync.
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,          // cell size
    dt: f32,         // time step (positive here: the combine re-traces forward)
    omega: f32,      // (unused here)
    density: f32,    // (unused here)
    color: u32,      // (unused here)
    nu: f32,         // (unused here)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> u: array<f32>;         // advecting velocity
@group(0) @binding(2) var<storage, read> v: array<f32>;         // advecting velocity
@group(0) @binding(3) var<storage, read> cmN: array<f32>;       // phi^n
@group(0) @binding(4) var<storage, read> cmHat: array<f32>;     // phi^  (forward)
@group(0) @binding(5) var<storage, read_write> cmTilde: array<f32>; // phi~ in, phi^{n+1} out

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

// maccormack_smoke: phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped to the
// min/max of the forward trace's bilinear stencil so no new extrema appear.
// Writes in place into the phi~ slot; elementwise, so aliasing is safe.
@compute @workgroup_size(8, 8)
fn maccormack_smoke(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }

    let idx = i * n + j;
    let corrected = cmHat[idx] + 0.5 * (cmN[idx] - cmTilde[idx]);

    // Re-trace to recover the stencil bounds. Cheaper than carrying min/max
    // through two extra full-grid buffers.
    let d = smoke_departure(i, j);
    let st = scalar_stencil(d.x, d.y);
    let a = cmN[st.i0 * n + st.j0];
    let b = cmN[st.i1 * n + st.j0];
    let c = cmN[st.i0 * n + st.j1];
    let e = cmN[st.i1 * n + st.j1];
    let lo = min(min(a, b), min(c, e));
    let hi = max(max(a, b), max(c, e));

    cmTilde[idx] = clamp(corrected, lo, hi);
}
