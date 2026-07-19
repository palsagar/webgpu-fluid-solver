// ============================================================================
// MacCormack combine (velocity)
//
// phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped to the corners of the forward
// trace's bilinear stencil PLUS phi^ itself, so the correction cannot create
// new extrema.
//
// Separate file from maccormack.wgsl (the smoke combine) for the same reason
// advect_smoke.wgsl is separate from advect.wgsl: bindings 3..6 mean different
// buffers here, and WGSL forbids two module-scope bindings at the same
// @group/@binding. The brief placed this entry point in maccormack.wgsl, which
// is not expressible -- bindings 3/4/5/6 there are already cmN/cmHat/cmTilde/s.
//
// `Params`, `Stencil`, `u_stencil`, `v_stencil`, `u_departure` and `v_departure`
// are re-declared verbatim from advect.wgsl -- WGSL has no modules, so
// duplication is the only way to share them. The limiter is correct ONLY while
// the re-trace here reproduces the forward pass's stencil exactly, and drift
// between the copies still yields plausible in-range velocities -- so it is
// invisible to every behavioural test. solver.spec.js asserts the two copies
// stay textually identical.
//
// THE SEED IS LOAD-BEARING. Seeding lo/hi with phi^ (rather than taking the
// stencil corners alone) is what makes a reverted face collapse to exactly
// first-order semi-Lagrangian:
//
//   advect_velocity reverts a face to phi^n on BOTH passes when either cell
//   sharing it is solid, so there phi^ == phi~ == phi^n bit-for-bit, hence
//   corrected == phi^ exactly. But the re-trace below is unconditional: it
//   computes a stencil for a trace the forward pass never took, whose corner
//   range need NOT contain phi^n[idx]. Without the seed the clamp would drag
//   the face off its boundary value.
//
// That is exactly what preserves the inflow BC: presets.js writes the inflow
// into column i=1 and makes i=0 solid, so every u-face at i=1 reverts. Note
// the revert is FACE-based, not cell-based -- cell (1, j) is fluid -- so the
// smoke combine's `if (s[idx] == 0.0)` guard would NOT fire at those faces.
// With the seed no guard is needed at all: the clamp is the identity wherever
// a revert happened, including inside solid cells, where the value is the
// (possibly moving) wall BC. Hence no `s` binding here.
//
// Solid stencil corners are NOT excluded, unlike the smoke combine. Velocity in
// a solid cell is a genuine wall BC that interaction.js rewrites in full every
// frame, and the forward interpolation legitimately blends it; there is no
// velocity analogue of the stale-dye problem that motivates smoke's exclusion.
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
@group(0) @binding(1) var<storage, read> u: array<f32>;    // u^n: advecting velocity AND phi^n
@group(0) @binding(2) var<storage, read> v: array<f32>;    // v^n: advecting velocity AND phi^n
@group(0) @binding(3) var<storage, read> uHat: array<f32>;         // phi^ (forward)
@group(0) @binding(4) var<storage, read> vHat: array<f32>;
@group(0) @binding(5) var<storage, read_write> uTilde: array<f32>; // phi~ in, phi^{n+1} out
@group(0) @binding(6) var<storage, read_write> vTilde: array<f32>;

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

// maccormack_velocity: phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped so no new
// extrema appear. Writes in place into the phi~ slots; every read of uTilde /
// vTilde is at this invocation's own index, so the aliasing is safe.
//
// The face branches carry the same index guards advect_velocity's do, because
// u_departure and v_departure reach one cell past (i, j). Faces outside those
// guards were already written by both advect passes with the reverted phi^n,
// which is the value the combine would have produced anyway.
@compute @workgroup_size(8, 8)
fn maccormack_velocity(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;

    if (i < 1u || i >= params.numX || j < 1u || j >= n) { return; }

    let idx = i * n + j;

    // u-face at (i*h, j*h + h/2).
    if (j < n - 1u) {
        let hat = uHat[idx];
        let corrected = hat + 0.5 * (u[idx] - uTilde[idx]);

        // Re-trace to recover the forward stencil. Cheaper than carrying
        // min/max through two extra full-grid buffers.
        let d = u_departure(i, j);
        let st = u_stencil(d.x, d.y);
        let a = u[st.i0 * n + st.j0];
        let b = u[st.i1 * n + st.j0];
        let c = u[st.i0 * n + st.j1];
        let e = u[st.i1 * n + st.j1];

        let lo = min(hat, min(min(a, b), min(c, e)));
        let hi = max(hat, max(max(a, b), max(c, e)));
        uTilde[idx] = clamp(corrected, lo, hi);
    }

    // v-face at (i*h + h/2, j*h).
    if (i < params.numX - 1u) {
        let hat = vHat[idx];
        let corrected = hat + 0.5 * (v[idx] - vTilde[idx]);

        let d = v_departure(i, j);
        let st = v_stencil(d.x, d.y);
        let a = v[st.i0 * n + st.j0];
        let b = v[st.i1 * n + st.j0];
        let c = v[st.i0 * n + st.j1];
        let e = v[st.i1 * n + st.j1];

        let lo = min(hat, min(min(a, b), min(c, e)));
        let hi = max(hat, max(max(a, b), max(c, e)));
        vTilde[idx] = clamp(corrected, lo, hi);
    }
}
