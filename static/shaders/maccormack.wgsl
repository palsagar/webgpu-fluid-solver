// ============================================================================
// MacCormack combine (smoke)
//
// phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped to the fluid corners of the
// forward trace's bilinear stencil (plus phi^ itself) so the correction cannot
// create new extrema. See maccormack_smoke for why solid corners are dropped
// and why phi^ has to be in the range regardless.
//
// Separate file from advect_smoke.wgsl on purpose: bindings 3/4/5 mean
// different buffers here, and WGSL forbids two module-scope bindings at the
// same @group/@binding. `Params`, `Stencil`, `scalar_stencil` and
// `smoke_departure` are re-declared verbatim below -- WGSL has no modules, so
// duplication is the only way to share them. The limiter is correct ONLY while
// the re-trace here reproduces the forward stencil exactly, and drift between
// the copies still yields values in [0,1] -- so it is invisible to the bounds
// test. solver.spec.js asserts the two copies stay textually identical.
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
@group(0) @binding(6) var<storage, read> s: array<f32>;         // solid mask

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

// maccormack_smoke: phi^{n+1} = phi^ + (phi^n - phi~)/2, clamped so no new
// extrema appear. Writes in place into the phi~ slot; elementwise, so aliasing
// is safe.
@compute @workgroup_size(8, 8)
fn maccormack_smoke(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }

    let idx = i * n + j;

    // Solid cells carry no transported field -- they carry a boundary
    // condition, so they must come out of the combine holding phi^n.
    //
    // This branch is a PROVABLE NO-OP, not a correctness requirement. An
    // earlier version of this comment claimed the clamp would otherwise
    // corrupt solid cells during an obstacle DRAG (when interaction.js writes
    // the obstacle's own velocity into them and the re-traced departure lands
    // far away, so the corner range need not contain phi^n[idx]). That is
    // wrong: advect_smoke reverts solid cells to mOrig on BOTH passes, so here
    // cmHat[idx] == cmTilde[idx] == cmN[idx] bit-for-bit, `corrected` is
    // exactly cmHat, and the bounds below are SEEDED with cmHat -- so
    // lo <= corrected <= hi holds identically and the clamp is the identity.
    // The seed, not this branch, is what protects solid cells; Task 6's
    // velocity combine relies on that and carries no such branch (and no `s`
    // binding) at all. Kept here only as a cheap early-out that states the
    // intent locally.
    if (s[idx] == 0.0) {
        cmTilde[idx] = cmN[idx];
        return;
    }

    let corrected = cmHat[idx] + 0.5 * (cmN[idx] - cmTilde[idx]);

    // Re-trace to recover the stencil bounds. Cheaper than carrying min/max
    // through two extra full-grid buffers.
    let d = smoke_departure(i, j);
    let st = scalar_stencil(d.x, d.y);

    // Bounds come from the FLUID corners of the stencil, plus phi^ itself.
    //
    // Excluding solid corners: interaction.js clears smoke only in the
    // PREVIOUS obstacle bbox, so cells newly covered by a dragged obstacle
    // keep stale dye. Under semi-Lagrangian that leaked into neighbours only
    // in proportion to its bilinear weight; a clamp against the raw corner
    // value admits the FULL excursion to it, letting stale dye bleed off the
    // obstacle surface far harder than the first-order scheme ever did.
    //
    // Seeding with phi^ (the first-order result) rather than dropping solid
    // corners outright is what keeps that exclusion from walling dye out of
    // the domain. This solver injects dye THROUGH a solid: the preset writes
    // the inlet band into column i=0, part of the solid left wall, and the
    // forward trace at i=1 clamps back into it. A fluid-only range there is
    // lo = hi = 1.0 (clear), which would pin i=1 to clear forever and kill the
    // smoke field outright -- the same failure mode as testing stencil corners
    // on the forward advection pass. phi^ already carries each solid corner's
    // contribution at exactly its bilinear weight, so admitting it restores
    // precisely the semi-Lagrangian leakage rate and no more.
    //
    // Boundedness is unaffected: phi^ is a convex combination of phi^n samples,
    // so lo/hi stay inside the range of phi^n. This also handles the
    // all-corners-solid case without a special branch -- the bounds collapse to
    // lo = hi = phi^, i.e. no correction at all.
    var ks = array<u32, 4>(
        st.i0 * n + st.j0,
        st.i1 * n + st.j0,
        st.i0 * n + st.j1,
        st.i1 * n + st.j1
    );
    var lo = cmHat[idx];
    var hi = cmHat[idx];
    for (var q = 0u; q < 4u; q = q + 1u) {
        let k = ks[q];
        if (s[k] == 0.0) { continue; }
        let val = cmN[k];
        lo = min(lo, val);
        hi = max(hi, val);
    }

    cmTilde[idx] = clamp(corrected, lo, hi);
}
