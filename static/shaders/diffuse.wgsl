// ============================================================================
// Explicit Viscous Diffusion — u += nu * dt_sub * laplacian(u)
//
// The host splits the frame's dt into N substeps chosen so that
// nu * dt_sub / h^2 <= 1/4, the 2D five-point stability limit.
//
// ---------------------------------------------------------------------------
// FACE CLASSIFICATION — three cases, not two
//
// A u-face at (i, j) separates cells (i-1, j) and (i, j). A v-face at (i, j)
// separates cells (i, j-1) and (i, j). Each face is one of:
//
//   FLUID   both flanking cells fluid. Diffused.
//   WALL    exactly one flanking cell solid. This is a wall-NORMAL face; the
//           projection pins it at the no-penetration value (zero, or the
//           obstacle's own velocity during a drag). Copied through untouched,
//           and read at face value when it appears in a neighbour's stencil --
//           which is exactly right, since the obstacle surface passes through
//           it and the velocity there really is the wall's.
//   BURIED  both flanking cells solid. The stored value is meaningless. Read
//           as a GHOST, -center, which places the field's zero on the wall
//           line half a cell away. This is what grows the boundary layer:
//           returning 0 (or skipping the face) instead would leave the
//           cylinder frictionless, and with it no separation and no shedding.
//
// The FLUID predicate is deliberately the same face-based test advect.wgsl
// uses to decide whether a face may advect (`s[idx] != 0 && s[(i-1)*n+j] != 0`).
// Keeping the two in agreement is what stops diffusion writing into solid
// cells or into the i=1 inflow column, both of which carry a BC rather than a
// transported value. Note this is the COMPLEMENT of "both cells solid" only on
// a grid with no wall-normal faces -- the WALL case is precisely the gap, and
// collapsing it into either neighbour would be a physics bug.
//
// ---------------------------------------------------------------------------
// THE STALE RING
//
// advect_velocity and maccormack_velocity both return for i < 1 or j < 1, so
// the i=0 column and the j=0 row are never written by advection. Under the
// three-slot rotation the slot that goes live after advection therefore carries
// an i=0/j=0 ring three steps old. A diffusion stencil reads its neighbours, so
// left alone it would pull that ring inward once per substep -- up to 32 times
// per frame.
//
// Both helpers below classify i == 0 and j == 0 as BURIED **by index**, before
// consulting the solid mask. Every preset already marks those lines solid
// (presets.js:94,119) so the mask would usually agree, but making it structural
// rather than contingent means the viscous stencil cannot read a stale entry
// even if a future preset opens one of those lines. The index test also avoids
// the u32 underflow the mask test would hit at i-1 == -1 / j-1 == -1.
//
// The remaining ring lines, i = numX-1 and j = numY-1, ARE read by the stencil.
// Advection writes both every step, so they are fresh -- but the substep loop
// ping-pongs between two slots, and a line written in neither would alternate
// between two different states from one substep to the next. Hence the ring
// copy-through in the entry point below.
// ============================================================================

struct Params {
    numX: u32,
    numY: u32,
    h: f32,
    dt: f32,         // dt_sub, already divided by the substep count
    omega: f32,      // (unused here)
    density: f32,    // (unused here)
    color: u32,      // (unused here)
    nu: f32,         // kinematic viscosity
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> uIn: array<f32>;
@group(0) @binding(3) var<storage, read> vIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> uOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> vOut: array<f32>;

// FLUID: both cells flanking the u-face are fluid. Callers hold i >= 1.
fn u_face_fluid(i: u32, j: u32) -> bool {
    let n = params.numY;
    return s[i * n + j] != 0.0 && s[(i - 1u) * n + j] != 0.0;
}

// FLUID: both cells flanking the v-face are fluid. Callers hold j >= 1.
fn v_face_fluid(i: u32, j: u32) -> bool {
    let n = params.numY;
    return s[i * n + j] != 0.0 && s[i * n + j - 1u] != 0.0;
}

// BURIED: both flanking cells solid, or the face sits on a line advection never
// writes (see THE STALE RING above).
fn u_face_buried(i: u32, j: u32) -> bool {
    if (i == 0u || j == 0u) { return true; }
    let n = params.numY;
    return s[i * n + j] == 0.0 && s[(i - 1u) * n + j] == 0.0;
}

fn v_face_buried(i: u32, j: u32) -> bool {
    if (i == 0u || j == 0u) { return true; }
    let n = params.numY;
    return s[i * n + j] == 0.0 && s[i * n + j - 1u] == 0.0;
}

fn u_neighbor(i: u32, j: u32, center: f32) -> f32 {
    if (u_face_buried(i, j)) { return -center; }   // ghost: u = 0 on the wall
    return uIn[i * params.numY + j];
}

fn v_neighbor(i: u32, j: u32, center: f32) -> f32 {
    if (v_face_buried(i, j)) { return -center; }
    return vIn[i * params.numY + j];
}

@compute @workgroup_size(8, 8)
fn diffuse(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    let nx = params.numX;
    if (i >= nx || j >= n) { return; }

    let idx = i * n + j;

    // Domain ring: copy through so the two ping-ponged slots stay in agreement
    // across substeps. Not an update -- these lines are BC storage, not flow.
    if (i < 1u || i >= nx - 1u || j < 1u || j >= n - 1u) {
        uOut[idx] = uIn[idx];
        vOut[idx] = vIn[idx];
        return;
    }

    let coeff = params.nu * params.dt / (params.h * params.h);

    // u
    if (u_face_fluid(i, j)) {
        let c = uIn[idx];
        let lap = u_neighbor(i + 1u, j, c) + u_neighbor(i - 1u, j, c) +
                  u_neighbor(i, j + 1u, c) + u_neighbor(i, j - 1u, c) - 4.0 * c;
        uOut[idx] = c + coeff * lap;
    } else {
        uOut[idx] = uIn[idx];
    }

    // v
    if (v_face_fluid(i, j)) {
        let c = vIn[idx];
        let lap = v_neighbor(i + 1u, j, c) + v_neighbor(i - 1u, j, c) +
                  v_neighbor(i, j + 1u, c) + v_neighbor(i, j - 1u, c) - 4.0 * c;
        vOut[idx] = c + coeff * lap;
    } else {
        vOut[idx] = vIn[idx];
    }
}
