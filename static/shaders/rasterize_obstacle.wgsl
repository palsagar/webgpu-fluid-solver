// ============================================================================
// Obstacle rasterizer — writes the Solid Mask, obstacle velocity, and the
// vacated-footprint restore entirely on the GPU (ADR-0010). Replaces the CPU
// rasterizer whose stale whole-field mirrors reset the flow on every drag.
//
// One dispatch per rotation slot (three total), each binding that slot's
// velocity pair and smoke buffer; writes to `s` and `p` are idempotent across
// the three. 7 storage buffers per dispatch (8 is the per-stage limit); the
// extra read-only buffer is a frozen snapshot of `s` so the smoke-clear test
// sees the original mask on every slot.
//
// OWNERSHIP RULE (what makes a single pass race-free): each thread writes
// ONLY its own idx in every array. The CPU rasterizer's neighbour write —
// `u[(i+1)*n+j] = vx` for an inside cell, and the symmetric zeroing of that
// face on restore — becomes a READ of the left neighbour's inside/restore
// state here. Face (i,j) is owned by cell (i,j) and decided by cells (i,j)
// and (i-1,j).
//
// RESTORE SEMANTICS (spec note 3): a vacated non-boundary cell returns to
// fluid with zero velocity and zero pressure; smoke is cleared to 1.0
// (absence) only where the OLD mask was solid obstacle. Boundary cells
// (sBoundary == 0) are never carved and never restored — they keep their BC
// values. Pressure is zeroed across the whole previous bbox, boundary cells
// included (the CPU path zeroed full column slices).
// ============================================================================

struct RasterParams {
    numX: u32,
    numY: u32,
    h: f32,
    shape: u32,      // 0 circle, 1 square, 2 airfoil (NACA 0012), 3 wedge
    center: vec2f,   // obstacle centre, sim units
    vel: vec2f,      // obstacle (drag) velocity, sim units/s
    radius: f32,
    angle: f32,      // compass-needle rotation, radians
    prevBBox: vec4i, // iMin, iMax, jMin, jMax; iMin > iMax = no previous
};

@group(0) @binding(0) var<uniform> params: RasterParams;
@group(0) @binding(1) var<storage, read_write> s: array<f32>;
@group(0) @binding(2) var<storage, read> sBoundary: array<f32>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: array<f32>;
@group(0) @binding(6) var<storage, read_write> smoke: array<f32>;
@group(0) @binding(7) var<storage, read> sOld: array<f32>;

// Cell-centre inside-test, ported 1:1 from the deleted CPU rasterizer.
fn inside(i: u32, j: u32) -> bool {
    let dx = (f32(i) + 0.5) * params.h - params.center.x;
    let dy = (f32(j) + 0.5) * params.h - params.center.y;
    let cosA = cos(-params.angle);
    let sinA = sin(-params.angle);
    let ldx = dx * cosA - dy * sinA;
    let ldy = dx * sinA + dy * cosA;
    let r = params.radius;
    switch params.shape {
        case 0u: {
            return dx * dx + dy * dy < r * r;
        }
        case 1u: {
            return abs(ldx) < r && abs(ldy) < r;
        }
        case 2u: {
            let chord = r * 4.0;
            let lx = ldx + chord * 0.5;
            if (lx < 0.0 || lx > chord) { return false; }
            let xc = lx / chord;
            let yt = 5.0 * 0.12 * chord * (
                0.2969 * sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc
                + 0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
            return abs(ldy) < yt;
        }
        default: { // 3u: wedge, apex at centre, pointing right in local frame
            let wedgeLen = r * 3.0;
            let tanHA = tan(15.0 * 3.14159265 / 180.0);
            let lx = ldx + wedgeLen * 0.5;
            return lx >= 0.0 && lx < wedgeLen && abs(ldy) < lx * tanHA;
        }
    }
}

fn inPrevBBox(i: u32, j: u32) -> bool {
    let b = params.prevBBox;
    if (b.x > b.y) { return false; } // sentinel: no previous footprint
    return i >= u32(b.x) && i <= u32(b.y) && j >= u32(b.z) && j <= u32(b.w);
}

@compute @workgroup_size(8, 8)
fn rasterize(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i >= params.numX || j >= n) { return; }
    let idx = i * n + j;

    let boundaryHere = sBoundary[idx] == 0.0;
    let insideHere = !boundaryHere && inside(i, j);
    let zeroHere = !boundaryHere && inPrevBBox(i, j);

    // The left neighbour decides this thread's u face as its right face.
    var insideLeft = false;
    var zeroLeft = false;
    if (i > 0u) {
        let leftIdx = (i - 1u) * n + j;
        if (sBoundary[leftIdx] != 0.0) {
            insideLeft = inside(i - 1u, j);
            zeroLeft = inPrevBBox(i - 1u, j);
        }
    }

    // Solid mask
    if (!boundaryHere) {
        if (insideHere) { s[idx] = 0.0; }
        else if (zeroHere) { s[idx] = 1.0; }
    }

    // u: inside cells and faces right of an inside cell carry the wall
    // velocity; the restore zeroes the same set. Obstacle wins over restore
    // where they overlap (the CPU ordering: restore first, then rasterize).
    if (insideHere || insideLeft) { u[idx] = params.vel.x; }
    else if (zeroHere || zeroLeft) { u[idx] = 0.0; }

    // v: cell-owned only — the CPU rasterizer writes no neighbour v face.
    if (!boundaryHere) {
        if (insideHere) { v[idx] = params.vel.y; }
        else if (zeroHere) { v[idx] = 0.0; }
    }

    // Pressure: zeroed across the whole previous bbox, boundary included.
    if (inPrevBBox(i, j)) { p[idx] = 0.0; }

    // Smoke: cleared only where the OLD mask was solid obstacle (read from
    // the frozen sOld snapshot, because s itself is updated across the three
    // per-slot dispatches).
    if (zeroHere && sOld[idx] == 0.0) { smoke[idx] = 1.0; }
}
