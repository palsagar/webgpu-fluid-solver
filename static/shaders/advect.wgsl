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
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
// Binding 4: u_new (advect_velocity) or m (advect_smoke) — same slot, different bind group per dispatch
@group(0) @binding(4) var<storage, read_write> buf4: array<f32>;
// Binding 5: v_new (advect_velocity) or m_new (advect_smoke)
@group(0) @binding(5) var<storage, read_write> buf5: array<f32>;

// sample_u: u-field, dy = h/2 offset
fn sample_u(x_in: f32, y_in: f32) -> f32 {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;
    let n = ny;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    let x0f = floor(x * h1);
    let x0 = min(u32(x0f), nx - 1u);
    let tx = (x - x0f * h) * h1;
    let x1 = min(x0 + 1u, nx - 1u);

    let y0f = floor((y - h2) * h1);
    let y0 = min(u32(y0f), ny - 1u);
    let ty = ((y - h2) - y0f * h) * h1;
    let y1 = min(y0 + 1u, ny - 1u);

    let sx = 1.0 - tx;
    let sy = 1.0 - ty;

    return sx * sy * u[x0 * n + y0] +
           tx * sy * u[x1 * n + y0] +
           tx * ty * u[x1 * n + y1] +
           sx * ty * u[x0 * n + y1];
}

// sample_v: v-field, dx = h/2 offset
fn sample_v(x_in: f32, y_in: f32) -> f32 {
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let nx = params.numX;
    let ny = params.numY;
    let n = ny;

    let x = clamp(x_in, h, f32(nx) * h);
    let y = clamp(y_in, h, f32(ny) * h);

    let x0f = floor((x - h2) * h1);
    let x0 = min(u32(x0f), nx - 1u);
    let tx = ((x - h2) - x0f * h) * h1;
    let x1 = min(x0 + 1u, nx - 1u);

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

// sample_scalar: cell-center field, dx = h/2, dy = h/2 offsets
// Used by advect_smoke via buf4 (m). Since we cannot pass storage buffers as
// function args in WGSL, this is inlined inside advect_smoke below.

// advect_velocity:
//   buf4 = u_new, buf5 = v_new
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

    // Copy current values (equivalent to CPU's newU.set(u), newV.set(v))
    buf4[idx] = u[idx];
    buf5[idx] = v[idx];

    // u component: face at (i*h, j*h + h/2)
    if (s[idx] != 0.0 && s[(i - 1u) * n + j] != 0.0 && j < n - 1u) {
        let x = f32(i) * h;
        let y = f32(j) * h + h2;
        let cu = u[idx];
        // avgV(i, j) = average of v at four surrounding v-faces
        let cv = (v[(i - 1u) * n + j] + v[idx] +
                  v[(i - 1u) * n + j + 1u] + v[i * n + j + 1u]) * 0.25;
        buf4[idx] = sample_u(x - dt * cu, y - dt * cv);
    }

    // v component: face at (i*h + h/2, j*h)
    if (s[idx] != 0.0 && s[i * n + j - 1u] != 0.0 && i < params.numX - 1u) {
        let x = f32(i) * h + h2;
        let y = f32(j) * h;
        // avgU(i, j) = average of u at four surrounding u-faces
        let cu = (u[i * n + j - 1u] + u[idx] +
                  u[(i + 1u) * n + j - 1u] + u[(i + 1u) * n + j]) * 0.25;
        let cv = v[idx];
        buf5[idx] = sample_v(x - dt * cu, y - dt * cv);
    }
}

// advect_smoke:
//   buf4 = m (read), buf5 = m_new (write)
// Note: buf4 is declared read_write above but the JS caller will bind the same
// m buffer to both slots if a read-only view is unavailable; in practice
// advect_smoke only reads buf4 and writes buf5 so there is no hazard.
@compute @workgroup_size(8, 8)
fn advect_smoke(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    let h = params.h;
    let h1 = 1.0 / h;
    let h2 = 0.5 * h;
    let dt = params.dt;
    let nx = params.numX;
    let ny = params.numY;

    if (i < 1u || i >= nx - 1u || j < 1u || j >= n - 1u) { return; }

    let idx = i * n + j;

    // Copy current value (equivalent to CPU's newM.set(m))
    buf5[idx] = buf4[idx];

    if (s[idx] != 0.0) {
        let cu = (u[idx] + u[(i + 1u) * n + j]) * 0.5;
        let cv = (v[idx] + v[i * n + j + 1u]) * 0.5;
        let x_in = f32(i) * h + h2 - dt * cu;
        let y_in = f32(j) * h + h2 - dt * cv;

        // Inline bilinear sample of buf4 (smoke field m) with dx=h/2, dy=h/2
        let x = clamp(x_in, h, f32(nx) * h);
        let y = clamp(y_in, h, f32(ny) * h);

        let x0f = floor((x - h2) * h1);
        let x0 = min(u32(x0f), nx - 1u);
        let tx = ((x - h2) - x0f * h) * h1;
        let x1 = min(x0 + 1u, nx - 1u);

        let y0f = floor((y - h2) * h1);
        let y0 = min(u32(y0f), ny - 1u);
        let ty = ((y - h2) - y0f * h) * h1;
        let y1 = min(y0 + 1u, ny - 1u);

        let sx = 1.0 - tx;
        let sy = 1.0 - ty;

        buf5[idx] = sx * sy * buf4[x0 * n + y0] +
                    tx * sy * buf4[x1 * n + y0] +
                    tx * ty * buf4[x1 * n + y1] +
                    sx * ty * buf4[x0 * n + y1];
    }
}
