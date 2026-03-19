struct Params {
    numX: u32,
    numY: u32,
    h: f32,
    dt: f32,
    gravity: f32,
    omega: f32,
    density: f32,
    color: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;

    if (i < 1u || i >= params.numX - 1u || j < 1u || j >= n - 1u) { return; }
    if ((i + j) % 2u != params.color) { return; }

    let idx = i * n + j;
    if (s[idx] == 0.0) { return; }

    let sx0 = s[(i - 1u) * n + j];
    let sx1 = s[(i + 1u) * n + j];
    let sy0 = s[i * n + j - 1u];
    let sy1 = s[i * n + j + 1u];
    let sTotal = sx0 + sx1 + sy0 + sy1;
    if (sTotal == 0.0) { return; }

    let div = u[(i + 1u) * n + j] - u[idx] + v[i * n + j + 1u] - v[idx];
    let pCorr = -div / sTotal * params.omega;

    p[idx] += params.density * params.h / params.dt * pCorr;

    u[idx]               -= sx0 * pCorr;
    u[(i + 1u) * n + j] += sx1 * pCorr;
    v[idx]               -= sy0 * pCorr;
    v[i * n + j + 1u]   += sy1 * pCorr;
}
