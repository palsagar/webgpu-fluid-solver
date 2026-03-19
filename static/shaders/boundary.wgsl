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

@compute @workgroup_size(64)
fn extrapolate_horizontal(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = params.numY;
    if (i >= params.numX) { return; }
    u[i * n + 0u] = u[i * n + 1u];
    u[i * n + n - 1u] = u[i * n + n - 2u];
}

@compute @workgroup_size(64)
fn extrapolate_vertical(@builtin(global_invocation_id) id: vec3u) {
    let j = id.x;
    let n = params.numY;
    if (j >= n) { return; }
    v[0u * n + j] = v[1u * n + j];
    v[(params.numX - 1u) * n + j] = v[(params.numX - 2u) * n + j];
}
