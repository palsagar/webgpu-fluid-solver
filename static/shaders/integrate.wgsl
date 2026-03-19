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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let j = id.y;
    let n = params.numY;
    if (i < 1u || i >= params.numX || j < 1u || j >= n - 1u) { return; }
    let idx = i * n + j;
    if (s[idx] != 0.0 && s[i * n + j - 1u] != 0.0) {
        v[idx] += params.gravity * params.dt;
    }
}
