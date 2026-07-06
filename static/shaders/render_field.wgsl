// Fullscreen Field View renderer (ADR-0005). Samples the scalar field
// storage buffer with manual bilinear filtering, maps values through a
// colormap LUT texture, and draws solid cells dark gray in-shader.
// Replaces the CPU putImageData path.

struct RenderParams {
  numX: u32,
  numY: u32,
  minVal: f32,
  maxVal: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> solid: array<f32>;
@group(1) @binding(0) var lut: texture_2d<f32>;
@group(1) @binding(1) var lutSampler: sampler;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  // Fullscreen triangle — no vertex buffer needed
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var out: VSOut;
  let p = corners[vi];
  out.pos = vec4<f32>(p, 0.0, 1.0);
  // uv.y follows NDC y (up), matching sim j increasing upward — no flip needed
  out.uv = p * 0.5 + vec2<f32>(0.5, 0.5);
  return out;
}

fn cellIndex(i: i32, j: i32) -> u32 {
  let ci = clamp(i, 0, i32(params.numX) - 1);
  let cj = clamp(j, 0, i32(params.numY) - 1);
  return u32(ci) * params.numY + u32(cj);
}

// Field value at cell (i, j); solid cells fall back to `fb` so bilinear
// filtering doesn't bleed stale in-solid values across obstacle edges.
fn fluidValue(i: i32, j: i32, fb: f32) -> f32 {
  let idx = cellIndex(i, j);
  if (solid[idx] == 0.0) { return fb; }
  return field[idx];
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // Cell-center space: cell (i, j) center sits at (i + 0.5, j + 0.5)
  let gx = in.uv.x * f32(params.numX) - 0.5;
  let gy = in.uv.y * f32(params.numY) - 0.5;

  // Nearest-cell solid test keeps obstacle edges crisp
  let ni = i32(round(gx));
  let nj = i32(round(gy));
  let nearestIdx = cellIndex(ni, nj);
  if (solid[nearestIdx] == 0.0) {
    return vec4<f32>(50.0 / 255.0, 50.0 / 255.0, 60.0 / 255.0, 1.0);
  }

  let center = field[nearestIdx];
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy));
  let tx = fract(gx);
  let ty = fract(gy);

  let v00 = fluidValue(i0,     j0,     center);
  let v10 = fluidValue(i0 + 1, j0,     center);
  let v01 = fluidValue(i0,     j0 + 1, center);
  let v11 = fluidValue(i0 + 1, j0 + 1, center);
  let value = mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);

  let t = clamp((value - params.minVal) / (params.maxVal - params.minVal + 1e-10), 0.0, 1.0);
  // textureSampleLevel is legal in non-uniform control flow (after the solid early-return);
  // plain textureSample would be a compile error here.
  let color = textureSampleLevel(lut, lutSampler, vec2<f32>(t, 0.5), 0.0);
  return vec4<f32>(color.rgb, 1.0);
}
