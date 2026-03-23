/**
 * GPU-based 2D incompressible fluid solver using WebGPU compute shaders.
 *
 * Runs pressure solve (red-black Gauss-Seidel), boundary extrapolation,
 * velocity advection, and smoke advection entirely on the GPU. Uses
 * ping-pong buffer pairs for advection to avoid read/write hazards.
 */
export class FluidSolver {
  /**
   * @param {GPUDevice} device - WebGPU device handle
   * @param {number} numX - Grid width in cells
   * @param {number} numY - Grid height in cells
   * @param {number} h - Cell spacing (world units per cell)
   */
  constructor(device, numX, numY, h) {
    this.device = device;
    this.numX = numX;
    this.numY = numY;
    this.h = h;
    this.paused = false;

    this.params = { numX, numY, h, dt: 1 / 60, omega: 1.9, density: 1000, color: 0 };

    this._createBuffers(numX, numY);
  }

  /**
   * Allocates all GPU buffers for the simulation grid.
   *
   * Creates primary field buffers (u, v, p, s, m) and ping-pong
   * destination buffers (uNew, vNew, mNew) for advection. Also
   * creates three uniform buffers: one general-purpose and two for
   * red/black pressure solve (which differ only in the color flag).
   *
   * @param {number} numX - Grid width in cells
   * @param {number} numY - Grid height in cells
   */
  _createBuffers(numX, numY) {
    const device = this.device;
    const size = numX * numY;
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Primary field buffers: horizontal velocity (u), vertical velocity (v),
    // pressure (p), solid mask (s), smoke/dye density (m)
    this.u    = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.v    = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.p    = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.s    = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.m    = device.createBuffer({ size: size * 4, usage: storageUsage });

    // Ping-pong destination buffers for advection (avoids read/write hazards)
    this.uNew = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.vNew = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.mNew = device.createBuffer({ size: size * 4, usage: storageUsage });

    // Uniform buffers: red/black variants carry color=0 and color=1 respectively
    const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this.uniformBuf      = device.createBuffer({ size: 32, usage: uniformUsage });
    this.uniformBufRed   = device.createBuffer({ size: 32, usage: uniformUsage });
    this.uniformBufBlack = device.createBuffer({ size: 32, usage: uniformUsage });
  }

  /**
   * Packs simulation parameters into a 32-byte ArrayBuffer and uploads
   * to the main uniform buffer. Layout must match the WGSL struct:
   * [numX(u32), numY(u32), h(f32), dt(f32), omega(f32), density(f32), color(u32), pad].
   *
   * @param {number} [colorOverride] - If provided, overrides the color field (0=red, 1=black)
   */
  writeParams(colorOverride) {
    const p = this.params;
    const color = colorOverride !== undefined ? colorOverride : p.color;
    const ab = new ArrayBuffer(32);
    const dv = new DataView(ab);
    dv.setUint32(0,  p.numX,   true);
    dv.setUint32(4,  p.numY,   true);
    dv.setFloat32(8,  p.h,     true);
    dv.setFloat32(12, p.dt,    true);
    dv.setFloat32(16, p.omega,  true);
    dv.setFloat32(20, p.density, true);
    dv.setUint32(24, color,    true);
    this.device.queue.writeBuffer(this.uniformBuf, 0, ab);
  }

  /**
   * Packs and uploads simulation parameters to a specific uniform buffer.
   * Used to write distinct color values to the red and black uniform buffers.
   *
   * @param {GPUBuffer} buf - Target uniform buffer
   * @param {number} [colorOverride] - If provided, overrides the color field
   */
  _writeParamsTo(buf, colorOverride) {
    const p = this.params;
    const color = colorOverride !== undefined ? colorOverride : p.color;
    const ab = new ArrayBuffer(32);
    const dv = new DataView(ab);
    dv.setUint32(0,  p.numX,   true);
    dv.setUint32(4,  p.numY,   true);
    dv.setFloat32(8,  p.h,     true);
    dv.setFloat32(12, p.dt,    true);
    dv.setFloat32(16, p.omega,  true);
    dv.setFloat32(20, p.density, true);
    dv.setUint32(24, color,    true);
    this.device.queue.writeBuffer(buf, 0, ab);
  }

  /** Releases all GPU buffers. Must be called before resize or disposal. */
  destroy() {
    this.u.destroy();
    this.v.destroy();
    this.p.destroy();
    this.s.destroy();
    this.m.destroy();
    this.uNew.destroy();
    this.vNew.destroy();
    this.mNew.destroy();
    this.uniformBuf.destroy();
    this.uniformBufRed.destroy();
    this.uniformBufBlack.destroy();
  }

  /**
   * Async factory that creates a FluidSolver, loads WGSL shaders, builds
   * compute pipelines with explicit bind group layouts, and initializes
   * all bind groups and uniform data.
   *
   * @param {GPUDevice} device - WebGPU device handle
   * @param {number} numX - Grid width in cells
   * @param {number} numY - Grid height in cells
   * @param {number} h - Cell spacing (world units per cell)
   * @returns {Promise<FluidSolver>}
   */
  static async create(device, numX, numY, h) {
    const solver = new FluidSolver(device, numX, numY, h);

    const [pressureWgsl, boundaryWgsl, advectWgsl] = await Promise.all([
      fetch('/shaders/pressure.wgsl').then(r => r.text()),
      fetch('/shaders/boundary.wgsl').then(r => r.text()),
      fetch('/shaders/advect.wgsl').then(r => r.text()),
    ]);

    const pressureMod  = device.createShaderModule({ code: pressureWgsl });
    const boundaryMod  = device.createShaderModule({ code: boundaryWgsl });
    const advectMod    = device.createShaderModule({ code: advectWgsl });

    // Create explicit bind group layouts so all declared bindings are included
    // (auto-layout only includes statically-used bindings, which breaks shared bind groups)
    const UNIFORM  = 'uniform';
    const STORAGE  = 'storage';
    const RO_STORAGE = 'read-only-storage';

    const bglEntry = (binding, type) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });

    // Layout for pressure: uniform(0) + storage(1,2) + read-only-storage(3) + storage(4)
    solver._pressureBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, STORAGE), bglEntry(2, STORAGE), bglEntry(3, RO_STORAGE), bglEntry(4, STORAGE)],
    });

    // Layout for boundary: uniform(0) + storage(1,2)
    solver._boundaryBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, STORAGE), bglEntry(2, STORAGE)],
    });

    // Layout for advect: uniform(0) + read-only(1,2,3) + read-write(4,5)
    solver._advectBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE), bglEntry(2, RO_STORAGE), bglEntry(3, RO_STORAGE), bglEntry(4, STORAGE), bglEntry(5, STORAGE)],
    });

    const makePipelineLayout = (bgl) => device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    solver.pressurePipeline    = device.createComputePipeline({ layout: makePipelineLayout(solver._pressureBGL),  compute: { module: pressureMod,  entryPoint: 'main' } });
    solver.boundaryHPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._boundaryBGL),  compute: { module: boundaryMod,  entryPoint: 'extrapolate_horizontal' } });
    solver.boundaryVPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._boundaryBGL),  compute: { module: boundaryMod,  entryPoint: 'extrapolate_vertical' } });
    solver.advectVelPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._advectBGL),    compute: { module: advectMod,    entryPoint: 'advect_velocity' } });
    solver.advectSmokePipeline = device.createComputePipeline({ layout: makePipelineLayout(solver._advectBGL),    compute: { module: advectMod,    entryPoint: 'advect_smoke' } });

    solver._createBindGroups();

    // Write initial uniform data
    solver._writeParamsTo(solver.uniformBuf, 0);
    solver._writeParamsTo(solver.uniformBufRed, 0);
    solver._writeParamsTo(solver.uniformBufBlack, 1);

    return solver;
  }

  /**
   * Creates all bind groups for the compute pipelines. Sets up two bind groups
   * per advection pass (A and B) for ping-pong: A reads from primary buffers and
   * writes to *New buffers, B does the reverse. Initializes flip state to A.
   */
  _createBindGroups() {
    const device = this.device;
    const entry = (binding, buffer) => ({ binding, resource: { buffer } });

    // Pressure red/black: [uniformBufRed/Black, u, v, s, p]
    this.pressureRedBindGroup = device.createBindGroup({
      layout: this._pressureBGL,
      entries: [entry(0, this.uniformBufRed), entry(1, this.u), entry(2, this.v), entry(3, this.s), entry(4, this.p)],
    });
    this.pressureBlackBindGroup = device.createBindGroup({
      layout: this._pressureBGL,
      entries: [entry(0, this.uniformBufBlack), entry(1, this.u), entry(2, this.v), entry(3, this.s), entry(4, this.p)],
    });

    // Boundary: [uniformBuf, u, v]
    this.boundaryBindGroup = device.createBindGroup({
      layout: this._boundaryBGL,
      entries: [entry(0, this.uniformBuf), entry(1, this.u), entry(2, this.v)],
    });

    // Advect velocity A/B (ping-pong)
    this.advectVelBindGroupA = device.createBindGroup({
      layout: this._advectBGL,
      entries: [entry(0, this.uniformBuf), entry(1, this.u), entry(2, this.v), entry(3, this.s), entry(4, this.uNew), entry(5, this.vNew)],
    });
    this.advectVelBindGroupB = device.createBindGroup({
      layout: this._advectBGL,
      entries: [entry(0, this.uniformBuf), entry(1, this.uNew), entry(2, this.vNew), entry(3, this.s), entry(4, this.u), entry(5, this.v)],
    });

    // Advect smoke A/B (ping-pong)
    this.advectSmokeBindGroupA = device.createBindGroup({
      layout: this._advectBGL,
      entries: [entry(0, this.uniformBuf), entry(1, this.u), entry(2, this.v), entry(3, this.s), entry(4, this.m), entry(5, this.mNew)],
    });
    this.advectSmokeBindGroupB = device.createBindGroup({
      layout: this._advectBGL,
      entries: [entry(0, this.uniformBuf), entry(1, this.u), entry(2, this.v), entry(3, this.s), entry(4, this.mNew), entry(5, this.m)],
    });

    // Start with A
    this._advectVelBindGroup   = this.advectVelBindGroupA;
    this._advectSmokeBindGroup = this.advectSmokeBindGroupA;
    this._advectVelFlip   = false;
    this._advectSmokeFlip = false;
  }

  /**
   * Runs one full simulation time step: pressure solve, boundary extrapolation,
   * velocity advection, and smoke advection. Encodes all passes into a single
   * command buffer and submits to the GPU queue, then swaps ping-pong state.
   *
   * @param {number} numIters - Number of red-black Gauss-Seidel pressure iterations
   */
  step(numIters) {
    const { device, numX, numY } = this;

    // Write params to all uniform buffers
    this.writeParams();
    this._writeParamsTo(this.uniformBufRed, 0);
    this._writeParamsTo(this.uniformBufBlack, 1);

    const encoder = device.createCommandEncoder();

    // Workgroup size is 8x8, so dispatch enough groups to cover the grid
    const dx = Math.ceil(numX / 8);
    const dy = Math.ceil(numY / 8);

    // Pressure solve (red-black Gauss-Seidel)
    for (let i = 0; i < numIters; i++) {
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pressurePipeline);
        pass.setBindGroup(0, this.pressureRedBindGroup);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pressurePipeline);
        pass.setBindGroup(0, this.pressureBlackBindGroup);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
    }

    // Boundary
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.boundaryHPipeline);
      pass.setBindGroup(0, this.boundaryBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numX / 64), 1, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.boundaryVPipeline);
      pass.setBindGroup(0, this.boundaryBindGroup);
      pass.dispatchWorkgroups(Math.ceil(numY / 64), 1, 1);
      pass.end();
    }

    // Advect velocity
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.advectVelPipeline);
      pass.setBindGroup(0, this._advectVelBindGroup);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    // Advect smoke
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.advectSmokePipeline);
      pass.setBindGroup(0, this._advectSmokeBindGroup);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);

    // Swap ping-pong bind groups so the next step reads from the buffers
    // that were just written to, and writes to the ones that were read from
    this._advectVelFlip = !this._advectVelFlip;
    this._advectVelBindGroup = this._advectVelFlip
      ? this.advectVelBindGroupB
      : this.advectVelBindGroupA;

    this._advectSmokeFlip = !this._advectSmokeFlip;
    this._advectSmokeBindGroup = this._advectSmokeFlip
      ? this.advectSmokeBindGroupB
      : this.advectSmokeBindGroupA;
  }

  /**
   * Destroys existing GPU buffers and recreates them for a new grid size.
   * Also rebuilds all bind groups. Callers must re-upload field data
   * (solid mask, velocities, smoke) after calling this.
   *
   * @param {number} numX - New grid width in cells
   * @param {number} numY - New grid height in cells
   * @param {number} h - New cell spacing
   */
  resize(numX, numY, h) {
    this.destroy();
    this.numX = numX;
    this.numY = numY;
    this.h = h;
    this.params.numX = numX;
    this.params.numY = numY;
    this.params.h = h;
    this._createBuffers(numX, numY);
    this._createBindGroups();
  }

  /**
   * Merges overrides into the simulation parameters and immediately
   * uploads to all three uniform buffers.
   *
   * @param {Object} overrides - Key/value pairs to merge (e.g., { dt: 1/120 })
   */
  setParams(overrides) {
    Object.assign(this.params, overrides);
    this.writeParams();
    this._writeParamsTo(this.uniformBufRed, 0);
    this._writeParamsTo(this.uniformBufBlack, 1);
  }

  /**
   * Resets ping-pong state so the next step reads from the primary buffers (u, v, m).
   * Call after uploading new field data to ensure the solver reads the correct buffers.
   */
  resetFlipState() {
    this._advectVelFlip = false;
    this._advectSmokeFlip = false;
    this._advectVelBindGroup = this.advectVelBindGroupA;
    this._advectSmokeBindGroup = this.advectSmokeBindGroupA;
  }

  writeSolidMask(data) { this.device.queue.writeBuffer(this.s, 0, data); }
  writeVelocityU(data) { this.device.queue.writeBuffer(this.u, 0, data); }
  writeVelocityV(data) { this.device.queue.writeBuffer(this.v, 0, data); }
  writeSmoke(data)     { this.device.queue.writeBuffer(this.m, 0, data); }

  get pressureBuffer()  { return this.p; }
  /** Returns the smoke buffer that holds the most recent advection output. */
  get smokeBuffer()     { return this._advectSmokeFlip ? this.mNew : this.m; }
  /** Returns the velocity buffers that hold the most recent advection output. */
  get velocityBuffers() {
    return this._advectVelFlip
      ? { u: this.uNew, v: this.vNew }
      : { u: this.u, v: this.v };
  }
  get solidBuffer()     { return this.s; }
}
