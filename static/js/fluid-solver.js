/**
 * GPU-based 2D incompressible fluid solver using WebGPU compute shaders.
 *
 * Runs pressure solve (red-black Gauss-Seidel), boundary extrapolation,
 * velocity advection, and smoke advection entirely on the GPU. Velocity and
 * smoke rotate through three buffer slots to avoid read/write hazards.
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
   * Velocity and smoke live in a 3-slot rotation (see `velPairs`/`smokeBufs`).
   * Pressure (p) and the solid mask (s) are single buffers. Also creates three
   * uniform buffers: one general-purpose and two for the red/black pressure
   * solve (which differ only in the color flag).
   *
   * @param {number} numX - Grid width in cells
   * @param {number} numY - Grid height in cells
   */
  _createBuffers(numX, numY) {
    const device = this.device;
    const size = numX * numY;
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

    // Three velocity pairs and three smoke buffers. MacCormack rotates through
    // them: current -> forward -> backward, with combine writing in place into
    // the backward pair. A 2-cycle would need a fourth pair and a distinct
    // combine output, which exceeds maxStorageBuffersPerShaderStage (8).
    this.velPairs = [];
    this.smokeBufs = [];
    for (let k = 0; k < 3; k++) {
      this.velPairs.push({
        u: device.createBuffer({ size: size * 4, usage: storageUsage }),
        v: device.createBuffer({ size: size * 4, usage: storageUsage }),
      });
      this.smokeBufs.push(device.createBuffer({ size: size * 4, usage: storageUsage }));
    }
    this._velCur = 0;
    this._smokeCur = 0;

    this.p = device.createBuffer({ size: size * 4, usage: storageUsage });
    this.s = device.createBuffer({ size: size * 4, usage: storageUsage });

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
    for (const pair of this.velPairs) { pair.u.destroy(); pair.v.destroy(); }
    for (const b of this.smokeBufs) b.destroy();
    this.p.destroy();
    this.s.destroy();
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
   * Creates all bind groups for the compute pipelines, indexed by rotation slot.
   *
   * Pressure and boundary get one variant per velocity pair; advection gets one
   * variant per (source, destination) slot pairing. Smoke advection is indexed
   * by BOTH the velocity slot and the smoke slot, because the velocity field
   * that carries the dye is not tied to the smoke rotation.
   */
  _createBindGroups() {
    const device = this.device;
    const entry = (binding, buffer) => ({ binding, resource: { buffer } });

    // One variant per velocity pair — pressure and boundary write velocity in
    // place, so they must target whichever pair is live this step.
    this.pressureRed = [];
    this.pressureBlack = [];
    this.boundary = [];
    for (const pair of this.velPairs) {
      this.pressureRed.push(device.createBindGroup({
        layout: this._pressureBGL,
        entries: [entry(0, this.uniformBufRed), entry(1, pair.u), entry(2, pair.v), entry(3, this.s), entry(4, this.p)],
      }));
      this.pressureBlack.push(device.createBindGroup({
        layout: this._pressureBGL,
        entries: [entry(0, this.uniformBufBlack), entry(1, pair.u), entry(2, pair.v), entry(3, this.s), entry(4, this.p)],
      }));
      this.boundary.push(device.createBindGroup({
        layout: this._boundaryBGL,
        entries: [entry(0, this.uniformBuf), entry(1, pair.u), entry(2, pair.v)],
      }));
    }

    // Velocity advection, interim 2-cycle form: read pair c, write pair (c+1)%3.
    // Task 5 replaces these with the three MacCormack passes.
    this.advectVel = [];
    for (let c = 0; c < 3; c++) {
      const src = this.velPairs[c], dst = this.velPairs[(c + 1) % 3];
      this.advectVel.push(device.createBindGroup({
        layout: this._advectBGL,
        entries: [entry(0, this.uniformBuf), entry(1, src.u), entry(2, src.v), entry(3, this.s), entry(4, dst.u), entry(5, dst.v)],
      }));
    }

    // Smoke advection is a 3x3 table indexed [velCur][smokeCur]. The velocity
    // slot and the smoke slot advance independently once viscous substepping
    // lands, so binding smoke's advecting velocity to the smoke index would
    // silently trace dye through the wrong velocity field.
    this.advectSmoke = [];
    for (let vc = 0; vc < 3; vc++) {
      this.advectSmoke.push([]);
      for (let sc = 0; sc < 3; sc++) {
        this.advectSmoke[vc].push(device.createBindGroup({
          layout: this._advectBGL,
          entries: [entry(0, this.uniformBuf),
                    entry(1, this.velPairs[vc].u),      // live velocity
                    entry(2, this.velPairs[vc].v),
                    entry(3, this.s),
                    entry(4, this.smokeBufs[sc]),
                    entry(5, this.smokeBufs[(sc + 1) % 3])],
        }));
      }
    }

    this._velCur = 0;
    this._smokeCur = 0;
  }

  /**
   * Runs one full simulation time step: pressure solve, boundary extrapolation,
   * velocity advection, and smoke advection. Encodes all passes into a single
   * command buffer and submits to the GPU queue, then advances the rotation.
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
        pass.setBindGroup(0, this.pressureRed[this._velCur]);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
      {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pressurePipeline);
        pass.setBindGroup(0, this.pressureBlack[this._velCur]);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
      }
    }

    // Boundary
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.boundaryHPipeline);
      pass.setBindGroup(0, this.boundary[this._velCur]);
      pass.dispatchWorkgroups(Math.ceil(numX / 64), 1, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.boundaryVPipeline);
      pass.setBindGroup(0, this.boundary[this._velCur]);
      pass.dispatchWorkgroups(Math.ceil(numY / 64), 1, 1);
      pass.end();
    }

    // Advect velocity: reads pair _velCur, writes pair (_velCur + 1) % 3.
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.advectVelPipeline);
      pass.setBindGroup(0, this.advectVel[this._velCur]);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    // Advect smoke through the SAME velocity pair the passes above worked on —
    // the projected, boundary-corrected time-n field in slot _velCur. Slot
    // _velCur + 1 now holds the advected, unprojected time-(n+1) velocity, and
    // tracing dye through that would be a semi-Lagrangian step out of sync with
    // the velocity it is supposed to follow. Hence both indices only advance
    // after the whole command buffer is encoded.
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.advectSmokePipeline);
      pass.setBindGroup(0, this.advectSmoke[this._velCur][this._smokeCur]);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);

    // Advance the rotation. Advection wrote into the next slot.
    this._velCur   = (this._velCur + 1) % 3;
    this._smokeCur = (this._smokeCur + 1) % 3;
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

  /** Resets the rotation so the next step reads pair 0. Call after uploading fields. */
  resetFlipState() {
    this._velCur = 0;
    this._smokeCur = 0;
  }

  writeSolidMask(data) { this.device.queue.writeBuffer(this.s, 0, data); }

  /** Writes u to every velocity pair, so no pair holds stale data. */
  writeVelocityU(data) {
    for (const p of this.velPairs) this.device.queue.writeBuffer(p.u, 0, data);
  }
  writeVelocityV(data) {
    for (const p of this.velPairs) this.device.queue.writeBuffer(p.v, 0, data);
  }
  writeSmoke(data) {
    for (const b of this.smokeBufs) this.device.queue.writeBuffer(b, 0, data);
  }

  /**
   * Re-applies one column of u to every velocity pair. Used for the inflow BC,
   * which must survive whichever pair the rotation currently reads.
   * @param {number} col - column index (i)
   * @param {Float32Array} data - source array
   * @param {number} srcOffset - element offset into `data`
   * @param {number} count - element count to copy
   */
  writeInflowColumn(col, data, srcOffset, count) {
    const byteOffset = col * this.numY * 4;
    for (const p of this.velPairs) {
      this.device.queue.writeBuffer(p.u, byteOffset, data, srcOffset, count);
    }
  }

  /**
   * Writes smoke values at a single cell index to every smoke buffer.
   * @param {number} index - flat cell index (i * numY + j)
   * @param {Float32Array} data - values to write at that index
   */
  writeSmokeCell(index, data) {
    for (const b of this.smokeBufs) this.device.queue.writeBuffer(b, index * 4, data);
  }

  get pressureBuffer()  { return this.p; }
  get solidBuffer()     { return this.s; }
  /** Returns the smoke buffer that holds the most recent advection output. */
  get smokeBuffer()     { return this.smokeBufs[this._smokeCur]; }
  /** Returns the velocity buffers that hold the most recent advection output. */
  get velocityBuffers() { return this.velPairs[this._velCur]; }
}
