/**
 * GPU-based 2D incompressible fluid solver using WebGPU compute shaders.
 *
 * Runs pressure solve (red-black Gauss-Seidel), boundary extrapolation,
 * velocity advection, smoke advection, and explicit viscous diffusion entirely
 * on the GPU. Velocity and smoke rotate through three buffer slots to avoid
 * read/write hazards; the viscous substep count's parity is what makes the two
 * rotations diverge.
 */
export class FluidSolver {
  /** Maximum viscous substeps per frame.
   *
   *  Beyond this the requested viscosity is SATURATED at `viscNuMax` rather
   *  than the substep count being truncated. Truncating the count while
   *  keeping `dt_sub = dt / N` leaves the explicit coefficient above the 1/4
   *  five-point limit by the factor `want / N_MAX`, and the worst mode then
   *  amplifies by `|1 - 8*coeff|` per substep -- at tier 256, nu = 0.1 that is
   *  ~12.7^32, i.e. Inf then NaN inside a single frame, propagated everywhere
   *  by the next pressure solve and unrecoverable without a preset reload.
   *
   *  Saturating nu instead keeps the coefficient at exactly 1/4: stable and
   *  bounded, under-diffusive rather than divergent. `viscClamped` then means
   *  "the requested nu was saturated, so the EFFECTIVE Reynolds number is
   *  HIGHER than the one requested" -- see `viscNuMax` / `viscNuEff`. */
  static N_MAX = 32;

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

    this.params = { numX, numY, h, dt: 1 / 60, omega: 1.9, density: 1000, color: 0, nu: 0 };

    /**
     * Simulation time elapsed: accumulated steps * dt, in the same seconds the
     * timestep is expressed in.
     *
     * Any measurement of a FREQUENCY must be timestamped from here, never from
     * wall time. The loop takes exactly one step per displayed frame and
     * `adaptive` changes the grid tier under load, so wall time and simulation
     * time run at a ratio that is neither fixed nor 1 — at dt = 1/240 and
     * 60 fps it is a quarter, and it moves whenever the frame rate does.
     */
    this.simTime = 0;

    /** Viscous substeps dispatched by the last step(). 0 when nu == 0. */
    this.viscSubsteps = 0;
    /** True when the last step() saturated nu at `viscNuMax`. The field is
     *  still stable; the effective viscosity is LOWER (effective Re HIGHER)
     *  than requested. */
    this.viscClamped = false;
    /** The nu the last step() actually applied: min(params.nu, viscNuMax). */
    this.viscNuEff = 0;

    this._createBuffers(numX, numY);
  }

  /**
   * Allocates all GPU buffers for the simulation grid.
   *
   * Velocity and smoke live in a 3-slot rotation (see `velPairs`/`smokeBufs`).
   * Pressure (p) and the solid mask (s) are single buffers. Also creates five
   * uniform buffers: one general-purpose, two for the red/black pressure solve
   * (which differ only in the color flag), one with dt negated for MacCormack's
   * backward pass, and one carrying the viscous substep dt.
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
    // Boundary mask: the preset's permanent solids (walls, step), uploaded
    // once per preset load. The obstacle rasterizer reads it to restore
    // vacated cells and to never carve permanent boundary cells.
    this.sBoundary = device.createBuffer({ size: size * 4, usage: storageUsage });
    // Frozen snapshot of s at the start of rasterizeObstacle. The three
    // per-slot dispatches all update s, so the smoke-clear test cannot read
    // the live s without seeing progressively restored values; sOld gives
    // every dispatch the original mask.
    this.sOld = device.createBuffer({ size: size * 4, usage: storageUsage });

    // Uniform buffers: red/black variants carry color=0 and color=1 respectively.
    // COPY_SRC so tests can read back what was actually uploaded -- the sign of
    // dt in uniformBufNegDt is not observable any other way.
    const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.uniformBuf      = device.createBuffer({ size: 32, usage: uniformUsage });
    this.uniformBufRed   = device.createBuffer({ size: 32, usage: uniformUsage });
    this.uniformBufBlack = device.createBuffer({ size: 32, usage: uniformUsage });
    // Identical to uniformBuf but with dt negated, so the MacCormack backward
    // pass can reuse the advect_smoke entry point unchanged.
    this.uniformBufNegDt = device.createBuffer({ size: 32, usage: uniformUsage });
    // Identical to uniformBuf but with dt replaced by the viscous substep dt.
    // Written per step, only when the viscous pass is actually dispatched.
    this.uniformBufVisc  = device.createBuffer({ size: 32, usage: uniformUsage });
    // Rasterizer uniforms: 64 bytes. Layout must match RasterParams in
    // rasterize_obstacle.wgsl: numX(0) numY(4) h(8) shape(12) center(16,20)
    // vel(24,28) radius(32) angle(36) pad(40-48) prevBBox vec4i(48-64).
    this.uniformBufRaster = device.createBuffer({ size: 64, usage: uniformUsage });
  }

  /**
   * Packs simulation parameters into a 32-byte ArrayBuffer and uploads
   * to the main uniform buffer. Layout must match the WGSL struct:
   * [numX(u32), numY(u32), h(f32), dt(f32), omega(f32), density(f32), color(u32), nu(f32)].
   * The older 7-field shaders (pressure/boundary) simply ignore nu; advect and
   * the rest declare all 8 fields.
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
    dv.setFloat32(28, p.nu ?? 0, true);
    this.device.queue.writeBuffer(this.uniformBuf, 0, ab);
  }

  /**
   * Packs and uploads simulation parameters to a specific uniform buffer.
   * Used to write distinct color values to the red and black uniform buffers.
   *
   * @param {GPUBuffer} buf - Target uniform buffer
   * @param {number} [colorOverride] - If provided, overrides the color field
   * @param {number} [dtOverride] - If provided, overrides dt (negated for MacCormack's backward pass)
   * @param {number} [nuOverride] - If provided, overrides nu (saturated for the viscous pass)
   */
  _writeParamsTo(buf, colorOverride, dtOverride, nuOverride) {
    const p = this.params;
    const color = colorOverride !== undefined ? colorOverride : p.color;
    const dt = dtOverride !== undefined ? dtOverride : p.dt;
    const nu = nuOverride !== undefined ? nuOverride : (p.nu ?? 0);
    const ab = new ArrayBuffer(32);
    const dv = new DataView(ab);
    dv.setUint32(0,  p.numX,   true);
    dv.setUint32(4,  p.numY,   true);
    dv.setFloat32(8,  p.h,     true);
    dv.setFloat32(12, dt,      true);
    dv.setFloat32(16, p.omega,  true);
    dv.setFloat32(20, p.density, true);
    dv.setUint32(24, color,    true);
    dv.setFloat32(28, nu,      true);
    this.device.queue.writeBuffer(buf, 0, ab);
  }

  /** Uploads params to every uniform buffer, including the negated-dt variant.
   *
   *  uniformBufVisc is seeded here too, even though step() rewrites it with the
   *  substep dt before dispatching: an unseeded buffer is all zeros, so numX =
   *  numY = 0 would make every diffuse thread return at the bounds check -- a
   *  silent no-op rather than a validation error if some future path ever
   *  dispatches the viscous pass without the per-step write. */
  _writeAllParams() {
    this.writeParams();
    this._writeParamsTo(this.uniformBufRed, 0);
    this._writeParamsTo(this.uniformBufBlack, 1);
    this._writeParamsTo(this.uniformBufNegDt, undefined, -this.params.dt);
    this._writeParamsTo(this.uniformBufVisc, 0);
  }

  /**
   * The largest kinematic viscosity the explicit viscous pass can represent at
   * the current grid spacing and timestep: N_MAX substeps each sitting exactly
   * on the 1/4 five-point stability limit.
   *
   * Requests above this are saturated here rather than run unstably, so this is
   * also the honest floor of the achievable Reynolds number: for a body of
   * diameter D in a free stream U, `Re_min = U * D / viscNuMax`. A UI offering
   * a Reynolds control must clamp to that floor (or surface it) rather than
   * accept a value it cannot deliver.
   */
  get viscNuMax() {
    return FluidSolver.N_MAX * 0.25 * this.h * this.h / this.params.dt;
  }

  /** Releases all GPU buffers. Must be called before resize or disposal. */
  destroy() {
    for (const pair of this.velPairs) { pair.u.destroy(); pair.v.destroy(); }
    for (const b of this.smokeBufs) b.destroy();
    this.p.destroy();
    this.s.destroy();
    this.sBoundary.destroy();
    this.sOld.destroy();
    this.uniformBuf.destroy();
    this.uniformBufRed.destroy();
    this.uniformBufBlack.destroy();
    this.uniformBufNegDt.destroy();
    this.uniformBufVisc.destroy();
    this.uniformBufRaster.destroy();
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

    const [pressureWgsl, boundaryWgsl, advectWgsl, advectSmokeWgsl, maccormackWgsl, maccormackVelWgsl, diffuseWgsl, rasterizeWgsl] = await Promise.all([
      fetch('/shaders/pressure.wgsl').then(r => r.text()),
      fetch('/shaders/boundary.wgsl').then(r => r.text()),
      fetch('/shaders/advect.wgsl').then(r => r.text()),
      fetch('/shaders/advect_smoke.wgsl').then(r => r.text()),
      fetch('/shaders/maccormack.wgsl').then(r => r.text()),
      fetch('/shaders/maccormack_velocity.wgsl').then(r => r.text()),
      fetch('/shaders/diffuse.wgsl').then(r => r.text()),
      fetch('/shaders/rasterize_obstacle.wgsl').then(r => r.text()),
    ]);

    const pressureMod      = device.createShaderModule({ code: pressureWgsl });
    const boundaryMod      = device.createShaderModule({ code: boundaryWgsl });
    const advectMod        = device.createShaderModule({ code: advectWgsl });
    const advectSmokeMod   = device.createShaderModule({ code: advectSmokeWgsl });
    const maccormackMod    = device.createShaderModule({ code: maccormackWgsl });
    const maccormackVelMod = device.createShaderModule({ code: maccormackVelWgsl });
    const diffuseMod       = device.createShaderModule({ code: diffuseWgsl });
    const rasterizeMod     = device.createShaderModule({ code: rasterizeWgsl });

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

    // Velocity advect: uniform + u^n,v^n + s + fu,fv + outU,outV = 7 storage.
    // Bindings 1/2 are the advecting velocity AND phi^n; 4/5 are the field
    // being advected. On the forward pass the same buffers land on both pairs.
    solver._advectVelBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, RO_STORAGE), bglEntry(4, RO_STORAGE), bglEntry(5, RO_STORAGE),
                bglEntry(6, STORAGE), bglEntry(7, STORAGE)],
    });

    // Velocity combine: uniform + u^n,v^n + uHat,vHat + uTilde,vTilde (rw, in
    // place) = 6 storage. No `s`: unlike the smoke combine, the velocity
    // limiter keeps solid stencil corners, and the phi^ seed alone makes the
    // clamp the identity wherever a face reverted. See maccormack_velocity.wgsl.
    solver._mcVelBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, RO_STORAGE), bglEntry(4, RO_STORAGE),
                bglEntry(5, STORAGE), bglEntry(6, STORAGE)],
    });

    // Smoke advect: uniform + u,v,s + mIn,mOrig + mOut = 6 storage buffers.
    solver._advectSmokeBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, RO_STORAGE), bglEntry(4, RO_STORAGE), bglEntry(5, RO_STORAGE),
                bglEntry(6, STORAGE)],
    });

    // Smoke combine: uniform + u,v + phi^n + phi^ + phi~ (rw, in place) + s
    // = 6 storage buffers. `s` lets the combine skip solid cells and drop
    // solid corners from the limiter bounds.
    solver._mcSmokeBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, RO_STORAGE), bglEntry(4, RO_STORAGE), bglEntry(5, STORAGE),
                bglEntry(6, RO_STORAGE)],
    });

    // Viscous diffusion: uniform + s + uIn,vIn + uOut,vOut = 5 storage buffers.
    // Well under maxStorageBuffersPerShaderStage (8); the pass reads the whole
    // stencil from one velocity slot and writes another, so no aliasing.
    solver._diffuseBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, RO_STORAGE),
                bglEntry(2, RO_STORAGE), bglEntry(3, RO_STORAGE),
                bglEntry(4, STORAGE), bglEntry(5, STORAGE)],
    });

    // Obstacle rasterizer: uniform + s(rw) + sBoundary(ro) + u,v(rw) + p(rw)
    // + smoke(rw) + sOld(ro) = 7 storage buffers, at but not over
    // maxStorageBuffersPerShaderStage (8). One dispatch per rotation slot;
    // writes to s and p are idempotent. sOld is a frozen snapshot of s so
    // the smoke-clear test sees the original mask on every slot.
    solver._rasterizeBGL = device.createBindGroupLayout({
      entries: [bglEntry(0, UNIFORM), bglEntry(1, STORAGE), bglEntry(2, RO_STORAGE),
                bglEntry(3, STORAGE), bglEntry(4, STORAGE), bglEntry(5, STORAGE),
                bglEntry(6, STORAGE), bglEntry(7, RO_STORAGE)],
    });

    const makePipelineLayout = (bgl) => device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    solver.pressurePipeline    = device.createComputePipeline({ layout: makePipelineLayout(solver._pressureBGL),    compute: { module: pressureMod,     entryPoint: 'main' } });
    solver.boundaryHPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._boundaryBGL),    compute: { module: boundaryMod,     entryPoint: 'extrapolate_horizontal' } });
    solver.boundaryVPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._boundaryBGL),    compute: { module: boundaryMod,     entryPoint: 'extrapolate_vertical' } });
    solver.advectVelPipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._advectVelBGL),   compute: { module: advectMod,       entryPoint: 'advect_velocity' } });
    solver.advectSmokePipeline = device.createComputePipeline({ layout: makePipelineLayout(solver._advectSmokeBGL), compute: { module: advectSmokeMod,  entryPoint: 'advect_smoke' } });
    solver.mcSmokePipeline     = device.createComputePipeline({ layout: makePipelineLayout(solver._mcSmokeBGL),     compute: { module: maccormackMod,   entryPoint: 'maccormack_smoke' } });
    solver.mcVelPipeline       = device.createComputePipeline({ layout: makePipelineLayout(solver._mcVelBGL),      compute: { module: maccormackVelMod, entryPoint: 'maccormack_velocity' } });
    solver.diffusePipeline     = device.createComputePipeline({ layout: makePipelineLayout(solver._diffuseBGL),   compute: { module: diffuseMod,      entryPoint: 'diffuse' } });
    solver.rasterizePipeline   = device.createComputePipeline({ layout: makePipelineLayout(solver._rasterizeBGL), compute: { module: rasterizeMod,   entryPoint: 'rasterize' } });

    solver._createBindGroups();

    // Write initial uniform data
    solver._writeAllParams();

    return solver;
  }

  /**
   * Creates all bind groups for the compute pipelines, indexed by rotation slot.
   *
   * Pressure and boundary get one variant per velocity pair. Velocity
   * MacCormack is indexed by the velocity slot alone. Smoke MacCormack is
   * indexed by BOTH the velocity slot and the smoke slot, because the velocity
   * field that carries the dye is not tied to the smoke rotation.
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

    // MacCormack velocity: forward -> backward -> combine. Indexed by the
    // velocity slot alone -- velocity is both the advecting field and the
    // advected one, so there is no second axis (unlike smoke below).
    //
    // Slot roles for velocity slot c: phi^n = c, phi^ = hat = (c+1)%3,
    // phi~ = tilde = (c+2)%3. The combine writes phi^{n+1} in place into tilde.
    this.velFwd = [];
    this.velBack = [];
    this.velCombine = [];
    for (let c = 0; c < 3; c++) {
      const nPair = this.velPairs[c];
      const hat = this.velPairs[(c + 1) % 3];
      const tilde = this.velPairs[(c + 2) % 3];

      // Forward: dt > 0, the advected field (4/5) IS phi^n (1/2), writes phi^.
      // Aliasing one buffer onto two read-only bindings is legal.
      this.velFwd.push(device.createBindGroup({
        layout: this._advectVelBGL,
        entries: [entry(0, this.uniformBuf),
                  entry(1, nPair.u), entry(2, nPair.v), entry(3, this.s),
                  entry(4, nPair.u), entry(5, nPair.v),
                  entry(6, hat.u), entry(7, hat.v)],
      }));

      // Backward: dt < 0 via uniformBufNegDt, advects phi^ (4/5) with the
      // time-n velocity still on 1/2, writes phi~. Keeping phi^n on 1/2 is
      // what makes a reverted face write phi^n rather than phi^ -- the
      // velocity equivalent of the smoke path's separate mOrig binding.
      this.velBack.push(device.createBindGroup({
        layout: this._advectVelBGL,
        entries: [entry(0, this.uniformBufNegDt),
                  entry(1, nPair.u), entry(2, nPair.v), entry(3, this.s),
                  entry(4, hat.u), entry(5, hat.v),
                  entry(6, tilde.u), entry(7, tilde.v)],
      }));

      // Combine: phi^ + (phi^n - phi~)/2, limited, in place into phi~.
      // uniformBuf (positive dt): the re-trace must reproduce the FORWARD
      // stencil, so this one must never bind uniformBufNegDt.
      this.velCombine.push(device.createBindGroup({
        layout: this._mcVelBGL,
        entries: [entry(0, this.uniformBuf),
                  entry(1, nPair.u), entry(2, nPair.v),
                  entry(3, hat.u), entry(4, hat.v),
                  entry(5, tilde.u), entry(6, tilde.v)],
      }));
    }

    // MacCormack smoke: forward -> backward -> combine. Each is a 3x3 table
    // indexed [velCur][smokeCur]. All three bind u/v as the advecting velocity,
    // so all three need the velocity axis. The two slots advance by +2 together
    // today (see step()), but they are independent indices and will diverge once
    // viscous substepping lands, so keying the advecting velocity off the smoke
    // index would silently trace dye through the wrong velocity field.
    //
    // Slot roles for smoke slot sc: phi^n = sc, phi^ = hat = (sc+1)%3,
    // phi~ = tilde = (sc+2)%3. The combine writes phi^{n+1} in place into tilde.
    this.smokeFwd = [];
    this.smokeBack = [];
    this.smokeCombine = [];
    for (let vc = 0; vc < 3; vc++) {
      this.smokeFwd.push([]);
      this.smokeBack.push([]);
      this.smokeCombine.push([]);
      const vel = this.velPairs[vc];
      for (let sc = 0; sc < 3; sc++) {
        const n = this.smokeBufs[sc];
        const hat = this.smokeBufs[(sc + 1) % 3];
        const tilde = this.smokeBufs[(sc + 2) % 3];

        // Forward: dt > 0, mIn == mOrig == phi^n, writes phi^.
        this.smokeFwd[vc].push(device.createBindGroup({
          layout: this._advectSmokeBGL,
          entries: [entry(0, this.uniformBuf),
                    entry(1, vel.u), entry(2, vel.v), entry(3, this.s),
                    entry(4, n), entry(5, n), entry(6, hat)],
        }));

        // Backward: dt < 0 via uniformBufNegDt, mIn == phi^, mOrig == phi^n,
        // writes phi~. mOrig is what makes unreliable cells revert cleanly.
        this.smokeBack[vc].push(device.createBindGroup({
          layout: this._advectSmokeBGL,
          entries: [entry(0, this.uniformBufNegDt),
                    entry(1, vel.u), entry(2, vel.v), entry(3, this.s),
                    entry(4, hat), entry(5, n), entry(6, tilde)],
        }));

        // Combine: phi^ + (phi^n - phi~)/2, limited, in place into phi~.
        this.smokeCombine[vc].push(device.createBindGroup({
          layout: this._mcSmokeBGL,
          entries: [entry(0, this.uniformBuf),
                    entry(1, vel.u), entry(2, vel.v),
                    entry(3, n), entry(4, hat), entry(5, tilde), entry(6, this.s)],
        }));
      }
    }

    // Viscous diffusion: a 3x3 table indexed [src][dst] over velocity slots.
    // The substep loop alternates between the combine's output pair and the one
    // pair the step has finished with, so which off-diagonal entries get used
    // depends on _velCur; the diagonal is never used (a pass cannot read and
    // write the same buffer).
    this.diffuse = [];
    for (let src = 0; src < 3; src++) {
      this.diffuse.push([]);
      for (let dst = 0; dst < 3; dst++) {
        if (src === dst) { this.diffuse[src].push(null); continue; }
        this.diffuse[src].push(device.createBindGroup({
          layout: this._diffuseBGL,
          entries: [entry(0, this.uniformBufVisc), entry(1, this.s),
                    entry(2, this.velPairs[src].u), entry(3, this.velPairs[src].v),
                    entry(4, this.velPairs[dst].u), entry(5, this.velPairs[dst].v)],
        }));
      }
    }

    // One rasterizer bind group per rotation slot.
    this.rasterize = [];
    for (let k = 0; k < 3; k++) {
      this.rasterize.push(device.createBindGroup({
        layout: this._rasterizeBGL,
        entries: [entry(0, this.uniformBufRaster), entry(1, this.s), entry(2, this.sBoundary),
                  entry(3, this.velPairs[k].u), entry(4, this.velPairs[k].v),
                  entry(5, this.p), entry(6, this.smokeBufs[k]), entry(7, this.sOld)],
      }));
    }

    this._velCur = 0;
    this._smokeCur = 0;
  }

  /**
   * Runs one full simulation time step: pressure solve, boundary extrapolation,
   * velocity advection, smoke advection, and (when nu > 0) viscous diffusion.
   * Encodes all passes into a single command buffer and submits to the GPU
   * queue, then advances the rotation.
   *
   * Sets `viscSubsteps` and `viscClamped` as a side effect.
   *
   * @param {number} numIters - Number of red-black Gauss-Seidel pressure iterations
   */
  step(numIters) {
    const { device, numX, numY } = this;

    // Write params to all uniform buffers
    this._writeAllParams();

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

    // Advect velocity by MacCormack: forward -> backward -> limited combine.
    // Result lands in the tilde pair, (_velCur + 2) % 3. Pair _velCur itself is
    // never written, so the smoke passes below still read the projected,
    // boundary-corrected time-n velocity.
    for (const [pipeline, group] of [
      [this.advectVelPipeline, this.velFwd[this._velCur]],
      [this.advectVelPipeline, this.velBack[this._velCur]],
      [this.mcVelPipeline,     this.velCombine[this._velCur]],
    ]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    // Advect smoke by MacCormack: forward -> backward -> limited combine.
    // Result lands in the tilde slot, (_smokeCur + 2) % 3.
    //
    // All three passes trace through the SAME velocity pair the passes above
    // worked on — the projected, boundary-corrected time-n field in slot
    // _velCur. Slots _velCur + 1 and + 2 now hold the advected, unprojected
    // time-(n+1) velocity, and tracing dye through that would be a
    // semi-Lagrangian step out of sync with the velocity it is supposed to
    // follow. Hence both indices only advance after the whole command buffer
    // is encoded.
    for (const [pipeline, group] of [
      [this.advectSmokePipeline, this.smokeFwd[this._velCur][this._smokeCur]],
      [this.advectSmokePipeline, this.smokeBack[this._velCur][this._smokeCur]],
      [this.mcSmokePipeline,     this.smokeCombine[this._velCur][this._smokeCur]],
    ]) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }

    // Both MacCormack chains land their result two slots ahead: the forward
    // pass writes hat = c+1, the backward pass and the in-place combine write
    // tilde = c+2. Viscosity may move the velocity result off that slot.
    let velNext = (this._velCur + 2) % 3;

    // Viscous diffusion, encoded last so it runs after the smoke passes have
    // read pair _velCur. Within step() the order is
    //   project -> extrapolate -> advect -> diffuse,
    // which composes across successive calls into the standard splitting
    //   advect -> diffuse -> project.
    // Diffusion therefore acts on the advected, not-yet-projected field, and
    // the projection that follows removes whatever divergence it introduced.
    const nu = this.params.nu ?? 0;
    let nSub = 0, nuEff = 0;
    if (nu > 0) {
      // Five-point explicit stability limit: nu * dt_sub / h^2 <= 1/4.
      //
      // Past N_MAX substeps we saturate nu rather than truncating N. Truncating
      // N leaves the coefficient ABOVE 1/4 and the scheme divergent (see N_MAX);
      // saturating pins it at exactly 1/4, which is stable. The cost is
      // under-diffusion -- the flow runs at a higher effective Re than asked
      // for -- which viscClamped reports so the UI can say so.
      const nuMax = this.viscNuMax;
      if (nu > nuMax) {
        nuEff = nuMax;
        nSub = FluidSolver.N_MAX;
        this.viscClamped = true;
      } else {
        nuEff = nu;
        nSub = Math.max(1, Math.ceil(nu * this.params.dt / (0.25 * this.h * this.h)));
        this.viscClamped = false;
      }
    } else {
      this.viscClamped = false;
    }
    this.viscSubsteps = nSub;
    this.viscNuEff = nuEff;

    if (nSub > 0) {
      this._writeParamsTo(this.uniformBufVisc, 0, this.params.dt / nSub, nuEff);
      // Ping-pong between the combine's output (tilde) and the hat pair, which
      // this step has finished with. Pair _velCur is never touched, so the
      // smoke passes above keep reading the time-n field they were encoded
      // against even though these passes share the command buffer.
      let src = velNext, dst = (this._velCur + 1) % 3;
      for (let k = 0; k < nSub; k++) {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.diffusePipeline);
        pass.setBindGroup(0, this.diffuse[src][dst]);
        pass.dispatchWorkgroups(dx, dy, 1);
        pass.end();
        const next = src;
        src = dst;
        dst = next;
      }
      velNext = src;   // parity decides where the result landed
    }

    device.queue.submit([encoder.finish()]);

    // Advance the rotation. An ODD substep count leaves the velocity result on
    // the hat pair rather than tilde, so _velCur and _smokeCur diverge -- which
    // is exactly why the smoke bind groups are a [velCur][smokeCur] table.
    this._velCur   = velNext;
    this._smokeCur = (this._smokeCur + 2) % 3;

    this.simTime += this.params.dt;
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
   * uploads to all five uniform buffers (via `_writeAllParams`).
   *
   * @param {Object} overrides - Key/value pairs to merge (e.g., { dt: 1/120 })
   */
  setParams(overrides) {
    Object.assign(this.params, overrides);
    this._writeAllParams();
  }

  /** Resets the rotation so the next step reads pair 0. Call after uploading fields.
   *  Also clears the viscous report, which otherwise describes the previous
   *  run's last step until the next one lands, and the simulation clock — the
   *  fields are being replaced, so the run this clock was counting is over.
   *  Anything holding a timestamped series across this must clear it too, or a
   *  fresh sample will land BEFORE the samples already in the window. */
  resetFlipState() {
    this._velCur = 0;
    this._smokeCur = 0;
    this.viscSubsteps = 0;
    this.viscClamped = false;
    this.viscNuEff = 0;
    this.simTime = 0;
  }

  writeSolidMask(data) { this.device.queue.writeBuffer(this.s, 0, data); }
  writeBoundaryMask(data) { this.device.queue.writeBuffer(this.sBoundary, 0, data); }

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
   * Rasterizes the obstacle on the GPU: writes the new footprint into the
   * solid mask with the drag velocity, restores the previous footprint from
   * the boundary mask (zero velocity/pressure, smoke cleared), in every
   * rotation slot. One uniform upload + three dispatches; no CPU field
   * arrays are involved, so the live field outside the footprints is
   * untouched (the field-reset defect, ADR-0010).
   *
   * @param {Object} o
   * @param {number} o.shape - 0 circle, 1 square, 2 airfoil, 3 wedge
   * @param {number} o.centerX - obstacle centre X, sim units
   * @param {number} o.centerY - obstacle centre Y, sim units
   * @param {number} o.vx - obstacle velocity X (drag), sim units/s
   * @param {number} o.vy - obstacle velocity Y (drag), sim units/s
   * @param {number} o.radius - shape radius, sim units
   * @param {number} o.angle - rotation, radians
   * @param {number[]|null} o.prevBBox - [iMin,iMax,jMin,jMax] or null for
   *   "no previous footprint" (first rasterize after preset load; `s` must
   *   already equal the boundary mask).
   */
  rasterizeObstacle(o) {
    const ab = new ArrayBuffer(64);
    const dv = new DataView(ab);
    dv.setUint32(0, this.numX, true);
    dv.setUint32(4, this.numY, true);
    dv.setFloat32(8, this.h, true);
    dv.setUint32(12, o.shape, true);
    dv.setFloat32(16, o.centerX, true);
    dv.setFloat32(20, o.centerY, true);
    dv.setFloat32(24, o.vx, true);
    dv.setFloat32(28, o.vy, true);
    dv.setFloat32(32, o.radius, true);
    dv.setFloat32(36, o.angle, true);
    const bb = o.prevBBox ?? [1, 0, 0, 0]; // iMin > iMax = no previous
    dv.setInt32(48, bb[0], true);
    dv.setInt32(52, bb[1], true);
    dv.setInt32(56, bb[2], true);
    dv.setInt32(60, bb[3], true);

    const encoder = this.device.createCommandEncoder();
    // Snapshot the solid mask so every per-slot dispatch sees the original
    // mask when deciding where to clear smoke. s itself is updated by each
    // dispatch (idempotently), so without the snapshot the second and third
    // dispatches would see an already-restored mask and skip the clear.
    encoder.copyBufferToBuffer(this.s, 0, this.sOld, 0, this.numX * this.numY * 4);
    this.device.queue.writeBuffer(this.uniformBufRaster, 0, ab);

    const dx = Math.ceil(this.numX / 8);
    const dy = Math.ceil(this.numY / 8);
    for (let k = 0; k < 3; k++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rasterizePipeline);
      pass.setBindGroup(0, this.rasterize[k]);
      pass.dispatchWorkgroups(dx, dy, 1);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  get pressureBuffer()  { return this.p; }
  get solidBuffer()     { return this.s; }
  /** Returns the smoke buffer that holds the most recent advection output. */
  get smokeBuffer()     { return this.smokeBufs[this._smokeCur]; }
  /** Returns the velocity buffers that hold the most recent advection output. */
  get velocityBuffers() { return this.velPairs[this._velCur]; }
}
