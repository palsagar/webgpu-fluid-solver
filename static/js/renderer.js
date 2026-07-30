import { FieldRenderer, backingSize } from './field-renderer.js';
import { probeCell } from './diagnostics.js';

/**
 * Renderer for the 2D flow simulation.
 * Field View (pressure/smoke) is drawn by FieldRenderer via a WebGPU render pass
 * onto a bottom canvas. This class owns the transparent, display-resolution 2D
 * overlay canvas on top (streamlines, velocity arrows, particles, obstacle) and
 * the GPU readbacks feeding it and the particle system: velocity, solid mask,
 * and a throttled pressure-range readback for auto-ranging.
 */
export class Renderer {
  /**
   * @param {HTMLElement} container - DOM element to attach the canvas to
   * @param {GPUDevice} device - WebGPU device for buffer operations
   * @param {Object} solver - Flow solver instance providing simulation buffers and parameters
   */
  constructor(container, device, solver) {
    this.device = device;
    this.solver = solver;
    this.numX = solver.numX;
    this.numY = solver.numY;
    this.h = solver.h;

    this.showPressure = false;
    this.showSmoke = true;
    this.showStreamlines = false;
    this.showVelocities = false;
    this.showObstacle = true;
    this.interaction = null;
    this.particleSystem = null;
    this.showParticles = true;
    /** Strouhal probe, assigned by the UI. Cleared here alongside particles. */
    this.probe = null;
    /** Gates the velocity readback the Strouhal probe feeds on, and its marker.
     *  Without this in the readback condition below the probe never receives a
     *  sample and its readout sits at `measuring...` forever. */
    this.showProbe = true;

    this.readbackPending = false;
    this.solidData = null;
    this._solidReadbackDone = false;
    this._solidReadbackPending = false;
    // Bumped by invalidateSolid(). A readback issued before an invalidation
    // carries a pre-invalidation mask, so it must not mark the mask fresh.
    this._solidGen = 0;

    // Bumped on every grid resize. Readbacks capture it before mapAsync and
    // discard themselves on resolve if it moved — otherwise a readback in
    // flight across a resize writes old-sized arrays that later index out of
    // bounds against the new grid, poisoning particle positions with NaN.
    this._gridGen = 0;

    this._velReadbackPending = false;
    this.uData = null;
    this.vData = null;
    this._velDataGen = 0;
    // Simulation time of the frame the current vData was COPIED (not the frame
    // its mapAsync resolved). The probe stamps samples with this so the series
    // carries capture-frame time; stamping with the live simTime at completion
    // adds a variable readback latency and the gaps stop being whole slots.
    this._velDataSimTime = 0;
    this._velDataVersion = -1;
    this._cachedStreamlines = null;
    this._cachedArrows = null;

    // Overlay canvas (top layer): transparent, holds streamlines/arrows/particles/obstacle.
    // The Field View is drawn by FieldRenderer on a WebGPU canvas underneath.
    this._canvas = document.createElement('canvas');
    this._canvas.id = 'overlay-canvas';
    this._canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;z-index:1;display:block;';
    this.container = container;
    container.appendChild(this._canvas);
    this.resizeCanvas();

    this._ctx = this._canvas.getContext('2d');

    this._pressureRange = null; // [min, max] from the throttled pressure readback
  }

  /**
   * Async factory: creates the Renderer plus its GPU FieldRenderer.
   * DOM append order is overlay first, field canvas second — explicit
   * z-index (overlay 1, field 0) enforces the stacking either way.
   * @returns {Promise<Renderer>}
   */
  static async create(container, device, solver) {
    const renderer = new Renderer(container, device, solver);
    renderer.fieldRenderer = await FieldRenderer.create(container, device, solver);
    return renderer;
  }

  get canvas() {
    return this._canvas;
  }

  /**
   * Matches the overlay canvas backing store to the container's display size.
   * Cached overlay geometry is baked in canvas pixels, so it is discarded here.
   * @returns {boolean} True if the dimensions actually changed.
   */
  resizeCanvas() {
    const { w, h } = backingSize(this.container);
    if (w === this._canvas.width && h === this._canvas.height) return false;
    this._canvas.width = w;
    this._canvas.height = h;
    this._cachedStreamlines = null;
    this._cachedArrows = null;
    this._velDataVersion = -1; // force overlay recompute against the new pixel size
    return true;
  }

  /**
   * Ratio of overlay canvas pixels to grid cells — used to scale stroke
   * widths so overlays keep their visual weight at display resolution.
   */
  get _overlayScale() {
    return Math.max(1, this._canvas.height / this.numY);
  }

  /**
   * Creates a GPU staging buffer for reading back simulation data to the CPU.
   * @param {number} numX - Grid width
   * @param {number} numY - Grid height
   * @returns {GPUBuffer} Staging buffer with MAP_READ | COPY_DST usage
   */
  _createStagingBuffer(numX, numY) {
    return this.device.createBuffer({
      size: numX * numY * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Main per-frame render method. Orchestrates GPU readback, field rendering,
   * overlay computation/drawing, and particle advection.
   * Called every frame from the requestAnimationFrame loop.
   */
  draw() {
    const { device, solver } = this;
    const showField = this.showSmoke || this.showPressure;
    const usePressure = this.showPressure && !this.showSmoke;

    this._frameCount = (this._frameCount || 0) + 1;

    // Pressure needs a CPU-side display range (symmetric about the mean) —
    // read back every 10 frames. Smoke uses the fixed [0,1] range: no
    // field readback at all (ADR-0005).
    if (usePressure && !this.readbackPending && this._frameCount % 10 === 1) {
      this.readbackPending = true;
      // Fresh staging buffer per call, destroyed on both paths — self-contained
      // like readbackVelocity/readbackSolid. A persistent buffer here could be
      // destroyed by resize() while this mapAsync was in flight, landing
      // getMappedRange() on a freed buffer.
      const staging = this._createStagingBuffer(this.numX, this.numY);
      const gen = this._gridGen;
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(solver.pressureBuffer, 0, staging, 0, this.numX * this.numY * 4);
      device.queue.submit([encoder.finish()]);

      staging.mapAsync(GPUMapMode.READ).then(() => {
        const raw = staging.getMappedRange();
        if (gen === this._gridGen) {
          this._pressureRange = this._computePressureRange(new Float32Array(raw.slice(0)));
        }
        staging.unmap();
        staging.destroy();
        this.readbackPending = false;
      }).catch(() => {
        staging.destroy();
        this.readbackPending = false;
      });
    }

    // Read solid mask once (refreshed on invalidateSolid()) — particles need it
    if (!this._solidReadbackDone) {
      this.readbackSolid();
    }

    // GPU field render — every frame
    if (showField) {
      if (usePressure) {
        const [minV, maxV] = this._pressureRange || [-1, 1];
        this.fieldRenderer.draw(solver.pressureBuffer, 'coolwarm', minV, maxV);
      } else {
        this.fieldRenderer.draw(solver.smokeBuffer, 'magma', 0, 1);
      }
    } else {
      this.fieldRenderer.clear();
    }
    this._updateColorbar();

    // Overlay canvas: clear to transparent, then draw overlays on top
    this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);

    // Velocity readback every 10 frames (not every frame) to reduce GPU stalls.
    // Needed for streamlines, arrows, and particle advection.
    if (this._frameCount % 10 === 0 && (this.showStreamlines || this.showVelocities || this.showParticles || this.showProbe)) {
      this.readbackVelocity();
    }

    // Recompute overlays only when new velocity data arrives
    if (this.uData && this._velDataVersion !== this._velDataGen) {
      this._velDataVersion = this._velDataGen;
      if (this.showStreamlines) this._cachedStreamlines = this._computeStreamlines();
      if (this.showVelocities) this._cachedArrows = this._computeArrows();
    }

    // Draw cached overlays every frame (cheap)
    if (this.showStreamlines && this._cachedStreamlines) {
      this._drawCachedStreamlines(this._ctx, this._cachedStreamlines);
    }
    if (this.showVelocities && this._cachedArrows) {
      this._drawCachedArrows(this._ctx, this._cachedArrows);
    }
    // Particle advection and rendering — freeze when solver is paused
    if (this.particleSystem) {
      if (!this.solver.paused) {
        const dt = this.solver.params.dt;
        this.particleSystem.step(
          this.uData, this.vData, dt,
          this.h, this.numX, this.numY, this.solidData
        );
      }
      this.particleSystem.draw(this._ctx, this.numX, this.numY, this.h, this._overlayScale);
    }
    if (this.interaction && this.interaction.showObstacle) {
      this.drawObstacle(this._ctx, this.interaction);
      this.drawProbe(this._ctx, this.interaction);
    }
  }

  setInteraction(interaction) {
    this.interaction = interaction;
  }

  /**
   * Marks the solid cell mask as stale, triggering a fresh GPU readback on the next frame.
   * Call when obstacles move, presets change, or the grid is resized.
   */
  invalidateSolid() {
    this._solidReadbackDone = false;
    this._solidGen++;
    if (this.particleSystem) this.particleSystem.clear();
    // The solid mask only changes when the geometry does — an obstacle drag, a
    // shape switch, or a preset load. A Strouhal series spanning such a change
    // is a frequency fitted across two different flows, so it is discarded for
    // exactly the same reason and in exactly the same place as the particles.
    if (this.probe) this.probe.clear();
  }

  /**
   * Draws the obstacle shape on the canvas overlay.
   * Supports circle, square, NACA 0012 airfoil, and wedge geometries.
   * Coordinates are converted from simulation space to canvas pixel space.
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Object} interaction - Interaction state with obstacle position, radius, and shape
   */
  drawObstacle(ctx, interaction) {
    const { numX, numY, h } = this;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;
    // Coordinate transforms: simulation space -> canvas pixels (y-axis flipped)
    const cX = x => x / domainWidth * cw;
    const cY = y => (1 - y / domainHeight) * ch;

    const cx = interaction.obstacleX;
    const cy = interaction.obstacleY;
    const r = interaction.obstacleRadius;
    const shape = interaction.activeShape;

    const fillColor = this.showPressure ? '#000000' : '#DDDDDD';
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1 * this._overlayScale;

    const angle = interaction.obstacleAngle || 0;
    const pcx = cX(cx);
    const pcy = cY(cy);

    ctx.save();
    ctx.translate(pcx, pcy);
    ctx.rotate(-angle);
    ctx.translate(-pcx, -pcy);

    if (shape === 'circle') {
      ctx.beginPath();
      ctx.arc(cX(cx), cY(cy), r / domainWidth * cw, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    } else if (shape === 'square') {
      const hw = r / domainWidth * cw;
      const hh = r / domainHeight * ch;
      ctx.fillRect(cX(cx) - hw, cY(cy) - hh, 2 * hw, 2 * hh);
      ctx.strokeRect(cX(cx) - hw, cY(cy) - hh, 2 * hw, 2 * hh);
    } else if (shape === 'airfoil') {
      // NACA 0012 symmetric airfoil: thickness distribution as a function of chord position
      const chord = r * 4;
      const n = 20;
      const upperPts = [];
      const lowerPts = [];
      for (let k = 0; k <= n; k++) {
        const xc = k / n;
        const lx = xc * chord - chord * 0.5; // sim coords relative to center
        const yt = 5 * 0.12 * chord * (
          0.2969 * Math.sqrt(xc)
          - 0.1260 * xc
          - 0.3516 * xc * xc
          + 0.2843 * xc * xc * xc
          - 0.1015 * xc * xc * xc * xc
        );
        upperPts.push([cx + lx, cy + yt]);
        lowerPts.push([cx + lx, cy - yt]);
      }
      ctx.beginPath();
      ctx.moveTo(cX(upperPts[0][0]), cY(upperPts[0][1]));
      for (let k = 1; k <= n; k++) {
        ctx.lineTo(cX(upperPts[k][0]), cY(upperPts[k][1]));
      }
      for (let k = n; k >= 0; k--) {
        ctx.lineTo(cX(lowerPts[k][0]), cY(lowerPts[k][1]));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else if (shape === 'wedge') {
      // Symmetric wedge with 15-degree half-angle, apex facing upstream
      const wedgeLen = r * 3;
      const tanHA = Math.tan(15 * Math.PI / 180);
      const apexX = cx - wedgeLen * 0.5;
      const baseX = cx + wedgeLen * 0.5;
      const halfH = wedgeLen * tanHA;
      ctx.beginPath();
      ctx.moveTo(cX(apexX), cY(cy));
      ctx.lineTo(cX(baseX), cY(cy + halfH));
      ctx.lineTo(cX(baseX), cY(cy - halfH));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();

    if (interaction._shiftHeld) {
      const lineLen = 1.5 * r;
      const ex = cx + lineLen * Math.cos(angle);
      const ey = cy + lineLen * Math.sin(angle);
      ctx.save();
      ctx.setLineDash([3 * this._overlayScale, 3 * this._overlayScale]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.lineWidth = 1.5 * this._overlayScale;
      ctx.beginPath();
      ctx.moveTo(cX(cx), cY(cy));
      ctx.lineTo(cX(ex), cY(ey));
      ctx.stroke();
      ctx.restore();
    }
  }

  /**
   * Draws the Strouhal probe as a ringed dot at the cell it actually samples.
   *
   * Drawn from `probeCell` — the same function `ui.js` samples through — and at
   * the CELL CENTRE rather than the ideal 2D-downstream point, so the marker
   * shows where the number comes from rather than where it was asked for. When
   * `probeCell` returns null (obstacle dragged too close to the outflow) there
   * is no marker, which is what makes the readout dropping to `measuring...`
   * legible instead of mysterious.
   *
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Object} interaction - Interaction state with obstacle position and radius
   */
  drawProbe(ctx, interaction) {
    if (!this.showProbe) return;
    const { numX, numY, h } = this;
    const cell = probeCell({
      obstacleX: interaction.obstacleX,
      obstacleY: interaction.obstacleY,
      D: 2 * interaction.obstacleRadius,
      h, numX, numY,
    });
    if (!cell) return;

    const px = ((cell.i + 0.5) * h) / (numX * h) * this._canvas.width;
    const py = (1 - ((cell.j + 0.5) * h) / (numY * h)) * this._canvas.height;
    const scale = this._overlayScale;

    ctx.save();
    // Spring green: absent from both the magma field ramp and the ice-blue
    // particle trails, so the marker stays findable over either.
    ctx.strokeStyle = 'rgba(120, 255, 170, 0.95)';
    ctx.fillStyle = 'rgba(120, 255, 170, 0.95)';
    ctx.lineWidth = 1.5 * scale;
    ctx.beginPath();
    ctx.arc(px, py, 5 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 1.5 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  /**
   * Displays a brief on-screen indicator when grid resolution changes.
   * @param {number} tier - New grid resolution tier (e.g., 100, 200)
   * @param {number} direction - Positive for upscale, negative for downscale
   */
  showTierChange(tier, direction) {
    const div = document.createElement('div');
    div.className = 'tier-indicator';
    div.textContent = (direction > 0 ? '↑ ' : '↓ ') + tier + '×' + tier;
    this._canvas.parentElement.appendChild(div);
    setTimeout(() => div.remove(), 1500);
  }

  /**
   * Reads the solid cell mask (s-field) from GPU to CPU via a temporary staging buffer.
   * The mask is used to render solid cells as dark gray and to block particle advection.
   * Uses a one-shot staging buffer that is destroyed after readback completes.
   * Guarded against re-entry: dragging an obstacle calls invalidateSolid() on
   * every mousemove, which without the guard allocates a full-grid staging
   * buffer per frame (~9 MB at the 1024 tier).
   */
  readbackSolid() {
    if (this._solidReadbackPending) return;
    this._solidReadbackPending = true;

    const { device, solver, numX, numY } = this;
    const size = numX * numY * 4;
    const gen = this._gridGen;
    const solidGen = this._solidGen;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(solver.solidBuffer, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    staging.mapAsync(GPUMapMode.READ).then(() => {
      // An invalidateSolid() landing mid-flight means this copy predates the
      // new mask; leave _solidReadbackDone false so the next frame re-reads.
      if (gen === this._gridGen && solidGen === this._solidGen) {
        this.solidData = new Float32Array(staging.getMappedRange().slice(0));
        this._solidReadbackDone = true;
      }
      staging.unmap();
      staging.destroy();
      this._solidReadbackPending = false;
    }).catch(() => {
      staging.destroy();
      this._solidReadbackPending = false;
    });
  }

  /**
   * Reads u and v velocity fields from GPU to CPU via temporary staging buffers.
   * Increments _velDataGen on completion to signal that overlay geometry
   * (streamlines, arrows) should be recomputed. Only one readback in-flight at a time.
   */
  readbackVelocity() {
    if (this._velReadbackPending) return;
    this._velReadbackPending = true;

    const { device, solver, numX, numY } = this;
    const size = numX * numY * 4;
    const gen = this._gridGen;
    // Snapshot the simulation time NOW, when the buffers are copied — this is
    // the frame the sampled velocity belongs to, regardless of how many frames
    // the mapAsync takes to resolve.
    const simTimeAtCapture = solver.simTime;
    const { u: uBuf, v: vBuf } = solver.velocityBuffers;

    const stagingU = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const stagingV = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(uBuf, 0, stagingU, 0, size);
    encoder.copyBufferToBuffer(vBuf, 0, stagingV, 0, size);
    device.queue.submit([encoder.finish()]);

    Promise.all([stagingU.mapAsync(GPUMapMode.READ), stagingV.mapAsync(GPUMapMode.READ)]).then(() => {
      if (gen === this._gridGen) {
        this.uData = new Float32Array(stagingU.getMappedRange().slice(0));
        this.vData = new Float32Array(stagingV.getMappedRange().slice(0));
        this._velDataSimTime = simTimeAtCapture;
        this._velDataGen++;
      }
      stagingU.unmap();
      stagingV.unmap();
      stagingU.destroy();
      stagingV.destroy();
      this._velReadbackPending = false;
    }).catch(() => {
      try { stagingU.destroy(); } catch (_) {}
      try { stagingV.destroy(); } catch (_) {}
      this._velReadbackPending = false;
    });
  }

  /**
   * Bilinearly interpolates a velocity component at an arbitrary simulation-space position.
   * Handles the MAC (marker-and-cell) grid staggering via dx/dy offsets.
   * @param {number} x - X position in simulation coordinates
   * @param {number} y - Y position in simulation coordinates
   * @param {Float32Array} field - Velocity component data (u or v)
   * @param {number} dx - Stagger offset in x (0 for u, h/2 for v)
   * @param {number} dy - Stagger offset in y (h/2 for u, 0 for v)
   * @returns {number} Interpolated velocity value
   */
  _sampleVel(x, y, field, dx, dy) {
    const { numX, numY, h } = this;
    const h1 = 1.0 / h;
    x = Math.max(Math.min(x, numX * h), h);
    y = Math.max(Math.min(y, numY * h), h);
    const x0 = Math.max(0, Math.min(Math.floor((x - dx) * h1), numX - 1));
    const tx = ((x - dx) - x0 * h) * h1;
    const x1 = Math.min(x0 + 1, numX - 1);
    const y0 = Math.max(0, Math.min(Math.floor((y - dy) * h1), numY - 1));
    const ty = ((y - dy) - y0 * h) * h1;
    const y1 = Math.min(y0 + 1, numY - 1);
    const sx = 1.0 - tx, sy = 1.0 - ty;
    const n = numY;
    return sx*sy*field[x0*n+y0] + tx*sy*field[x1*n+y0] + tx*ty*field[x1*n+y1] + sx*ty*field[x0*n+y1];
  }

  /**
   * Computes streamline paths by integrating the velocity field from seed points.
   * Seeds are placed on a regular grid (every 5 cells). Each streamline is traced
   * forward using Euler integration with a fixed step scale.
   * @returns {Array<number[]>|null} Array of flat [x0,y0,x1,y1,...] paths in canvas pixels, or null
   */
  _computeStreamlines() {
    if (!this.uData) return null;
    const { numX, numY, h, uData, vData } = this;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;
    const numSegs = 25;
    const stepScale = 0.01;
    const paths = [];

    for (let i = 1; i < numX - 1; i += 5) {
      for (let j = 1; j < numY - 1; j += 5) {
        let x = (i + 0.5) * h;
        let y = (j + 0.5) * h;
        const pts = [x / domainWidth * cw, (1 - y / domainHeight) * ch];

        for (let s = 0; s < numSegs; s++) {
          const u = this._sampleVel(x, y, uData, 0, h / 2);
          const v = this._sampleVel(x, y, vData, h / 2, 0);
          if (u === 0 && v === 0) break;
          x += u * stepScale;
          y += v * stepScale;
          if (x < 0 || x > domainWidth || y < 0 || y > domainHeight) break;
          pts.push(x / domainWidth * cw, (1 - y / domainHeight) * ch);
        }
        if (pts.length > 2) paths.push(pts);
      }
    }
    return paths;
  }

  /**
   * Draws pre-computed streamline paths onto the canvas.
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Array<number[]>} paths - Flat coordinate arrays from _computeStreamlines
   */
  _drawCachedStreamlines(ctx, paths) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1.5 * this._overlayScale;
    for (const pts of paths) {
      ctx.beginPath();
      ctx.moveTo(pts[0], pts[1]);
      for (let k = 2; k < pts.length; k += 2) {
        ctx.lineTo(pts[k], pts[k + 1]);
      }
      ctx.stroke();
    }
  }

  /**
   * Computes velocity arrow geometry from the readback velocity field.
   * Arrows are placed on a regular grid (every 8 cells), sized proportionally
   * to velocity magnitude, and colored on a blue-to-green gradient.
   * @returns {Array<Object>|null} Array of arrow descriptors {px,py,ex,ey,r,g,b,angle,headLen}, or null
   */
  _computeArrows() {
    if (!this.uData) return null;
    const { numX, numY, h, uData, vData } = this;
    const n = numY;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;

    // First pass: find max velocity magnitude for normalization
    let maxMag = 0;
    for (let i = 0; i < numX; i += 8) {
      for (let j = 0; j < numY; j += 8) {
        const u = uData[i * n + j], v = vData[i * n + j];
        const m = Math.sqrt(u * u + v * v);
        if (m > maxMag) maxMag = m;
      }
    }
    if (maxMag === 0) return null;

    const maxArrowPx = 12 * this._overlayScale;
    const spacing = 8;
    const arrows = [];

    for (let i = spacing; i < numX - 1; i += spacing) {
      for (let j = spacing; j < numY - 1; j += spacing) {
        const u = uData[i * n + j];
        const v = vData[i * n + j];
        const mag = Math.sqrt(u * u + v * v);
        if (mag < maxMag * 0.01) continue;

        const frac = mag / maxMag;
        const arrowPx = maxArrowPx * frac;
        const px = (i + 0.5) * h / domainWidth * cw;
        const py = (1 - (j + 0.5) * h / domainHeight) * ch;
        const angle = Math.atan2(-v, u);
        const ex = px + arrowPx * Math.cos(angle);
        const ey = py + arrowPx * Math.sin(angle);
        const r = Math.floor(30 * (1 - frac));
        const g = Math.floor(80 + 175 * frac);
        const b = Math.floor(120 + 135 * frac);
        const headLen = Math.max(3 * this._overlayScale, arrowPx * 0.4);
        arrows.push({ px, py, ex, ey, r, g, b, angle, headLen });
      }
    }
    return arrows;
  }

  /**
   * Draws pre-computed velocity arrows with triangular arrowheads onto the canvas.
   * @param {CanvasRenderingContext2D} ctx - Canvas 2D context
   * @param {Array<Object>} arrows - Arrow descriptors from _computeArrows
   */
  _drawCachedArrows(ctx, arrows) {
    ctx.lineWidth = 1.5 * this._overlayScale;
    for (const a of arrows) {
      const col = `rgb(${a.r},${a.g},${a.b})`;
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(a.px, a.py);
      ctx.lineTo(a.ex, a.ey);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(a.ex, a.ey);
      ctx.lineTo(a.ex - a.headLen * Math.cos(a.angle - 0.5), a.ey - a.headLen * Math.sin(a.angle - 0.5));
      ctx.lineTo(a.ex - a.headLen * Math.cos(a.angle + 0.5), a.ey - a.headLen * Math.sin(a.angle + 0.5));
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * Computes the pressure display range: symmetric about the field mean so
   * the diverging coolwarm colormap centers on zero gauge pressure.
   * @param {Float32Array} data - Pressure field readback
   * @returns {[number, number]} [minVal, maxVal]
   */
  _computePressureRange(data) {
    let minVal = data[0], maxVal = data[0], sum = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < minVal) minVal = data[i];
      if (data[i] > maxVal) maxVal = data[i];
      sum += data[i];
    }
    const mean = sum / data.length;
    const range = Math.max(Math.abs(maxVal - mean), Math.abs(minVal - mean));
    // A uniform field (all zeros before the solver runs) would give a zero-width
    // range, which the shader maps to t=0 — a saturated blue screen instead of
    // the neutral center of the diverging colormap. Widen it to keep t at 0.5.
    if (range < 1e-10) return [mean - 1, mean + 1];
    return [mean - range, mean + range];
  }

  /**
   * Updates the colorbar labels and gradient for the active field.
   */
  _updateColorbar() {
    const maxEl = document.getElementById('colorbar-max');
    const minEl = document.getElementById('colorbar-min');
    const unitEl = document.getElementById('colorbar-unit');
    const gradient = document.getElementById('colorbar-gradient');
    const usePressure = this.showPressure && !this.showSmoke;
    const showField = this.showSmoke || this.showPressure;
    if (!showField) {
      if (maxEl) maxEl.textContent = '';
      if (minEl) minEl.textContent = '';
      if (unitEl) unitEl.textContent = '';
      if (gradient) gradient.style.background = 'linear-gradient(to bottom, #000000, #000000)';
    } else if (usePressure) {
      const [minVal, maxVal] = this._pressureRange || [-1, 1];
      const fmt = v => (Math.abs(v) > 1000 || Math.abs(v) < -1000) ? v.toExponential(1) : v.toFixed(0);
      if (maxEl) maxEl.textContent = fmt(maxVal);
      if (minEl) minEl.textContent = fmt(minVal);
      if (unitEl) unitEl.textContent = 'N/m²';
      if (gradient) gradient.style.background = 'linear-gradient(to bottom, #b40426, #f7f7f7, #3b4cc0)';
    } else {
      if (maxEl) maxEl.textContent = 'clear';
      if (minEl) minEl.textContent = 'dye';
      if (unitEl) unitEl.textContent = '';
      if (gradient) gradient.style.background = 'linear-gradient(to bottom, #fcfdbf, #fc8961, #b73779, #51127c, #000004)';
    }
  }

  /**
   * Resizes the renderer to match a new grid resolution.
   * Invalidates readbacks in flight, resets the canvas dimensions, and clears
   * all cached readback data and overlay geometry.
   * @param {number} numX - New grid width
   * @param {number} numY - New grid height
   * @param {number} h - New cell size
   */
  resize(numX, numY, h) {
    // Invalidate readbacks already in flight — they carry old-grid data.
    this._gridGen++;
    this.readbackPending = false;
    this._velReadbackPending = false;
    this._solidReadbackPending = false;

    this.numX = numX;
    this.numY = numY;
    this.h = h;
    this._pressureRange = null;
    this.fieldRenderer.resize();
    this.solidData = null;
    this._solidReadbackDone = false;
    this.uData = null;
    this.vData = null;
    this._velDataGen = 0;
    this._velDataSimTime = 0;
    this._velDataVersion = -1;
    this._cachedStreamlines = null;
    this._cachedArrows = null;
    if (this.particleSystem) this.particleSystem.clear();
    // h changed, so the probe cell moved and the flow is about to be reloaded
    // onto a different grid — the series describes neither.
    if (this.probe) this.probe.clear();
  }
}
