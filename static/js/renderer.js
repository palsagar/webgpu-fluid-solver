/**
 * Renderer for the 2D flow simulation.
 * Uses a 2D canvas with putImageData for field visualization (pressure/smoke)
 * and canvas drawing for overlays (streamlines, velocity arrows, particles, obstacles).
 * GPU data is read back via staging buffers for CPU-side rendering.
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

    this.readbackPending = false;
    this.fieldData = null;
    this.solidData = null;
    this._solidReadbackDone = false;

    this._velReadbackPending = false;
    this.uData = null;
    this.vData = null;
    this._velDataGen = 0;
    this._velDataVersion = -1;
    this._cachedStreamlines = null;
    this._cachedArrows = null;

    this.activeColormap = 'viridis';
    this.colormaps = {};

    // Create 2D canvas
    this._canvas = document.createElement('canvas');
    this._canvas.style.width = '100%';
    this._canvas.style.height = '100%';
    this._canvas.style.display = 'block';
    container.appendChild(this._canvas);

    this._canvas.width = this.numX;
    this._canvas.height = this.numY;

    this._ctx = this._canvas.getContext('2d');
    this._imageData = this._ctx.createImageData(this.numX, this.numY);

    this._stagingBuffer = this._createStagingBuffer(this.numX, this.numY);

    this._loadColormaps();
  }

  /**
   * Loads colormap PNG images (256x1 pixel strips) and converts them
   * to Uint8Array lookup tables for fast per-pixel color mapping.
   */
  async _loadColormaps() {
    const names = ['viridis', 'coolwarm', 'magma'];
    const offscreen = document.createElement('canvas');
    offscreen.width = 256;
    offscreen.height = 1;
    const ctx = offscreen.getContext('2d');

    for (const name of names) {
      const resp = await fetch(`/colormaps/${name}.png`);
      const blob = await resp.blob();
      const bitmap = await createImageBitmap(blob);
      ctx.drawImage(bitmap, 0, 0);
      const imageData = ctx.getImageData(0, 0, 256, 1);
      this.colormaps[name] = new Uint8Array(imageData.data.buffer);
    }
  }

  get canvas() {
    return this._canvas;
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
    // Choose which field to visualize
    const srcBuffer = this.showSmoke ? solver.smokeBuffer : solver.pressureBuffer;

    // Asynchronous GPU-to-CPU readback of the active field (smoke or pressure).
    // Only one readback is in-flight at a time to avoid mapping conflicts.
    if (!this.readbackPending) {
      this.readbackPending = true;
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(srcBuffer, 0, this._stagingBuffer, 0, this.numX * this.numY * 4);
      device.queue.submit([encoder.finish()]);

      this._stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const raw = this._stagingBuffer.getMappedRange();
        this.fieldData = new Float32Array(raw.slice(0));
        this._stagingBuffer.unmap();
        this.readbackPending = false;
      }).catch(() => { this.readbackPending = false; });
    }

    // Read solid mask once (refreshed on invalidateSolid())
    if (!this._solidReadbackDone) {
      this.readbackSolid();
    }

    if (this.fieldData) {
      this._renderField(this.fieldData);
    }

    // Velocity readback every 10 frames (not every frame) to reduce GPU stalls.
    // Needed for streamlines, arrows, and particle advection.
    this._frameCount = (this._frameCount || 0) + 1;
    if (this._frameCount % 10 === 0 && (this.showStreamlines || this.showVelocities || this.showParticles)) {
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
      this.particleSystem.draw(this._ctx, this.numX, this.numY, this.h);
    }
    if (this.interaction && this.interaction.showObstacle) {
      this.drawObstacle(this._ctx, this.interaction);
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
    if (this.particleSystem) this.particleSystem.clear();
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
    ctx.lineWidth = 1;

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
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cX(cx), cY(cy));
      ctx.lineTo(cX(ex), cY(ey));
      ctx.stroke();
      ctx.restore();
    }
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
   */
  readbackSolid() {
    const { device, solver, numX, numY } = this;
    const size = numX * numY * 4;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(solver.solidBuffer, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    staging.mapAsync(GPUMapMode.READ).then(() => {
      this.solidData = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      this._solidReadbackDone = true;
    }).catch(() => { staging.destroy(); });
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
    const { u: uBuf, v: vBuf } = solver.velocityBuffers;

    const stagingU = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const stagingV = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(uBuf, 0, stagingU, 0, size);
    encoder.copyBufferToBuffer(vBuf, 0, stagingV, 0, size);
    device.queue.submit([encoder.finish()]);

    Promise.all([stagingU.mapAsync(GPUMapMode.READ), stagingV.mapAsync(GPUMapMode.READ)]).then(() => {
      this.uData = new Float32Array(stagingU.getMappedRange().slice(0));
      this.vData = new Float32Array(stagingV.getMappedRange().slice(0));
      stagingU.unmap();
      stagingV.unmap();
      stagingU.destroy();
      stagingV.destroy();
      this._velReadbackPending = false;
      this._velDataGen++;
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
    ctx.lineWidth = 1.5;
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

    const maxArrowPx = 12;
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
        const headLen = Math.max(3, arrowPx * 0.4);
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
    ctx.lineWidth = 1.5;
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
   * Renders the scalar field (smoke or pressure) to the canvas via putImageData.
   * Maps field values through a colormap LUT, renders solid cells as dark gray,
   * and updates the colorbar UI labels and gradient.
   * @param {Float32Array} data - Scalar field values indexed as [i * numY + j]
   */
  _renderField(data) {
    const { numX, numY } = this;
    let minVal, maxVal;

    if (this.showSmoke) {
      // Smoke has a fixed [0, 1] range — 0 = dye, 1 = clear
      minVal = 0;
      maxVal = 1;
    } else {
      // Pressure: center range around the mean for diverging colormap
      minVal = data[0];
      maxVal = data[0];
      let sum = 0;
      for (let i = 1; i < data.length; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
        sum += data[i];
      }
      sum += data[0];
      const mean = sum / data.length;
      const range = Math.max(Math.abs(maxVal - mean), Math.abs(minVal - mean));
      minVal = mean - range;
      maxVal = mean + range;
    }

    const colormapName = this.showSmoke ? 'magma' : 'coolwarm';
    const colormapData = this.colormaps[colormapName];

    const pixels = this._imageData.data;

    const solid = this.solidData;

    // data is indexed [i * numY + j], display row j from bottom to top
    for (let j = 0; j < numY; j++) {
      for (let i = 0; i < numX; i++) {
        const idx = i * numY + j;
        const pixelIdx = ((numY - 1 - j) * numX + i) * 4;

        // Solid cells rendered as dark gray
        if (solid && solid[idx] === 0.0) {
          pixels[pixelIdx]     = 50;
          pixels[pixelIdx + 1] = 50;
          pixels[pixelIdx + 2] = 60;
          pixels[pixelIdx + 3] = 255;
          continue;
        }

        // Normalize value to [0,1] and look up RGBA in the colormap LUT
        const value = data[idx];
        if (colormapData) {
          const t = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal + 1e-10)));
          const lutIdx = Math.floor(t * 255) * 4;
          pixels[pixelIdx]     = colormapData[lutIdx];
          pixels[pixelIdx + 1] = colormapData[lutIdx + 1];
          pixels[pixelIdx + 2] = colormapData[lutIdx + 2];
          pixels[pixelIdx + 3] = 255;
        } else {
          const gray = Math.floor(Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal + 1e-10))) * 255);
          pixels[pixelIdx]     = gray;
          pixels[pixelIdx + 1] = gray;
          pixels[pixelIdx + 2] = gray;
          pixels[pixelIdx + 3] = 255;
        }
      }
    }

    this._ctx.putImageData(this._imageData, 0, 0);

    // Update colorbar labels and gradient
    const maxEl = document.getElementById('colorbar-max');
    const minEl = document.getElementById('colorbar-min');
    const unitEl = document.getElementById('colorbar-unit');
    const gradient = document.getElementById('colorbar-gradient');
    if (this.showSmoke) {
      if (maxEl) maxEl.textContent = 'clear';
      if (minEl) minEl.textContent = 'dye';
      if (unitEl) unitEl.textContent = '';
      if (gradient) gradient.style.background = 'linear-gradient(to bottom, #fcfdbf, #fc8961, #b73779, #51127c, #000004)';
    } else {
      const fmt = v => (Math.abs(v) > 1000 || Math.abs(v) < -1000) ? v.toExponential(1) : v.toFixed(0);
      if (maxEl) maxEl.textContent = fmt(maxVal);
      if (minEl) minEl.textContent = fmt(minVal);
      if (unitEl) unitEl.textContent = 'N/m²';
      if (gradient) gradient.style.background = 'linear-gradient(to bottom, #b40426, #f7f7f7, #3b4cc0)';
    }
  }

  /**
   * Resizes the renderer to match a new grid resolution.
   * Destroys and recreates the staging buffer, resets the canvas dimensions,
   * and clears all cached readback data and overlay geometry.
   * @param {number} numX - New grid width
   * @param {number} numY - New grid height
   * @param {number} h - New cell size
   */
  resize(numX, numY, h) {
    this._stagingBuffer.destroy();
    this.numX = numX;
    this.numY = numY;
    this.h = h;
    this._canvas.width = numX;
    this._canvas.height = numY;
    this._imageData = this._ctx.createImageData(numX, numY);
    this._stagingBuffer = this._createStagingBuffer(numX, numY);
    this.fieldData = null;
    this.solidData = null;
    this._solidReadbackDone = false;
    this.uData = null;
    this.vData = null;
    this._velDataGen = 0;
    this._velDataVersion = -1;
    this._cachedStreamlines = null;
    this._cachedArrows = null;
    if (this.particleSystem) this.particleSystem.clear();
  }
}
