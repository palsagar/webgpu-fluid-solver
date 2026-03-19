export class Renderer {
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

    this.readbackPending = false;
    this.fieldData = null;
    this.solidData = null;
    this._solidReadbackDone = false;

    this._velReadbackPending = false;
    this.uData = null;
    this.vData = null;

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

  _createStagingBuffer(numX, numY) {
    return this.device.createBuffer({
      size: numX * numY * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  draw() {
    const { device, solver } = this;
    // Choose which field to visualize
    const srcBuffer = this.showSmoke ? solver.smokeBuffer : solver.pressureBuffer;

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

    this._frameCount = (this._frameCount || 0) + 1;
    if (this._frameCount % 5 === 0 && (this.showStreamlines || this.showVelocities)) {
      this.readbackVelocity();
    }

    if (this.showStreamlines && this.uData) {
      this._drawStreamlines(this._ctx);
    }
    if (this.showVelocities && this.uData) {
      this._drawVelocityArrows(this._ctx);
    }
    if (this.interaction && this.interaction.showObstacle) {
      this.drawObstacle(this._ctx, this.interaction);
    }
  }

  setInteraction(interaction) {
    this.interaction = interaction;
  }

  invalidateSolid() {
    this._solidReadbackDone = false;
  }

  drawObstacle(ctx, interaction) {
    const { numX, numY, h } = this;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;
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
  }

  showTierChange(tier, direction) {
    const div = document.createElement('div');
    div.className = 'tier-indicator';
    div.textContent = (direction > 0 ? '↑ ' : '↓ ') + tier + '×' + tier;
    this._canvas.parentElement.appendChild(div);
    setTimeout(() => div.remove(), 1500);
  }

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
    }).catch(() => {
      try { stagingU.destroy(); } catch (_) {}
      try { stagingV.destroy(); } catch (_) {}
      this._velReadbackPending = false;
    });
  }

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

  _drawStreamlines(ctx) {
    if (!this.uData) return;

    const { numX, numY, h, uData, vData } = this;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;
    const cX = x => x / domainWidth * cw;
    const cY = y => (1 - y / domainHeight) * ch;

    const numSegs = 25;
    const stepScale = 0.01; // velocity-proportional step (like original solver)

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1.5;

    for (let i = 1; i < numX - 1; i += 5) {
      for (let j = 1; j < numY - 1; j += 5) {
        let x = (i + 0.5) * h;
        let y = (j + 0.5) * h;

        ctx.beginPath();
        ctx.moveTo(cX(x), cY(y));

        for (let s = 0; s < numSegs; s++) {
          const u = this._sampleVel(x, y, uData, 0, h / 2);
          const v = this._sampleVel(x, y, vData, h / 2, 0);
          if (u === 0 && v === 0) break;
          // Velocity-proportional step (matches original solver approach)
          x += u * stepScale;
          y += v * stepScale;
          if (x < 0 || x > domainWidth || y < 0 || y > domainHeight) break;
          ctx.lineTo(cX(x), cY(y));
        }
        ctx.stroke();
      }
    }
  }

  _drawVelocityArrows(ctx) {
    if (!this.uData) return;

    const { numX, numY, h, uData, vData } = this;
    const n = numY;
    const domainWidth = numX * h;
    const domainHeight = numY * h;
    const cw = this._canvas.width;
    const ch = this._canvas.height;
    const cX = x => x / domainWidth * cw;
    const cY = y => (1 - y / domainHeight) * ch;

    // Compute max velocity for scaling
    let maxMag = 0;
    for (let i = 0; i < numX; i += 8) {
      for (let j = 0; j < numY; j += 8) {
        const u = uData[i * n + j], v = vData[i * n + j];
        const m = Math.sqrt(u * u + v * v);
        if (m > maxMag) maxMag = m;
      }
    }
    if (maxMag === 0) return;

    // Arrow length in pixels, scaled by velocity magnitude
    const maxArrowPx = 12;
    const spacing = 8; // every 8th cell

    for (let i = spacing; i < numX - 1; i += spacing) {
      for (let j = spacing; j < numY - 1; j += spacing) {
        const u = uData[i * n + j];
        const v = vData[i * n + j];
        const mag = Math.sqrt(u * u + v * v);
        if (mag < maxMag * 0.01) continue;

        const frac = mag / maxMag;
        const arrowPx = maxArrowPx * frac;

        const px = cX((i + 0.5) * h);
        const py = cY((j + 0.5) * h);
        const angle = Math.atan2(-v, u); // negative v because canvas Y is flipped

        const ex = px + arrowPx * Math.cos(angle);
        const ey = py + arrowPx * Math.sin(angle);

        // Color by magnitude: dark blue → cyan
        const r = Math.floor(30 * (1 - frac));
        const g = Math.floor(80 + 175 * frac);
        const b = Math.floor(120 + 135 * frac);
        ctx.strokeStyle = `rgb(${r},${g},${b})`;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 1.5;

        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Arrowhead
        const headLen = Math.max(3, arrowPx * 0.4);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle - 0.5), ey - headLen * Math.sin(angle - 0.5));
        ctx.lineTo(ex - headLen * Math.cos(angle + 0.5), ey - headLen * Math.sin(angle + 0.5));
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  _renderField(data) {
    const { numX, numY } = this;
    let minVal, maxVal;

    if (this.showSmoke) {
      // Smoke has a fixed [0, 1] range — 0 = dye, 1 = clear
      minVal = 0;
      maxVal = 1;
    } else {
      // Pressure: auto-range from data
      minVal = data[0];
      maxVal = data[0];
      for (let i = 1; i < data.length; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
      }
    }

    const colormapName = this.showSmoke ? 'magma' : 'viridis';
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
          pixels[pixelIdx]     = 40;
          pixels[pixelIdx + 1] = 40;
          pixels[pixelIdx + 2] = 48;
          pixels[pixelIdx + 3] = 255;
          continue;
        }

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

    // Update colorbar labels
    const maxEl = document.getElementById('colorbar-max');
    const minEl = document.getElementById('colorbar-min');
    const unitEl = document.getElementById('colorbar-unit');
    if (this.showSmoke) {
      if (maxEl) maxEl.textContent = 'clear';
      if (minEl) minEl.textContent = 'dye';
      if (unitEl) unitEl.textContent = '';
    } else {
      if (maxEl) maxEl.textContent = maxVal.toFixed(0);
      if (minEl) minEl.textContent = minVal.toFixed(0);
      if (unitEl) unitEl.textContent = 'N/m²';
    }
  }

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
  }
}
