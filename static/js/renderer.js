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

    const segLen = h * 0.2;
    const numSegs = 15;

    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;

    for (let i = 1; i < numX - 1; i += 5) {
      for (let j = 1; j < numY - 1; j += 5) {
        let x = (i + 0.5) * h;
        let y = (j + 0.5) * h;

        ctx.beginPath();
        ctx.moveTo(cX(x), cY(y));

        for (let s = 0; s < numSegs; s++) {
          const u = this._sampleVel(x, y, uData, 0, h / 2);
          const v = this._sampleVel(x, y, vData, h / 2, 0);
          const l = Math.sqrt(u * u + v * v);
          if (l === 0) break;
          x += (u / l) * segLen;
          y += (v / l) * segLen;
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

    ctx.strokeStyle = 'rgba(68,68,68,0.6)';
    ctx.fillStyle = 'rgba(68,68,68,0.6)';
    ctx.lineWidth = 1;

    for (let i = 0; i < numX; i += 5) {
      for (let j = 0; j < numY; j += 5) {
        const u = uData[i * n + j];
        const v = vData[i * n + j];
        const mag = Math.sqrt(u * u + v * v);
        if (mag === 0) continue;

        const len = Math.min(mag * 0.02, h * 2);
        const cx = (i + 0.5) * h;
        const cy = (j + 0.5) * h;
        const ux = u / mag, uy = v / mag;

        const x0 = cX(cx);
        const y0 = cY(cy);
        const x1 = cX(cx + ux * len);
        const y1 = cY(cy + uy * len);

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        // Arrowhead
        const headLen = Math.max(2, len * cw / domainWidth * 0.3);
        const angle = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - headLen * Math.cos(angle - Math.PI / 6), y1 - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x1 - headLen * Math.cos(angle + Math.PI / 6), y1 - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  _renderField(data) {
    const { numX, numY } = this;
    let minVal = data[0];
    let maxVal = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] < minVal) minVal = data[i];
      if (data[i] > maxVal) maxVal = data[i];
    }

    const colormapName = this.showSmoke ? 'magma' : 'viridis';
    const colormapData = this.colormaps[colormapName];

    const pixels = this._imageData.data;

    // data is indexed [i * numY + j], display row j from bottom to top
    for (let j = 0; j < numY; j++) {
      for (let i = 0; i < numX; i++) {
        const value = data[i * numY + j];
        const pixelIdx = ((numY - 1 - j) * numX + i) * 4;

        if (colormapData) {
          const t = Math.max(0, Math.min(1, (value - minVal) / (maxVal - minVal + 1e-10)));
          const lutIdx = Math.floor(t * 255) * 4;
          pixels[pixelIdx]     = colormapData[lutIdx];
          pixels[pixelIdx + 1] = colormapData[lutIdx + 1];
          pixels[pixelIdx + 2] = colormapData[lutIdx + 2];
          pixels[pixelIdx + 3] = 255;
        } else {
          // fallback grayscale before colormaps load
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
    if (maxEl) maxEl.textContent = maxVal.toFixed(0);
    if (minEl) minEl.textContent = minVal.toFixed(0);
    if (unitEl) unitEl.textContent = this.showPressure ? 'N/m²' : '';
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
    this.uData = null;
    this.vData = null;
  }
}
