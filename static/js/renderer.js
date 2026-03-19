export class Renderer {
  constructor(container, device, solver) {
    this.device = device;
    this.solver = solver;
    this.numX = solver.numX;
    this.numY = solver.numY;
    this.h = solver.h;

    this.showPressure = false;
    this.showSmoke = true;

    this.readbackPending = false;
    this.fieldData = null;

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
    const srcBuffer = this.showPressure ? solver.pressureBuffer : solver.smokeBuffer;

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
      });
    }

    if (this.fieldData) {
      this._renderField(this.fieldData);
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

    const colormapName = this.showPressure ? 'viridis' : 'magma';
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
  }
}
