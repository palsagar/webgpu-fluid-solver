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
    let min = data[0];
    let max = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }

    const range = max - min || 1;
    const pixels = this._imageData.data;

    // data is indexed [i * numY + j], display row j from bottom to top
    for (let j = 0; j < numY; j++) {
      for (let i = 0; i < numX; i++) {
        const val = (data[i * numY + j] - min) / range;
        const gray = Math.floor(val * 255);
        // Canvas pixel row 0 = top, simulation j=0 = bottom — flip vertically
        const pixelIdx = ((numY - 1 - j) * numX + i) * 4;
        pixels[pixelIdx]     = gray;
        pixels[pixelIdx + 1] = gray;
        pixels[pixelIdx + 2] = gray;
        pixels[pixelIdx + 3] = 255;
      }
    }

    this._ctx.putImageData(this._imageData, 0, 0);

    // Update colorbar labels
    const maxEl = document.getElementById('colorbar-max');
    const minEl = document.getElementById('colorbar-min');
    if (maxEl) maxEl.textContent = max.toFixed(3);
    if (minEl) minEl.textContent = min.toFixed(3);
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
