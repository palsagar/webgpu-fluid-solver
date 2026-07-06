/**
 * GPU renderer for the Field View (ADR-0005). Draws the colormapped scalar
 * field (smoke or pressure) via a WebGPU render pass with bilinear filtering
 * and a colormap LUT texture. Solid cells are drawn dark gray in-shader.
 * Overlays are NOT drawn here — they live on the 2D canvas layered above.
 */
export class FieldRenderer {
  /**
   * @param {GPUDevice} device - WebGPU device handle
   * @param {HTMLElement} container - DOM element to attach the canvas to
   * @param {Object} solver - Solver providing numX, numY, and solidBuffer
   */
  constructor(device, container, solver) {
    this.device = device;
    this.solver = solver;

    // Bottom layer of the canvas stack — sized to display pixels, not grid cells
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'field-canvas';
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;z-index:0;';
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.round(container.clientWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(container.clientHeight * dpr));
    container.appendChild(this.canvas);

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx = this.canvas.getContext('webgpu');
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

    // Uniforms: [numX u32, numY u32, minVal f32, maxVal f32] — must match render_field.wgsl
    this.uniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._fieldBindGroups = new Map(); // GPUBuffer -> GPUBindGroup, cleared on resize
    this._lutBindGroups = {};          // colormap name -> GPUBindGroup
  }

  /**
   * Async factory: loads the render shader, builds the pipeline with explicit
   * bind group layouts, and uploads the colormap LUT textures.
   * @returns {Promise<FieldRenderer>}
   */
  static async create(device, container, solver) {
    const fr = new FieldRenderer(device, container, solver);

    const code = await fetch('/shaders/render_field.wgsl').then(r => r.text());
    const module = device.createShaderModule({ code });

    // Explicit bind group layouts (ADR-0002)
    fr._fieldBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });
    fr._lutBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    fr.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [fr._fieldBGL, fr._lutBGL] }),
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: fr.format }] },
      primitive: { topology: 'triangle-list' },
    });

    fr.sampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });
    await fr._loadLuts(['magma', 'coolwarm', 'viridis']);
    return fr;
  }

  /**
   * Loads 256x1 colormap PNGs as GPU textures and builds one LUT bind group each.
   * @param {string[]} names - Colormap names matching /colormaps/<name>.png
   */
  async _loadLuts(names) {
    for (const name of names) {
      const blob = await fetch(`/colormaps/${name}.png`).then(r => r.blob());
      const bitmap = await createImageBitmap(blob);
      const texture = this.device.createTexture({
        size: [256, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [256, 1]);
      this._lutBindGroups[name] = this.device.createBindGroup({
        layout: this._lutBGL,
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: this.sampler },
        ],
      });
    }
  }

  /**
   * Returns (and caches) the bind group for a specific field buffer.
   * Smoke ping-pongs between two buffers, so the cache ends up holding one
   * bind group per distinct buffer. Cleared on resize (buffers are recreated).
   * @param {GPUBuffer} fieldBuffer
   */
  _bindGroupFor(fieldBuffer) {
    let bg = this._fieldBindGroups.get(fieldBuffer);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this._fieldBGL,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuf } },
          { binding: 1, resource: { buffer: fieldBuffer } },
          { binding: 2, resource: { buffer: this.solver.solidBuffer } },
        ],
      });
      this._fieldBindGroups.set(fieldBuffer, bg);
    }
    return bg;
  }

  /**
   * Renders one frame of the Field View.
   * @param {GPUBuffer} fieldBuffer - Scalar field to display (smoke or pressure)
   * @param {string} colormapName - 'magma' | 'coolwarm' | 'viridis'
   * @param {number} minVal - Field value mapped to LUT t=0
   * @param {number} maxVal - Field value mapped to LUT t=1
   */
  draw(fieldBuffer, colormapName, minVal, maxVal) {
    const lutBG = this._lutBindGroups[colormapName];
    if (!lutBG) return; // LUT textures still loading — skip this frame

    const ab = new ArrayBuffer(16);
    const dv = new DataView(ab);
    dv.setUint32(0, this.solver.numX, true);
    dv.setUint32(4, this.solver.numY, true);
    dv.setFloat32(8, minVal, true);
    dv.setFloat32(12, maxVal, true);
    this.device.queue.writeBuffer(this.uniformBuf, 0, ab);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroupFor(fieldBuffer));
    pass.setBindGroup(1, lutBG);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Call after solver.resize(): the old field/solid buffers were destroyed,
   * so every cached bind group is stale.
   */
  resize() {
    this._fieldBindGroups.clear();
  }
}
