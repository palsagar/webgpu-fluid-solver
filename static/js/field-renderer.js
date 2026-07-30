/**
 * GPU renderer for the Field View (ADR-0005). Draws the colormapped scalar
 * field (smoke or pressure) via a WebGPU render pass with bilinear filtering
 * and a colormap LUT texture. Solid cells are drawn dark gray in-shader.
 * Overlays are NOT drawn here — they live on the 2D canvas layered above.
 */

/** Byte offsets of the uniform block — must match `RenderParams` in render_field.wgsl. */
const UNIFORMS = { numX: 0, numY: 4, minVal: 8, maxVal: 12, size: 16 };

/** Colormap LUT PNGs are 256x1 strips; the shader samples them as a 1D ramp. */
const LUT_WIDTH = 256;

/**
 * Upper bound on either canvas dimension. The field pass shades one fragment
 * per backing-store pixel, and that cost is independent of the grid tier —
 * adaptive resolution cannot claw it back. Caps the worst case on very large
 * or high-DPI displays; CSS upscales beyond this.
 * Renderer.resizeCanvas applies the identical formula — keep them in step.
 */
export const MAX_BACKING_DIM = 3840;

/** Backing-store size for a container, in device pixels, clamped. */
export function backingSize(container) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(container.clientWidth * dpr));
  const h = Math.max(1, Math.round(container.clientHeight * dpr));
  const scale = Math.min(1, MAX_BACKING_DIM / Math.max(w, h));
  return { w: Math.max(1, Math.round(w * scale)), h: Math.max(1, Math.round(h * scale)) };
}

export class FieldRenderer {
  /**
   * Argument order matches Renderer.create — keep them in step.
   * @param {HTMLElement} container - DOM element to attach the canvas to
   * @param {GPUDevice} device - WebGPU device handle
   * @param {Object} solver - Solver providing numX, numY, and solidBuffer
   */
  constructor(container, device, solver) {
    this.device = device;
    this.solver = solver;
    this.container = container;

    // Bottom layer of the canvas stack — sized to display pixels, not grid cells
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'field-canvas';
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;z-index:0;';
    this.resizeCanvas();
    container.appendChild(this.canvas);

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx = this.canvas.getContext('webgpu');
    // COPY_SRC enables captureNextFrame() — a presented WebGPU canvas cannot be
    // read back with drawImage or a page screenshot, so tests need the texture.
    this.ctx.configure({
      device,
      format: this.format,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    this.uniformBuf = device.createBuffer({
      size: UNIFORMS.size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uniformData = new DataView(new ArrayBuffer(UNIFORMS.size));

    this._fieldBindGroups = new Map(); // GPUBuffer -> GPUBindGroup, cleared on resize
    this._lutBindGroups = {};          // colormap name -> GPUBindGroup
  }

  /**
   * Matches the canvas backing store to the container's display size.
   * Called on construction and whenever the container resizes — without this
   * the backing store keeps its launch-time size and the field is CSS-stretched.
   * @returns {boolean} True if the dimensions actually changed.
   */
  resizeCanvas() {
    const { w, h } = backingSize(this.container);
    if (w === this.canvas.width && h === this.canvas.height) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    return true;
  }

  /**
   * Async factory: loads the render shader, builds the pipeline with explicit
   * bind group layouts, and uploads the colormap LUT textures.
   * @returns {Promise<FieldRenderer>}
   */
  static async create(container, device, solver) {
    const fr = new FieldRenderer(container, device, solver);

    const resp = await fetch('/shaders/render_field.wgsl');
    if (!resp.ok) throw new Error(`render_field.wgsl: HTTP ${resp.status}`);
    const module = device.createShaderModule({ code: await resp.text() });

    // createShaderModule does not throw on bad WGSL — it yields an invalid
    // module and the failure only surfaces later as a black field.
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) {
      throw new Error('render_field.wgsl failed to compile:\n' + errors.map(m => m.message).join('\n'));
    }

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
    await fr._loadLuts(['magma', 'coolwarm']);
    return fr;
  }

  /**
   * Loads 256x1 colormap PNGs as GPU textures and builds one LUT bind group each.
   * @param {string[]} names - Colormap names matching /colormaps/<name>.png
   */
  async _loadLuts(names) {
    for (const name of names) {
      const resp = await fetch(`/colormaps/${name}.png`);
      if (!resp.ok) throw new Error(`colormap ${name}: HTTP ${resp.status}`);
      const bitmap = await createImageBitmap(await resp.blob());
      if (bitmap.width !== LUT_WIDTH || bitmap.height !== 1) {
        throw new Error(`colormap ${name}: expected ${LUT_WIDTH}x1, got ${bitmap.width}x${bitmap.height}`);
      }
      const texture = this.device.createTexture({
        size: [LUT_WIDTH, 1],
        format: 'rgba8unorm',
        // RENDER_ATTACHMENT is required: copyExternalImageToTexture is
        // implemented as a render pass, and omitting it silently yields a
        // black LUT (and therefore a black field).
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [LUT_WIDTH, 1]);
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
   * Smoke rotates through three buffers (`smokeBufs`), so the cache ends up
   * holding one bind group per distinct buffer. Cleared on resize (buffers are
   * recreated).
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
   * @param {string} colormapName - 'magma' (smoke) | 'coolwarm' (pressure)
   * @param {number} minVal - Field value mapped to LUT t=0
   * @param {number} maxVal - Field value mapped to LUT t=1
   */
  draw(fieldBuffer, colormapName, minVal, maxVal) {
    const lutBG = this._lutBindGroups[colormapName];
    if (!lutBG) {
      // LUTs are loaded before create() resolves, so this only fires on an
      // unknown name — warn once rather than rendering nothing forever.
      if (!this._warnedLuts?.has(colormapName)) {
        (this._warnedLuts ??= new Set()).add(colormapName);
        console.warn(`FieldRenderer: no LUT named "${colormapName}"; field not drawn`);
      }
      return;
    }

    const dv = this._uniformData;
    dv.setUint32(UNIFORMS.numX, this.solver.numX, true);
    dv.setUint32(UNIFORMS.numY, this.solver.numY, true);
    dv.setFloat32(UNIFORMS.minVal, minVal, true);
    dv.setFloat32(UNIFORMS.maxVal, maxVal, true);
    this.device.queue.writeBuffer(this.uniformBuf, 0, dv.buffer);

    const encoder = this.device.createCommandEncoder();
    const texture = this.ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
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

    const finishCapture = this._pendingCapture ? this._encodeCapture(encoder, texture) : null;
    this.device.queue.submit([encoder.finish()]);
    if (finishCapture) finishCapture();
  }

  /**
   * Clear the field canvas to black when no scalar field is selected.
   * Mirrors draw()'s loadOp so the previous field frame does not persist.
   */
  clear() {
    const encoder = this.device.createCommandEncoder();
    const texture = this.ctx.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: 'clear',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: 'store',
      }],
    });
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Arms a one-shot pixel readback of the next rendered frame.
   * Test seam: a presented canvas texture is unreadable via drawImage or a
   * screenshot, so the copy must ride in the render pass's own encoder.
   * @returns {Promise<{width: number, height: number, data: Uint8Array}>} RGBA8
   */
  captureNextFrame() {
    return new Promise((resolve) => { this._pendingCapture = resolve; });
  }

  /**
   * Appends the capture copy to `encoder` and returns a function that starts
   * the map-and-resolve once the encoder has been submitted.
   */
  _encodeCapture(encoder, texture) {
    const resolve = this._pendingCapture;
    this._pendingCapture = null;

    const { width, height } = texture;
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256; // WebGPU row alignment
    const buffer = this.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);

    const bgra = this.format.startsWith('bgra');
    return () => {
      buffer.mapAsync(GPUMapMode.READ).then(() => {
        const padded = new Uint8Array(buffer.getMappedRange());
        const data = new Uint8Array(width * height * 4);
        for (let y = 0; y < height; y++) {
          const src = y * bytesPerRow;
          const dst = y * width * 4;
          for (let x = 0; x < width * 4; x += 4) {
            // Canvas format is platform-dependent; normalize to RGBA
            data[dst + x]     = padded[src + x + (bgra ? 2 : 0)];
            data[dst + x + 1] = padded[src + x + 1];
            data[dst + x + 2] = padded[src + x + (bgra ? 0 : 2)];
            data[dst + x + 3] = padded[src + x + 3];
          }
        }
        buffer.unmap();
        buffer.destroy();
        resolve({ width, height, data });
      });
    };
  }

  /**
   * Call after solver.resize(): the old field/solid buffers were destroyed,
   * so every cached bind group is stale.
   */
  resize() {
    this._fieldBindGroups.clear();
  }
}
