import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';
import { UI } from './ui.js';
import { ParticleSystem } from './particles.js';
import { AdaptiveController } from './adaptive.js';

/**
 * Bootstrap the application: request a WebGPU device, create the solver/renderer/interaction
 * stack, wire up the UI, and start the simulation loop.
 */
async function init() {
    // Guard: bail early with a user-visible message if WebGPU isn't available
    if (!navigator.gpu) {
        document.getElementById('no-webgpu').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        document.getElementById('no-webgpu').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
        return;
    }
    const device = await adapter.requestDevice();

    // Populate welcome modal with adapter info
    const adapterInfo = adapter.info;
    const adapterName = adapterInfo.device || adapterInfo.description
        || [adapterInfo.vendor, adapterInfo.architecture].filter(Boolean).join(' ') || 'Unknown GPU';
    document.getElementById('gpu-adapter-name').textContent = adapterName;

    // Dismiss welcome modal
    const overlay = document.getElementById('welcome-overlay');
    const dismissWelcome = () => {
        overlay.classList.add('welcome-hidden');
        overlay.addEventListener('transitionend', () => {
            overlay.style.display = 'none';
        }, { once: true });
    };
    document.getElementById('start-sim-btn').addEventListener('click', dismissWelcome);
    document.getElementById('welcome-close-btn').addEventListener('click', dismissWelcome);

    // Surface validation/OOM errors that WebGPU would otherwise swallow
    device.addEventListener('uncapturederror', (e) => {
        console.error('WebGPU uncaptured error:', e.error);
    });

    let rafId = null;
    device.lost.then((info) => {
        console.error('GPU device lost:', info.message);
        // Stop the loop — every GPU call from here on fails, and the readbacks
        // would otherwise re-allocate staging buffers every frame forever.
        if (rafId !== null) cancelAnimationFrame(rafId);
        document.getElementById('device-lost-banner').style.display = 'block';
    });

    // Grid dimensions: numY is the vertical resolution, numX is derived from
    // the container's aspect ratio so cells are roughly square
    const container = document.getElementById('canvas-container');
    const numY = 256;
    const numX = Math.round(numY * container.clientWidth / container.clientHeight);
    const h = 1.0 / numY;

    const solver = await FluidSolver.create(device, numX, numY, h);
    const renderer = await Renderer.create(container, device, solver);
    const interaction = new Interaction(renderer.canvas, solver);

    renderer.setInteraction(interaction);
    interaction._renderer = renderer;

    const particles = new ParticleSystem();
    renderer.particleSystem = particles;
    interaction._particleSystem = particles;

    const ui = new UI(solver, renderer, interaction);

    // Welcome modal → Guide link
    document.getElementById('open-guide-from-welcome')?.addEventListener('click', (e) => {
        e.preventDefault();
        dismissWelcome();
        setTimeout(() => ui.openGuide?.(), 350);
    });

    const adaptive = new AdaptiveController(solver, renderer, interaction, ui);
    ui.adaptive = adaptive;

    // Test handle for browser-driven verification (Playwright)
    window.__flowlab = { device, solver, renderer, interaction, ui, adaptive, particles };

    // Canvas backing stores are display-resolution and set at construction, so
    // they need re-sizing when the container changes. Debounced: a drag-resize
    // fires continuously and each change reallocates the swapchain.
    //
    // Deliberately does NOT re-tier the grid. applyTier() destroys every GPU
    // buffer, reloads the preset, and clears particle emitters — running that
    // on a window resize would silently discard the user's obstacle position,
    // dye field, and emitters. Cells go slightly non-square until the next
    // explicit tier change, which is exactly how master behaved.
    let resizeTimer = null;
    new ResizeObserver(() => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            renderer.fieldRenderer.resizeCanvas();
            renderer.resizeCanvas();
        }, 150);
    }).observe(container);

    // Exponentially-smoothed frame time for the performance HUD
    let frameTimeSmoothed = 0;
    let hudCounter = 0;
    const perfHud = document.getElementById('perf-hud');

    /**
     * Main simulation loop — called once per display frame via requestAnimationFrame.
     * Sequence: re-apply boundary conditions, solver step, render, update HUD.
     */
    function frame() {
        const t0 = performance.now();
        if (!solver.paused) {
            // Re-apply smoke inlet BEFORE step (so advection picks it up)
            if (ui.smokeInletData) {
                solver.device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
            }
            solver.step(ui.numIters);
            // Re-apply inflow velocity AFTER step — the pressure solver can drift
            // the i=1 column values, so we force them back each frame.
            if (ui.boundaryVelData) {
                const bv = ui.boundaryVelData;
                const n = solver.numY;
                solver.writeInflowColumn(1, bv.uData, 1 * n, n);
            }
        }
        renderer.draw();
        const frameTime = performance.now() - t0;
        adaptive.tick(frameTime);

        // EMA smoothing (alpha = 0.1) to dampen frame-to-frame jitter in the HUD
        frameTimeSmoothed = frameTimeSmoothed * 0.9 + frameTime * 0.1;
        hudCounter++;
        if (hudCounter % 10 === 0) {
            perfHud.textContent =
                frameTimeSmoothed.toFixed(1) + ' ms/frame | ' +
                Math.round(1000 / frameTimeSmoothed) + ' fps\n' +
                'grid: ' + solver.numX + '×' + solver.numY +
                ' | iters: ' + ui.numIters;
        }

        rafId = requestAnimationFrame(frame);
    }
    rafId = requestAnimationFrame(frame);
}

init().catch((err) => {
    console.error('Initialization failed:', err);
    const banner = document.getElementById('fatal-banner');
    document.getElementById('fatal-banner-message').textContent = err.message ?? String(err);
    banner.style.display = 'block';
});
