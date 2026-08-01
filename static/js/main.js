import { trackEvent } from './analytics.js';
import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';
import { UI } from './ui.js';
import { ParticleSystem } from './particles.js';
import { Tour, STEPS } from './tour.js';
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

    // First-visit CTA: wire Skip early so clicks during fallible boot are not
    // lost, and so the button is reachable even if boot fails. The tour/start
    // path is still wired after the tour instance exists below.
    if (!Tour.readFlag()) {
        document.getElementById('start-sim-btn').addEventListener('click', () => {
            Tour.writeFlag('skipped');
            dismissWelcome();
        });
    }

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
    // Analytics: obstacle-inserted fires only on the insert-on-click path in
    // obstacle-less presets (interaction.js invokes this callback there).
    interaction.onObstacleInserted = () => trackEvent('obstacle-inserted');

    const ui = new UI(solver, renderer, interaction);

    const adaptive = new AdaptiveController(solver, renderer, interaction, ui);
    ui.adaptive = adaptive;

    const tour = new Tour({ ui, interaction, solver, steps: STEPS });
    ui.tour = tour;

    // First visit: wire the slim welcome's two paths. Returning visitors had
    // the overlay hidden by the inline script before first paint.
    if (!Tour.readFlag()) {
        document.getElementById('start-tour-btn').addEventListener('click', () => {
            dismissWelcome();
            tour.start();
        });
    }

    // Test handle for browser-driven verification (Playwright)
    window.__flowlab = { device, solver, renderer, interaction, ui, adaptive, particles, tour };

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

    // Wall-clock frame time comes from successive rAF timestamps, NOT from a
    // performance.now() bracket around the step+draw calls. Those calls only
    // ENCODE GPU work; the GPU runs it asynchronously, so the bracket measured
    // ~0.4 ms at every tier — the HUD read ~2600 fps while the display was
    // delivering 56, and `adaptive` saw a number that could never cross either
    // of its thresholds in the useful direction. The rAF delta is the interval
    // the user actually sees.
    let lastFrameTs = 0;

    // A hidden tab stops rAF entirely, so the first timestamp after it returns
    // carries the whole gap. Dropping one sample beats feeding `adaptive` a
    // multi-second "frame" that would downscale the grid on tab focus.
    document.addEventListener('visibilitychange', () => { lastFrameTs = 0; });

    /**
     * Main simulation loop — called once per display frame via requestAnimationFrame.
     * Sequence: re-apply boundary conditions, solver step, render, update HUD.
     * @param {number} ts - rAF timestamp (ms), the start of this display frame
     */
    function frame(ts) {
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
        // After draw(), so a readback landing this frame is sampled this frame.
        ui.tick();

        // The first frame, and the first after a hidden tab, have no interval
        // to report — measure from the next one rather than invent this one.
        const frameTime = lastFrameTs ? ts - lastFrameTs : 0;
        lastFrameTs = ts;

        if (frameTime > 0) {
            adaptive.tick(frameTime);

            // EMA smoothing (alpha = 0.1) to dampen frame-to-frame jitter in the
            // HUD, seeded from the first real sample so the readout does not
            // climb out of a fictitious zero over its first second.
            frameTimeSmoothed = frameTimeSmoothed > 0
                ? frameTimeSmoothed * 0.9 + frameTime * 0.1
                : frameTime;
            hudCounter++;
            if (hudCounter % 10 === 0) {
                perfHud.textContent =
                    frameTimeSmoothed.toFixed(1) + ' ms/frame | ' +
                    Math.round(1000 / frameTimeSmoothed) + ' fps\n' +
                    'grid: ' + solver.numX + '×' + solver.numY +
                    ' | iters: ' + ui.numIters;
            }
        }

        rafId = requestAnimationFrame(frame);
    }
    rafId = requestAnimationFrame(frame);
}

init().catch((err) => {
    console.error('Initialization failed:', err);
    // Hide the welcome overlay so the fatal banner and its Reload control are
    // reachable. The overlay's z-index is above the banner, and its close
    // button was only wired inside init(), which just failed.
    document.getElementById('welcome-overlay').style.display = 'none';
    const banner = document.getElementById('fatal-banner');
    document.getElementById('fatal-banner-message').textContent = err.message ?? String(err);
    banner.style.display = 'block';
});
