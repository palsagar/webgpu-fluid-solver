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
    document.querySelector('.welcome-close').addEventListener('click', dismissWelcome);

    device.lost.then((info) => {
        console.error('GPU device lost:', info.message);
        document.getElementById('device-lost-banner').style.display = 'block';
    });

    // Grid dimensions: numY is the vertical resolution, numX is derived from
    // the container's aspect ratio so cells are roughly square
    const container = document.getElementById('canvas-container');
    const numY = 256;
    const numX = Math.round(numY * container.clientWidth / container.clientHeight);
    const h = 1.0 / numY;

    const solver = await FluidSolver.create(device, numX, numY, h);
    const renderer = new Renderer(container, device, solver);
    const interaction = new Interaction(renderer.canvas, solver);

    renderer.setInteraction(interaction);
    interaction._renderer = renderer;

    const particles = new ParticleSystem();
    renderer.particleSystem = particles;
    interaction._particleSystem = particles;

    const ui = new UI(solver, renderer, interaction);
    const adaptive = new AdaptiveController(solver, renderer, interaction, ui);
    ui.adaptive = adaptive;

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
            // Must write to BOTH ping-pong buffers since the solver alternates reads.
            if (ui.boundaryVelData) {
                const bv = ui.boundaryVelData;
                const n = solver.numY;
                solver.device.queue.writeBuffer(solver.u, 1 * n * 4, bv.uData, 1 * n, n);
                solver.device.queue.writeBuffer(solver.uNew, 1 * n * 4, bv.uData, 1 * n, n);
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

        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

init();
