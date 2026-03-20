import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';
import { UI } from './ui.js';
import { ParticleSystem } from './particles.js';
import { AdaptiveController } from './adaptive.js';

async function init() {
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

    device.lost.then((info) => {
        console.error('GPU device lost:', info.message);
        document.getElementById('device-lost-banner').style.display = 'block';
    });

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

    let frameTimeSmoothed = 0;
    let hudCounter = 0;
    const perfHud = document.getElementById('perf-hud');

    function frame() {
        const t0 = performance.now();
        if (!solver.paused) {
            // Re-apply smoke inlet BEFORE step (so advection picks it up)
            if (ui.smokeInletData) {
                solver.device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
            }
            solver.step(ui.numIters);
            // Re-apply inflow velocity AFTER step (so advection can't overwrite it)
            if (ui.boundaryVelData) {
                const bv = ui.boundaryVelData;
                const n = solver.numY;
                // Write inflow column i=1 to both ping-pong buffers
                solver.device.queue.writeBuffer(solver.u, 1 * n * 4, bv.uData, 1 * n, n);
                solver.device.queue.writeBuffer(solver.uNew, 1 * n * 4, bv.uData, 1 * n, n);
            }
        }
        renderer.draw();
        const frameTime = performance.now() - t0;
        adaptive.tick(frameTime);

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
