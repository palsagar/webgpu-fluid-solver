import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';
import { UI } from './ui.js';
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

    const ui = new UI(solver, renderer, interaction);
    const adaptive = new AdaptiveController(solver, renderer, interaction, ui);
    ui.adaptive = adaptive;

    let frameTimeSmoothed = 0;
    let hudCounter = 0;
    const perfHud = document.getElementById('perf-hud');

    function frame() {
        const t0 = performance.now();
        if (!solver.paused) solver.step(ui.numIters);
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
