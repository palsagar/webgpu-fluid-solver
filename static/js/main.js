import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';
import { UI } from './ui.js';

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

    const ui = new UI(solver, renderer, interaction);

    function frame() {
        if (!solver.paused) {
            solver.step(ui.numIters);
        }
        renderer.draw();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

init();
