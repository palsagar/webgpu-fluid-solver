import { FluidSolver } from './fluid-solver.js';
import { Renderer } from './renderer.js';
import { Interaction } from './interaction.js';

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

    // Set up wind tunnel initial conditions (hardcoded, presets come in Task 8)
    initWindTunnel(solver, interaction);

    function frame() {
        if (!solver.paused) {
            solver.step(40);
        }
        renderer.draw();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

function initWindTunnel(solver, interaction) {
    const { numX, numY } = solver;
    const n = numY;
    const sData = new Float32Array(numX * numY);
    const uData = new Float32Array(numX * numY);
    const mData = new Float32Array(numX * numY);
    mData.fill(1.0); // smoke = 1.0 is clear, 0.0 is dark dye

    for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
            let s = 1.0;
            if (i === 0 || j === 0 || j === numY - 1) s = 0.0;
            sData[i * n + j] = s;
            if (i === 1) uData[i * n + j] = 2.0;
        }
    }

    // Smoke inlet: central 10% of cells at x=0
    const pipeH = 0.1 * numY;
    const minJ = Math.floor(0.5 * numY - 0.5 * pipeH);
    const maxJ = Math.floor(0.5 * numY + 0.5 * pipeH);
    for (let j = minJ; j < maxJ; j++) {
        mData[j] = 0.0; // dark dye at inlet
    }

    // Set boundary mask so interaction never overwrites permanent walls
    interaction.boundaryMask = sData.slice();
    // Copy inlet velocities into interaction's uData
    interaction._uData.set(uData);

    solver.writeVelocityU(uData);
    solver.writeSmoke(mData);
    solver.setParams({ gravity: 0, omega: 1.9, dt: 1/60, density: 1000 });

    // Place initial circle obstacle at (0.4, 0.5)
    interaction.rasterizeObstacle(0.4, 0.5, 0, 0);
}

init();
