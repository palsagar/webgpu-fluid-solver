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
        if (!solver.paused) {
            // Re-apply smoke inlet BEFORE step (so advection picks it up)
            if (ui.smokeInletData) {
                solver.device.queue.writeBuffer(solver.smokeBuffer, 0, ui.smokeInletData);
            }
            solver.step(ui.numIters);
            // Re-apply boundary velocities AFTER step (so advection can't overwrite them)
            if (ui.boundaryVelData) {
                const bv = ui.boundaryVelData;
                const n = solver.numY;
                if (bv.type === 'inflow') {
                    // Write column i=1 to both u and uNew
                    solver.device.queue.writeBuffer(
                        solver.u, 1 * n * 4, bv.uData, 1 * n, n
                    );
                    solver.device.queue.writeBuffer(
                        solver.uNew, 1 * n * 4, bv.uData, 1 * n, n
                    );
                } else if (bv.type === 'lid') {
                    // Write lid velocity at j=numY-2 to both u and uNew
                    if (!bv._lidBuf) {
                        bv._lidBuf = new Float32Array(1);
                        bv._lidBuf[0] = bv.lidVel;
                    }
                    for (let i = 1; i < bv.numX - 1; i++) {
                        const offset = (i * n + (n - 2)) * 4;
                        solver.device.queue.writeBuffer(solver.u, offset, bv._lidBuf);
                        solver.device.queue.writeBuffer(solver.uNew, offset, bv._lidBuf);
                    }
                }
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
