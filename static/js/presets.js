/**
 * Preset configurations for the fluid simulation.
 * Each preset defines solver parameters, boundary conditions, obstacle
 * placement, and default visualization toggles.
 *
 * - numIters: pressure solver Jacobi iterations per step (higher = more accurate)
 * - dt: simulation timestep in seconds
 * - omega: SOR over-relaxation factor for the pressure solver
 * - inVel: horizontal inflow velocity at the left boundary
 * - obstacle: optional obstacle definition (shape, position in [0,1] domain coords, radius)
 * - boundaryType: 'windTunnel' (open right) or 'backwardStep' (step geometry on left wall)
 */
export const PRESETS = {
  windTunnel: {
    name: 'Wind Tunnel',
    numIters: 40, dt: 1/60, omega: 1.9, inVel: 2.0,
    obstacle: { shape: 'circle', x: 0.4, y: 0.5, radius: 0.15 },
    boundaryType: 'windTunnel',
    show: { pressure: false, smoke: true, streamlines: false, velocities: false },
  },
  karmanVortex: {
    name: 'Kármán Vortex',
    // Higher iteration count and smaller timestep for resolving vortex shedding
    numIters: 80, dt: 1/120, omega: 1.9, inVel: 1.0,
    // Small obstacle to trigger periodic vortex shedding
    obstacle: { shape: 'circle', x: 0.3, y: 0.5, radius: 0.06 },
    boundaryType: 'windTunnel',
    show: { pressure: false, smoke: true, streamlines: false, velocities: false },
  },
  backwardStep: {
    name: 'Backward Step',
    numIters: 60, dt: 1/60, omega: 1.9, inVel: 1.5,
    obstacle: null,
    boundaryType: 'backwardStep',
    // Step occupies left 30% of domain, bottom 50% height
    stepGeometry: { x0: 0, x1: 0.3, y0: 0, y1: 0.5 },
    show: { pressure: false, smoke: true, streamlines: false, velocities: false },
  },
};

/**
 * Initializes the solver, interaction handler, and GPU buffers for a named preset.
 * Resets all velocity/pressure/smoke fields, configures boundary conditions and
 * solid masks, places obstacles, and returns per-frame data needed by the main loop.
 *
 * @param {string} name - Key into PRESETS (e.g. 'karmanVortex')
 * @param {Object} solver - The GPU fluid solver (must already be sized)
 * @param {Object} interaction - The obstacle/interaction handler
 * @returns {{ show: Object, numIters: number, smokeInletData: Float32Array|null, boundaryVelData: Object|null }}
 *   - show: which visualization layers to enable
 *   - numIters: pressure iterations per solver step
 *   - smokeInletData: first-column dye values to re-inject each frame (null if no inflow)
 *   - boundaryVelData: inflow velocity array to re-apply each frame after the pressure solve
 */
export function loadPreset(name, solver, interaction) {
  const preset = PRESETS[name];
  const { numX, numY, h } = solver;
  const n = numY;

  solver.setParams({ dt: preset.dt, omega: preset.omega, density: 1000 });

  // Reset fields — zero velocity, pressure; clear smoke (m=1.0 = no dye)
  const zeros = new Float32Array(numX * numY);
  solver.writeVelocityU(zeros);
  solver.writeVelocityV(zeros);
  solver.device.queue.writeBuffer(solver.p, 0, zeros);

  const mData = new Float32Array(numX * numY);
  mData.fill(1.0); // 1.0 = clear, 0.0 = dark dye

  const sData = new Float32Array(numX * numY); // solid mask: 0 = solid, 1 = fluid
  const uData = new Float32Array(numX * numY);

  const domainWidth  = numX * h;
  const domainHeight = numY * h;

  const bt = preset.boundaryType;

  if (bt === 'windTunnel') {
    const inVel = preset.inVel;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        let s = 1.0;
        // Solid walls on left, top, and bottom edges (right edge is open outflow)
        if (i === 0 || j === 0 || j === numY - 1) s = 0.0;
        sData[i * n + j] = s;
        // Set inflow velocity at column i=1 (survives advection because i=0 is solid)
        if (i === 1) uData[i * n + j] = inVel;
      }
    }

    // Smoke inlet: central 10% of cells at x=0
    const pipeH = 0.1 * numY;
    const minJ = Math.floor(0.5 * numY - 0.5 * pipeH);
    const maxJ = Math.floor(0.5 * numY + 0.5 * pipeH);
    for (let j = minJ; j < maxJ; j++) {
      mData[j] = 0.0;
    }

  } else if (bt === 'backwardStep') {
    const sg = preset.stepGeometry;
    const inVel = preset.inVel;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        // Cell-center coords normalized to [0, 1]
        const cx = (i + 0.5) * h / domainWidth;
        const cy = (j + 0.5) * h / domainHeight;
        let s = 1.0;
        // Left wall, top and bottom walls
        if (i === 0 || j === 0 || j === numY - 1) s = 0.0;
        // Step block — solid region in lower-left corner
        if (cx < sg.x1 && cy < sg.y1) s = 0.0;
        sData[i * n + j] = s;
        // Inflow only above the step (upper portion of left edge)
        if (i === 1 && cy >= sg.y1) uData[i * n + j] = inVel;
      }
    }
    // Smoke inlet: central band of inflow cells (not all, for visual clarity)
    const inflowMinJ = Math.floor(numY * 0.45);
    const inflowMaxJ = Math.floor(numY * 0.55);
    for (let j = inflowMinJ; j < inflowMaxJ; j++) {
      if (uData[1 * n + j] > 0) mData[j] = 0.0;
    }

  }

  // Write all fields to both ping-pong buffers to avoid stale data
  solver.resetFlipState();
  solver.writeSolidMask(sData);
  solver.writeVelocityU(uData);
  solver.writeVelocityV(new Float32Array(numX * numY));
  solver.writeSmoke(mData);
  solver.device.queue.writeBuffer(solver.uNew, 0, new Float32Array(numX * numY));
  solver.device.queue.writeBuffer(solver.vNew, 0, new Float32Array(numX * numY));
  solver.device.queue.writeBuffer(solver.mNew, 0, mData);

  // Resize interaction arrays if grid size changed
  const iSize = numX * numY;
  if (!interaction._sData || interaction._sData.length !== iSize) {
    interaction._sData = new Float32Array(iSize);
    interaction._uData = new Float32Array(iSize);
    interaction._vData = new Float32Array(iSize);
  }
  interaction.boundaryMask = sData.slice();
  interaction._uData.set(uData);

  // Rasterize obstacle(s), or hide obstacle overlay if none
  interaction._prevBBox = null; // Clear stale bbox from previous grid size
  interaction.obstacleAngle = 0;
  interaction.showObstacle = false;
  if (preset.obstacle) {
    interaction.showObstacle = true;
    interaction.activeShape = preset.obstacle.shape;
    interaction.obstacleRadius = preset.obstacle.radius;
    // Convert normalized [0,1] obstacle position to domain coordinates
    const ox = preset.obstacle.x * domainWidth;
    const oy = preset.obstacle.y * domainHeight;
    interaction.rasterizeObstacle(ox, oy, 0, 0);
  }

  // Smoke inlet data (first column) — re-injected each frame by the main loop
  let smokeInletData = null;
  if (preset.inVel > 0) {
    smokeInletData = mData.slice(0, numY);
  }

  // Boundary velocity data (inflow at i=1, re-applied each frame after pressure solve)
  const boundaryVelData = preset.inVel > 0
    ? { type: 'inflow', uData: uData.slice() }
    : null;

  return { show: preset.show, numIters: preset.numIters, smokeInletData, boundaryVelData };
}
