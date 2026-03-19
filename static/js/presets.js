export const PRESETS = {
  windTunnel: {
    name: 'Wind Tunnel',
    numIters: 40, dt: 1/60, gravity: 0, omega: 1.9, inVel: 2.0,
    obstacle: { shape: 'circle', x: 0.4, y: 0.5, radius: 0.15 },
    boundaryType: 'windTunnel',
    show: { pressure: true, smoke: true, streamlines: false, velocities: false },
  },
  karmanVortex: {
    name: 'Kármán Vortex',
    numIters: 80, dt: 1/120, gravity: 0, omega: 1.9, inVel: 1.0,
    obstacle: { shape: 'circle', x: 0.3, y: 0.5, radius: 0.06 },
    boundaryType: 'windTunnel',
    show: { pressure: false, smoke: true, streamlines: false, velocities: false },
  },
  lidCavity: {
    name: 'Lid Cavity',
    numIters: 60, dt: 1/60, gravity: 0, omega: 1.9, inVel: 0,
    lidVel: 1.0,
    obstacle: null,
    boundaryType: 'cavity',
    show: { pressure: true, smoke: false, streamlines: true, velocities: false },
  },
  backwardStep: {
    name: 'Backward Step',
    numIters: 60, dt: 1/60, gravity: 0, omega: 1.9, inVel: 1.5,
    obstacle: null,
    boundaryType: 'backwardStep',
    stepGeometry: { x0: 0, x1: 0.3, y0: 0, y1: 0.5 },
    show: { pressure: true, smoke: false, streamlines: true, velocities: false },
  },
  channelFlow: {
    name: 'Channel Flow',
    numIters: 40, dt: 1/60, gravity: 0, omega: 1.9, inVel: 1.5,
    obstacles: [
      { shape: 'circle', x: 0.25, y: 0.35, radius: 0.08 },
      { shape: 'circle', x: 0.45, y: 0.65, radius: 0.08 },
      { shape: 'circle', x: 0.65, y: 0.35, radius: 0.08 },
    ],
    boundaryType: 'windTunnel',
    show: { pressure: false, smoke: true, streamlines: true, velocities: false },
  },
  sandbox: {
    name: 'Sandbox',
    numIters: 40, dt: 1/60, gravity: 0, omega: 1.0, inVel: 0,
    obstacle: null,
    boundaryType: 'box',
    paintMode: true,
    show: { pressure: false, smoke: true, streamlines: false, velocities: false },
  },
};

export function loadPreset(name, solver, interaction) {
  const preset = PRESETS[name];
  const { numX, numY, h } = solver;
  const n = numY;

  solver.setParams({ dt: preset.dt, gravity: preset.gravity, omega: preset.omega, density: 1000 });

  // Reset fields
  const zeros = new Float32Array(numX * numY);
  solver.writeVelocityU(zeros);
  solver.writeVelocityV(zeros);
  solver.device.queue.writeBuffer(solver.p, 0, zeros);

  const mData = new Float32Array(numX * numY);
  mData.fill(1.0);

  const sData = new Float32Array(numX * numY);
  const uData = new Float32Array(numX * numY);

  const domainWidth  = numX * h;
  const domainHeight = numY * h;

  const bt = preset.boundaryType;

  if (bt === 'windTunnel') {
    const inVel = preset.inVel;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        let s = 1.0;
        if (i === 0 || j === 0 || j === numY - 1) s = 0.0;
        sData[i * n + j] = s;
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

  } else if (bt === 'cavity') {
    const lidVel = preset.lidVel || 0;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        let s = 1.0;
        if (i === 0 || i === numX - 1 || j === 0 || j === numY - 1) s = 0.0;
        sData[i * n + j] = s;
        // Top wall (j === numY-2): set u = lidVel
        if (j === numY - 2) uData[i * n + j] = lidVel;
      }
    }

  } else if (bt === 'backwardStep') {
    const sg = preset.stepGeometry;
    const inVel = preset.inVel;
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        const cx = (i + 0.5) * h / domainWidth;
        const cy = (j + 0.5) * h / domainHeight;
        let s = 1.0;
        // Top and bottom walls
        if (j === 0 || j === numY - 1) s = 0.0;
        // Step block
        if (cx < sg.x1 && cy < sg.y1) s = 0.0;
        sData[i * n + j] = s;
        // Inflow on upper half of left edge
        if (i === 1 && cy >= sg.y1) uData[i * n + j] = inVel;
      }
    }

  } else if (bt === 'box') {
    for (let i = 0; i < numX; i++) {
      for (let j = 0; j < numY; j++) {
        let s = 1.0;
        if (i === 0 || i === numX - 1 || j === 0 || j === numY - 1) s = 0.0;
        sData[i * n + j] = s;
      }
    }
  }

  solver.writeSolidMask(sData);
  solver.writeVelocityU(uData);
  solver.writeSmoke(mData);

  interaction.boundaryMask = sData.slice();
  interaction._uData.set(uData);

  // Rasterize obstacle(s)
  if (preset.obstacle) {
    interaction.activeShape = preset.obstacle.shape;
    interaction.obstacleRadius = preset.obstacle.radius;
    const ox = preset.obstacle.x * domainWidth;
    const oy = preset.obstacle.y * domainHeight;
    interaction.rasterizeObstacle(ox, oy, 0, 0);
  } else if (preset.obstacles) {
    for (const obs of preset.obstacles) {
      interaction.activeShape = obs.shape;
      interaction.obstacleRadius = obs.radius;
      const ox = obs.x * domainWidth;
      const oy = obs.y * domainHeight;
      interaction.rasterizeObstacle(ox, oy, 0, 0);
    }
  }

  return { show: preset.show, numIters: preset.numIters, paintMode: preset.paintMode || false };
}
