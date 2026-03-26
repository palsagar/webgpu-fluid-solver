/**
 * Handles user interaction with the simulation canvas — obstacle dragging
 * and particle emitter placement.
 */
export class Interaction {
    /**
     * @param {HTMLCanvasElement} canvas - The simulation canvas element.
     * @param {Object} solver - The GPU fluid solver instance.
     */
    constructor(canvas, solver) {
        this.canvas = canvas;
        this.solver = solver;

        this.activeShape = 'circle';
        this.obstacleX = 0;
        this.obstacleY = 0;
        this.obstacleRadius = 0.15;
        this.dragging = false;
        this.prevX = 0;
        this.prevY = 0;
        this.boundaryMask = null;
        this.paintMode = false;
        this._paintFrame = 0;
        this._particleSystem = null;
        this.obstacleAngle = 0;
        this._shiftHeld = false;

        const size = solver.numX * solver.numY;
        this._sData = new Float32Array(size);
        this._uData = new Float32Array(size);
        this._vData = new Float32Array(size);

        this.mode = 'obstacle'; // 'obstacle' or 'particles'

        canvas.addEventListener('mousedown', e => this._onPointerDown(e.clientX, e.clientY, e.shiftKey));
        canvas.addEventListener('mousemove', e => this._onPointerMove(e.clientX, e.clientY, e.shiftKey));
        canvas.addEventListener('mouseup',   () => this._endDrag());
        canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches[0]; this._onPointerDown(t.clientX, t.clientY); }, { passive: false });
        canvas.addEventListener('touchmove',  e => { e.preventDefault(); const t = e.touches[0]; this._onPointerMove(t.clientX, t.clientY, false); }, { passive: false });
        canvas.addEventListener('touchend',   () => this._endDrag());

        document.addEventListener('keydown', e => { if (e.key === 'Shift') this._shiftHeld = true; });
        document.addEventListener('keyup', e => { if (e.key === 'Shift') this._shiftHeld = false; });
    }

    /**
     * Converts screen (client) coordinates to simulation-domain coordinates.
     * The simulation domain has (0,0) at the bottom-left corner.
     * @param {number} clientX - Mouse/touch X in client pixels.
     * @param {number} clientY - Mouse/touch Y in client pixels.
     * @returns {{ x: number, y: number }} Position in simulation units.
     */
    screenToSim(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = clientX - rect.left;
        const my = clientY - rect.top;
        // X maps directly; Y is flipped (screen top = sim top)
        const x = mx / rect.width  * this.solver.numX * this.solver.h;
        const y = (1.0 - my / rect.height) * this.solver.numY * this.solver.h;
        return { x, y };
    }

    /**
     * Rasterizes the active obstacle shape onto the solver's grid at the given
     * center position. Clears the previous obstacle footprint, writes the new
     * solid mask, and sets obstacle velocity in both ping-pong buffers.
     *
     * Three-step process:
     *   1. Restore cells from the previous bounding box to their boundary-mask state.
     *   2. Mark new obstacle cells as solid (s=0) and assign obstacle velocity.
     *   3. Save the new bounding box for the next call.
     *
     * @param {number} centerX - Obstacle center X in simulation units.
     * @param {number} centerY - Obstacle center Y in simulation units.
     * @param {number} [vx=0] - Obstacle velocity X (from drag motion).
     * @param {number} [vy=0] - Obstacle velocity Y (from drag motion).
     */
    rasterizeObstacle(centerX, centerY, vx = 0, vy = 0) {
        this.obstacleX = centerX;
        this.obstacleY = centerY;

        const { numX, numY, h } = this.solver;
        const n = numY;
        const sData = this._sData;
        const uData = this._uData;
        const vData = this._vData;

        const r = this.obstacleRadius;
        const shape = this.activeShape;

        // Shape-specific geometry constants used for inside-test and bounding box
        const chord = r * 4;            // airfoil chord length
        const wedgeLen = r * 3;         // wedge length
        const tanHA = Math.tan(15 * Math.PI / 180); // wedge half-angle (15 degrees)

        // Conservative bounding extent covering all possible shapes
        const maxExtent = Math.max(r, chord * 0.5, wedgeLen * 0.5);

        // Grid-cell bounding box for the new obstacle, clamped to interior cells
        const newIMin = Math.max(1, Math.floor((centerX - maxExtent) / h - 1));
        const newIMax = Math.min(numX - 2, Math.ceil((centerX + maxExtent) / h + 1));
        const newJMin = Math.max(1, Math.floor((centerY - maxExtent) / h - 1));
        const newJMax = Math.min(numY - 2, Math.ceil((centerY + maxExtent) / h + 1));

        // Step 1: Restore the previous obstacle bounding box to boundary mask values
        // and clear stale velocity/pressure/smoke imprint left by the old obstacle position.
        if (this._prevBBox) {
            const { iMin, iMax, jMin, jMax } = this._prevBBox;
            const clearSmoke = new Float32Array([1.0]);
            for (let i = iMin; i <= iMax; i++) {
                for (let j = jMin; j <= jMax; j++) {
                    const idx = i * n + j;
                    if (this.boundaryMask && this.boundaryMask[idx] === 0) continue;
                    const wasObstacle = sData[idx] === 0.0;
                    sData[idx] = this.boundaryMask ? this.boundaryMask[idx] : 1.0;
                    uData[idx] = 0.0;
                    vData[idx] = 0.0;
                    if (i + 1 < numX) {
                        uData[(i + 1) * n + j] = 0.0;
                    }
                    // Clear smoke in former obstacle cells to prevent stale dye imprints
                    if (wasObstacle) {
                        this.solver.device.queue.writeBuffer(this.solver.m, idx * 4, clearSmoke);
                        this.solver.device.queue.writeBuffer(this.solver.mNew, idx * 4, clearSmoke);
                    }
                }
                // Zero pressure for this column slice (contiguous in memory)
                const colStart = i * n + jMin;
                const colLen = jMax - jMin + 1;
                this.solver.device.queue.writeBuffer(
                    this.solver.p, colStart * 4,
                    new Float32Array(colLen)
                );
            }
        } else {
            // First call: initialise from boundaryMask (or all-fluid)
            if (this.boundaryMask) {
                sData.set(this.boundaryMask);
            } else {
                sData.fill(1.0);
            }
        }

        // Step 2: Rasterize new obstacle within its bounding box
        const obstacleCells = [];

        for (let i = newIMin; i <= newIMax; i++) {
            for (let j = newJMin; j <= newJMax; j++) {
                const idx = i * n + j;

                // Skip permanent boundary cells
                if (this.boundaryMask && this.boundaryMask[idx] === 0) continue;

                // Cell center in simulation coordinates
                const cx = (i + 0.5) * h;
                const cy = (j + 0.5) * h;
                // Offset from obstacle center
                const dx = cx - centerX;
                const dy = cy - centerY;

                // Test whether this cell falls inside the active shape
                let inside = false;

                if (shape === 'circle') {
                    inside = dx * dx + dy * dy < r * r;
                } else if (shape === 'square') {
                    inside = Math.abs(dx) < r && Math.abs(dy) < r;
                } else if (shape === 'airfoil') {
                    // NACA 0012 symmetric airfoil at zero angle of attack.
                    // lx is the chordwise coordinate (0 at leading edge, chord at trailing edge).
                    const lx = dx + chord * 0.5; // shift so leading edge is at lx=0
                    const ly = dy;
                    if (lx >= 0 && lx <= chord) {
                        const xc = lx / chord; // normalized chordwise position [0,1]
                        // NACA 0012 thickness distribution (half-thickness at xc)
                        const yt = 5 * 0.12 * chord * (
                            0.2969 * Math.sqrt(xc)
                            - 0.1260 * xc
                            - 0.3516 * xc * xc
                            + 0.2843 * xc * xc * xc
                            - 0.1015 * xc * xc * xc * xc
                        );
                        inside = Math.abs(ly) < yt;
                    }
                } else if (shape === 'wedge') {
                    // Wedge points right; apex at center
                    const lx = dx + wedgeLen * 0.5;
                    const ly = dy;
                    inside = lx >= 0 && lx < wedgeLen && Math.abs(ly) < lx * tanHA;
                }

                if (inside) {
                    sData[idx] = 0.0; // mark cell as solid
                    uData[idx] = vx;
                    vData[idx] = vy;
                    // Also set u at the right face of this cell for staggered grid consistency
                    if (i + 1 < numX) {
                        uData[(i + 1) * n + j] = vx;
                    }
                    if (this.paintMode) obstacleCells.push(idx);
                }
            }
        }

        // Step 3: Save the new bounding box for the next call
        this._prevBBox = { iMin: newIMin, iMax: newIMax, jMin: newJMin, jMax: newJMax };

        this.solver.writeSolidMask(sData);
        // Write velocity to BOTH ping-pong buffers so the active one always gets it
        this.solver.writeVelocityU(uData);
        this.solver.writeVelocityV(vData);
        this.solver.device.queue.writeBuffer(this.solver.uNew, 0, uData);
        this.solver.device.queue.writeBuffer(this.solver.vNew, 0, vData);

        // Notify renderer that solid mask changed
        if (this._renderer) this._renderer.invalidateSolid();

        // Paint mode: write oscillating dye values to obstacle cells for visual feedback
        if (this.paintMode && obstacleCells.length > 0) {
            const val = 0.5 + 0.5 * Math.sin(0.1 * this._paintFrame);
            const buf = new Float32Array(1);
            buf[0] = val;
            for (const idx of obstacleCells) {
                this.solver.device.queue.writeBuffer(this.solver.smokeBuffer, idx * 4, buf);
            }
            this._paintFrame++;
        }
    }

    /**
     * Handles pointer-down events. In particle mode, places a new emitter
     * at the clicked location. In obstacle mode, begins obstacle dragging.
     * @param {number} clientX - Client X coordinate.
     * @param {number} clientY - Client Y coordinate.
     */
    _onPointerDown(clientX, clientY) {
        if (this.mode === 'particles') {
            if (this._particleSystem) {
                const { x, y } = this.screenToSim(clientX, clientY);
                this._particleSystem.addEmitter(x, y);
                const hint = document.getElementById('canvas-hint');
                if (hint) hint.remove();
            }
            return; // Never fall through to drag in particles mode
        }
        this._startDrag(clientX, clientY);
    }

    /**
     * Initiates obstacle dragging at the given screen position.
     * Records the starting simulation-space position and rasterizes
     * the obstacle with zero velocity.
     */
    _startDrag(clientX, clientY) {
        if (this.mode !== 'obstacle') return;
        const { x, y } = this.screenToSim(clientX, clientY);
        this.prevX = x;
        this.prevY = y;
        this.dragging = true;
        this.rasterizeObstacle(x, y, 0, 0);
    }

    /**
     * Continues an active drag. Computes obstacle velocity from the
     * frame-to-frame displacement divided by the solver timestep,
     * then re-rasterizes at the new position.
     */
    _drag(clientX, clientY) {
        if (!this.dragging || this.mode !== 'obstacle') return;
        const { x, y } = this.screenToSim(clientX, clientY);
        const dt = this.solver.params.dt;
        // Finite-difference velocity estimate for moving-wall boundary condition
        const vx = (x - this.prevX) / dt;
        const vy = (y - this.prevY) / dt;
        this.rasterizeObstacle(x, y, vx, vy);
        this.prevX = x;
        this.prevY = y;
    }

    /** Ends the current drag interaction. */
    _endDrag() {
        this.dragging = false;
    }
}
