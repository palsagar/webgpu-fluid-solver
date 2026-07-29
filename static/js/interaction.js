/**
 * Handles user interaction with the simulation canvas — obstacle dragging
 * and particle emitter placement.
 */
export class Interaction {
    /** Shape enum order — index is the WGSL `shape` in rasterize_obstacle.wgsl. */
    static SHAPES = ['circle', 'square', 'airfoil', 'wedge'];

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
        this._particleSystem = null;
        this.obstacleAngle = 0;
        this._shiftHeld = false;

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
     * center position — on the GPU, via solver.rasterizeObstacle (ADR-0010).
     * The previous bounding box is handed over as uniforms; the shader
     * restores it from the boundary mask and writes the new footprint with
     * the drag velocity, in every rotation slot. No CPU field mirrors exist:
     * the live field outside both footprints is untouched.
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
        const r = this.obstacleRadius;

        // Same conservative bounding extent the CPU rasterizer used
        const maxExtent = Math.max(r, r * 4 * 0.5, r * 3 * 0.5);
        const iMin = Math.max(1, Math.floor((centerX - maxExtent) / h - 1));
        const iMax = Math.min(numX - 2, Math.ceil((centerX + maxExtent) / h + 1));
        const jMin = Math.max(1, Math.floor((centerY - maxExtent) / h - 1));
        const jMax = Math.min(numY - 2, Math.ceil((centerY + maxExtent) / h + 1));

        this.solver.rasterizeObstacle({
            shape: Interaction.SHAPES.indexOf(this.activeShape),
            centerX, centerY, vx, vy,
            radius: r,
            angle: this.obstacleAngle,
            prevBBox: this._prevBBox
                ? [this._prevBBox.iMin, this._prevBBox.iMax, this._prevBBox.jMin, this._prevBBox.jMax]
                : null,
        });

        this._prevBBox = { iMin, iMax, jMin, jMax };
        if (this._renderer) this._renderer.invalidateSolid();
    }

    /**
     * Handles pointer-down events. In particle mode, places a new emitter
     * at the clicked location. In obstacle mode, begins obstacle dragging
     * or (if Shift) rotates the obstacle.
     * @param {number} clientX - Client X coordinate.
     * @param {number} clientY - Client Y coordinate.
     * @param {boolean} shiftKey - Whether the Shift key is held.
     */
    _onPointerDown(clientX, clientY, shiftKey) {
        if (this.mode === 'particles') {
            if (this._particleSystem) {
                const { x, y } = this.screenToSim(clientX, clientY);
                this._particleSystem.addEmitter(x, y);
                const hint = document.getElementById('canvas-hint');
                if (hint) hint.remove();
            }
            return; // Never fall through to drag in particles mode
        }
        if (shiftKey || this._shiftHeld) {
            this._rotate(clientX, clientY);
            return;
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
     * Handles pointer movement. If Shift is held, rotates the obstacle.
     * Otherwise continues an active drag (translate).
     */
    _onPointerMove(clientX, clientY, shiftKey) {
        if (this.mode !== 'obstacle') return;
        if (shiftKey || this._shiftHeld) {
            this._rotate(clientX, clientY);
            return;
        }
        if (!this.dragging) return;
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
        if (!this.dragging) return;
        this.dragging = false;
        // ADR-0011: the stored wall velocity must not outlive the drag — the
        // viscous ghost reads it every frame. One dispatch round, the same
        // cost as a mousemove; matches the zero-velocity rasterize in
        // _startDrag and _rotate.
        this.rasterizeObstacle(this.obstacleX, this.obstacleY, 0, 0);
    }

    /**
     * Sets the obstacle rotation angle from the mouse position.
     * Angle is computed as atan2 from obstacle center to cursor (compass-needle).
     * Re-rasterizes at the current position with zero velocity.
     */
    _rotate(clientX, clientY) {
        if (this.mode !== 'obstacle') return;
        const { x, y } = this.screenToSim(clientX, clientY);
        this.obstacleAngle = Math.atan2(y - this.obstacleY, x - this.obstacleX);
        this.rasterizeObstacle(this.obstacleX, this.obstacleY, 0, 0);
    }
}
