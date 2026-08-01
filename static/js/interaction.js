/**
 * Handles user interaction with the simulation canvas — obstacle dragging
 * and particle emitter placement.
 */
export class Interaction {
    /** Maximum stored drag velocity, in sim units/s.
     *
     *  The drag velocity is estimated from per-event pointer displacement divided
     *  by the solver timestep, which explodes for high-frequency micro-drags: a
     *  1-px pointer wiggle divided by dt = 1/240 yields a wall velocity of
     *  hundreds of sim units/s. Injecting that through the moving-wall boundary
     *  drives the pressure solve to multi-million-Pa magnitudes (D2). Bounding the
     *  stored wall velocity to the fastest flow speed the UI allows (the inflow
     *  slider max = 5.0) keeps the momentum injection physically sane without
     *  touching the fluid itself or the ADR-0011 moderate-drag gate (VX = 1.0,
     *  real moderate drags ≈ 3.3).
     */
    static MAX_DRAG_VELOCITY = 5.0;

    /** Shape enum order — index is the WGSL `shape` in rasterize_obstacle.wgsl. */
    static SHAPES = ['circle', 'square', 'airfoil', 'wedge'];

    /** Clamps a scalar to [-MAX_DRAG_VELOCITY, MAX_DRAG_VELOCITY]. */
    static _clampDragVel(v) {
        return Math.max(-Interaction.MAX_DRAG_VELOCITY, Math.min(Interaction.MAX_DRAG_VELOCITY, v));
    }

    /**
     * Distance threshold (sim units) above which an obstacle reposition is
     * treated as a teleport rather than a frame-to-frame drag. When crossed,
     * the pressure field is zeroed so the projection solve starts fresh and
     * does not inherit stale/artifacted pressure from a far-away footprint.
     */
    static PRESSURE_CLEAR_TELEPORT_DISTANCE = 0.5;

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

        canvas.addEventListener('mousedown', e => {
            if (e.button !== 0) return;
            this._onPointerDown(e.clientX, e.clientY, e.shiftKey);
        });
        canvas.addEventListener('contextmenu', e => e.preventDefault());
        canvas.addEventListener('mousemove', e => this._onPointerMove(e.clientX, e.clientY, e.shiftKey));
        window.addEventListener('mouseup',    e => { if (e.button !== 0) return; this._endDrag(); });
        canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches[0]; this._onPointerDown(t.clientX, t.clientY); }, { passive: false });
        canvas.addEventListener('touchmove',  e => { e.preventDefault(); const t = e.touches[0]; this._onPointerMove(t.clientX, t.clientY, false); }, { passive: false });
        window.addEventListener('touchend',   () => this._endDrag());
        window.addEventListener('touchcancel', () => this._endDrag());
        window.addEventListener('blur', () => this._endDrag());

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
        const prevX = this.obstacleX;
        const prevY = this.obstacleY;
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

        const dx = centerX - prevX;
        const dy = centerY - prevY;
        const clearPressure = (dx * dx + dy * dy) >
            Interaction.PRESSURE_CLEAR_TELEPORT_DISTANCE * Interaction.PRESSURE_CLEAR_TELEPORT_DISTANCE;

        this.solver.rasterizeObstacle({
            shape: Interaction.SHAPES.indexOf(this.activeShape),
            centerX, centerY, vx, vy,
            radius: r,
            angle: this.obstacleAngle,
            prevBBox: this._prevBBox
                ? [this._prevBBox.iMin, this._prevBBox.iMax, this._prevBBox.jMin, this._prevBBox.jMax]
                : null,
            clearPressure,
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
        // Obstacle-less preset (backwardStep): the first pointerdown INSERTS the
        // obstacle at the click point and un-hides it, so the overlay ring, the
        // shape buttons, and the Re/St badges all acknowledge the body now in
        // the flow — no phantom solid the UI pretends does not exist.
        if (!this.showObstacle) this.showObstacle = true;
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
            // Guard the rotation path symmetrically with the pointer-down guard:
            // an obstacle-less preset hides the obstacle, so Shift+mousemove must
            // not rasterize a stale/phantom solid or mutate the hidden geometry.
            if (!this.showObstacle) return;
            this._rotate(clientX, clientY);
            return;
        }
        if (!this.dragging) return;
        const { x, y } = this.screenToSim(clientX, clientY);
        const dt = this.solver.params.dt;
        // Finite-difference velocity estimate for moving-wall boundary condition
        const vx = Interaction._clampDragVel((x - this.prevX) / dt);
        const vy = Interaction._clampDragVel((y - this.prevY) / dt);
        this.rasterizeObstacle(x, y, vx, vy);
        this.prevX = x;
        this.prevY = y;
    }

    /**
     * Ends the current drag interaction.
     * Bound on `window` so a pointer released outside the canvas still ends
     * the drag and zeroes the stored wall velocity.
     */
    _endDrag() {
        if (!this.dragging) return;
        this.dragging = false;
        // ADR-0011: the stored wall velocity must not outlive the drag — the
        // viscous ghost reads it every frame. Zero-rasterize only while the
        // obstacle is still shown: a preset switch to an obstacle-less preset
        // hides the obstacle and resets the field, so a late release must not
        // re-rasterize at the old centre.
        if (this.showObstacle) {
            this.rasterizeObstacle(this.obstacleX, this.obstacleY, 0, 0);
        }
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
