export class Interaction {
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

        const size = solver.numX * solver.numY;
        this._sData = new Float32Array(size);
        this._uData = new Float32Array(size);
        this._vData = new Float32Array(size);

        canvas.addEventListener('mousedown', e => this._startDrag(e.clientX, e.clientY));
        canvas.addEventListener('mousemove', e => this._drag(e.clientX, e.clientY));
        canvas.addEventListener('mouseup',   () => this._endDrag());
        canvas.addEventListener('touchstart', e => { e.preventDefault(); const t = e.touches[0]; this._startDrag(t.clientX, t.clientY); }, { passive: false });
        canvas.addEventListener('touchmove',  e => { e.preventDefault(); const t = e.touches[0]; this._drag(t.clientX, t.clientY); }, { passive: false });
        canvas.addEventListener('touchend',   () => this._endDrag());
    }

    screenToSim(clientX, clientY) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = clientX - rect.left;
        const my = clientY - rect.top;
        const x = mx / rect.width  * this.solver.numX * this.solver.h;
        const y = (1.0 - my / rect.height) * this.solver.numY * this.solver.h;
        return { x, y };
    }

    rasterizeObstacle(centerX, centerY, vx = 0, vy = 0) {
        this.obstacleX = centerX;
        this.obstacleY = centerY;

        const { numX, numY, h } = this.solver;
        const n = numY;
        const sData = this._sData;
        const uData = this._uData;
        const vData = this._vData;

        // Start from boundaryMask if available, else all fluid
        if (this.boundaryMask) {
            sData.set(this.boundaryMask);
        } else {
            sData.fill(1.0);
        }

        const r = this.obstacleRadius;
        const shape = this.activeShape;

        // Shape-specific constants
        const chord = r * 4;            // airfoil chord length
        const wedgeLen = r * 3;         // wedge length
        const tanHA = Math.tan(15 * Math.PI / 180); // wedge half-angle

        for (let i = 0; i < numX; i++) {
            for (let j = 0; j < numY; j++) {
                const idx = i * n + j;

                // Skip permanent boundary cells
                if (this.boundaryMask && this.boundaryMask[idx] === 0) continue;

                const cx = (i + 0.5) * h;
                const cy = (j + 0.5) * h;
                const dx = cx - centerX;
                const dy = cy - centerY;

                let inside = false;

                if (shape === 'circle') {
                    inside = dx * dx + dy * dy < r * r;
                } else if (shape === 'square') {
                    inside = Math.abs(dx) < r && Math.abs(dy) < r;
                } else if (shape === 'airfoil') {
                    // NACA 0012, no angle of attack (lx = dx, ly = dy in local frame)
                    // Local x runs from leading edge (centerX) forward (positive dx direction)
                    const lx = dx + chord * 0.5; // shift so leading edge is at lx=0
                    const ly = dy;
                    if (lx >= 0 && lx <= chord) {
                        const xc = lx / chord;
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
                    sData[idx] = 0.0;
                    uData[idx] = vx;
                    vData[idx] = vy;
                    if (i + 1 < numX) {
                        uData[(i + 1) * n + j] = vx;
                    }
                }
            }
        }

        this.solver.writeSolidMask(sData);
        this.solver.writeVelocityU(uData);
        this.solver.writeVelocityV(vData);
    }

    _startDrag(clientX, clientY) {
        const { x, y } = this.screenToSim(clientX, clientY);
        this.prevX = x;
        this.prevY = y;
        this.dragging = true;
        this.rasterizeObstacle(x, y, 0, 0);
    }

    _drag(clientX, clientY) {
        if (!this.dragging) return;
        const { x, y } = this.screenToSim(clientX, clientY);
        const dt = this.solver.params.dt;
        const vx = (x - this.prevX) / dt;
        const vy = (y - this.prevY) / dt;
        this.rasterizeObstacle(x, y, vx, vy);
        this.prevX = x;
        this.prevY = y;
    }

    _endDrag() {
        this.dragging = false;
    }
}
