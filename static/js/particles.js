/**
 * CPU-based Lagrangian particle system for flow visualization.
 * Particles are emitted from user-placed emitters, advected using
 * bilinearly-interpolated velocity readback data from the GPU solver,
 * and rendered as fading polyline trails on the 2D canvas overlay.
 */
export class ParticleSystem {
    /**
     * @param {number} [maxParticles=5000] - Maximum live particles before oldest are culled.
     * @param {number} [trailLen=20] - Number of past positions stored per particle trail.
     * @param {number} [maxAge=300] - Particle lifetime in frames before removal.
     */
    constructor(maxParticles = 5000, trailLen = 20, maxAge = 300) {
        this.maxParticles = maxParticles;
        this.trailLen = trailLen;
        this.maxAge = maxAge;
        this.particles = [];
        this.emitters = []; // { x, y } — continuous emission points
        this.emitRate = 3;  // particles per emitter per frame
    }

    /**
     * Registers a new continuous particle emitter at the given simulation position.
     * Oldest emitter is removed if the cap (10) is exceeded.
     * @param {number} x - Emitter X in simulation units.
     * @param {number} y - Emitter Y in simulation units.
     */
    addEmitter(x, y) {
        this.emitters.push({ x, y });
        if (this.emitters.length > 10) this.emitters.shift();
    }

    /**
     * Advances the particle system by one frame: spawns new particles from
     * emitters, advects all particles using bilinearly-sampled velocity, and
     * removes dead particles (aged out, out of bounds, or inside solids).
     *
     * @param {Float32Array} uData - Horizontal velocity field (staggered grid).
     * @param {Float32Array} vData - Vertical velocity field (staggered grid).
     * @param {number} dt - Simulation timestep.
     * @param {number} h - Grid cell size.
     * @param {number} numX - Grid width in cells.
     * @param {number} numY - Grid height in cells.
     * @param {Float32Array|null} solidData - Solid mask (0 = solid, 1 = fluid).
     */
    step(uData, vData, dt, h, numX, numY, solidData) {
        if (!uData || !vData) return;

        // Spawn new particles with slight jitter around each emitter
        for (const em of this.emitters) {
            for (let i = 0; i < this.emitRate; i++) {
                const px = em.x + (Math.random() - 0.5) * 0.01;
                const py = em.y + (Math.random() - 0.5) * 0.01;
                this.particles.push({ x: px, y: py, age: 0, trail: [[px, py]] });
            }
        }
        // Drop oldest particles if over capacity
        if (this.particles.length > this.maxParticles) {
            this.particles.splice(0, this.particles.length - this.maxParticles);
        }

        const domainW = numX * h;
        const domainH = numY * h;
        const n = numY;

        // Iterate in reverse so splice() doesn't shift unvisited indices
        for (let k = this.particles.length - 1; k >= 0; k--) {
            const p = this.particles[k];
            // Sample velocity at particle position (staggered offsets differ for u and v)
            const u = this._sample(p.x, p.y, uData, 0, h / 2, h, numX, numY);
            const v = this._sample(p.x, p.y, vData, h / 2, 0, h, numX, numY);
            // Forward-Euler advection
            p.x += u * dt;
            p.y += v * dt;
            p.age++;

            // Append new position to trail, trim oldest point
            p.trail.push([p.x, p.y]);
            if (p.trail.length > this.trailLen) p.trail.shift();

            // Remove particles that left the domain or entered a solid cell
            const outOfBounds = p.x < h || p.x > domainW - h || p.y < h || p.y > domainH - h;
            let inSolid = false;
            if (solidData) {
                const si = Math.floor(p.x / h);
                const sj = Math.floor(p.y / h);
                if (si >= 0 && si < numX && sj >= 0 && sj < numY) {
                    inSolid = solidData[si * n + sj] === 0.0;
                }
            }
            if (p.age > this.maxAge || outOfBounds || inSolid) {
                this.particles.splice(k, 1);
            }
        }
    }

    /**
     * Renders all particle trails and emitter markers onto the 2D canvas overlay.
     * Trails fade from opaque to transparent as particles age.
     *
     * @param {CanvasRenderingContext2D} ctx - 2D drawing context for the overlay canvas.
     * @param {number} numX - Grid width in cells.
     * @param {number} numY - Grid height in cells.
     * @param {number} h - Grid cell size.
     */
    draw(ctx, numX, numY, h) {
        const domainW = numX * h;
        const domainH = numY * h;
        const cw = ctx.canvas.width;
        const ch = ctx.canvas.height;
        // Coordinate transforms: simulation -> canvas pixels (Y is flipped)
        const toX = x => x / domainW * cw;
        const toY = y => (1 - y / domainH) * ch;

        ctx.lineWidth = 1.2;
        for (const p of this.particles) {
            if (p.trail.length < 2) continue;
            // Alpha decreases linearly with age for a fade-out effect
            const life = 1 - p.age / this.maxAge;
            const alpha = Math.max(0, life) * 0.7;
            // Ice blue — contrasts with warm magma palette on both light and dark regions
            ctx.strokeStyle = `rgba(100, 200, 255, ${alpha.toFixed(2)})`;
            ctx.beginPath();
            ctx.moveTo(toX(p.trail[0][0]), toY(p.trail[0][1]));
            for (let i = 1; i < p.trail.length; i++) {
                ctx.lineTo(toX(p.trail[i][0]), toY(p.trail[i][1]));
            }
            ctx.stroke();
        }

        // Draw emitter markers — small ring with soft glow
        ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
        ctx.fillStyle = 'rgba(100, 200, 255, 0.2)';
        ctx.lineWidth = 1.5;
        for (const em of this.emitters) {
            ctx.beginPath();
            ctx.arc(toX(em.x), toY(em.y), 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }
    }

    /** Removes all particles and emitters. */
    clear() {
        this.particles = [];
        this.emitters = [];
    }

    /**
     * Bilinear interpolation of a staggered-grid field at an arbitrary point.
     * The field origin is offset by (dx, dy) to account for MAC grid staggering
     * (u lives on vertical cell faces, v on horizontal cell faces).
     *
     * @param {number} x - Sample X in simulation units.
     * @param {number} y - Sample Y in simulation units.
     * @param {Float32Array} field - The velocity component array.
     * @param {number} dx - X offset of field samples from cell corner (0 for u, h/2 for v).
     * @param {number} dy - Y offset of field samples from cell corner (h/2 for u, 0 for v).
     * @param {number} h - Grid cell size.
     * @param {number} numX - Grid width in cells.
     * @param {number} numY - Grid height in cells.
     * @returns {number} Interpolated field value at (x, y).
     */
    _sample(x, y, field, dx, dy, h, numX, numY) {
        const h1 = 1.0 / h;
        // Clamp position to interior of domain
        x = Math.max(Math.min(x, numX * h), h);
        y = Math.max(Math.min(y, numY * h), h);
        // Find the four surrounding grid nodes and interpolation weights
        const x0 = Math.max(0, Math.min(Math.floor((x - dx) * h1), numX - 1));
        const tx = ((x - dx) - x0 * h) * h1;
        const x1 = Math.min(x0 + 1, numX - 1);
        const y0 = Math.max(0, Math.min(Math.floor((y - dy) * h1), numY - 1));
        const ty = ((y - dy) - y0 * h) * h1;
        const y1 = Math.min(y0 + 1, numY - 1);
        const sx = 1.0 - tx, sy = 1.0 - ty;
        const n = numY;
        // Standard bilinear interpolation from four corner values
        return sx*sy*field[x0*n+y0] + tx*sy*field[x1*n+y0] + tx*ty*field[x1*n+y1] + sx*ty*field[x0*n+y1];
    }
}
