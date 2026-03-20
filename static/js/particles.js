export class ParticleSystem {
    constructor(maxParticles = 5000, trailLen = 20, maxAge = 300) {
        this.maxParticles = maxParticles;
        this.trailLen = trailLen;
        this.maxAge = maxAge;
        this.particles = [];
    }

    emit(x, y, count = 150) {
        for (let i = 0; i < count; i++) {
            // Small random offset so particles don't all stack
            const px = x + (Math.random() - 0.5) * 0.02;
            const py = y + (Math.random() - 0.5) * 0.02;
            this.particles.push({
                x: px, y: py, age: 0,
                trail: [[px, py]],
            });
        }
        // Enforce cap — drop oldest
        if (this.particles.length > this.maxParticles) {
            this.particles.splice(0, this.particles.length - this.maxParticles);
        }
    }

    step(uData, vData, dt, h, numX, numY, solidData) {
        if (!uData || !vData) return;
        const domainW = numX * h;
        const domainH = numY * h;
        const n = numY;

        for (let k = this.particles.length - 1; k >= 0; k--) {
            const p = this.particles[k];
            // Sample velocity via bilinear interpolation
            const u = this._sample(p.x, p.y, uData, 0, h / 2, h, numX, numY);
            const v = this._sample(p.x, p.y, vData, h / 2, 0, h, numX, numY);
            p.x += u * dt;
            p.y += v * dt;
            p.age++;

            // Push to trail
            p.trail.push([p.x, p.y]);
            if (p.trail.length > this.trailLen) p.trail.shift();

            // Kill conditions: age, out of bounds, inside solid
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

    draw(ctx, numX, numY, h) {
        const domainW = numX * h;
        const domainH = numY * h;
        const cw = ctx.canvas.width;
        const ch = ctx.canvas.height;
        const toX = x => x / domainW * cw;
        const toY = y => (1 - y / domainH) * ch;

        ctx.lineWidth = 1;
        for (const p of this.particles) {
            if (p.trail.length < 2) continue;
            // Per-trail alpha based on particle age
            const alpha = Math.max(0, 1 - p.age / this.maxAge) * 0.8;
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha.toFixed(2)})`;
            ctx.beginPath();
            ctx.moveTo(toX(p.trail[0][0]), toY(p.trail[0][1]));
            for (let i = 1; i < p.trail.length; i++) {
                ctx.lineTo(toX(p.trail[i][0]), toY(p.trail[i][1]));
            }
            ctx.stroke();
        }
    }

    clear() {
        this.particles = [];
    }

    // Bilinear velocity sampling (same logic as renderer._sampleVel)
    _sample(x, y, field, dx, dy, h, numX, numY) {
        const h1 = 1.0 / h;
        x = Math.max(Math.min(x, numX * h), h);
        y = Math.max(Math.min(y, numY * h), h);
        const x0 = Math.max(0, Math.min(Math.floor((x - dx) * h1), numX - 1));
        const tx = ((x - dx) - x0 * h) * h1;
        const x1 = Math.min(x0 + 1, numX - 1);
        const y0 = Math.max(0, Math.min(Math.floor((y - dy) * h1), numY - 1));
        const ty = ((y - dy) - y0 * h) * h1;
        const y1 = Math.min(y0 + 1, numY - 1);
        const sx = 1.0 - tx, sy = 1.0 - ty;
        const n = numY;
        return sx*sy*field[x0*n+y0] + tx*sy*field[x1*n+y0] + tx*ty*field[x1*n+y1] + sx*ty*field[x0*n+y1];
    }
}
