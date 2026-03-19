export class AdaptiveController {
    constructor(solver, renderer, interaction, ui) {
        this.solver = solver;
        this.renderer = renderer;
        this.interaction = interaction;
        this.ui = ui;

        this.tiers = [64, 128, 256, 512, 1024];
        this.currentTierIndex = 2;
        this.frameTimes = [];
        this.warmupFrames = 0;
        this.lastUpscaleTime = 0;
        this.manualOverride = false;
    }

    tick(frameTimeMs) {
        if (this.manualOverride) return;
        this.warmupFrames++;
        if (this.warmupFrames <= 30) return;
        this.frameTimes.push(frameTimeMs);
        if (this.frameTimes.length > 60) this.frameTimes.shift();
        const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        if (avg > 18 && this.currentTierIndex > 0) {
            this.downscale();
        } else if (avg < 10 && this.currentTierIndex < this.tiers.length - 1) {
            if (Date.now() - this.lastUpscaleTime > 2000) {
                this.upscale();
            }
        }
    }

    applyTier() {
        const tier = this.tiers[this.currentTierIndex];
        const container = this.renderer.canvas.parentElement;
        const numY = tier;
        const numX = Math.round(tier * container.clientWidth / container.clientHeight);
        const h = 1.0 / numY;

        this.solver.resize(numX, numY, h);
        this.ui.reapplyCurrentPreset();
        this.renderer.resize(numX, numY, h);

        this.frameTimes = [];
        this.warmupFrames = 0;
    }

    downscale() {
        this.currentTierIndex--;
        this.applyTier();
    }

    upscale() {
        this.currentTierIndex++;
        this.lastUpscaleTime = Date.now();
        this.applyTier();
    }
}
