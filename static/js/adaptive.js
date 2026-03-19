export class AdaptiveController {
    constructor(solver, renderer, interaction, ui) {
        this.solver = solver;
        this.renderer = renderer;
        this.interaction = interaction;
        this.ui = ui;

        this.tiers = [64, 128, 256, 512];
        this.currentTierIndex = 2; // start at 256
        this.frameTimes = [];
        this.warmupFrames = 0;
        this.lastUpscaleTime = 0;
        this.manualOverride = false;
        this.enabled = false; // disabled by default — user can enable manually
    }

    tick(frameTimeMs) {
        if (!this.enabled || this.manualOverride) return;
        this.warmupFrames++;
        if (this.warmupFrames <= 120) return; // wait 2 seconds before measuring
        this.frameTimes.push(frameTimeMs);
        if (this.frameTimes.length > 120) this.frameTimes.shift();
        if (this.frameTimes.length < 60) return; // need enough samples
        const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        if (avg > 20 && this.currentTierIndex > 0) {
            this.downscale();
        } else if (avg < 8 && this.currentTierIndex < this.tiers.length - 1) {
            if (Date.now() - this.lastUpscaleTime > 5000) {
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
        this.renderer.showTierChange(this.tiers[this.currentTierIndex], -1);
    }

    upscale() {
        this.currentTierIndex++;
        this.lastUpscaleTime = Date.now();
        this.applyTier();
        this.renderer.showTierChange(this.tiers[this.currentTierIndex], 1);
    }
}
