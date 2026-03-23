/**
 * Monitors frame times and automatically adjusts grid resolution to maintain
 * smooth performance. Downscales quickly when frames are slow, upscales
 * cautiously (with a cooldown) when headroom exists.
 */
export class AdaptiveController {
    /**
     * @param {Object} solver - The GPU fluid solver instance
     * @param {Object} renderer - The canvas renderer instance
     * @param {Object} interaction - The obstacle/interaction handler
     * @param {Object} ui - The UI controller (used to reapply presets after resize)
     */
    constructor(solver, renderer, interaction, ui) {
        this.solver = solver;
        this.renderer = renderer;
        this.interaction = interaction;
        this.ui = ui;

        // Grid resolution tiers (cell count along Y axis)
        this.tiers = [64, 128, 256, 512];
        this.currentTierIndex = 2; // start at 256
        this.frameTimes = [];
        this.warmupFrames = 0;
        this.lastUpscaleTime = 0;
        this.manualOverride = false;
        this.enabled = false; // disabled by default — user can enable manually
    }

    /**
     * Called once per frame with the elapsed frame time. Collects samples
     * into a rolling window and triggers resolution changes when thresholds
     * are crossed.
     * @param {number} frameTimeMs - Duration of the last frame in milliseconds
     */
    tick(frameTimeMs) {
        if (!this.enabled || this.manualOverride) return;
        this.warmupFrames++;
        if (this.warmupFrames <= 120) return; // wait ~2s for GPU/JIT to stabilize
        this.frameTimes.push(frameTimeMs);
        if (this.frameTimes.length > 120) this.frameTimes.shift(); // rolling window of 120 samples
        if (this.frameTimes.length < 60) return; // need enough samples for stable average
        const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        // >20ms avg (~<50 FPS): drop resolution immediately
        if (avg > 20 && this.currentTierIndex > 0) {
            this.downscale();
        // <8ms avg (~>125 FPS) with 5s cooldown: try higher resolution
        } else if (avg < 8 && this.currentTierIndex < this.tiers.length - 1) {
            if (Date.now() - this.lastUpscaleTime > 5000) {
                this.upscale();
            }
        }
    }

    /**
     * Applies the current tier by resizing solver, renderer, and reapplying
     * the active preset. Resets frame-time history so the next measurement
     * window reflects the new resolution.
     */
    applyTier() {
        const tier = this.tiers[this.currentTierIndex];
        const container = this.renderer.canvas.parentElement;
        const numY = tier;
        // Scale X cells proportionally to canvas aspect ratio
        const numX = Math.round(tier * container.clientWidth / container.clientHeight);
        const h = 1.0 / numY;

        this.solver.resize(numX, numY, h);
        this.ui.reapplyCurrentPreset();
        this.renderer.resize(numX, numY, h);

        // Reset measurement state for the new resolution
        this.frameTimes = [];
        this.warmupFrames = 0;
    }

    /** Drops to the next lower resolution tier and notifies the renderer. */
    downscale() {
        this.currentTierIndex--;
        this.applyTier();
        this.renderer.showTierChange(this.tiers[this.currentTierIndex], -1);
    }

    /** Promotes to the next higher resolution tier, recording the time for cooldown. */
    upscale() {
        this.currentTierIndex++;
        this.lastUpscaleTime = Date.now();
        this.applyTier();
        this.renderer.showTierChange(this.tiers[this.currentTierIndex], 1);
    }
}
