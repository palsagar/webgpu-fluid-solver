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
        this.tiers = [64, 128, 256, 512, 1024];
        this.currentTierIndex = 2; // start at 256
        // Ceiling for automatic promotion. 1024 is manual-only: it measures
        // ~58 ms/frame (~17 fps) even on the dev machine, so auto-promoting
        // into it would stall for seconds, drop back, and promote again.
        // Lowered further by downscale() when a tier proves too slow.
        this.maxAutoTierIndex = this.tiers.indexOf(512);
        this.frameTimes = [];
        this.tierStartTime = 0;
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
        // Warmup is wall-clock, not frame-counted: at 17 fps a 120-frame gate
        // takes 7s, so a bad tier would hold the screen hostage before the
        // downscale check could even run.
        if (this.tierStartTime === 0) this.tierStartTime = Date.now();
        if (Date.now() - this.tierStartTime < 2000) return; // let GPU/JIT stabilize
        this.frameTimes.push(frameTimeMs);
        if (this.frameTimes.length > 120) this.frameTimes.shift(); // rolling window of 120 samples
        if (this.frameTimes.length < 20) return; // need enough samples for stable average
        const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        // >20ms avg (~<50 FPS): drop resolution immediately
        if (avg > 20 && this.currentTierIndex > 0) {
            this.downscale();
        // <8ms avg (~>125 FPS) with 5s cooldown: try higher resolution
        } else if (avg < 8 && this.currentTierIndex < this.maxAutoTierIndex) {
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
        // A collapsed container (hidden tab, zero-height layout) would make
        // numX Infinity/NaN. solver.resize() destroys buffers before creating
        // new ones, so throwing here would leave the solver unrecoverable.
        if (!(container.clientHeight > 0) || !(container.clientWidth > 0)) return;
        // Scale X cells proportionally to canvas aspect ratio
        const numX = Math.round(tier * container.clientWidth / container.clientHeight);
        const h = 1.0 / numY;

        this.solver.resize(numX, numY, h);
        this.ui.reapplyCurrentPreset();
        this.renderer.resize(numX, numY, h);

        // Reset measurement state for the new resolution
        this.frameTimes = [];
        this.tierStartTime = 0;
    }

    /** Drops to the next lower resolution tier and notifies the renderer. */
    downscale() {
        const failedTier = this.currentTierIndex;
        this.currentTierIndex--;
        // Remember that this tier was too slow. Without it the controller
        // promotes straight back once the cooldown expires, and oscillates.
        this.maxAutoTierIndex = Math.min(this.maxAutoTierIndex, failedTier - 1);
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
