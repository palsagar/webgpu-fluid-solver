/**
 * Monitors frame times and automatically adjusts grid resolution to maintain
 * smooth performance. Downscales quickly when frames are slow, upscales
 * cautiously (with a cooldown) when headroom exists.
 */
export class AdaptiveController {
    /**
     * Average frame time below which a tier is judged to have headroom.
     *
     * `tick` is now fed WALL-CLOCK frame time (rAF deltas). Under that signal a
     * tier with headroom does not report "fast" — it reports the display's
     * refresh period, because vsync is what it is waiting on. The old 8 ms
     * threshold could therefore never be crossed on any display: 8 ms is below
     * the 8.33 ms period of the 120 Hz dev machine and half the 16.67 ms of a
     * 60 Hz one. It only ever fired because the value passed in was CPU encode
     * time (~0.4 ms), which is not a frame time at all.
     *
     * Measured wall-clock averages per tier on the dev machine (120 Hz, Karman
     * at 256 iterations, 120-frame windows):
     *
     *   tier        64      128      256      512     1024
     *   ms/frame   8.33     8.34    17.75    62.01   257.36
     *
     * Frame time carries run-to-run variance: the presets.js numIters sweep
     * timed this same tier-256 @ 256-iters point at 14.4-16.6 ms against the
     * 17.75 ms here — separate runs, not a contradiction. The 12 and 20 ms
     * thresholds are chosen with margin for exactly that spread.
     *
     * 12 ms separates the two vsync-capped tiers from the first GPU-bound one
     * with ~40% margin on either side. Both directions are then reachable: 64
     * and 128 promote, 512 and 1024 trip the 20 ms downscale, and the 256
     * startup tier sits between the two and holds.
     *
     * On a 60 Hz display no tier can beat 16.67 ms, so nothing auto-promotes.
     * That is the conservative direction and it is deliberate — promoting into
     * a tier the display cannot sustain is the failure this constant exists to
     * avoid, and the manual tier buttons remain.
     */
    static UPSCALE_MS = 12;

    /**
     * Average wall-clock frame time above which the current tier is judged too
     * slow and dropped immediately (no cooldown — a stalling tier should not be
     * held for five seconds before being abandoned).
     *
     * 20 ms is ~50 fps. Against the measured per-tier table on the UPSCALE_MS
     * block above, it sits above both vsync-capped tiers (8.33 / 8.34 ms) and
     * above the 256 startup tier (17.75 ms), so none of the three trips it,
     * while 512 (62.01 ms) and 1024 (257.36 ms) clear it by 3.1x and 12.9x.
     *
     * Exported as a named constant rather than left inline because the tests
     * assert the two thresholds do not overlap and that this one stays
     * reachable from the slow tiers. A literal typed into the test instead
     * would assert nothing about the value the controller actually uses.
     */
    static DOWNSCALE_MS = 20;

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
        // 257 ms/frame (3.9 fps) on the dev machine at the Karman preset's 256
        // pressure iterations — it was ~58 ms when that preset ran 80 — so
        // auto-promoting into it would stall for seconds, drop back, and
        // promote again.
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
     * @param {number} frameTimeMs - Wall-clock interval between display frames
     *   (rAF timestamp delta), in ms — not CPU encode time, which is ~0.4 ms at
     *   every tier and carries no information about whether the GPU is keeping up
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
        // >DOWNSCALE_MS avg (~<50 FPS): drop resolution immediately
        if (avg > AdaptiveController.DOWNSCALE_MS && this.currentTierIndex > 0) {
            this.downscale();
        // <UPSCALE_MS avg with 5s cooldown: try higher resolution
        } else if (avg < AdaptiveController.UPSCALE_MS && this.currentTierIndex < this.maxAutoTierIndex) {
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
