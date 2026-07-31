import { loadPreset, PRESETS } from './presets.js';
import {
    honestWindow, windowState, fmtRe, reFromSliderPos,
    nuNumConverged, NU_NUM_ITERS256, NU_NUM_ITERS256_DT, NU_NUM_MEASURED_U,
    PROJECTION_ITERS_MEASURED,
    StrouhalProbe, probeCell,
} from './diagnostics.js';

// How close the live dt must sit to NU_NUM_ITERS256_DT for that table to be
// the right one to quote. `nu_num` is linear in dt, so a relative dt error
// carries straight into a relative nu error — 0.1% is four orders of magnitude
// below the ±10% fit-window uncertainty the table already carries, while the
// dt slider's own step near 1/240 is a 2.4% jump, so no neighbouring slider
// position sneaks through. This exists so a slider dragged back to the bottom
// (0.0041667, the preset's dt to within 8e-6) still counts as the anchor.
const ANCHOR_DT_REL_TOL = 1e-3;

// How close the live inflow must sit to NU_NUM_MEASURED_U for the projection
// table to be a measurement of THIS flow. Same 0.1% as the dt tolerance and for
// the same reason: it exists so a slider returned to the preset value counts as
// the measured point despite float round-trips through the DOM, not so a
// neighbouring position sneaks through. The inflow slider steps by 0.1 on a
// U = 1.0 anchor, so the nearest other position is 10% away — a hundred times
// this tolerance.
const MEASURED_U_REL_TOL = 1e-3;

// Map kebab-case data-preset attribute values to PRESETS object keys
const PRESET_KEY_MAP = {
    'karman-vortex':  'karmanVortex',
    'backward-step':  'backwardStep',
};

/**
 * UI controller — wires DOM controls (presets, sliders, toggles, keyboard shortcuts)
 * to the solver, renderer, and interaction layer.
 */
export class UI {
    /**
     * @param {FluidSolver} solver - GPU-backed fluid solver instance
     * @param {Renderer} renderer - Canvas renderer instance
     * @param {Interaction} interaction - Mouse/touch interaction handler
     */
    constructor(solver, renderer, interaction) {
        this.solver = solver;
        this.renderer = renderer;
        this.interaction = interaction;

        // Downstream wake probe. Owned here, but referenced by the renderer so
        // it is cleared by the same invalidateSolid()/resize() calls that clear
        // the particles — obstacle drag, shape switch, preset load, tier change.
        this.probe = new StrouhalProbe();
        renderer.probe = this.probe;
        // Last velocity readback generation fed to the probe, so each readback
        // contributes exactly one sample no matter how many frames it survives.
        this._probeVelGen = -1;
        // Simulation time of the last sample, so a readback delivered while the
        // field is frozen cannot pad the window with duplicates. See _sampleProbe.
        this._probeSimTime = -1;

        // Initial preset
        this.currentPreset = 'karmanVortex';
        this.smokeInletData = null;
        this.boundaryVelData = null;
        const config = loadPreset(this.currentPreset, solver, interaction);
        this.numIters = config.numIters;
        this.smokeInletData = config.smokeInletData ?? null;
        this.boundaryVelData = config.boundaryVelData ?? null;
        this._applyShow(config.show);

        this._bindPresetButtons();
        this._bindVizToggles();
        this._bindShapePicker();
        this._bindModeSwitcher();
        this._bindPlayback();
        this._bindAdvancedPanel();
        this._bindKeyboard();
        this._bindGuideModal();

        // Sync slider displays to current preset values
        this._syncSliders();
    }

    /**
     * Re-loads the current preset from scratch — used after grid resize
     * or when the user clicks "Restore Defaults".
     */
    reapplyCurrentPreset() {
        const config = loadPreset(this.currentPreset, this.solver, this.interaction);
        this.numIters = config.numIters;
        this.smokeInletData = config.smokeInletData ?? null;
        this.boundaryVelData = config.boundaryVelData ?? null;
        this.renderer.invalidateSolid();
        this._applyShow(config.show);
        this._updateVizCheckboxes(config.show);
        this._syncSliders();
    }

    /**
     * Programmatically set visualization flags and update the corresponding checkboxes.
     * @param {Object} show - Map of visualization layer names to booleans
     */
    setVisualization(show) {
        this._applyShow(show);
        this._updateVizCheckboxes(show);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Push visualization flags from a preset config into the renderer.
     * @param {Object} show - { pressure, smoke, streamlines, velocities }
     */
    _applyShow(show) {
        this.renderer.showPressure    = show.pressure    ?? false;
        this.renderer.showSmoke       = show.smoke       ?? false;
        this.renderer.showStreamlines = show.streamlines ?? false;
        this.renderer.showVelocities  = show.velocities  ?? false;
    }

    /** Sync the DOM checkboxes to match the given visualization flags. */
    _updateVizCheckboxes(show) {
        document.querySelectorAll('[data-viz]').forEach(cb => {
            const key = cb.dataset.viz;
            if (key in show) cb.checked = show[key];
        });
    }

    /** Sync all slider positions and displayed values to current solver/preset state. */
    _syncSliders() {
        const p = this.solver.params;
        const preset = PRESETS[this.currentPreset];

        const setSlider = (id, valId, value, decimals) => {
            const el = document.getElementById(id);
            const valEl = document.getElementById(valId);
            if (el) el.value = value;
            if (valEl) valEl.textContent = value.toFixed(decimals);
        };

        setSlider('slider-dt',    'val-dt',    p.dt,      4);
        setSlider('slider-omega', 'val-omega', p.omega,   2);
        const itersEl = document.getElementById('slider-iters');
        const itersValEl = document.getElementById('val-iters');
        if (itersEl) itersEl.value = this.numIters;
        if (itersValEl) itersValEl.textContent = this.numIters;

        const inVel = preset?.inVel ?? 0;
        const invelEl = document.getElementById('slider-invel');
        const invelValEl = document.getElementById('val-invel');
        if (invelEl) invelEl.value = inVel;
        if (invelValEl) invelValEl.textContent = inVel.toFixed(2);

        // Sync resolution buttons to current tier
        if (this.adaptive) {
            const tierIdx = this.adaptive.currentTierIndex;
            document.querySelectorAll('[data-tier]').forEach(b => {
                b.classList.toggle('active', parseInt(b.dataset.tier) === tierIdx);
            });
        }

        this._updateReBadge();
    }

    /** The inflow velocity currently driving the flow — the live slider, not the preset default. */
    _inflowVelocity() {
        const el = document.getElementById('slider-invel');
        const fromSlider = el ? parseFloat(el.value) : NaN;
        return Number.isFinite(fromSlider) ? fromSlider : (PRESETS[this.currentPreset]?.inVel ?? 0);
    }

    /**
     * Push the Re slider into the solver as a real viscosity (nu = U*D/Re), then
     * report whether this grid can actually deliver that Re.
     *
     * Both bounds are measured (see diagnostics.js): the floor is the solver's
     * own `viscNuMax`, the ceiling is the Taylor-Green calibration. Nothing here
     * falls back to an estimate — if a tier has no measured operating-point
     * viscosity the projection ceiling is simply not claimed.
     */
    _updateReBadge() {
        const el = document.getElementById('val-re');
        const badge = document.getElementById('re-badge');
        const slider = document.getElementById('slider-re');
        if (!el || !slider) return;

        const hideBadge = () => {
            if (badge) {
                badge.textContent = '';
                badge.classList.remove('visible', 'empty');
            }
        };

        // Every caller of this method has just changed something the wake
        // depends on — Re, dt, the iteration count, or the inflow speed — so
        // the samples already in the probe describe a flow that no longer
        // exists. Dropping them costs a fresh ~43 s window; keeping them costs
        // a number averaged across two different simulations and presented as
        // a measurement of the current one.
        //
        // The inflow slider is the sharpest case: U scales the stored samples
        // AND sits in the shedding gate (`rms < THRESHOLD * U`), so raising it
        // could flip a shedding wake to `steady` with no change in the physics.
        this.probe.clear();

        const re = reFromSliderPos(parseFloat(slider.value));
        const D = 2 * this.interaction.obstacleRadius;
        const U = this._inflowVelocity();
        el.textContent = fmtRe(re);
        this._updateFlowInfo();

        // No obstacle or no free stream — Re is undefined, so claim nothing.
        // The showObstacle gate is load-bearing: obstacle-less presets
        // (backwardStep) never reassign interaction.obstacleRadius, so D here is
        // a phantom inherited from whatever preset was viewed before. Without
        // this clause the badge would apply a viscosity derived from a body that
        // is not in the flow. Mirrors the same guard in _updateStrouhal.
        if (!(D > 0) || !(U > 0) || !this.interaction.showObstacle) {
            this.solver.setParams({ nu: 0 });
            el.textContent = '--';
            hideBadge();
            return;
        }

        const nuReq = (U * D) / re;
        this.solver.setParams({ nu: nuReq });

        const w = honestWindow({
            h: this.solver.h,
            dt: this.solver.params.dt,
            D, U,
            nMax: this.solver.constructor.N_MAX,
            // Scaled by the LIVE dt, not a constant: nu_num is linear in dt and
            // the presets do not share one. Karman runs 1/240, backwardStep
            // 1/60, where a fixed constant made the ceiling ~2x optimistic.
            nuNum: nuNumConverged(this.solver.params.dt),
            // The solver's own saturation limit rather than a re-derivation of
            // it. Identical formula today; passing it keeps the floor tied to
            // the number the solver actually clamps at if that ever moves.
            nuMax: this.solver.viscNuMax,
        });

        // The operating-point ceiling is measured on ONE slice: dt = 1/240 at
        // PROJECTION_ITERS_MEASURED iterations and an amplitude of
        // NU_NUM_MEASURED_U, across the tiers. It is rescaled to none of the
        // three — three lines through the (tier x numIters x dt x U) surface
        // are measured, not the surface, and interpolating it would be
        // inventing the number this branch exists to measure.
        //
        // So it is quoted ONLY where it was measured. Off that slice — the
        // backwardStep preset (dt = 1/60 at 60 iters, U = 1.5), or
        // any move of the dt / iterations / inflow sliders — no projection
        // ceiling is supplied, and `windowState` reports the ceiling as
        // unmeasured instead. Carrying it off-slice used to produce an
        // impossible pair: on a since-removed preset (ADR-0009), a
        // projection ceiling of Re 771 against a converged ceiling of
        // Re 297 — an under-converged solve dissipating less than a
        // converged one.
        //
        // The AMPLITUDE axis is the one added last and the one with the least
        // behind it. dt and numIters are each measured at two settings, so the
        // shape of the dependence is at least known. `U` is measured at ONE —
        // every Taylor-Green fit ran at A = 1.0 — so off it not even the SIGN
        // of the correction is established, while the inflow slider spans
        // 0.5 .. 5.0 live on the flagship preset. Gating here is what stops one
        // drag of that slider leaving a measured ceiling on screen for a flow
        // nothing measured.
        const atMeasuredPoint =
            Math.abs(this.solver.params.dt - NU_NUM_ITERS256_DT)
                <= ANCHOR_DT_REL_TOL * NU_NUM_ITERS256_DT
            && this.numIters === PROJECTION_ITERS_MEASURED
            && Math.abs(U - NU_NUM_MEASURED_U)
                <= MEASURED_U_REL_TOL * NU_NUM_MEASURED_U;
        const nuProjection = atMeasuredPoint ? NU_NUM_ITERS256[this.solver.numY] : undefined;
        const reMaxProjection = nuProjection ? (U * D) / nuProjection : Infinity;

        // Saturation must be read from `viscNuMax`, a getter that is always
        // current, NOT from `solver.viscClamped` — that flag describes the last
        // step, so between a slider move and the next frame it reports the
        // previous request. Trusting it made the badge claim "substep budget
        // saturated" while the slider sat at the top of its range.
        const nuMaxNow = this.solver.viscNuMax;
        const clamped = nuReq > nuMaxNow;

        // The Re the solver is actually running, from the viscosity it applied
        // — not from params.nu, which differs exactly when saturated. Same
        // staleness caveat, so viscNuEff is only used once it matches what this
        // request implies; until the next step it would describe the old one.
        const nuExpected = Math.min(nuReq, nuMaxNow);
        const nuEff = Math.abs(this.solver.viscNuEff - nuExpected) <= 1e-9 * nuExpected
            ? this.solver.viscNuEff
            : nuExpected;
        const reEff = nuEff > 0 ? (U * D) / nuEff : re;

        const st = windowState({
            re, reEff,
            reMin: w.reMin, reMax: w.reMax, reMaxProjection,
            viscClamped: clamped,
        });

        if (!badge) return;
        badge.textContent = st.ok ? '' : st.reason;
        badge.classList.toggle('visible', !st.ok);
        badge.classList.toggle('empty', st.code === 'empty-grid' || st.code === 'empty-iters');
    }

    /**
     * Per-frame hook, called from the rAF loop: feed the wake probe from the
     * latest velocity readback and repaint its readout.
     *
     * Lives on the frame loop rather than on a slider handler because the probe
     * is a MEASUREMENT — it accumulates while the flow evolves, and nothing the
     * user does marks the moment it becomes valid.
     */
    tick() {
        this._sampleProbe();
        this._updateStrouhal();
    }

    /**
     * Push one transverse-velocity sample per velocity readback.
     *
     * Gated on simulation time having ADVANCED since the last sample, not on
     * `solver.paused`. `readbackVelocity` keeps firing every 10 frames on a
     * frozen field, and the identical samples it would deliver are not 10 steps
     * of flow — they would pad the window with a flat line, drag the RMS under
     * the shedding gate, and turn a shedding wake into 'steady' just by leaving
     * the app paused. A `paused` test catches that case but also catches the
     * single-step button, which advances the field 10 steps between readbacks
     * exactly as the running loop does; keying off `simTime` rejects the frozen
     * field and admits the stepped one.
     */
    _sampleProbe() {
        const { renderer, solver, interaction } = this;
        if (solver.simTime === this._probeSimTime) return;
        if (renderer._velDataGen === this._probeVelGen) return;
        if (!renderer.vData || !interaction.showObstacle) return;
        this._probeVelGen = renderer._velDataGen;
        this._probeSimTime = solver.simTime;

        const cell = probeCell({
            obstacleX: interaction.obstacleX,
            obstacleY: interaction.obstacleY,
            D: 2 * interaction.obstacleRadius,
            h: solver.h,
            numX: solver.numX,
            numY: solver.numY,
        });
        if (!cell) return;

        // v, not u: on the centreline the streamwise component dips once per
        // shed vortex from EITHER side and so carries 2f. See StrouhalProbe.
        //
        // Stamp with the time the velocity was COPIED (renderer._velDataSimTime),
        // not the live simTime at which this readback's mapAsync happened to
        // resolve. Copies fire every 10 steps, so capture-times are exact
        // multiples of 10*dt and the series is evenly spaced; the live simTime
        // carries a variable readback latency that makes the gaps jitter.
        this.probe.push(renderer.vData[cell.i * solver.numY + cell.j], renderer._velDataSimTime);
    }

    /**
     * Write the probe's verdict into the readout — a number only when the wake
     * is actually shedding.
     *
     * D and U are read HERE rather than cached at push time, so dragging the
     * obstacle to a new size rescales St to the geometry the flow currently
     * has. A drag clears the series as well, so in practice that path re-reads
     * a window that already belongs to the new geometry; the inflow slider is
     * cleared explicitly (see `_updateReBadge`) because it does NOT go through
     * the renderer's invalidation path.
     */
    _updateStrouhal() {
        const el = document.getElementById('val-st');
        if (!el) return;
        if (!this.interaction.showObstacle) { el.textContent = '--'; return; }

        const D = 2 * this.interaction.obstacleRadius;
        const U = this._inflowVelocity();
        const { state, st } = this.probe.read({ D, U });
        el.textContent = state === 'shedding'   ? st.toFixed(2)
                       : state === 'steady'     ? 'steady — no shedding'
                       : state === 'unresolved' ? 'under-sampled — lower dt'
                       : state === 'no-signal'  ? 'no signal'
                       : 'measuring…';
    }

    /** Update the flow-info overlay text with the preset-specific physics description. */
    _updateFlowInfo() {
        const el = document.getElementById('flow-info');
        if (!el) return;
        const info = {
            karmanVortex:  'Periodic vortex shedding behind a small cylinder',
            backwardStep:  'Sudden expansion — recirculation and flow reattachment',
        };
        el.textContent = info[this.currentPreset] || '';
    }

    /** Attach click handlers to preset buttons, mapping kebab-case attributes to preset keys. */
    _bindPresetButtons() {
        document.querySelectorAll('[data-preset]').forEach(btn => {
            btn.addEventListener('click', () => {
                const attrName = btn.dataset.preset;
                const presetKey = PRESET_KEY_MAP[attrName];
                if (!presetKey) return;
                this._loadAndApplyPreset(presetKey);
                document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    /**
     * Load a preset by key, apply its config to solver/renderer/interaction,
     * and update all UI elements (checkboxes, sliders, shape buttons).
     * @param {string} presetKey - Key into the PRESETS object (e.g. 'karmanVortex')
     */
    _loadAndApplyPreset(presetKey) {
        this.currentPreset = presetKey;
        const config = loadPreset(presetKey, this.solver, this.interaction);
        this.numIters = config.numIters;
        this.smokeInletData = config.smokeInletData ?? null;
        this.boundaryVelData = config.boundaryVelData ?? null;
        this.renderer.invalidateSolid();
        this._applyShow(config.show);
        this._updateVizCheckboxes(config.show);
        this._syncSliders();
        // Update shape button active state from preset
        const preset = PRESETS[presetKey];
        const shape = preset.obstacle?.shape ?? preset.obstacles?.[0]?.shape;
        if (shape) {
            this.interaction.activeShape = shape;
            document.querySelectorAll('[data-shape]').forEach(b => {
                b.classList.toggle('active', b.dataset.shape === shape);
            });
        }
    }

    /** Initialize visualization checkboxes from renderer state and bind change handlers. */
    _bindVizToggles() {
        document.querySelectorAll('[data-viz]').forEach(cb => {
            const key = cb.dataset.viz;
            if (key === 'pressure')    cb.checked = this.renderer.showPressure;
            if (key === 'smoke')       cb.checked = this.renderer.showSmoke;
            if (key === 'streamlines') cb.checked = this.renderer.showStreamlines ?? false;
            if (key === 'velocities')  cb.checked = this.renderer.showVelocities  ?? false;

            cb.addEventListener('change', () => {
                if (key === 'pressure')    this.renderer.showPressure    = cb.checked;
                if (key === 'smoke')       this.renderer.showSmoke       = cb.checked;
                if (key === 'streamlines') this.renderer.showStreamlines = cb.checked;
                if (key === 'velocities')  this.renderer.showVelocities  = cb.checked;
            });
        });
    }

    /** Bind the mode toggle button to switch between obstacle-drag and particle-emit modes. */
    _bindModeSwitcher() {
        const btn = document.getElementById('btn-mode');
        if (!btn) return;
        btn.addEventListener('click', () => {
            const isParticles = this.interaction.mode === 'particles';
            this.interaction.mode = isParticles ? 'obstacle' : 'particles';
            btn.classList.toggle('active', !isParticles);
        });
    }

    /** Bind obstacle shape buttons — clicking re-rasterizes the obstacle immediately. */
    _bindShapePicker() {
        document.querySelectorAll('[data-shape]').forEach(btn => {
            btn.addEventListener('click', () => {
                this.interaction.activeShape = btn.dataset.shape;
                document.querySelectorAll('[data-shape]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                // Re-rasterize at current position so the old shape is cleared immediately
                if (this.interaction.showObstacle) {
                    this.interaction.rasterizeObstacle(
                        this.interaction.obstacleX,
                        this.interaction.obstacleY,
                        0, 0
                    );
                }
            });
        });
    }

    /** Bind play/pause, single-step, reset, and restart buttons. */
    _bindPlayback() {
        const btnPlay  = document.getElementById('btn-play');
        const btnStep  = document.getElementById('btn-step');
        const btnReset = document.getElementById('btn-reset');

        const togglePause = () => {
            this.solver.paused = !this.solver.paused;
            if (btnPlay) btnPlay.textContent = this.solver.paused ? '\u25B6 Play' : '\u23F8 Pause';
        };

        // Mirrors the rAF loop's order — step, draw, tick — so single-stepping
        // is self-contained rather than relying on the loop to repaint after it.
        //
        // The loop DOES tick unconditionally (main.js ticks even while paused),
        // so removing this call alone does not change what the user sees, and
        // mutating it away does not fail the suite. What actually froze the
        // readout during single-stepping was `_sampleProbe`'s `paused` gate,
        // which is now a simulation-time gate; see there. This call stays so
        // that stepping does not depend on a separate loop for its readout.
        const stepOnce = () => {
            if (this.solver.paused) {
                this.solver.step(this.numIters);
                this.renderer.draw();
                this.tick();
            }
        };

        btnPlay?.addEventListener('click', togglePause);
        btnStep?.addEventListener('click', stepOnce);
        btnReset?.addEventListener('click', () => this.reapplyCurrentPreset());
        document.getElementById('btn-restart')?.addEventListener('click', () => location.reload());

        this._togglePause = togglePause;
        this._stepOnce = stepOnce;
    }

    /** Bind open/close for the advanced settings panel and its child sliders. */
    _bindAdvancedPanel() {
        const panel = document.getElementById('advanced-panel');

        document.getElementById('btn-advanced')?.addEventListener('click', () => {
            panel?.classList.toggle('visible');
        });

        document.getElementById('btn-close-advanced')?.addEventListener('click', () => {
            panel?.classList.remove('visible');
        });

        document.querySelectorAll('.panel-close').forEach(btn => {
            btn.addEventListener('click', () => panel?.classList.remove('visible'));
        });

        this._bindSliders();
    }

    /** Wire up sliders for dt, omega, iterations, inflow velocity, and resolution tier buttons. */
    _bindSliders() {
        const bind = (id, valId, decimals, onChange) => {
            const el = document.getElementById(id);
            const valEl = document.getElementById(valId);
            if (!el) return;
            el.addEventListener('input', () => {
                if (valEl) valEl.textContent = parseFloat(el.value).toFixed(decimals);
                onChange(el.value);
            });
        };

        // dt and numIters both move the honest window, so both must repaint the
        // badge. dt sets BOTH bounds — the ceiling through nuNumConverged(dt)
        // and the floor through viscNuMax, which divides by dt — and numIters
        // decides whether the measured projection ceiling applies at all.
        // Without these the badge silently describes the previous timestep.
        bind('slider-dt',      'val-dt',      4, v => {
            this.solver.setParams({ dt: parseFloat(v) });
            this._updateReBadge();
        });
        bind('slider-omega',   'val-omega',   2, v => this.solver.setParams({ omega:   parseFloat(v) }));
        bind('slider-iters',   'val-iters',   0, v => {
            this.numIters = parseInt(v);
            this._updateReBadge();
        });

        // The Re slider carries a log-spaced position, so val-re is written by
        // _updateReBadge() rather than by bind()'s generic formatter.
        document.getElementById('slider-re')
            ?.addEventListener('input', () => this._updateReBadge());

        const invelEl  = document.getElementById('slider-invel');
        const invelVal = document.getElementById('val-invel');
        if (invelEl) {
            invelEl.addEventListener('input', () => {
                const inVel = parseFloat(invelEl.value);
                if (invelVal) invelVal.textContent = inVel.toFixed(2);
                this._setInflowVelocity(inVel);
                // U changed, so nu = U*D/Re and both window bounds move with it.
                this._updateReBadge();
            });
        }

        document.querySelectorAll('[data-tier]').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.tier);
                if (!Number.isInteger(idx) || idx < 0 || idx >= this.adaptive?.tiers.length) return;
                if (this.adaptive) {
                    this.adaptive.manualOverride = true;
                    this.adaptive.currentTierIndex = idx;
                    this.adaptive.applyTier();
                    document.querySelectorAll('[data-tier]').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                }
            });
        });

        document.getElementById('btn-defaults')?.addEventListener('click', () => {
            if (this.adaptive) this.adaptive.manualOverride = false;
            this.reapplyCurrentPreset();
        });
    }

    /**
     * Update the inflow velocity at column i=1 on the GPU and in the
     * persistent boundaryVelData so it survives across frames. Bounded: one
     * column write per rotation slot via writeInflowColumn — the old path
     * rebuilt a whole field from interaction's stale CPU mirror and pushed
     * it over the live one (a second instance of the field-reset defect).
     * @param {number} inVel - New inflow velocity value
     */
    _setInflowVelocity(inVel) {
        if (!this.boundaryVelData) return;
        const n = this.solver.numY;
        const col = this.boundaryVelData.uData;
        const mask = this.boundaryVelData.col1Mask;
        for (let j = 0; j < n; j++) col[1 * n + j] = mask[j] ? inVel : 0;
        this.solver.writeInflowColumn(1, col, 1 * n, n);
    }

    /**
     * Bind keyboard shortcuts: 'p' = play/pause, 'm' = step once, digit keys =
     * switch preset by 1-based index into PRESETS (no-op past the preset count).
     * Ignores keypresses inside input fields.
     */
    _bindKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (document.getElementById('guide-overlay')?.classList.contains('guide-visible')) return;
            if (this.tour?.active) return;
            switch (e.key) {
                case 'p': this._togglePause?.(); break;
                case 'm': this._stepOnce?.(); break;
                case '1': case '2': case '3': case '4': case '5': case '6': {
                    const idx = parseInt(e.key) - 1;
                    const keys = Object.keys(PRESETS);
                    if (idx < keys.length) {
                        const presetKey = keys[idx];
                        this._loadAndApplyPreset(presetKey);
                        const attrName = Object.keys(PRESET_KEY_MAP).find(k => PRESET_KEY_MAP[k] === presetKey);
                        document.querySelectorAll('[data-preset]').forEach(b => {
                            b.classList.toggle('active', b.dataset.preset === attrName);
                        });
                    }
                    break;
                }
            }
        });
    }

    /** Bind open/close for the guide modal and accordion section toggling. */
    _bindGuideModal() {
        const overlay = document.getElementById('guide-overlay');
        if (!overlay) return;

        const openGuide = () => overlay.classList.add('guide-visible');
        const closeGuide = () => overlay.classList.remove('guide-visible');

        document.getElementById('btn-guide')?.addEventListener('click', openGuide);
        document.getElementById('guide-close')?.addEventListener('click', closeGuide);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) closeGuide();
        });

        // Replay entry point — the guide is the help home, so re-entry lives
        // here. ui.tour is assigned by main.js after construction; the
        // click-time lookup makes bind order irrelevant.
        document.getElementById('btn-replay-tour')?.addEventListener('click', () => {
            this.tour?.start();
        });

        // Accordion: single-open behavior
        overlay.querySelectorAll('.guide-section-header').forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                const wasOpen = section.classList.contains('guide-section-open');
                overlay.querySelectorAll('.guide-section').forEach(s => s.classList.remove('guide-section-open'));
                if (!wasOpen) section.classList.add('guide-section-open');
            });
        });

        // Learn-more expanders: independent toggle
        overlay.querySelectorAll('.guide-learn-more-toggle').forEach(toggle => {
            toggle.addEventListener('click', () => {
                const lm = toggle.parentElement;
                lm.classList.toggle('learn-more-open');
                toggle.textContent = lm.classList.contains('learn-more-open')
                    ? toggle.textContent.replace('\u25B8', '\u25BE')
                    : toggle.textContent.replace('\u25BE', '\u25B8');
            });
        });

        // Expose for main.js to call from welcome modal link
        this.openGuide = openGuide;
    }
}
