import { loadPreset, PRESETS } from './presets.js';

// Map kebab-case data-preset attribute values to PRESETS object keys
const PRESET_KEY_MAP = {
    'wind-tunnel':    'windTunnel',
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

        this._updateRe(inVel);
    }

    /**
     * Compute and display the Reynolds number based on inflow velocity and obstacle diameter.
     * Re = U * D / h, where D = 2 * obstacleRadius and h is the cell size.
     * @param {number} inVel - Inflow velocity magnitude
     */
    _updateRe(inVel) {
        const el = document.getElementById('val-re');
        if (!el) return;
        const h = this.solver.h;
        const D = 2 * this.interaction.obstacleRadius;
        if (inVel === 0 || D === 0) {
            el.textContent = '--';
            this._updateFlowInfo();
            return;
        }
        const Re = inVel * D / h;
        el.textContent = Re.toFixed(0);
        this._updateFlowInfo();
    }

    /** Update the flow-info overlay text with preset-specific physics description and Re. */
    _updateFlowInfo() {
        const el = document.getElementById('flow-info');
        if (!el) return;
        const info = {
            windTunnel:    'Uniform flow past a bluff body — wake separation and drag',
            karmanVortex:  'Periodic vortex shedding behind a small cylinder',
            backwardStep:  'Sudden expansion — recirculation and flow reattachment',
        };
        const reEl = document.getElementById('val-re');
        const re = reEl ? reEl.textContent : '--';
        const desc = info[this.currentPreset] || '';
        el.textContent = desc + (re !== '--' ? `  · Re ≈ ${re}` : '');
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

        const stepOnce = () => {
            if (this.solver.paused) {
                this.solver.step(this.numIters);
                this.renderer.draw();
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

        bind('slider-dt',      'val-dt',      4, v => this.solver.setParams({ dt:      parseFloat(v) }));
        bind('slider-omega',   'val-omega',   2, v => this.solver.setParams({ omega:   parseFloat(v) }));
        bind('slider-iters',   'val-iters',   0, v => { this.numIters = parseInt(v); });


        const invelEl  = document.getElementById('slider-invel');
        const invelVal = document.getElementById('val-invel');
        if (invelEl) {
            invelEl.addEventListener('input', () => {
                const inVel = parseFloat(invelEl.value);
                if (invelVal) invelVal.textContent = inVel.toFixed(2);
                this._setInflowVelocity(inVel);
                this._updateRe(inVel);
            });
        }

        document.querySelectorAll('[data-tier]').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.tier);
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
     * Update the inflow velocity at column i=1 in both the GPU buffers and
     * the persistent boundaryVelData so it survives across frames.
     * @param {number} inVel - New inflow velocity value
     */
    _setInflowVelocity(inVel) {
        const { solver, interaction } = this;
        const { numX, numY } = solver;
        // Build a full velocity array based on interaction's stored _uData,
        // then override the i=1 inflow column.
        const full = new Float32Array(interaction._uData);
        const mask = interaction.boundaryMask;
        for (let j = 0; j < numY; j++) {
            const idx = 1 * numY + j;
            if (!mask || mask[idx] !== 0) {
                full[idx] = inVel;
            }
        }
        interaction._uData.set(full);
        solver.writeVelocityU(full);
        // Update boundary vel data so it persists across frames
        if (this.boundaryVelData) {
            this.boundaryVelData.uData = full.slice();
        }
    }

    /**
     * Bind keyboard shortcuts: 'p' = play/pause, 'm' = step once,
     * '1'-'6' = switch preset by index. Ignores keypresses inside input fields.
     */
    _bindKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (document.getElementById('guide-overlay')?.classList.contains('guide-visible')) return;
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
