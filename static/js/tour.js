/**
 * Onboarding tour — spotlight walkthrough of FlowLab's features.
 * Spec: docs/superpowers/specs/2026-07-31-onboarding-tour-design.md
 *
 * The engine is step-agnostic: steps are injected, so tests can drive tiny
 * custom step arrays. The real 12-step script (STEPS, Task 3) is what main.js
 * wires up.
 *
 * The overlay is four dim rects framing a GENUINE hole over the target (not a
 * translucent sheet): the spotlighted control is the only clickable element,
 * and nested panels (Advanced) show through the hole with no z-index
 * negotiation. Do-it steps are detected purely from DOM events — the tour
 * never instruments interaction.js or ui.js internals.
 */

const STORAGE_KEY = 'flowlab.tour.v1';
const PAD = 8; // px of breathing room around the spotlight hole
const DRAG_THRESHOLD = 5; // px of pointer travel that turns a click into a drag

export class Tour {
    /**
     * @param {Object} ctx
     * @param {Object} ctx.ui - UI controller.
     * @param {Object} ctx.interaction - Interaction instance (mode reads).
     * @param {Object} ctx.solver - FluidSolver (paused reads).
     * @param {Array} ctx.steps - [
     *   { target?: string, title: string, body: string,
     *     action?: { type: 'click' | 'change' | 'drag' | 'click-then-canvas',
     *                selector?: string, advanceWhen?: (ctx) => boolean },
     *     onLeave?: (ctx) => void }
     * ]
     *
     * `action.selector` narrows which descendant counts as the do-it gesture
     * (e.g. click steps use it to ignore buttons that are not the intended
     * control). `action.advanceWhen` is an optional post-gesture gate that
     * decides whether the step should advance after a qualifying gesture (used
     * by toggle buttons whose state changes after the click). `onLeave` fires
     * both when navigating between steps and when the tour exits (finish or
     * skip).
     */
    constructor({ ui, interaction, solver, steps }) {
        this._ctx = { ui, interaction, solver };
        this._steps = steps;
        this._active = false;
        this._index = -1;
        this._els = null;            // { dims: [top,right,bottom,left], ring, tooltip }
        this._teardownAction = null; // removes the current step's action listeners
        this._overrideTarget = null; // click-then-canvas retargets the hole mid-step
        this._relayoutTimer = null;  // one deferred layout pass, covers target CSS transitions
        this._targetTransitionListener = null; // { target, fn } removes the relayout transition listener
        this._onKeydown = null;
        this._onResize = null;
    }

    get active() { return this._active; }
    get stepIndex() { return this._index; }

    static readFlag() {
        try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
    }

    static writeFlag(value) {
        // Private-mode failure just means the welcome re-appears next visit.
        try { localStorage.setItem(STORAGE_KEY, value); } catch { /* ignore */ }
    }

    /** Start (or restart) the tour from step 0. */
    start() {
        if (this._active) return;
        document.getElementById('guide-overlay')?.classList.remove('guide-visible');
        this._resetToCleanState();
        this._buildDom();
        this._active = true;
        this._onKeydown = (e) => { if (e.key === 'Escape') this.skip(); };
        document.addEventListener('keydown', this._onKeydown);
        let resizeTimer = null;
        this._onResize = () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => this._layout(), 150);
        };
        window.addEventListener('resize', this._onResize);
        this._goTo(0);
    }

    /** Skip the tour — same teardown as finishing, different flag value. */
    skip() { this._end('skipped'); }

    /** Advance a read step. Last read step finishes the tour. */
    next() {
        const step = this._steps[this._index];
        if (!step || step.action) return;
        if (this._index === this._steps.length - 1) this._end('done');
        else this._goTo(this._index + 1);
    }

    /** Back is navigation, not time travel: undo nothing, retrace one step. */
    back() {
        if (this._index > 0) this._goTo(this._index - 1);
    }

    // ── Internals ─────────────────────────────────────────────────────────

    _end(result) {
        if (!this._active) return;
        this._teardownCurrentAction();
        const step = this._steps[this._index];
        if (step?.onLeave) step.onLeave(this._ctx);
        this._resetToCleanState();
        clearTimeout(this._relayoutTimer);
        this._detachTargetRelayoutListener();
        Tour.writeFlag(result);
        this._removeDom();
        document.removeEventListener('keydown', this._onKeydown);
        window.removeEventListener('resize', this._onResize);
        this._active = false;
        this._index = -1;
    }

    _goTo(i) {
        if (i < 0 || i >= this._steps.length) return;
        const prev = this._steps[this._index];
        this._teardownCurrentAction();
        if (prev?.onLeave) prev.onLeave(this._ctx);
        this._overrideTarget = null;
        this._index = i;
        this._renderStep();
    }

    _renderStep() {
        const step = this._steps[this._index];
        const { tooltip } = this._els;
        const isLast = this._index === this._steps.length - 1;

        tooltip.innerHTML = '';
        const title = document.createElement('div');
        title.className = 'tour-title';
        title.textContent = step.title;
        const body = document.createElement('div');
        body.className = 'tour-body';
        body.textContent = step.body;
        const footer = document.createElement('div');
        footer.className = 'tour-footer';

        const counter = document.createElement('span');
        counter.className = 'tour-counter';
        counter.textContent = `${this._index + 1} / ${this._steps.length}`;

        const skipBtn = document.createElement('button');
        skipBtn.className = 'tour-skip';
        skipBtn.textContent = 'Skip tour';
        skipBtn.addEventListener('click', () => this.skip());

        const nav = document.createElement('span');
        nav.className = 'tour-nav';
        if (this._index > 0) {
            const backBtn = document.createElement('button');
            backBtn.className = 'tour-btn tour-btn-secondary';
            backBtn.textContent = 'Back';
            backBtn.addEventListener('click', () => this.back());
            nav.appendChild(backBtn);
        }
        if (!step.action || !this._targetEl(step.target)) {
            const nextBtn = document.createElement('button');
            nextBtn.className = 'tour-btn tour-btn-primary';
            nextBtn.textContent = isLast ? 'Done' : 'Next';
            nextBtn.addEventListener('click', () => step.action ? this._advance() : this.next());
            nav.appendChild(nextBtn);
        }

        footer.append(counter, skipBtn, nav);
        tooltip.append(title, body, footer);

        this._els.ring.classList.toggle('tour-pulse', Boolean(step.action));
        this._installAction(step);

        this._layout();
        // Targets with CSS transitions (the Advanced panel slides in over
        // 0.3 s) get one deferred pass so the ring lands on their final rect.
        clearTimeout(this._relayoutTimer);
        this._relayoutTimer = setTimeout(() => this._layout(), 350);
    }

    /** Resolve the step's target element, falling back to centered on failure. */
    _targetEl(selector) {
        if (!selector) return null;
        const el = document.querySelector(selector);
        if (!el) console.warn(`[tour] target not found: ${selector} — falling back to centered`);
        return el;
    }

    _layout() {
        if (!this._active || !this._els) return;
        const { dims, ring, tooltip } = this._els;
        const step = this._steps[this._index];
        const W = window.innerWidth, H = window.innerHeight;
        const [dt, dr, db, dl] = dims;

        let el = this._targetEl(this._overrideTarget ?? step.target);
        if (el) {
            const r = el.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0 || r.bottom <= 0 || r.right <= 0 || r.top >= H || r.left >= W) {
                el = null;
            } else {
                this._attachTargetRelayoutListener(el);
            }
        }
        if (!el) this._detachTargetRelayoutListener();

        if (!el) {
            // Centered card: the top dim covers everything, ring hidden.
            Object.assign(dt.style, { left: '0px', top: '0px', width: `${W}px`, height: `${H}px` });
            for (const d of [dr, db, dl]) Object.assign(d.style, { left: '0px', top: '0px', width: '0px', height: '0px' });
            ring.style.display = 'none';
            tooltip.style.left = `${Math.max(12, (W - tooltip.offsetWidth) / 2)}px`;
            tooltip.style.top = `${Math.max(12, (H - tooltip.offsetHeight) / 2)}px`;
            return;
        }

        const r = el.getBoundingClientRect();
        const t = Math.max(0, r.top - PAD), l = Math.max(0, r.left - PAD);
        const b = Math.min(H, r.bottom + PAD), rt = Math.min(W, r.right + PAD);

        ring.style.display = 'block';
        Object.assign(ring.style, { left: `${l}px`, top: `${t}px`, width: `${rt - l}px`, height: `${b - t}px` });

        Object.assign(dt.style, { left: '0px', top: '0px', width: `${W}px`, height: `${t}px` });
        Object.assign(db.style, { left: '0px', top: `${b}px`, width: `${W}px`, height: `${H - b}px` });
        Object.assign(dl.style, { left: '0px', top: `${t}px`, width: `${l}px`, height: `${b - t}px` });
        Object.assign(dr.style, { left: `${rt}px`, top: `${t}px`, width: `${W - rt}px`, height: `${b - t}px` });

        // Tooltip: side with the most room; if no side plausibly fits (e.g. the
        // target is the whole canvas), float it at the top-center inside the hole.
        const tw = tooltip.offsetWidth, th = tooltip.offsetHeight, M = 12;
        const spaces = { bottom: H - b, top: t, right: W - rt, left: l };
        const side = Object.keys(spaces).sort((a, z) => spaces[z] - spaces[a])[0];
        let x, y;
        if (spaces[side] < 180) {
            x = Math.min(Math.max(l + (rt - l - tw) / 2, M), W - tw - M);
            y = Math.min(t + 16, H - th - M);
        } else if (side === 'bottom') {
            x = Math.min(Math.max(l, M), W - tw - M); y = b + M;
        } else if (side === 'top') {
            x = Math.min(Math.max(l, M), W - tw - M); y = Math.max(t - th - M, M);
        } else if (side === 'right') {
            x = rt + M; y = Math.min(Math.max(t, M), H - th - M);
        } else { // left
            x = Math.max(l - tw - M, M); y = Math.min(Math.max(t, M), H - th - M);
        }
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
    }

    /**
     * Re-layout when the current target's CSS transition finishes, so a
     * target that slides off-screen (e.g. the Advanced panel) falls back to the
     * centered-card layout instead of framing a hidden element.
     */
    _attachTargetRelayoutListener(el) {
        if (this._targetTransitionListener && this._targetTransitionListener.target === el) return;
        this._detachTargetRelayoutListener();
        const fn = () => this._layout();
        el.addEventListener('transitionend', fn);
        this._targetTransitionListener = { target: el, fn };
    }

    _detachTargetRelayoutListener() {
        if (!this._targetTransitionListener) return;
        const { target, fn } = this._targetTransitionListener;
        target.removeEventListener('transitionend', fn);
        this._targetTransitionListener = null;
    }

    _buildDom() {
        const mk = (cls) => {
            const d = document.createElement('div');
            d.className = cls;
            document.body.appendChild(d);
            return d;
        };
        const dims = ['top', 'right', 'bottom', 'left'].map(side => mk(`tour-dim tour-dim-${side}`));
        const ring = mk('tour-ring');
        const tooltip = mk('tour-tooltip');
        // Clicking a dim instead of the target: re-trigger the ring pulse,
        // nothing else. No nag tooltips, no auto-advance.
        dims.forEach(d => d.addEventListener('click', () => {
            if (!this._steps[this._index]?.action) return;
            ring.classList.remove('tour-pulse');
            void ring.offsetWidth; // restart the CSS animation
            ring.classList.add('tour-pulse');
        }));
        this._els = { dims, ring, tooltip };
    }

    _removeDom() {
        if (!this._els) return;
        const { dims, ring, tooltip } = this._els;
        [...dims, ring, tooltip].forEach(el => el.remove());
        this._els = null;
    }

    // Removes the current step's action listeners (installed only for do-it steps).
    _teardownCurrentAction() {
        this._teardownAction?.();
        this._teardownAction = null;
    }

    /**
     * Drive the app's own UI paths back to the clean Kármán state. Runs on
     * start (so Replay begins clean) and on finish AND skip (so the tour can
     * never leak its mess into the user's session). Only .click() on existing
     * controls — no solver internals.
     */
    _resetToCleanState() {
        // Restore the default resolution tier first: tier selection reloads the
        // current preset at the chosen resolution, so this must happen before
        // the Kármán preset click below.
        const tierBtn = document.querySelector('[data-tier="2"]');
        if (tierBtn && !tierBtn.classList.contains('active')) tierBtn.click();
        document.querySelector('[data-preset="karman-vortex"]')?.click();
        // loadPreset sets interaction.activeShape but not the button chrome.
        const circleBtn = document.querySelector('[data-shape="circle"]');
        if (circleBtn && !circleBtn.classList.contains('active')) circleBtn.click();
        if (this._ctx.solver.paused) document.getElementById('btn-play')?.click();
        if (document.getElementById('advanced-panel')?.classList.contains('visible')) {
            document.getElementById('btn-close-advanced')?.click();
        }
        if (this._ctx.interaction.mode === 'particles') {
            document.getElementById('btn-mode')?.click();
        }
    }

    /** A do-it step's gesture was observed: advance, or finish on the last step. */
    _advance() {
        if (!this._active) return;
        if (this._index === this._steps.length - 1) this._end('done');
        else this._goTo(this._index + 1);
    }

    /**
     * Wire the DOM listeners that detect a do-it gesture. Pure DOM events —
     * the only app state read is interaction.mode, through the injected ref.
     */
    _installAction(step) {
        if (!step.action) return;
        const el = this._targetEl(step.target);
        const canvas = document.getElementById('overlay-canvas');
        const offs = [];
        const on = (target, type, fn) => {
            target.addEventListener(type, fn);
            offs.push(() => target.removeEventListener(type, fn));
        };

        switch (step.action.type) {
            case 'click': {
                if (!el) break;
                // Delegated targets (toolbar groups) count only real controls.
                // An optional `selector` narrows which descendant counts, and
                // an optional `advanceWhen` gate lets the step decide after a
                // qualifying click whether it really should advance (used by
                // #btn-advanced, which toggles the panel open and closed).
                on(el, 'click', (e) => {
                    const selector = step.action.selector ?? 'button, input, a';
                    if (!e.target.closest(selector)) return;
                    if (!step.action.advanceWhen || step.action.advanceWhen(this._ctx)) {
                        this._advance();
                    }
                });
                break;
            }
            case 'change': {
                if (!el) break;
                on(el, 'change', () => this._advance());
                break;
            }
            case 'drag': {
                if (!canvas) break;
                let dragging = false, moved = false, sx = 0, sy = 0;
                on(canvas, 'pointerdown', (e) => {
                    if (e.button !== 0) return;
                    dragging = true; moved = false; sx = e.clientX; sy = e.clientY;
                    try { e.target.setPointerCapture(e.pointerId); } catch (_) {}
                });
                on(canvas, 'pointermove', (e) => {
                    if (dragging && Math.hypot(e.clientX - sx, e.clientY - sy) > DRAG_THRESHOLD) moved = true;
                });
                const endDrag = () => {
                    if (dragging && moved) this._advance();
                    dragging = false; moved = false;
                };
                const cancelDrag = () => { dragging = false; moved = false; };
                on(window, 'pointerup', endDrag);
                on(window, 'pointercancel', cancelDrag);
                break;
            }
            case 'click-then-canvas': {
                if (!el || !canvas) break;
                let armed = false;
                on(el, 'click', () => {
                    // The button toggles the mode. Only arm when the click lands
                    // us in particle mode; otherwise keep the hole on the
                    // button so the user can click again.
                    if (this._ctx.interaction.mode !== 'particles') {
                        armed = false;
                        this._overrideTarget = null;
                        this._layout();
                        return;
                    }
                    armed = true;
                    // The hole must move to the canvas, or the dims would block
                    // the very click the step is asking for.
                    this._overrideTarget = '#overlay-canvas';
                    this._layout();
                });
                on(canvas, 'pointerdown', (e) => {
                    if (e.button !== 0) return;
                    if (armed && this._ctx.interaction.mode === 'particles') this._advance();
                });
                break;
            }
            default:
                console.warn(`[tour] unknown action type: ${step.action.type}`);
        }
        this._teardownAction = () => { offs.forEach(off => off()); };
    }
}

/**
 * The 12-step onboarding script, in Guide order. Copy stays at one or two
 * sentences per step: the tour orients, the Guide (?) documents.
 */
export const STEPS = [
    {
        target: null,
        title: 'Welcome to FlowLab',
        body: 'This is a live fluid simulation running entirely on your GPU. The smoke is dye injected at the left inlet, carried by the flow. A quick tour — about a minute.',
    },
    {
        target: '#overlay-canvas',
        title: 'Move the obstacle',
        body: 'Click and drag the circle through the flow and watch the wake react — vortices shed and swirl downstream in real time.',
        action: { type: 'drag' },
    },
    {
        target: '#viz-group',
        title: 'Visualization modes',
        body: 'The field view can show dye (Smoke), Pressure, Streamlines, or a Velocity arrow grid, in any combination. Turn on Pressure now.',
        action: { type: 'change' },
    },
    {
        target: '#colorbar',
        title: 'The colorbar',
        body: 'The colorbar always tells you what the colors mean: the range and units of whichever field is on top. Pressure reads in N/m², smoke is dye concentration.',
    },
    {
        target: '#shape-group',
        title: 'Obstacle shapes',
        body: 'Circle, Square, Airfoil, Wedge — each sheds a different wake. Pick one now; the Airfoil is a favorite.',
        action: { type: 'click', selector: '[data-shape]' },
    },
    {
        target: '#btn-mode',
        title: 'Particle tracers',
        body: 'Click Particles, then click anywhere on the flow to place an emitter. Ice-blue tracers stream from it, showing where fluid parcels actually travel.',
        action: { type: 'click-then-canvas' },
    },
    {
        target: '#preset-bar',
        title: 'Presets',
        body: 'Two classic flows ship with FlowLab. Switch to Backward Step now — a sudden expansion with a trapped recirculation bubble behind the step.',
        action: { type: 'click' },
    },
    {
        target: '#btn-play',
        title: 'Pause and step',
        body: 'Pause the simulation. While paused, Step advances exactly one frame — handy for inspecting a vortex up close. The tour resumes play for you.',
        action: { type: 'click' },
        onLeave: ({ solver }) => {
            if (solver.paused) document.getElementById('btn-play')?.click();
        },
    },
    {
        target: '#btn-advanced',
        title: 'Advanced controls',
        body: 'Open the Advanced panel — this is where the numerical solver lives.',
        action: { type: 'click', advanceWhen: () => document.getElementById('advanced-panel')?.classList.contains('visible') },
    },
    {
        target: '#advanced-panel',
        title: 'Solver parameters',
        body: 'dt, ω, pressure iterations, inflow velocity, and the measured Re / St readouts. Each is documented in the Guide — nothing here is guesswork.',
    },
    {
        target: '#resolution-picker',
        title: 'Grid resolution',
        body: 'Grid tiers from 64 to 1024. Higher is prettier but heavier — if your GPU struggles, FlowLab downscales automatically and says so in the HUD.',
    },
    {
        target: null,
        title: 'You\u2019re all set',
        body: 'Shortcuts: P pauses, M steps, 1 / 2 load presets. Everything else is in the Guide (the ? button, top right). Pressing Done resets the tour\u2019s changes — you start from a clean Kármán vortex street. Enjoy!',
    },
];
