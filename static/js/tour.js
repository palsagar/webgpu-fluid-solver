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

export class Tour {
    /**
     * @param {Object} ctx
     * @param {Object} ctx.ui - UI controller.
     * @param {Object} ctx.interaction - Interaction instance (mode reads).
     * @param {Object} ctx.solver - FluidSolver (paused reads).
     * @param {Array} ctx.steps - [{ target, title, body, action?, onLeave? }]
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
        clearTimeout(this._relayoutTimer);
        Tour.writeFlag(result);
        this._removeDom();
        document.removeEventListener('keydown', this._onKeydown);
        window.removeEventListener('resize', this._onResize);
        this._active = false;
        this._index = -1;
    }

    _goTo(i) {
        if (i < 0 || i >= this._steps.length) return;
        this._teardownCurrentAction();
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
        const nextBtn = document.createElement('button');
        nextBtn.className = 'tour-btn tour-btn-primary';
        nextBtn.textContent = isLast ? 'Done' : 'Next';
        nextBtn.addEventListener('click', () => this.next());
        nav.appendChild(nextBtn);

        footer.append(counter, skipBtn, nav);
        tooltip.append(title, body, footer);

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
        const el = this._targetEl(this._overrideTarget ?? step.target);

        const W = window.innerWidth, H = window.innerHeight;
        const [dt, dr, db, dl] = dims;

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
}
