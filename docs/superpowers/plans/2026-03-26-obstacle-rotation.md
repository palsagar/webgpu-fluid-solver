# Obstacle Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to rotate obstacle shapes via Shift + mouse, with visual feedback.

**Architecture:** Add `obstacleAngle` state to Interaction. Inverse-rotate cell coordinates before existing inside-tests in `rasterizeObstacle()`. Apply canvas rotation transform in `drawObstacle()`. Track Shift key via document keydown/keyup + event.shiftKey fallback.

**Tech Stack:** Vanilla JS (ES modules), WebGPU, 2D canvas rendering. No build step, no test framework — testing via browser.

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `static/js/interaction.js` | Modify | Add `obstacleAngle`, `_shiftHeld`, Shift tracking, `_rotate()` method, inverse-rotate in `rasterizeObstacle`, modify pointer handlers |
| `static/js/renderer.js` | Modify | Canvas rotation in `drawObstacle`, orientation line |
| `static/js/presets.js` | Modify | Reset `obstacleAngle = 0` on preset load |

No new files. No shader changes. No uniform buffer changes.

---

### Task 1: Add rotation state and Shift tracking to Interaction

**Files:**
- Modify: `static/js/interaction.js:10-39` (constructor)

- [ ] **Step 1: Add `obstacleAngle` and `_shiftHeld` state**

In the constructor, after `this._particleSystem = null;` (line 24), add:

```js
this.obstacleAngle = 0;
this._shiftHeld = false;
```

- [ ] **Step 2: Add Shift key listeners**

In the constructor, after the touch event listeners (after line 38), add:

```js
document.addEventListener('keydown', e => { if (e.key === 'Shift') this._shiftHeld = true; });
document.addEventListener('keyup', e => { if (e.key === 'Shift') this._shiftHeld = false; });
```

- [ ] **Step 3: Modify mouse event handlers to pass shiftKey**

Replace the three mouse event listeners (lines 33-35):

```js
canvas.addEventListener('mousedown', e => this._onPointerDown(e.clientX, e.clientY, e.shiftKey));
canvas.addEventListener('mousemove', e => this._onPointerMove(e.clientX, e.clientY, e.shiftKey));
canvas.addEventListener('mouseup',   () => this._endDrag());
```

- [ ] **Step 4: Commit**

```bash
git -c commit.gpgsign=false add static/js/interaction.js
git -c commit.gpgsign=false commit -m "feat(rotation): add obstacleAngle state and Shift tracking"
```

---

### Task 2: Add rotation and pointer-move logic to Interaction

**Files:**
- Modify: `static/js/interaction.js:231-278` (pointer handlers)

- [ ] **Step 1: Add `_rotate()` method**

After `_endDrag()` (line 278), add:

```js
/**
 * Sets the obstacle rotation angle from the mouse position.
 * Angle is computed as atan2 from obstacle center to cursor (compass-needle).
 * Re-rasterizes at the current position with zero velocity.
 */
_rotate(clientX, clientY) {
    if (this.mode !== 'obstacle') return;
    const { x, y } = this.screenToSim(clientX, clientY);
    this.obstacleAngle = Math.atan2(y - this.obstacleY, x - this.obstacleX);
    this.rasterizeObstacle(this.obstacleX, this.obstacleY, 0, 0);
}
```

- [ ] **Step 2: Add `_onPointerMove()` method replacing `_drag`**

Rename `_drag` to `_onPointerMove` and add Shift-rotation handling. Replace the existing `_drag` method (lines 263-273) with:

```js
/**
 * Handles pointer movement. If Shift is held, rotates the obstacle.
 * Otherwise continues an active drag (translate).
 */
_onPointerMove(clientX, clientY, shiftKey) {
    if (this.mode !== 'obstacle') return;
    if (shiftKey || this._shiftHeld) {
        this._rotate(clientX, clientY);
        return;
    }
    if (!this.dragging) return;
    const { x, y } = this.screenToSim(clientX, clientY);
    const dt = this.solver.params.dt;
    const vx = (x - this.prevX) / dt;
    const vy = (y - this.prevY) / dt;
    this.rasterizeObstacle(x, y, vx, vy);
    this.prevX = x;
    this.prevY = y;
}
```

- [ ] **Step 3: Update `_onPointerDown` for Shift**

Replace `_onPointerDown` (lines 231-242) with:

```js
_onPointerDown(clientX, clientY, shiftKey) {
    if (this.mode === 'particles') {
        if (this._particleSystem) {
            const { x, y } = this.screenToSim(clientX, clientY);
            this._particleSystem.addEmitter(x, y);
            const hint = document.getElementById('canvas-hint');
            if (hint) hint.remove();
        }
        return;
    }
    if (shiftKey || this._shiftHeld) {
        this._rotate(clientX, clientY);
        return;
    }
    this._startDrag(clientX, clientY);
}
```

- [ ] **Step 4: Update touch event handler for mousemove rename**

The touch `touchmove` handler (line 37) calls `this._drag(...)`. Update it to call `this._onPointerMove(...)` instead. Touch doesn't have Shift, so pass `false`:

```js
canvas.addEventListener('touchmove',  e => { e.preventDefault(); const t = e.touches[0]; this._onPointerMove(t.clientX, t.clientY, false); }, { passive: false });
```

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false add static/js/interaction.js
git -c commit.gpgsign=false commit -m "feat(rotation): add _rotate() and Shift-aware pointer handlers"
```

---

### Task 3: Add inverse rotation to rasterizeObstacle

**Files:**
- Modify: `static/js/interaction.js:73-198` (rasterizeObstacle cell loop)

- [ ] **Step 1: Compute rotation matrix outside the cell loop**

After `const maxExtent = ...` (line 92), add:

```js
const angle = this.obstacleAngle;
const cosA = Math.cos(-angle);
const sinA = Math.sin(-angle);
```

- [ ] **Step 2: Apply inverse rotation inside the cell loop**

After the `dx`/`dy` computation (lines 153-154), add the inverse rotation and replace shape tests to use local-frame coordinates. Replace the block from `const dx = cx - centerX;` through the end of the wedge test (lines 153-185) with:

```js
const dx = cx - centerX;
const dy = cy - centerY;

// Inverse-rotate into obstacle's local frame
const ldx = dx * cosA - dy * sinA;
const ldy = dx * sinA + dy * cosA;

let inside = false;

if (shape === 'circle') {
    inside = dx * dx + dy * dy < r * r;
} else if (shape === 'square') {
    inside = Math.abs(ldx) < r && Math.abs(ldy) < r;
} else if (shape === 'airfoil') {
    const lx = ldx + chord * 0.5;
    const ly = ldy;
    if (lx >= 0 && lx <= chord) {
        const xc = lx / chord;
        const yt = 5 * 0.12 * chord * (
            0.2969 * Math.sqrt(xc)
            - 0.1260 * xc
            - 0.3516 * xc * xc
            + 0.2843 * xc * xc * xc
            - 0.1015 * xc * xc * xc * xc
        );
        inside = Math.abs(ly) < yt;
    }
} else if (shape === 'wedge') {
    const lx = ldx + wedgeLen * 0.5;
    const ly = ldy;
    inside = lx >= 0 && lx < wedgeLen && Math.abs(ly) < lx * tanHA;
}
```

Note: circle uses `dx`/`dy` (not `ldx`/`ldy`) since distance is rotation-invariant.

- [ ] **Step 3: Commit**

```bash
git -c commit.gpgsign=false add static/js/interaction.js
git -c commit.gpgsign=false commit -m "feat(rotation): inverse-rotate cell coords in rasterizeObstacle"
```

---

### Task 4: Add canvas rotation to drawObstacle in Renderer

**Files:**
- Modify: `static/js/renderer.js:193-268` (drawObstacle method)

- [ ] **Step 1: Add rotation transform wrapping all shape drawing**

In `drawObstacle`, after `ctx.lineWidth = 1;` (line 211), add the canvas rotation transform. Then wrap all shape drawing (lines 213-267) in a save/restore block with the rotation applied. Replace the entire shape-drawing block (lines 213-267) with:

```js
const angle = interaction.obstacleAngle || 0;
const pcx = cX(cx);
const pcy = cY(cy);

ctx.save();
ctx.translate(pcx, pcy);
ctx.rotate(-angle);
ctx.translate(-pcx, -pcy);

if (shape === 'circle') {
  ctx.beginPath();
  ctx.arc(cX(cx), cY(cy), r / domainWidth * cw, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
} else if (shape === 'square') {
  const hw = r / domainWidth * cw;
  const hh = r / domainHeight * ch;
  ctx.fillRect(cX(cx) - hw, cY(cy) - hh, 2 * hw, 2 * hh);
  ctx.strokeRect(cX(cx) - hw, cY(cy) - hh, 2 * hw, 2 * hh);
} else if (shape === 'airfoil') {
  const chord = r * 4;
  const n = 20;
  const upperPts = [];
  const lowerPts = [];
  for (let k = 0; k <= n; k++) {
    const xc = k / n;
    const lx = xc * chord - chord * 0.5;
    const yt = 5 * 0.12 * chord * (
      0.2969 * Math.sqrt(xc)
      - 0.1260 * xc
      - 0.3516 * xc * xc
      + 0.2843 * xc * xc * xc
      - 0.1015 * xc * xc * xc * xc
    );
    upperPts.push([cx + lx, cy + yt]);
    lowerPts.push([cx + lx, cy - yt]);
  }
  ctx.beginPath();
  ctx.moveTo(cX(upperPts[0][0]), cY(upperPts[0][1]));
  for (let k = 1; k <= n; k++) {
    ctx.lineTo(cX(upperPts[k][0]), cY(upperPts[k][1]));
  }
  for (let k = n; k >= 0; k--) {
    ctx.lineTo(cX(lowerPts[k][0]), cY(lowerPts[k][1]));
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
} else if (shape === 'wedge') {
  const wedgeLen = r * 3;
  const tanHA = Math.tan(15 * Math.PI / 180);
  const apexX = cx - wedgeLen * 0.5;
  const baseX = cx + wedgeLen * 0.5;
  const halfH = wedgeLen * tanHA;
  ctx.beginPath();
  ctx.moveTo(cX(apexX), cY(cy));
  ctx.lineTo(cX(baseX), cY(cy + halfH));
  ctx.lineTo(cX(baseX), cY(cy - halfH));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

ctx.restore();
```

- [ ] **Step 2: Add orientation line (visible while Shift held)**

After the `ctx.restore()` added in Step 1, add:

```js
if (interaction._shiftHeld) {
  const lineLen = 1.5 * r;
  const ex = cx + lineLen * Math.cos(angle);
  const ey = cy + lineLen * Math.sin(angle);
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cX(cx), cY(cy));
  ctx.lineTo(cX(ex), cY(ey));
  ctx.stroke();
  ctx.restore();
}
```

- [ ] **Step 3: Commit**

```bash
git -c commit.gpgsign=false add static/js/renderer.js
git -c commit.gpgsign=false commit -m "feat(rotation): add canvas rotation transform and orientation line to drawObstacle"
```

---

### Task 5: Reset obstacleAngle on preset load

**Files:**
- Modify: `static/js/presets.js:148-158` (obstacle setup in loadPreset)

- [ ] **Step 1: Add angle reset**

In `loadPreset`, after `interaction._prevBBox = null;` (line 148), add:

```js
interaction.obstacleAngle = 0;
```

- [ ] **Step 2: Commit**

```bash
git -c commit.gpgsign=false add static/js/presets.js
git -c commit.gpgsign=false commit -m "feat(rotation): reset obstacleAngle on preset load"
```

---

### Task 6: Manual browser testing

**Files:** None (read-only verification)

- [ ] **Step 1: Start the dev server**

```bash
uv run uvicorn server:app --reload --port 8000
```

- [ ] **Step 2: Open browser and verify basic simulation**

Navigate to `http://localhost:8000`. Dismiss welcome modal. Confirm Kármán Vortex preset runs normally with the circle obstacle visible.

- [ ] **Step 3: Test drag-to-translate still works**

Click and drag the obstacle without Shift held. Confirm it moves and the flow responds.

- [ ] **Step 4: Test Shift+mousemove rotation**

Hold Shift and move the mouse around the obstacle. Confirm:
- The orientation line appears (dashed white line from center)
- The obstacle does NOT translate
- The rasterized solid cells match the visual overlay rotation

- [ ] **Step 5: Test rotation with non-symmetric shapes**

Switch to airfoil shape, hold Shift, rotate. Confirm the airfoil rotates visually and the flow wraps around the rotated shape. Repeat with square and wedge.

- [ ] **Step 6: Test angle persistence**

Rotate the airfoil to ~45 degrees. Release Shift. Confirm angle holds. Switch to square shape. Confirm the square is rendered at the same angle.

- [ ] **Step 7: Test preset reset**

Click a different preset, then back. Confirm `obstacleAngle` resets to 0 (shape returns to default orientation).
