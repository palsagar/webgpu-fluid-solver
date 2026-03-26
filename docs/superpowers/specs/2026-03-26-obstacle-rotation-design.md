# Obstacle Rotation Design Spec

**Date**: 2026-03-26
**Feature**: Allow users to rotate solid obstacle shapes via Shift + mouse

## Summary

Add rotation capability to the obstacle system. Users hold Shift and move the mouse to set the obstacle's rotation angle based on the vector from obstacle center to cursor position (compass-needle style). Existing drag-to-translate behavior is unchanged when Shift is not held.

## Decisions

1. **Rotation control**: Angle = `atan2(mouseY - obstacleY, mouseX - obstacleX)` in simulation coordinates
2. **Activation key**: Shift (read via `e.shiftKey` on mouse events + `keydown`/`keyup` tracking)
3. **Shape scope**: All shapes (circle stores angle but is visually symmetric; angle persists across shape switches)
4. **Visual feedback**: Thin orientation line from obstacle center outward along rotation axis, visible only while Shift is held

## Approach

Inverse-rotate cell coordinates in the inside test. Before each shape's existing axis-aligned inside test, transform `(dx, dy)` by `-angle` into the obstacle's local frame. Shape test logic is unchanged. Same rotation applied to canvas drawing via `ctx.rotate()`.

## State

Single new property: `Interaction.obstacleAngle` (radians, default 0). No new GPU buffers, no shader changes, no uniform buffer layout changes.

## Interaction Layer (`interaction.js`)

### Shift-key tracking

- Track `_shiftHeld` via `keydown`/`keyup` on `document`
- Also read `e.shiftKey` on mouse events as fallback

### Mouse behavior

- **Shift + mousemove** (obstacle mode): compute angle from cursor-to-center vector, re-rasterize at current position with new angle, zero obstacle velocity. No translation.
- **Shift + mousedown**: set angle immediately, don't initiate drag.
- **mousemove without Shift**: existing drag-to-translate, unchanged.
- **Shift release**: angle persists, no action.

### `rasterizeObstacle()` rotation

Reads `this.obstacleAngle`. Before the per-cell shape test, compute inverse-rotated local coordinates:

```js
const cosA = Math.cos(-angle);
const sinA = Math.sin(-angle);
// Inside cell loop, after dx/dy:
const ldx = dx * cosA - dy * sinA;
const ldy = dx * sinA + dy * cosA;
```

All shape inside-tests use `(ldx, ldy)` instead of `(dx, dy)`. Trig computed once outside the loop.

## Rendering (`renderer.js`)

### `drawObstacle()` rotation

Use canvas transform stack:
```js
ctx.save();
ctx.translate(canvasCX, canvasCY);
ctx.rotate(-angle);  // negative because canvas Y is flipped
ctx.translate(-canvasCX, -canvasCY);
// ... existing shape drawing code unchanged ...
ctx.restore();
```

### Orientation line

When Shift is held, draw a thin dashed line from obstacle center outward along the angle direction. Length = `1.5 * obstacleRadius`. Color: semi-transparent white. Only drawn while `_shiftHeld` is true (exposed from interaction to renderer).

## Preset & Reset Behavior

- `loadPreset()` in `presets.js`: set `interaction.obstacleAngle = 0`
- Shape switching: angle persists
- Grid resize: angle persists

## Files Changed

| File | Change |
|------|--------|
| `interaction.js` | Add `obstacleAngle`, `_shiftHeld`, Shift tracking, rotation in pointer handlers, inverse-rotate in `rasterizeObstacle` |
| `renderer.js` | Canvas rotation transform in `drawObstacle`, orientation line drawing |
| `presets.js` | Reset `obstacleAngle = 0` in `loadPreset` |

No shader changes. No uniform buffer changes. No new files.
