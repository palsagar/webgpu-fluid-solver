---
status: accepted
---

# Moving-wall viscous boundary condition

The viscous stencil ghosts a mask-buried face against its own stored velocity — `w + (w − center)`, placing the wall's velocity on the wall line half a cell away — instead of against zero, and a drag that ends re-rasterizes once with zero velocity so the stored wall value cannot outlive the motion.

The defect this closes is measured and disclosed (ROADMAP Known gaps): `diffuse.wgsl` classified a velocity face flanked by two solid cells as buried and ghosted it to `-center`, pinning a *dragged* obstacle's wall at **zero** regardless of the drag velocity `vx`. With `ν > 0` during a drag, diffusion pulled near-wall fluid toward zero instead of toward `vx`, partially cancelling the shear the moving wall imparts — bounded (coeff ≤ 1/4), drag-only. The MacCormack advection path already preserves the moving-wall BC (pinned by PR A's tests); the viscous pass was the one that threw it away. Unblocked by ADR-0010: the field reset that masked the defect is gone, and stored solid-cell velocities are now load-bearing and correct in every rotation slot.

Face classes are unchanged: FLUID faces diffuse; WALL faces copy through and are read at face value (already correct — the rasterizer writes the wall velocity there); BURIED faces ghost. BURIED splits in two, and the split is load-bearing:

- **Buried by index** — the `i = 0` / `j = 0` ring advection never writes — keeps the zero ghost, and its stored value is never loaded. Those entries are three steps stale; reading them is the MacCormack stale-ring leak, measured **9.894e-3** inviscid. The index classification stays structural.
- **Buried by mask** — both flanking cells solid per the Solid Mask —
  ghosts to `w + (w − center)` with `w` the face's **own stored value**.
  Since ADR-0010 the rasterizer writes the drag velocity into every inside
  cell's own u-face and v-face, so a mask-buried face's stored value IS the
  wall velocity. **Exception:** u-faces on the domain top row
  (`j = numY-1`) join the index-buried classification because their stored
  `u` is `boundary.wgsl`'s zero-gradient (Neumann) free-stream
  extrapolation, not a wall velocity; an unguarded moving-wall ghost there
  measurably destroyed the no-slip boundary layer
  (`viscous.dom / inviscid.dom = 0.536` vs the `< 0.5` gate) during
  implementation. No new buffer: `uIn`/`vIn` are already bound, and the pass
  stays at 6 storage buffers.

The formulation is `w + (w − center)` because it reads as the ghost
construction: linear extrapolation placing `w` on the wall line half a cell
away. At `w = 0` it is `-center` EXACTLY for every nonzero center; at a
zero center the result can differ only in the sign of zero, which no
downstream observable can detect. Stationary obstacles and `ν = 0` runs are
therefore bit-identical to the pre-change shader in every observable output,
and every constant measured stationary — the `nu_num` table, onset
Re_c = 52.2, the St table, operator fidelity 0.99751 — is unaffected. The
change is live only in the defect's exact trigger condition: a moving
obstacle with `ν > 0`.

## The drag-end half of the decision

`_endDrag()` previously set `dragging = false` and nothing else: the last mousemove's velocity stayed in the solid cells until the next rasterize. With advection alone that was bounded and shipped; with the ghost reading it, a drag ending on a fast flick would leave the viscous pass pumping momentum toward a phantom moving wall on every subsequent frame — up to 32 substeps each — indefinitely. `_endDrag()` now re-rasterizes at the current position with zero velocity: one dispatch round, the same cost as a single mousemove, matching the zero-velocity pattern `_startDrag` and `_rotate` already use. Mid-drag *holds* (mouse down, not moving) keep the last drag velocity — that is shipped advection behavior pinned by PR A's tests, and redefining it is out of scope.

## Acceptance gate — deterministic delta, three legs

The measurement is the user-facing path itself: a scripted constant-velocity drag replayed deterministically (same rasterize calls, same dt, quieted inflow) on master and on this branch, with near-wall velocity read back per step. A synthetic planar Couette channel was considered and not taken: the defect lives on the drag path, and the gate exercises rasterizer → stored velocity → ghost end to end. Because advection already drags near-wall fluid toward `vx` on master, the before/after contrast is a fractional change, not 0 vs U/H — hence a delta gate, not an absolute threshold (an absolute fraction has no derivation and would be chosen to pass, the failure mode ADR-0008 documented). The near-wall set is exactly the faces the change can touch: fluid faces whose stencil reads at least one buried neighbour.

1. **Direction and magnitude (measured).** After K steps the near-wall fluid velocity ends strictly closer to `vx` than the identical replay on master. The closure factor is recorded here as a measured number. Measured on the scripted-drag harness (quieted field, `vx = 1.0`, 4 substeps at coeff 0.2, K = 40 steps, boot tier): near-wall mean `u` = **-0.0055895052864798345** on master @ `9e05cf7` against **0.576810504309833** on this branch — the ring closes from -0.56% to 57.7% of `vx`, a 2.38× reduction of the deficit. Per-step near-wall extrema stayed within [0.3887721598148346, 0.8068600296974182], inside the convexity bound. The pre-fix branch run reproduced the master value bit-exactly, so the delta is attributable to the stencil change alone.
2. **No overshoot (provable).** The explicit update with coeff ≤ 1/4 is a convex combination, so near-wall `u` stays within `[0, vx]` at every step — asserted in-suite per step, not sampled at the end.
3. **Zero control (bit-exact).** A mask-buried face with `w = 0` ghosts to exactly `-center` (unit pin), and the full pre-existing suite — PR A's mask oracle, bit-identity outside drag bounding boxes, the moving-wall advection tests, the stale-ring assertions — passes unchanged.

The mechanism itself is mutation-tested: a crafted buried face with stored `w`, one isolated diffuse dispatch, and the stencil neighbour must equal `w + (w − center)` to f32-replicated precision — with the comment recording that reverting the ghost to `-center` fails the test for any `w ≠ 0`.

## Considered options

- **All buried faces read the stored value** — one rule, simpler shader; rejected because the `i = 0` / `j = 0` ring's stored values are three steps stale, and the ghost would load exactly the leak the index classification exists to block.
- **A separate wall-velocity buffer** — explicit, but spends a seventh storage buffer duplicating data already in `u`/`v`, and adds a second writer contract to keep in sync with the rasterizer.
- **Synthetic planar Couette as the acceptance gate** — exact analytic target (the discrete steady state of a row-aligned channel IS the linear profile), but it bypasses the rasterizer and the stored-velocity path that constitute the defect; the scripted-drag delta exercises the real path and was chosen instead.
- **Absolute near-wall fraction threshold** — rejected; no derivation, chosen to pass.
- **Disclose-only on drag end** — rejected; the ghost makes the stale velocity a permanent momentum pump, a new defect this change would introduce.

## Consequences and limitations

- The viscous pass stays at 6 storage buffers; no pipeline, bind-group, or budget changes.
- **Inflow-column ownership contest:** `writeInflowColumn(1, …)` re-applies every frame and overwrites the stored drag velocity of any obstacle cell carved into column 1, so the ghost reads `inVel` there. Pre-existing (advection already read those values), bounded to one column; not fixed here.
- Rotation rasterizes with zero velocity, so a *spinning* obstacle's walls are stationary to the viscous pass — unchanged, disclosed.
- **Top-row corner:** an obstacle dragged against the domain top row has its buried u-faces ghosted to `-center` (stationary) because those faces are classified with the index-buried ring.
- The ghost places `w` on the wall line half a cell from the fluid face regardless of the surface's true position within the solid cell — first-order in surface position, same as the stationary wall today.
- PR D's eraser inherits the vacated-cell restore semantics (zero velocity/pressure, Smoke = 1.0); unaffected.
