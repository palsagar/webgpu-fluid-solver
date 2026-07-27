# Handoff: Execute the PR A plan (GPU obstacle rasterization) on a GPU machine

**Status:** Not started. The plan is fully spec'd, self-reviewed, and ready to
execute task-by-task. Execution was attempted on a GPU-less host on 2026-07-27
and cancelled before Task 1 — SwiftShader works but is far too slow to run the
suite (~7 s/frame at the default tier 256; ~3+ minutes per test even at tier
64). No plan code was written. The branch tip is still just the spec commit.

**Resume from: Task 1 of the plan, exactly as written.**

## Read first (in order)

1. `docs/superpowers/plans/2026-07-27-gpu-obstacle-rasterization.md` — the
   plan. Six tasks, TDD, full code included. This file (and this handoff)
   were force-added over the `.gitignore` rule for `docs/superpowers/`;
   the owner intends to untrack them after the PR lands — do not "clean up"
   the ignore rule.
2. `docs/adr/0010-gpu-side-obstacle-rasterization.md` — the decision record.
3. `docs/ROADMAP.md` PR A bullet (~line 45) — the seven build notes the plan
   implements.
4. `CONTEXT.md` — project glossary.

## How to execute

Use `superpowers:subagent-driven-development` (the plan's header names it):
one implementer subagent per task, task review after each, whole-branch
review at the end. Tasks are strictly sequential (T2 consumes T1, T3 must
land before T4, T5/T6 consume everything).

A stale SDD ledger exists at
`.superpowers/sdd/2026-07-27-gpu-obstacle-rasterization/progress.md`, marked
CANCELLED. Delete that directory before starting so the skill creates a
fresh ledger — do not resume from it.

One pre-flight note from the cancelled session's plan scan: Task 2's test
transcribes the deleted CPU inside-tests verbatim as its oracle. Review
rubrics flag verbatim logic duplication as a defect, but the plan's Global
Constraints mandate it ("The deleted CPU code survives as the test oracle";
"Port the CPU geometry exactly"). The plan text governs — decided already,
don't re-litigate.

## Environment (GPU machine)

- `npm ci && npx playwright install chromium` (plus
  `npx playwright install-deps chromium` on a fresh Linux box).
- Tests run HEADED Chromium with `--enable-unsafe-webgpu`. Headless machine:
  `xvfb-run -a npx playwright test ...` (same for `npm test`).
- Focused runs: `npx playwright test tests/solver.spec.js -g "<substring>"`.
  The Playwright config auto-starts the server on port 8321.
- Full suite before finishing (plan Task 6): `npm test`.

## House rules that bind the work

- Test style: `boot(page)` → `page.evaluate` against `window.__flowlab`,
  inline `readBuf` helpers, bit-exact assertions with comments recording
  what mutation each assertion catches. Match it.
- `interaction.rasterizeObstacle(centerX, centerY, vx, vy)` keeps its exact
  signature — `tests/diagnostics.spec.js:1337` and `ui.js`'s shape picker
  call it.
- No measured-number claims anywhere (branch rule: every number is
  measured). This PR changes no physics.

## Decisions already made (don't re-litigate)

Full GPU rasterizer (not bounded CPU upload, not readback); newly-fluid
cells get zero velocity/pressure and smoke = 1.0; boundary mask is a
solver-owned GPU buffer uploaded per preset load; 3 dispatches per
rasterize (one per rotation slot, forced by the 8-storage-buffer limit);
`paintMode` deleted; `_setInflowVelocity` fixed as a second instance of the
field-reset defect (bounded column-1 write).

## After PR A merges

The roadmap's next items are PR B (moving-wall viscous BC, depends on PR A)
and PRs C/D (Blow/Draw modes) — deliberately NOT spec'd yet; grill them in
a fresh session if asked.
