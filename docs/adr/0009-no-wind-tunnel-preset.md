---
status: accepted
---

# No Wind Tunnel preset

The Wind Tunnel preset (uniform flow past a large bluff body, `r = 0.15`, `U = 2.0`, `dt = 1/60`, 40 pressure iterations) was removed from the presets. Its interesting regimes — clean wake separation, drag, anything past the shedding transition at a large blockage — live at Reynolds numbers this solver cannot honestly deliver on consumer hardware: the scheme's measured ceiling is a flat Re ≈ 237 (ADR-0008), and the preset's own operating point sat off the one measured slice on all three axes (`dt`, iterations, `U`), so its Re badge could only ever report `unmeasured`. What the preset showed at high indicated Re was shaped by numerical, not physical, viscosity — exactly what the honest-numerics work exists to stop presenting as physics. Kármán Vortex already covers flow past a bluff body at a measured onset.

Only the preset is gone. The `windTunnel` *boundary type* (open right edge) stays — the Kármán preset uses it — and the preset's off-slice parameter set survives as an adversarial case in `tests/diagnostics.spec.js`, where it pins the refusal to quote an unmeasured ceiling (the Re 771 vs 297 impossible pair).

Revisit only if the honest ceiling rises by an order of magnitude — e.g. a multigrid pressure solve (deferred in ROADMAP) or an implicit viscous solve (considered and set aside in ADR-0008).
