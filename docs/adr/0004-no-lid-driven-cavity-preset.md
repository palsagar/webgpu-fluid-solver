---
status: accepted
---

# No lid-driven cavity preset

Lid-driven cavity — the classic CFD benchmark — was removed from the presets. The solver's boundary `extrapolate` step copies interior velocities into wall cells every step, overwriting any forced tangential velocity at the lid. Supporting the cavity would require modifying `boundary.wgsl` to exempt (or re-force) the lid wall, so all presets are restricted to inflow-driven topologies (`windTunnel`, `backwardStep`) where forced velocity lives at column `i=1`, protected from advection by the solid left wall.

Revisit only alongside a boundary-shader redesign.
