# CPU-side Lagrangian particles instead of a GPU particle pass

Particles are advected on the CPU in `particles.js`, reusing the renderer's existing velocity readback (`uData`/`vData`) — no dedicated GPU compute pass or extra buffers. At ~5000 particles the CPU cost is negligible, and reusing the readback means zero additional GPU↔CPU traffic.

## Considered options

- **GPU compute particles** — scales to millions, but requires new pipelines, its own buffers, and either GPU rendering or its own readback. Rejected as unnecessary complexity at this particle count.
- **Burst emission** — replaced by continuous emitters (3 particles/frame), which produce visible steady streams where bursts did not.

If particle counts ever need to grow by ~100×, this decision (and ADR-0001) is the thing to revisit.
