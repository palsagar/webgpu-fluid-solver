# Explicit bind group layouts instead of `layout: 'auto'`

Compute pipelines use hand-written `GPUBindGroupLayout`s, never `layout: 'auto'`. Auto-layout only includes bindings that are *statically used* by the entry point — e.g. `boundary.wgsl`'s `extrapolate_horizontal` never references `v`, so auto-layout drops that binding and shared bind group creation fails. Explicit layouts let multiple entry points share one bind group with all declared bindings.

Do not "simplify" this back to `layout: 'auto'` — it will break bind group creation for any shader that declares a binding it doesn't reference.
