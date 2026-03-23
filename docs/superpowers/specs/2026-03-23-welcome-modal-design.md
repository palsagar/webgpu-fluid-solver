# Welcome Modal Design Spec

## Overview

Add a welcome/splash modal to FlowLab that introduces the simulation, shows WebGPU detection status with GPU adapter info, and requires the user to click "Start Simulation" to begin. The simulation runs in the background behind a semi-transparent overlay.

## Design Decisions

- **Approach A (Pure HTML/CSS):** Modal markup lives in `index.html`, styled in `style.css`, with minimal JS in `main.js` to populate dynamic fields and handle dismissal.
- **Content framing:** Hybrid — experience hook first, then physics substance.
- **Sim runs behind the modal** — GPU initializes and the sim starts (unpaused), visible through the dimmed overlay. This gives an immediate visual taste of the fluid dynamics.

## Architecture

### Files Modified

1. **`static/index.html`** — Add modal markup (~40 lines) inside `#app`, before the existing content.
2. **`static/css/style.css`** — Add modal styles (~80 lines): overlay, container, badge, status card, button, close button, fade-out transition.
3. **`static/js/main.js`** — After WebGPU adapter/device acquisition: populate adapter name via textContent, attach click handler to dismiss modal with CSS transition.

No new files. No new modules.

### Modal HTML Structure

```
#welcome-overlay               — fixed overlay, dark semi-transparent bg
  #welcome-modal               — glassmorphism container (max-width 680px)
    button.welcome-close       — X close button (top-right)
    .welcome-title-row         — flex row: h1 "FlowLab" + .webgpu-badge "WebGPU"
    p.welcome-desc             — description paragraph with bold/italic rich text
    #gpu-status-card           — green-bordered status card
      .gpu-status-badge        — green pill with checkmark + "WebGPU is available"
      #gpu-adapter-name        — span inside a static template, populated via textContent
      .gpu-detail-card         — nested card explaining GPU acceleration
    button#start-sim-btn       — green "Start Simulation" button
```

### Styling

Matches existing GitHub dark theme (`#0d1117` bg, `#161b22` surfaces, `#30363d` borders):

| Element | Background | Border | Text |
|---------|-----------|--------|------|
| Overlay | `rgba(0,0,0,0.5)` | — | — |
| Container | `rgba(13,17,23,0.85)` + `backdrop-filter: blur(16px)` | `1px solid rgba(48,54,61,0.8)` | — |
| WebGPU badge | `rgba(31,111,235,0.2)` | `1px solid rgba(31,111,235,0.3)` | `#58a6ff` |
| Status card | `rgba(35,134,54,0.08)` | `1px solid rgba(35,134,54,0.5)` | — |
| Status badge | `rgba(35,134,54,0.5)` | `1px solid rgba(126,231,135,0.25)` | `#7ee787` |
| Detail card | `rgba(255,255,255,0.04)` | `1px solid rgba(48,54,61,0.6)` | `#8b949e` |
| Start button | `#238636` | `1px solid rgba(35,134,54,0.6)` | `#ffffff` |

Container: `border-radius: 12px`, `padding: 48px 48px 40px`, `max-width: 680px`, `max-height: 90vh` with `overflow-y: auto`.

### Dismissal Behavior

- Clicking "Start Simulation" or the X close button triggers dismissal.
- Dismissal: add a CSS class (`welcome-hidden`) that transitions `opacity` to 0 over ~300ms, then set `display: none` after the transition ends (via `transitionend` event).
- No `localStorage` persistence — the modal shows every page load. This is a simulation tool, not a SaaS app; users expect a "launch" action.

### Dynamic Content (JS)

In `main.js`, after `adapter = await navigator.gpu.requestAdapter()`:

```js
const adapterInfo = adapter.info;  // GPUAdapterInfo
const adapterName = adapterInfo.device || adapterInfo.description || 'Unknown GPU';
document.getElementById('gpu-adapter-name').textContent = adapterName;
```

Dismiss handler:

```js
const overlay = document.getElementById('welcome-overlay');
const dismiss = () => {
  overlay.classList.add('welcome-hidden');
  overlay.addEventListener('transitionend', () => {
    overlay.style.display = 'none';
  }, { once: true });
};
document.getElementById('start-sim-btn').addEventListener('click', dismiss);
document.querySelector('.welcome-close').addEventListener('click', dismiss);
```

### Description Text

> Watch real fluid dynamics unfold in your browser. **FlowLab** solves the *2D incompressible Navier-Stokes equations* entirely on your GPU — **pressure, advection, and boundary conditions** all run as WebGPU compute shaders for *real-time interactive* performance. Drag obstacles through flowing fluid and watch **vortex streets**, *recirculation zones*, and **flow separation** emerge naturally.

### GPU Detail Card Text

> The **pressure solver**, **velocity advection**, and **boundary conditions** all execute as compute shader dispatches on your graphics card. This delivers *real-time frame rates* even at high grid resolutions, with the CPU free for rendering overlays and particle tracing.

## Edge Cases

- **No WebGPU:** The existing `#no-webgpu` fallback div handles this. The welcome modal is never shown if `navigator.gpu` is falsy — the code bails before reaching modal population.
- **Adapter info unavailable:** Fall back to "Unknown GPU" if `adapter.info.device` and `adapter.info.description` are both empty.
- **Mobile/small screens:** Container uses `width: 90vw` and `max-height: 90vh` with `overflow-y: auto`. Padding reduces on small screens via a media query.
- **Keyboard accessibility:** Start button is a real `<button>`, focusable and Enter-activatable. Close button has `aria-label="Close"`.

## What This Does NOT Include

- No "Switch to CPU" button (we have no CPU fallback).
- No performance warning section (not applicable).
- No localStorage "don't show again" persistence.
- No keyboard shortcut tips or controls help.
