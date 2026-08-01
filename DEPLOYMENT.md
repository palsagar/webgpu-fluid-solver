# Deployment

FlowLab ships as a single Docker container (FastAPI + static files). `PORT` selects
the listen port (default 8000); `/api/health` backs the Docker HEALTHCHECK.

## Analytics (Umami) — wiring runbook

The Umami tracking tag is **injected at request time by the FastAPI app itself**,
driven by two runtime env vars — nothing about Umami lives in the repo or the
image. `UmamiInjectionMiddleware` (see `server.py`) inserts:

```html
<script defer src="${UMAMI_DOMAIN}/script.js" data-website-id="${UMAMI_ID}"></script>
```

into the HTML page immediately after `</title>`, with the values read from the
container env at process start.

### One-time setup

1. **In your Umami dashboard** (https://analytics.amalavidahomestay.com): add a
   website for `flow.gpuphysics.dev` (Settings → Websites → Add). Copy its
   **Website ID** (a UUID).
2. **In Coolify** → this app → **Environment Variables**, add two variables
   (plain **runtime** vars — do *not* tick "Build Variable"):

   | Var | Value | Notes |
   |---|---|---|
   | `UMAMI_DOMAIN` | `https://analytics.amalavidahomestay.com` | Full URL of the Umami instance. Must include `https://`; a trailing slash is stripped. |
   | `UMAMI_ID` | `xxxxxxxx-xxxx-...` | The Website ID (UUID) from step 1. |

3. **Redeploy** (or just Restart) the app in Coolify. Env-var changes take effect
   on process start — **no rebuild needed**.

### Verify

```bash
curl -s https://flow.gpuphysics.dev/ | grep -o '<script[^>]*script.js[^>]*>'
# → <script defer src="https://analytics.amalavidahomestay.com/script.js" data-website-id="...">
```

Then load the site and confirm a hit appears in Umami's Realtime view. To see a
custom event, clear the `flowlab.tour.v1` localStorage key, reload, and click
"Take the Tour" — a `tour-started` event should arrive.

### Events

Pageviews are automatic. Custom events: `tour-started`, `tour-completed`,
`tour-skipped`, `preset-changed` (`{ preset }`), `resolution-changed` (`{ tier }`),
`obstacle-inserted`.

### Notes

- **Change or disable analytics:** edit/remove the env vars + Restart. No rebuild.
- With either var unset the middleware is a pass-through — local dev and the
  Playwright suite are never tracked.
- Malformed values (bad scheme, whitespace/quotes, non-UUID ID) are treated as
  unconfigured — the app logs a warning at startup and serves untracked pages.
- **Umami is cookieless** — no consent banner required.
- If you enable *domain enforcement* on the Umami website, make sure
  `flow.gpuphysics.dev` is in its allowed-domains list, or events are dropped.
- FlowLab sends `Referrer-Policy: no-referrer`, so Umami's Referrers panel stays
  empty — deliberate, consistent with the site's privacy posture.
- If a `script-src` CSP is ever added, allowlist `analytics.amalavidahomestay.com`.
- The audience is technical (ad-blocker-heavy), so expect some undercount.
- Optional hardening: add `data-domains="flow.gpuphysics.dev"` to the injected tag
  in `server.py` if the env vars are ever set on a non-prod deployment.
