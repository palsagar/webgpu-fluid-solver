import logging
import os
import re

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_logger = logging.getLogger(__name__)


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Disable browser caching for JS/CSS/HTML, and set baseline security headers.

    Caching is off unconditionally: ES modules are cached aggressively enough
    that stale code silently defeats debugging, which is worth more here than
    the bandwidth saved on a single-page demo.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html", ".wgsl")) or path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # frame-ancestors only — index.html uses inline <script> blocks and calls
        # the GitHub API, so a default-src/script-src policy would need those
        # refactored out first to avoid breaking the page.
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        return response


def _umami_config() -> tuple[str, str]:
    """Read and validate the Umami env vars. Malformed values are treated as
    unconfigured (inert) — a typo in Coolify must never corrupt the page."""
    domain = os.environ.get("UMAMI_DOMAIN", "").rstrip("/")
    website_id = os.environ.get("UMAMI_ID", "")
    if not (domain and website_id):
        return "", ""
    if not re.fullmatch(r"https://[^\"'\u003c\u003e\s]+", domain) or not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        website_id,
    ):
        _logger.warning("Umami env vars are set but malformed — analytics disabled")
        return "", ""
    return domain, website_id


UMAMI_DOMAIN, UMAMI_ID = _umami_config()


class UmamiInjectionMiddleware(BaseHTTPMiddleware):
    """Inject the Umami analytics tag into HTML responses when configured.

    Same pattern as the sibling static site's nginx sub_filter: the tag never
    lives in the repo — it is spliced in at request time from the UMAMI_DOMAIN /
    UMAMI_ID env vars. Either var empty => pass-through, so local dev and CI
    are never tracked.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        if not (UMAMI_DOMAIN and UMAMI_ID):
            return response
        # HEAD passes through untouched: StaticFiles sends no body for HEAD,
        # so rebuilding would zero Content-Length (and its validators are
        # consistent with the uninjected representation, like nginx sub_filter).
        if request.method != "GET":
            return response
        if response.status_code != 200:
            return response
        if "text/html" not in response.headers.get("content-type", ""):
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        tag = (
            f'<script defer src="{UMAMI_DOMAIN}/script.js" '
            f'data-website-id="{UMAMI_ID}"></script>'
        ).encode()
        # Anchor: just before </body>, AFTER the app module tag. Deferred
        # scripts execute in document order, so a slow analytics fetch can
        # never delay main.js (the reverse ordering would stall app boot).
        body = body.replace(b"</body>", tag + b"</body>", 1)
        headers = dict(response.headers)
        # The rewritten body is a different representation than the file on
        # disk — StaticFiles' validators must not survive onto it.
        for h in ("etag", "last-modified", "accept-ranges"):
            headers.pop(h, None)
        headers["content-length"] = str(len(body))
        return Response(content=body, status_code=response.status_code, headers=headers)


app = FastAPI()
app.add_middleware(NoCacheMiddleware)
app.add_middleware(UmamiInjectionMiddleware)

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
