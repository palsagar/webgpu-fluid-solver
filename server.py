import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


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


UMAMI_DOMAIN = os.environ.get("UMAMI_DOMAIN", "").rstrip("/")
UMAMI_ID = os.environ.get("UMAMI_ID", "")


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
        if "text/html" not in response.headers.get("content-type", ""):
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        tag = (
            f'<script defer src="{UMAMI_DOMAIN}/script.js" '
            f'data-website-id="{UMAMI_ID}"></script>'
        ).encode()
        body = body.replace(b"</title>", b"</title>" + tag, 1)
        headers = dict(response.headers)
        headers["content-length"] = str(len(body))
        return Response(content=body, status_code=response.status_code, headers=headers)


app = FastAPI()
app.add_middleware(NoCacheMiddleware)
app.add_middleware(UmamiInjectionMiddleware)

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
