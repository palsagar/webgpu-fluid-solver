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


app = FastAPI()
app.add_middleware(NoCacheMiddleware)

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
