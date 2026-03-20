from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Disable browser caching for JS/CSS/HTML during development."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html", ".wgsl")) or path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


app = FastAPI()
app.add_middleware(NoCacheMiddleware)

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
