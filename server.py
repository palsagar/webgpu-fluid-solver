from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok"})

app.mount("/", StaticFiles(directory="static", html=True), name="static")
