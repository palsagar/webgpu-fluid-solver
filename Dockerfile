FROM python:3.12-slim

WORKDIR /app

# Install from the lockfile — an unpinned `pip install fastapi uvicorn` resolves
# whatever is newest on PyPI at build time, so images are not reproducible.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev --no-install-project

COPY server.py .
COPY static/ static/

RUN useradd -u 1000 app && chown -R app:app /app
USER app

ENV PORT=8000
# Umami analytics is injected at request time by UmamiInjectionMiddleware,
# driven by these runtime env vars — set them in Coolify. Empty default = inert.
ENV UMAMI_DOMAIN="" \
    UMAMI_ID=""
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

CMD .venv/bin/uvicorn server:app --host 0.0.0.0 --port ${PORT}
