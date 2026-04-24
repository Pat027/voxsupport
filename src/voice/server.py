"""
FastAPI server — mounts the web + Twilio transports, exposes Prometheus
metrics, and serves the browser demo page.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.observability import metrics as _metrics  # noqa: F401 — registers Prometheus collectors at import time
from src.voice.transports import twilio as twilio_transport
from src.voice.transports import web as web_transport

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("voxsupport starting up")
    yield
    logger.info("voxsupport shutting down")


app = FastAPI(title="voxsupport", version="0.1.0", lifespan=lifespan)

# Transports
app.include_router(web_transport.router)
app.include_router(twilio_transport.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("demo/web_client/index.html")


app.mount("/static", StaticFiles(directory="demo/web_client"), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.voice.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
