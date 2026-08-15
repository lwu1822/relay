"""relay — FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.cache import SemanticCache
from app.config import load_config
from app.logging_config import configure_logging, get_logger
from app.router import Router
from app.routes.chat import router as chat_router

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Config path is overridable via env var so Docker/Fly can point at a
    # deploy-specific config baked into the image. Defaults to config.yaml
    # for local dev.
    config_path = os.environ.get("RELAY_CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    app.state.router = Router(
        provider_configs=config.providers,
        routing=config.routing,
    )
    app.state.provider_configs = config.providers

    if config.cache.enabled:
        app.state.cache = SemanticCache(config=config.cache)
        logger.info("cache_enabled")
    else:
        app.state.cache = None
        logger.info("cache_disabled")

    logger.info("startup", providers=[p.name for p in config.providers])
    yield
    await app.state.router.close()
    if app.state.cache is not None:
        await app.state.cache.close()
    logger.info("shutdown")


app = FastAPI(
    title="relay",
    description="Self-hostable LLM inference gateway with an OpenAI-compatible API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """Readiness probe — true once lifespan has wired router + cache.

    Returns 503 until startup completes so orchestrators (Fly, Docker)
    don't route traffic to an instance that hasn't loaded the embedding
    model yet.
    """
    router_ready = hasattr(request.app.state, "router") and request.app.state.router is not None
    cache_resolved = hasattr(request.app.state, "cache")
    if not (router_ready and cache_resolved):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus exposition endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
