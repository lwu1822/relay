"""relay — FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_config
from app.logging_config import configure_logging, get_logger
from app.router import Router
from app.routes.chat import router as chat_router

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("config.yaml")
    app.state.router = Router(
        provider_configs=config.providers,
        routing=config.routing,
    )
    logger.info("startup", providers=[p.name for p in config.providers])
    yield
    await app.state.router.close()
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
async def ready(request: dict = {}):
    return {"status": "ready"}
