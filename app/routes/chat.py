"""POST /v1/chat/completions — OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.logging_config import get_logger
from app.metrics import (
    CACHE_HITS_TOTAL,
    COST_SAVED,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    estimate_cost,
)
from app.models import ChatCompletionRequest
from app.router import AllProvidersFailedError

router = APIRouter()
logger = get_logger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    chat_request = ChatCompletionRequest(**body)

    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    router_instance = request.app.state.router
    cache = getattr(request.app.state, "cache", None)
    provider_configs = getattr(request.app.state, "provider_configs", [])

    log = logger.bind(
        request_id=request_id,
        model=chat_request.model,
    )

    # --- Cache lookup (before routing) ---
    if cache is not None:
        cached = await cache.lookup(chat_request)
        if cached is not None:
            cached_response, cached_provider = cached
            latency_ms = int((time.monotonic() - start_time) * 1000)
            latency_s = latency_ms / 1000.0
            CACHE_HITS_TOTAL.inc()
            REQUESTS_TOTAL.labels(provider="cache", status="success").inc()
            REQUEST_DURATION.labels(provider="cache").observe(latency_s)
            cost_saved = estimate_cost(
                chat_request.model, cached_response.usage, provider_configs
            )
            COST_SAVED.inc(cost_saved)
            log.info(
                "chat_completion",
                provider=cached_provider,
                latency_ms=latency_ms,
                status="cache_hit",
                cost_saved=round(cost_saved, 6),
            )
            return cached_response.model_dump()

    # --- Route to provider ---
    try:
        response, provider_name = await router_instance.route(chat_request)
        latency_ms = int((time.monotonic() - start_time) * 1000)
        latency_s = latency_ms / 1000.0
        REQUESTS_TOTAL.labels(provider=provider_name, status="success").inc()
        REQUEST_DURATION.labels(provider=provider_name).observe(latency_s)
        log.info(
            "chat_completion",
            provider=provider_name,
            latency_ms=latency_ms,
            status="success",
        )

        # Insert into cache after a successful provider response.
        if cache is not None:
            await cache.insert(chat_request, response, provider_name)

        return response.model_dump()

    except AllProvidersFailedError as e:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        latency_s = latency_ms / 1000.0
        REQUESTS_TOTAL.labels(provider="none", status="all_providers_failed").inc()
        REQUEST_DURATION.labels(provider="none").observe(latency_s)
        log.error(
            "chat_completion",
            latency_ms=latency_ms,
            status="all_providers_failed",
            error=str(e),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "All providers are currently unavailable.",
                    "type": "all_providers_failed",
                    "request_id": request_id,
                }
            },
        )
