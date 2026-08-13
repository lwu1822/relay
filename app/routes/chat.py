"""POST /v1/chat/completions — OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.logging_config import get_logger
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
    log = logger.bind(
        request_id=request_id,
        model=chat_request.model,
    )

    try:
        response, provider_name = await router_instance.route(chat_request)
        latency_ms = int((time.monotonic() - start_time) * 1000)
        log.info(
            "chat_completion",
            provider=provider_name,
            latency_ms=latency_ms,
            status="success",
        )
        return response.model_dump()

    except AllProvidersFailedError as e:
        latency_ms = int((time.monotonic() - start_time) * 1000)
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
