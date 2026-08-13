"""Ollama provider — local LLM inference via OpenAI-compatible API."""

from __future__ import annotations

import httpx

from app.models import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import Provider, ProviderError, ProviderTimeoutError


class OllamaProvider(Provider):
    """Provider for local Ollama instances.

    Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
    No authentication required.
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(name="ollama", base_url=base_url, api_key=None)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self._client

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        client = await self._get_client()
        try:
            resp = await client.post(
                "/chat/completions",
                json=request.model_dump(exclude_none=True),
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Ollama timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(f"Cannot connect to Ollama at {self.base_url}: {e}") from e

        return ChatCompletionResponse(**resp.json())

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
