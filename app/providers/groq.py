"""Groq provider — cloud LLM inference via OpenAI-compatible API."""

from __future__ import annotations

import httpx

from app.models import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import Provider, ProviderError, ProviderTimeoutError


class GroqProvider(Provider):
    """Provider for Groq cloud API.

    Groq exposes an OpenAI-compatible endpoint at /v1/chat/completions.
    Requires GROQ_API_KEY environment variable.
    """

    def __init__(self, base_url: str = "https://api.groq.com/openai/v1", api_key: str | None = None):
        super().__init__(name="groq", base_url=base_url, api_key=api_key)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
            )
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
            raise ProviderTimeoutError(f"Groq timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Groq returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(f"Cannot connect to Groq at {self.base_url}: {e}") from e

        return ChatCompletionResponse(**resp.json())

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
