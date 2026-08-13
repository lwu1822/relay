"""Provider ABC — defines the interface every LLM provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Raised when a provider returns an error or is unreachable."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider times out."""


class Provider(ABC):
    """Abstract base class for LLM providers.

    Every provider speaks the OpenAI-compatible chat completions API,
    so implementations are thin: build the HTTP request, call the
    provider, parse the response, raise typed exceptions on failure.
    """

    def __init__(self, name: str, base_url: str, api_key: str | None = None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Send a chat completion request and return the parsed response.

        Raises:
            ProviderError: on non-2xx responses or connection errors.
            ProviderTimeoutError: on request timeout.
        """
        ...

    async def close(self) -> None:
        """Clean up resources (httpx clients, etc.). Override if needed."""
        pass
