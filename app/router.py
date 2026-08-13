"""Router — selects providers and handles retry/failover."""

from __future__ import annotations

import asyncio
import itertools
from typing import Protocol

from app.config import ProviderConfig, RoutingConfig
from app.logging_config import get_logger
from app.models import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import Provider, ProviderError, ProviderTimeoutError
from app.providers.groq import GroqProvider
from app.providers.ollama import OllamaProvider

logger = get_logger(__name__)


class AllProvidersFailedError(Exception):
    """Raised when every provider has been tried and all failed."""


def build_provider(cfg: ProviderConfig) -> Provider:
    """Instantiate a Provider from a ProviderConfig."""
    api_key = cfg.get_api_key()
    if cfg.name == "ollama":
        return OllamaProvider(base_url=cfg.base_url)
    if cfg.name == "groq":
        return GroqProvider(base_url=cfg.base_url, api_key=api_key)
    raise ValueError(f"Unknown provider: {cfg.name}")


class Router:
    """Routes requests across providers with retry and failover.

    Strategies:
        round_robin — cycle through providers on each request.
        cost_based  — try cheapest provider first (by static cost table).
    """

    def __init__(self, provider_configs: list[ProviderConfig], routing: RoutingConfig):
        self.provider_configs = provider_configs
        self.routing = routing
        self._providers: dict[str, Provider] = {}
        self._rr_counter = itertools.count()

        for cfg in provider_configs:
            self._providers[cfg.name] = build_provider(cfg)

    def _get_ordered_providers(self) -> list[Provider]:
        """Return providers in the order dictated by the routing strategy."""
        if self.routing.strategy == "cost_based":
            sorted_cfgs = sorted(
                self.provider_configs,
                key=lambda c: c.get_model().cost_per_1k_input_tokens,
            )
            return [self._providers[c.name] for c in sorted_cfgs]
        else:
            # round_robin: rotate starting index
            names = [c.name for c in self.provider_configs]
            start = next(self._rr_counter) % len(names)
            ordered = names[start:] + names[:start]
            return [self._providers[n] for n in ordered]

    async def route(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionResponse, str]:
        """Route a request, retrying and failing over across providers.

        Returns:
            (response, provider_name) on success.

        Raises:
            AllProvidersFailedError: if every provider fails.
        """
        providers = self._get_ordered_providers()
        last_error: Exception | None = None

        for provider in providers:
            for attempt in range(1, self.routing.max_retries + 1):
                try:
                    logger.info(
                        "routing_request",
                        provider=provider.name,
                        model=request.model,
                        attempt=attempt,
                    )
                    response = await provider.chat_completion(request)
                    logger.info(
                        "request_succeeded",
                        provider=provider.name,
                        model=request.model,
                        attempt=attempt,
                    )
                    return response, provider.name
                except (ProviderError, ProviderTimeoutError) as e:
                    last_error = e
                    logger.warning(
                        "request_failed",
                        provider=provider.name,
                        attempt=attempt,
                        error=str(e),
                    )
                    if attempt < self.routing.max_retries:
                        delay = min(
                            self.routing.retry_base_delay * (2 ** (attempt - 1)),
                            self.routing.retry_max_delay,
                        )
                        await asyncio.sleep(delay)

            logger.warning(
                "provider_exhausted",
                provider=provider.name,
                error=str(last_error) if last_error else "unknown",
            )

        raise AllProvidersFailedError(
            f"All providers failed. Last error: {last_error}"
        )

    async def close(self) -> None:
        """Close all provider HTTP clients."""
        for provider in self._providers.values():
            await provider.close()
