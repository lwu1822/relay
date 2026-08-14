"""Prometheus metrics for the relay gateway.

Exposes:
    gateway_requests_total{provider,status}            — Counter
    gateway_cache_hits_total                            — Counter
    gateway_request_duration_seconds{provider}          — Histogram
    gateway_cost_saved_dollars                          — Counter

The /metrics endpoint is mounted in app/main.py and returns the default
prometheus_client exposition format.
"""

from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Histogram

from app.config import ProviderConfig
from app.models import Usage

REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total gateway requests by provider and status.",
    ["provider", "status"],
)

CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total",
    "Total semantic cache hits.",
)

REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "Request latency in seconds, by provider.",
    ["provider"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

COST_SAVED = Counter(
    "gateway_cost_saved_dollars",
    "Estimated cost saved by cache hits, in dollars (approximate).",
)


def estimate_cost(
    model_name: str, usage: Usage, provider_configs: list[ProviderConfig]
) -> float:
    """Approximate cost (USD) of a single completion.

    Scans configured providers for a ModelConfig matching model_name and uses
    the cheapest matching rate. Returns 0.0 if no match is found. Rates are
    approximate public figures (see config.example.yaml).
    """
    cheapest_in: Optional[float] = None
    cheapest_out: Optional[float] = None
    for provider in provider_configs:
        for model in provider.models:
            if model.name == model_name:
                if cheapest_in is None or model.cost_per_1k_input_tokens < cheapest_in:
                    cheapest_in = model.cost_per_1k_input_tokens
                if cheapest_out is None or model.cost_per_1k_output_tokens < cheapest_out:
                    cheapest_out = model.cost_per_1k_output_tokens

    if cheapest_in is None or cheapest_out is None:
        return 0.0

    return (
        (usage.prompt_tokens / 1000.0) * cheapest_in
        + (usage.completion_tokens / 1000.0) * cheapest_out
    )
