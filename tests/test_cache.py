"""Unit tests for SemanticCache.

These tests use a deterministic fake encoder so they do not require the
sentence-transformers model download and run fast. The fake encoder maps
each distinct word in the input to a unique one-hot-ish dimension, so
identical inputs produce identical embeddings (similarity 1.0) and inputs
with no shared words are orthogonal (similarity 0.0).

Run with: pytest tests/test_cache.py -v
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.cache import SemanticCache
from app.config import CacheConfig
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)


class FakeEncoder:
    """Deterministic encoder: each unique word -> a unique basis vector.

    The embedding of a sentence is the sum of its word vectors, which after
    L2 normalization gives a usable similarity signal for tests:
      - identical sentences -> cosine similarity 1.0
      - sentences sharing some words -> 0 < similarity < 1
      - sentences with no shared words -> 0.0
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._vocab: dict[str, int] = {}

    def _word_index(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab) % self.dim
        return self._vocab[word]

    def encode(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            vec[self._word_index(word)] += 1.0
        return vec


def _make_request(content: str, model: str = "llama3.2:1b") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=10,
    )


def _make_response(content: str = "hello", model: str = "llama3.2:1b") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="test-id",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop")],
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )


@pytest.fixture
def cache():
    cfg = CacheConfig(
        enabled=True,
        similarity_threshold=0.95,
        ttl_seconds=3600,
        max_size=1000,
    )
    return SemanticCache(config=cfg, encoder=FakeEncoder())


@pytest.fixture
def small_cache():
    """Cache with max_size=2 for eviction tests."""
    cfg = CacheConfig(
        enabled=True,
        similarity_threshold=0.95,
        ttl_seconds=3600,
        max_size=2,
    )
    return SemanticCache(config=cfg, encoder=FakeEncoder())


@pytest.mark.asyncio
async def test_insert_then_lookup_identical_returns_hit(cache):
    """An identical last-user-message should produce a cache hit."""
    req = _make_request("what is the capital of france")
    resp = _make_response("paris")
    await cache.insert(req, resp, "ollama")

    hit = await cache.lookup(req)
    assert hit is not None
    cached_resp, provider = hit
    assert cached_resp.choices[0].message.content == "paris"
    assert provider == "ollama"


@pytest.mark.asyncio
async def test_lookup_empty_cache_returns_miss(cache):
    """A lookup against an empty cache returns None."""
    hit = await cache.lookup(_make_request("anything"))
    assert hit is None


@pytest.mark.asyncio
async def test_lookup_no_user_message_returns_miss(cache):
    """A request with no user message returns None (nothing to embed)."""
    req = ChatCompletionRequest(
        model="llama3.2:1b",
        messages=[ChatMessage(role="system", content="be helpful")],
    )
    resp = _make_response("ok")
    await cache.insert(req, resp, "ollama")
    hit = await cache.lookup(req)
    assert hit is None


@pytest.mark.asyncio
async def test_lookup_dissimilar_returns_miss(cache):
    """A request with no shared words to any cached entry returns None."""
    await cache.insert(
        _make_request("alpha beta gamma"),
        _make_response("a"),
        "ollama",
    )
    hit = await cache.lookup(_make_request("delta epsilon zeta"))
    assert hit is None


@pytest.mark.asyncio
async def test_lookup_scoped_by_model(cache):
    """A hit requires the same model on the cached entry and the lookup."""
    req = _make_request("shared text here")
    await cache.insert(req, _make_response("a"), "ollama")

    # Same text, different model -> miss (cache is scoped by model).
    other_model_req = _make_request("shared text here", model="other-model")
    hit = await cache.lookup(other_model_req)
    assert hit is None

    # Same text, same model -> hit.
    hit = await cache.lookup(req)
    assert hit is not None


@pytest.mark.asyncio
async def test_ttl_expiry_evicts_entries():
    """Entries older than ttl_seconds are evicted on the next lookup."""
    cfg = CacheConfig(
        enabled=True,
        similarity_threshold=0.95,
        ttl_seconds=1,  # 1 second
        max_size=1000,
    )
    cache = SemanticCache(config=cfg, encoder=FakeEncoder())

    req = _make_request("hello world")
    await cache.insert(req, _make_response("hi"), "ollama")

    # Immediate lookup should hit.
    hit = await cache.lookup(req)
    assert hit is not None

    # Wait for TTL to expire, then lookup should miss.
    await asyncio.sleep(1.1)
    hit = await cache.lookup(req)
    assert hit is None


@pytest.mark.asyncio
async def test_max_size_evicts_oldest(small_cache):
    """When max_size is exceeded, the oldest entry is evicted (FIFO)."""
    # Insert 3 distinct entries with max_size=2.
    await small_cache.insert(
        _make_request("first unique prompt"), _make_response("1"), "ollama"
    )
    await small_cache.insert(
        _make_request("second unique prompt"), _make_response("2"), "ollama"
    )
    await small_cache.insert(
        _make_request("third unique prompt"), _make_response("3"), "ollama"
    )

    # The first entry should have been evicted.
    hit = await small_cache.lookup(_make_request("first unique prompt"))
    assert hit is None

    # The second and third should still be present.
    hit = await small_cache.lookup(_make_request("second unique prompt"))
    assert hit is not None
    hit = await small_cache.lookup(_make_request("third unique prompt"))
    assert hit is not None


@pytest.mark.asyncio
async def test_close_is_noop(cache):
    """close() releases no resources and returns None."""
    result = await cache.close()
    assert result is None
