"""SemanticCache — in-memory cache over sentence-transformer embeddings.

A cache hit returns a previously stored response when the most recent user
message of an incoming request is semantically similar (cosine similarity
>= threshold) to a cached entry that was generated for the same requested
model. Entries expire after ttl_seconds and the cache evicts FIFO once it
exceeds max_size.

Embedding computation is synchronous/CPU-bound (sentence-transformers), so
encoding happens in a worker thread via asyncio.to_thread to avoid blocking
the event loop. Only the in-memory store scan/insert is guarded by a lock.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from app.config import CacheConfig
from app.logging_config import get_logger
from app.models import ChatCompletionRequest, ChatCompletionResponse

logger = get_logger(__name__)

# An encoder maps a string to a 1-D numpy array. Real production usage
# supplies a SentenceTransformer; tests can inject a deterministic fake.
Encoder = Callable[[str], np.ndarray]


@dataclass
class CacheEntry:
    embedding: np.ndarray  # L2-normalized so cosine similarity = dot product
    model: str
    response: ChatCompletionResponse
    provider: str
    created_at: float


class SemanticCache:
    """In-memory semantic cache keyed by embedding of the last user message."""

    def __init__(self, config: CacheConfig, encoder: Optional[Encoder] = None):
        self.config = config
        self._entries: list[CacheEntry] = []
        self._lock = asyncio.Lock()
        if encoder is not None:
            self._encoder = encoder
        else:
            # Lazy import so the module loads even if sentence-transformers
            # is not yet installed (e.g. during config-only smoke tests).
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(config.model_name)
        logger.info(
            "cache_loaded",
            model_name=config.model_name,
            threshold=config.similarity_threshold,
            ttl_seconds=config.ttl_seconds,
            max_size=config.max_size,
        )

    def _last_user_text(self, request: ChatCompletionRequest) -> Optional[str]:
        """Return the content of the most recent user message, or None."""
        for msg in reversed(request.messages):
            if msg.role == "user":
                return msg.content
        return None

    async def _encode(self, text: str) -> np.ndarray:
        """Encode text to a L2-normalized embedding in a worker thread.

        Supports both SentenceTransformer instances (which expose .encode)
        and plain callables (used by tests).
        """
        if hasattr(self._encoder, "encode"):
            vec = await asyncio.to_thread(self._encoder.encode, text)
        else:
            vec = await asyncio.to_thread(self._encoder, text)
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def _evict_expired(self, now: float) -> None:
        """Drop entries older than ttl_seconds. Caller must hold the lock."""
        ttl = self.config.ttl_seconds
        if ttl <= 0:
            return
        before = len(self._entries)
        self._entries = [e for e in self._entries if now - e.created_at < ttl]
        expired = before - len(self._entries)
        if expired:
            logger.info("cache_expired_entries", count=expired)

    def _enforce_max_size(self) -> None:
        """FIFO-evict oldest entries while over max_size. Caller holds lock."""
        max_size = self.config.max_size
        if max_size <= 0:
            return
        while len(self._entries) > max_size:
            evicted = self._entries.pop(0)
            logger.info("cache_evicted_entry", reason="max_size")

    async def lookup(
        self, request: ChatCompletionRequest
    ) -> Optional[tuple[ChatCompletionResponse, str]]:
        """Return (cached response, provider) on a hit, else None."""
        text = self._last_user_text(request)
        if not text:
            return None

        query = await self._encode(text)

        async with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None
            for entry in self._entries:
                if entry.model != request.model:
                    continue
                sim = float(np.dot(query, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self.config.similarity_threshold:
                logger.info(
                    "cache_hit",
                    similarity=round(best_sim, 4),
                    model=request.model,
                )
                return best_entry.response, best_entry.provider

            logger.info(
                "cache_miss",
                best_similarity=round(best_sim, 4) if best_entry else None,
                model=request.model,
            )
            return None

    async def insert(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        provider_name: str,
    ) -> None:
        """Store a response keyed by the embedding of the last user message."""
        text = self._last_user_text(request)
        if not text:
            return

        embedding = await self._encode(text)

        async with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            self._entries.append(
                CacheEntry(
                    embedding=embedding,
                    model=request.model,
                    response=response,
                    provider=provider_name,
                    created_at=now,
                )
            )
            self._enforce_max_size()

    async def close(self) -> None:
        """No external resources to release; present for symmetry with providers."""
        return None
