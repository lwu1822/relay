"""Smoke tests — one live call per provider.

Run with: pytest tests/test_providers.py -v

These are real integration tests (not mocked) per the Sprint 1 plan.
Groq test is skipped if GROQ_API_KEY is not set.
"""

from __future__ import annotations

import os

import pytest

from app.models import ChatCompletionRequest, ChatMessage
from app.providers.groq import GroqProvider
from app.providers.ollama import OllamaProvider


@pytest.fixture
def sample_request():
    return ChatCompletionRequest(
        model="llama3.2:1b",
        messages=[ChatMessage(role="user", content="Say hello in one word.")],
        max_tokens=10,
    )


@pytest.mark.asyncio
async def test_ollama_provider(sample_request):
    """Smoke test: one real call to local Ollama."""
    provider = OllamaProvider()
    try:
        response = await provider.chat_completion(sample_request)
        assert response.choices
        assert response.choices[0].message.content
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_groq_provider(sample_request):
    """Smoke test: one real call to Groq cloud API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    groq_request = ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="Say hello in one word.")],
        max_tokens=10,
    )
    provider = GroqProvider(api_key=api_key)
    try:
        response = await provider.chat_completion(groq_request)
        assert response.choices
        assert response.choices[0].message.content
    finally:
        await provider.close()
