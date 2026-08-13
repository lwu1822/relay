# relay

A self-hostable **LLM inference gateway** with an OpenAI-compatible API. Routes requests across multiple providers (Ollama, Groq), with semantic caching, quality-aware cost optimization, retry/failover, and Prometheus observability.

## Why

Every team using LLMs hits the same problems: vendor lock-in, unpredictable latency, no caching, no visibility into cost. `relay` sits between your application and LLM providers, giving you one stable endpoint with routing, caching, failover, and metrics.

## Features

- **OpenAI-compatible API** — drop-in replacement for `openai.ChatCompletion.create()`
- **Multi-provider routing** — round-robin or cost-based across Ollama (local) and Groq (cloud)
- **Semantic caching** — sentence-transformer embeddings + cosine similarity; skip the provider entirely on cache hits
- **Retry & failover** — exponential backoff per provider, automatic fallback to next provider
- **Quality-aware cost optimization** — route to the cheapest model that passes a per-category quality bar
- **Observability** — Prometheus metrics (p50/p95/p99 latency, cache hit rate, cost saved), structured JSON logging

## Quick start

```bash
# Clone
git clone https://github.com/lwu1822/relay.git
cd relay

# Install dependencies
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
# edit config.yaml with your provider settings

# Run
uvicorn app.main:app --reload

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Architecture

```
Client → relay gateway → [cache lookup] → [router] → provider (Ollama / Groq)
                         ↓ cache hit        ↓ fail
                         return cached      retry → fallback provider
```

<!-- Architecture diagram (PNG) will be added in Sprint 3 -->

## Configuration

See [`config.example.yaml`](config.example.yaml) for all options.

## Benchmark

<!-- Benchmark numbers will be added in Sprint 2 -->

## Tech stack

- **Python 3.11+** / FastAPI / httpx / pydantic v2
- **sentence-transformers** (`all-MiniLM-L6-v2`) for semantic cache embeddings
- **prometheus-client** for metrics
- **structlog** for structured logging
- **Docker** multi-stage build / **Fly.io** deployment
- **Go** for V2 hot-path rewrite (planned)

## License

TBD