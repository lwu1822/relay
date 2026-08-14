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

A benchmark script (`scripts/benchmark.py`) fires ~1000 requests at a running
relay instance with a mix of exact-duplicate, near-duplicate (paraphrased),
and unique prompts, then diffs the Prometheus counters at `/metrics` before
and after the run to report cache hit rate, latency percentiles, and
throughput.

### Run it

```bash
# Terminal 1: start the gateway (Ollama must be running on :11434)
uvicorn app.main:app --reload

# Terminal 2: run the benchmark
python scripts/benchmark.py --url http://localhost:8000 --count 1000
```

Flags: `--concurrency` (default 20), `--dup-ratio` (default 0.4),
`--near-dup-ratio` (default 0.3), `--model` (default `llama3.2:1b`).

### Results

| Requests | OK | Throughput (req/s) | Cache hit rate | p50 (ms) | p95 (ms) | p99 (ms) |
|----------|-----|--------------------|----------------|----------|----------|----------|
| 1000 | 1000 | 5.01 | 88.40% | 162.4 | 32919.3 | 35945.6 |

> The benchmark prints a ready-to-paste table row at the end of each run.

**Reading these numbers:** The 1000-request mix was 40% exact-duplicate,
30% near-duplicate (paraphrased), and 30% unique prompts, fired at
concurrency 20 against a single Ollama (`llama3.2:1b`) backend.

- **88.40% cache hit rate** — 884 of 1000 requests were served from the
  semantic cache, skipping the LLM provider entirely. This includes
  near-duplicate paraphrases (e.g. "What is the capital of France?" vs
  "Which city is the capital of France?") matched via cosine similarity
  over `all-MiniLM-L6-v2` embeddings at threshold 0.95.
- **p50 = 162 ms** — the median request is a cache hit; this is the
  embedding encode + similarity scan cost, not LLM inference.
- **p95/p99 = 33–36 s** — the tail is dominated by the ~116 unique
  requests that required a real Ollama call (`llama3.2:1b` on local CPU),
  queued behind concurrency-20 in-flight work. This is a property of the
  local single-process backend, not the gateway; a faster provider
  (e.g. Groq cloud) would compress the tail substantially.

## Tech stack

- **Python 3.11+** / FastAPI / httpx / pydantic v2
- **sentence-transformers** (`all-MiniLM-L6-v2`) for semantic cache embeddings
- **prometheus-client** for metrics
- **structlog** for structured logging
- **Docker** multi-stage build / **Fly.io** deployment
- **Go** for V2 hot-path rewrite (planned)

## License

TBD