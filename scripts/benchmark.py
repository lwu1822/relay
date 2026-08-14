"""Benchmark script for the relay gateway.

Fires a configurable mix of duplicate / near-duplicate / unique prompts at a
running relay instance, then diffs the Prometheus counters exposed at
/metrics before and after the run to report cache hit rate, latency
percentiles, and throughput.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --count 1000

The relay server must already be running (e.g. uvicorn app.main:app).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from dataclasses import dataclass

import httpx


# Base prompts and paraphrased near-duplicates used to build the request mix.
# Each tuple is (base_prompt, [near_duplicate_variants]).
PROMPT_POOL: list[tuple[str, list[str]]] = [
    ("What is the capital of France?", [
        "Which city is the capital of France?",
        "Tell me the capital city of France.",
    ]),
    ("Explain recursion in simple terms.", [
        "Can you describe recursion simply?",
        "How would you explain recursion to a beginner?",
    ]),
    ("Write a haiku about the ocean.", [
        "Compose a haiku about the sea.",
        "Give me a haiku on the ocean.",
    ]),
    ("What is the meaning of life?", [
        "What's the purpose of life?",
        "Explain the meaning of existence.",
    ]),
    ("How does a CPU work?", [
        "Explain how a computer processor functions.",
        "Describe the operation of a CPU.",
    ]),
    ("Translate 'hello' to Spanish.", [
        "How do you say 'hello' in Spanish?",
        "What is 'hello' translated to Spanish?",
    ]),
    ("Summarize the plot of Romeo and Juliet.", [
        "Give me a short summary of Romeo and Juliet.",
        "Briefly describe the story of Romeo and Juliet.",
    ]),
    ("What causes rainbows?", [
        "Why do rainbows appear?",
        "Explain the science behind rainbows.",
    ]),
    ("List three benefits of exercise.", [
        "Name three advantages of working out.",
        "What are three good reasons to exercise?",
    ]),
    ("How do I bake chocolate chip cookies?", [
        "What's the recipe for chocolate chip cookies?",
        "Give me instructions to bake chocolate chip cookies.",
    ]),
    ("What is the speed of light?", [
        "How fast does light travel?",
        "Tell me the speed of light in vacuum.",
    ]),
    ("Explain the theory of relativity.", [
        "Describe Einstein's theory of relativity.",
        "What does the theory of relativity say?",
    ]),
    ("Recommend a good sci-fi book.", [
        "Suggest a science fiction novel to read.",
        "What sci-fi book would you recommend?",
    ]),
    ("How do plants make food?", [
        "Explain how photosynthesis works.",
        "Describe the process plants use to make food.",
    ]),
    ("What is machine learning?", [
        "Define machine learning in simple terms.",
        "Explain what machine learning is.",
    ]),
    ("Why is the sky blue?", [
        "What makes the sky appear blue?",
        "Explain why the sky is blue.",
    ]),
    ("How do I learn Python?", [
        "What's the best way to learn Python?",
        "Give me advice on learning Python.",
    ]),
    ("What is the largest planet?", [
        "Which planet in our solar system is the biggest?",
        "Name the largest planet.",
    ]),
    ("Describe how the internet works.", [
        "Explain the workings of the internet.",
        "How does the internet function?",
    ]),
    ("What are the symptoms of dehydration?", [
        "List signs that someone is dehydrated.",
        "What indicates a person is dehydrated?",
    ]),
]


@dataclass
class Result:
    status: int
    latency_ms: float
    ok: bool


def build_prompt_list(count: int, dup_ratio: float, near_dup_ratio: float) -> list[str]:
    """Build a list of `count` prompts mixing duplicates, near-dups, and uniques.

    dup_ratio: fraction of prompts that are exact repeats of an earlier prompt.
    near_dup_ratio: fraction that are paraphrased variants of an earlier prompt.
    Remaining prompts are unique (drawn from the base pool, cycling as needed).
    """
    n_dup = int(count * dup_ratio)
    n_near = int(count * near_dup_ratio)
    n_unique = count - n_dup - n_near

    bases = [b for b, _ in PROMPT_POOL]
    variants = [v for _, vs in PROMPT_POOL for v in vs]

    prompts: list[str] = []
    # Unique prompts: cycle through base prompts.
    for i in range(n_unique):
        prompts.append(bases[i % len(bases)])
    # Near-duplicates: cycle through variant prompts (semantically close to a base).
    for i in range(n_near):
        prompts.append(variants[i % len(variants)])
    # Exact duplicates: repeat the first base prompt many times.
    for _ in range(n_dup):
        prompts.append(bases[0])

    # Shuffle deterministically so the run interleaves hit/miss patterns.
    import random
    rng = random.Random(42)
    rng.shuffle(prompts)
    return prompts


def parse_metric(text: str, name: str) -> float:
    """Sum all sample values for a Prometheus counter from plaintext exposition."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(name):
            # Lines look like: name{labels} value  OR  name value
            m = re.search(r"\s([0-9eE+\-.]+)\s*$", line)
            if m:
                total += float(m.group(1))
    return total


async def fetch_metrics(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    resp = await client.get(f"{url}/metrics")
    resp.raise_for_status()
    text = resp.text
    return {
        "requests_total": parse_metric(text, "gateway_requests_total"),
        "cache_hits_total": parse_metric(text, "gateway_cache_hits_total"),
    }


async def send_one(
    client: httpx.AsyncClient, url: str, prompt: str, model: str, sem: asyncio.Semaphore
) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
    }
    async with sem:
        start = time.monotonic()
        try:
            resp = await client.post(
                f"{url}/v1/chat/completions", json=payload, timeout=60.0
            )
            latency_ms = (time.monotonic() - start) * 1000.0
            return Result(
                status=resp.status_code,
                latency_ms=latency_ms,
                ok=resp.status_code == 200,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000.0
            print(f"  request error: {e}")
            return Result(status=0, latency_ms=latency_ms, ok=False)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


async def run(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    model = args.model
    count = args.count
    concurrency = args.concurrency

    prompts = build_prompt_list(count, args.dup_ratio, args.near_dup_ratio)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        print(f"Target: {url}")
        print(f"Requests: {count} (dup={args.dup_ratio:.0%} "
              f"near-dup={args.near_dup_ratio:.0%} unique={1 - args.dup_ratio - args.near_dup_ratio:.0%})")
        print(f"Concurrency: {concurrency}")
        print(f"Model: {model}")
        print("Capturing baseline /metrics ...")
        before = await fetch_metrics(client, url)

        print("Firing requests ...")
        wall_start = time.monotonic()
        results = await asyncio.gather(
            *[send_one(client, url, p, model, sem) for p in prompts]
        )
        wall_seconds = time.monotonic() - wall_start

        print("Capturing post-run /metrics ...")
        after = await fetch_metrics(client, url)

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    latencies = sorted(r.latency_ms for r in ok)

    requests_delta = after["requests_total"] - before["requests_total"]
    hits_delta = after["cache_hits_total"] - before["cache_hits_total"]
    hit_rate = (hits_delta / requests_delta) if requests_delta > 0 else 0.0
    rps = count / wall_seconds if wall_seconds > 0 else 0.0

    print()
    print("=" * 60)
    print("Benchmark results")
    print("=" * 60)
    print(f"Total requests sent      : {count}")
    print(f"Successful (HTTP 200)    : {len(ok)}")
    print(f"Failed                   : {len(failed)}")
    print(f"Wall-clock time          : {wall_seconds:.2f} s")
    print(f"Throughput               : {rps:.2f} req/s")
    print(f"Cache hits (delta)       : {int(hits_delta)}")
    print(f"Cache hit rate           : {hit_rate:.2%}")
    print(f"Latency p50              : {percentile(latencies, 0.50):.1f} ms")
    print(f"Latency p95              : {percentile(latencies, 0.95):.1f} ms")
    print(f"Latency p99              : {percentile(latencies, 0.99):.1f} ms")
    print("=" * 60)
    print()
    print("Copy-paste block for README:")
    print()
    print(f"| {count} | {len(ok)} | {rps:.2f} | {hit_rate:.2%} | "
          f"{percentile(latencies, 0.50):.1f} | "
          f"{percentile(latencies, 0.95):.1f} | "
          f"{percentile(latencies, 0.99):.1f} |")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the relay gateway.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="relay base URL (default: http://localhost:8000)")
    parser.add_argument("--model", default="llama3.2:1b",
                        help="model name to send in requests")
    parser.add_argument("--count", type=int, default=1000,
                        help="total number of requests (default: 1000)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="max concurrent in-flight requests (default: 20)")
    parser.add_argument("--dup-ratio", type=float, default=0.4,
                        help="fraction of exact-duplicate prompts (default: 0.4)")
    parser.add_argument("--near-dup-ratio", type=float, default=0.3,
                        help="fraction of near-duplicate prompts (default: 0.3)")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
