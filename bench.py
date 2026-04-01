#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The perf-llm Project Authors

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


@dataclass
class RequestResult:
    ok: bool
    provider: str
    concurrency: int
    round_id: int
    thinking_level: str | None
    latency_s: float
    output_tokens: int | None
    tokens_per_s: float | None
    status: int | None
    error: str | None


@dataclass
class PointSummary:
    provider: str
    model: str
    concurrency: int
    rounds: int
    thinking_level: str | None
    total_requests: int
    succeeded: int
    failed: int
    latency_min_s: float | None
    latency_mean_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    latency_max_s: float | None
    throughput_mean_tokens_s: float | None
    throughput_sum_tokens_s: float | None
    total_output_tokens: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str | None) -> list[str | None]:
    if value is None or not value.strip():
        return [None]
    return [x.strip() for x in value.split(",") if x.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] * (1 - frac) + values[high] * frac


DEFAULT_PROMPT = "Hello, tell me a joke."
LOGGER = logging.getLogger("perf_llm")


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def merge_extra_body(base: dict[str, Any], extra_json: str | None) -> dict[str, Any]:
    if not extra_json:
        return base
    extra = json.loads(extra_json)
    if not isinstance(extra, dict):
        raise SystemExit("--extra-body-json must be a JSON object")
    merged = dict(base)
    merged.update(extra)
    return merged


def configure_logging(debug: bool, quiet: bool) -> None:
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def make_headers(provider: str, api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider == "openai" and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_output_tokens(provider: str, payload: dict[str, Any]) -> int | None:
    if provider == "openai":
        usage = payload.get("usage") or {}
        value = usage.get("completion_tokens")
        return int(value) if isinstance(value, (int, float)) else None
    if provider == "ollama":
        value = payload.get("eval_count")
        return int(value) if isinstance(value, (int, float)) else None
    return None


def make_request_payload(
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
    thinking_level: str | None,
    thinking_key: str,
    ctx_size: int | None,
    extra_body_json: str | None,
) -> dict[str, Any]:
    if provider == "openai":
        if ctx_size is not None:
            LOGGER.warning(
                "Ignoring --ctx-size for provider=openai: no standard chat/completions field"
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
    elif provider == "ollama":
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if ctx_size is not None:
            options["num_ctx"] = ctx_size
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if thinking_level is not None:
        body[thinking_key] = thinking_level

    return merge_extra_body(body, extra_body_json)


def endpoint_url(provider: str, base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if provider == "openai":
        return f"{base_url}/v1/chat/completions"
    if provider == "ollama":
        return f"{base_url}/api/generate"
    raise ValueError(f"Unsupported provider: {provider}")


def models_url(provider: str, base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if provider == "openai":
        return f"{base_url}/v1/models"
    if provider == "ollama":
        return f"{base_url}/api/tags"
    raise ValueError(f"Unsupported provider: {provider}")


def extract_model_names(provider: str, payload: dict[str, Any]) -> list[str]:
    if provider == "openai":
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        names = [item.get("id") for item in data if isinstance(item, dict)]
        return [str(name) for name in names if name]

    if provider == "ollama":
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        names = [item.get("model") or item.get("name") for item in models if isinstance(item, dict)]
        return [str(name) for name in names if name]

    return []


async def list_models(
    *, provider: str, base_url: str, api_key: str | None, timeout_s: float
) -> list[str]:
    if aiohttp is None:
        raise SystemExit("Missing dependency: aiohttp. Install it with: pip install aiohttp")

    headers = make_headers(provider, api_key)
    url = models_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        LOGGER.debug("GET %s", url)
        async with session.get(url, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise SystemExit(f"Failed to list models: HTTP {resp.status}: {text[:500]}")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to decode model list response as JSON: {exc}") from exc

    return extract_model_names(provider, payload)


async def one_request(
    session: Any,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    concurrency: int,
    round_id: int,
    thinking_level: str | None,
) -> RequestResult:
    start = time.perf_counter()
    try:
        LOGGER.debug("POST %s", url)
        async with session.post(url, headers=headers, json=payload) as resp:
            status = resp.status
            text = await resp.text()
            latency_s = time.perf_counter() - start
            if status >= 400:
                return RequestResult(
                    ok=False,
                    provider=provider,
                    concurrency=concurrency,
                    round_id=round_id,
                    thinking_level=thinking_level,
                    latency_s=latency_s,
                    output_tokens=None,
                    tokens_per_s=None,
                    status=status,
                    error=text[:500],
                )
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return RequestResult(
                    ok=False,
                    provider=provider,
                    concurrency=concurrency,
                    round_id=round_id,
                    thinking_level=thinking_level,
                    latency_s=latency_s,
                    output_tokens=None,
                    tokens_per_s=None,
                    status=status,
                    error="Invalid JSON response",
                )

            output_tokens = extract_output_tokens(provider, data)
            tokens_per_s = (
                (output_tokens / latency_s) if output_tokens is not None and latency_s > 0 else None
            )
            return RequestResult(
                ok=True,
                provider=provider,
                concurrency=concurrency,
                round_id=round_id,
                thinking_level=thinking_level,
                latency_s=latency_s,
                output_tokens=output_tokens,
                tokens_per_s=tokens_per_s,
                status=status,
                error=None,
            )
    except Exception as exc:
        latency_s = time.perf_counter() - start
        return RequestResult(
            ok=False,
            provider=provider,
            concurrency=concurrency,
            round_id=round_id,
            thinking_level=thinking_level,
            latency_s=latency_s,
            output_tokens=None,
            tokens_per_s=None,
            status=None,
            error=str(exc),
        )


async def run_point(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    concurrency: int,
    rounds: int,
    thinking_level: str | None,
    thinking_key: str,
    max_tokens: int | None,
    temperature: float | None,
    ctx_size: int | None,
    timeout_s: float,
    extra_body_json: str | None,
    verbose: bool,
) -> tuple[PointSummary, list[RequestResult]]:
    if aiohttp is None:
        raise SystemExit("Missing dependency: aiohttp. Install it with: pip install aiohttp")

    headers = make_headers(provider, api_key)
    url = endpoint_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    payload = make_request_payload(
        provider=provider,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        thinking_key=thinking_key,
        ctx_size=ctx_size,
        extra_body_json=extra_body_json,
    )

    results: list[RequestResult] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for round_id in range(rounds):
            tasks = [
                one_request(
                    session,
                    provider=provider,
                    url=url,
                    headers=headers,
                    payload=payload,
                    concurrency=concurrency,
                    round_id=round_id,
                    thinking_level=thinking_level,
                )
                for _ in range(concurrency)
            ]
            batch = await asyncio.gather(*tasks)
            results.extend(batch)
            if verbose:
                for item in batch:
                    if not item.ok:
                        LOGGER.warning(
                            "provider=%s conc=%s round=%s thinking=%r status=%s error=%s",
                            provider,
                            concurrency,
                            round_id,
                            thinking_level,
                            item.status,
                            item.error,
                        )

    summary = summarize_point(provider, model, concurrency, rounds, thinking_level, results)
    return summary, results


def summarize_point(
    provider: str,
    model: str,
    concurrency: int,
    rounds: int,
    thinking_level: str | None,
    results: list[RequestResult],
) -> PointSummary:
    latencies = [r.latency_s for r in results]
    tps_values = [r.tokens_per_s for r in results if r.tokens_per_s is not None]
    output_tokens = [r.output_tokens for r in results if r.output_tokens is not None]
    succeeded = sum(1 for r in results if r.ok)
    failed = len(results) - succeeded

    return PointSummary(
        provider=provider,
        model=model,
        concurrency=concurrency,
        rounds=rounds,
        thinking_level=thinking_level,
        total_requests=len(results),
        succeeded=succeeded,
        failed=failed,
        latency_min_s=min(latencies) if latencies else None,
        latency_mean_s=statistics.fmean(latencies) if latencies else None,
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        latency_p99_s=percentile(latencies, 0.99),
        latency_max_s=max(latencies) if latencies else None,
        throughput_mean_tokens_s=statistics.fmean(tps_values) if tps_values else None,
        throughput_sum_tokens_s=sum(tps_values) if tps_values else None,
        total_output_tokens=sum(output_tokens) if output_tokens else 0,
    )


def print_summary_table(summaries: list[PointSummary]) -> None:
    headers = [
        "provider",
        "model",
        "thinking",
        "conc",
        "ok",
        "fail",
        "lat_mean",
        "p95",
        "p99",
        "tps_mean",
        "tokens",
    ]
    rows = []
    for s in summaries:
        rows.append(
            [
                s.provider,
                s.model,
                s.thinking_level or "-",
                str(s.concurrency),
                str(s.succeeded),
                str(s.failed),
                fmt_float(s.latency_mean_s),
                fmt_float(s.latency_p95_s),
                fmt_float(s.latency_p99_s),
                fmt_float(s.throughput_mean_tokens_s),
                str(s.total_output_tokens),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(parts: list[str]) -> str:
        return "  ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in rows:
        print(line(row))


def fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


async def run_warmup(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    thinking_level: str | None,
    thinking_key: str,
    max_tokens: int | None,
    temperature: float | None,
    ctx_size: int | None,
    timeout_s: float,
    extra_body_json: str | None,
    warmup_runs: int,
    verbose: bool,
) -> None:
    if warmup_runs <= 0:
        return
    if aiohttp is None:
        raise SystemExit("Missing dependency: aiohttp. Install it with: pip install aiohttp")

    headers = make_headers(provider, api_key)
    url = endpoint_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    payload = make_request_payload(
        provider=provider,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        thinking_key=thinking_key,
        ctx_size=ctx_size,
        extra_body_json=extra_body_json,
    )

    LOGGER.info(
        "warmup provider=%s model=%s thinking=%r runs=%s",
        provider,
        model,
        thinking_level,
        warmup_runs,
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for warmup_id in range(warmup_runs):
            result = await one_request(
                session,
                provider=provider,
                url=url,
                headers=headers,
                payload=payload,
                concurrency=1,
                round_id=-(warmup_id + 1),
                thinking_level=thinking_level,
            )
            if verbose or not result.ok:
                LOGGER.warning(
                    "warmup-result run=%s/%s ok=%s status=%s latency_s=%.4f error=%s",
                    warmup_id + 1,
                    warmup_runs,
                    result.ok,
                    result.status,
                    result.latency_s,
                    result.error,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible and Ollama text completion servers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--list-models",
        "--list-mnodels",
        action="store_true",
        help="List models available on the base URL and exit",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--thinking-level", default=None)
    parser.add_argument("--thinking-key", default="thinking_level")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-max-tokens", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-temperature", action="store_true")
    parser.add_argument("--ctx-size", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--extra-body-json", default=None)
    parser.add_argument("--output-json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    configure_logging(args.debug, args.quiet)

    if args.list_models:
        models = await list_models(
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
        )
        for model_name in models:
            print(model_name)
        return 0

    if not args.model:
        raise SystemExit("--model is required unless --list-models is used")

    effective_max_tokens = None if args.no_max_tokens else args.max_tokens
    effective_temperature = None if args.no_temperature else args.temperature

    prompt = load_prompt(args)
    concurrencies = parse_csv_ints(args.concurrency)
    thinking_levels = parse_csv_strings(args.thinking_level)

    await run_warmup(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        prompt=prompt,
        thinking_level=thinking_levels[0],
        thinking_key=args.thinking_key,
        max_tokens=effective_max_tokens,
        temperature=effective_temperature,
        ctx_size=args.ctx_size,
        timeout_s=args.timeout,
        extra_body_json=args.extra_body_json,
        warmup_runs=args.warmup_runs,
        verbose=args.verbose,
    )

    summaries: list[PointSummary] = []
    all_results: list[RequestResult] = []

    for thinking_level in thinking_levels:
        for concurrency in concurrencies:
            LOGGER.info(
                "run provider=%s model=%s thinking=%r concurrency=%s rounds=%s",
                args.provider,
                args.model,
                thinking_level,
                concurrency,
                args.rounds,
            )
            summary, results = await run_point(
                provider=args.provider,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                prompt=prompt,
                concurrency=concurrency,
                rounds=args.rounds,
                thinking_level=thinking_level,
                thinking_key=args.thinking_key,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                ctx_size=args.ctx_size,
                timeout_s=args.timeout,
                extra_body_json=args.extra_body_json,
                verbose=args.verbose,
            )
            summaries.append(summary)
            all_results.extend(results)

    print_summary_table(summaries)

    if args.output_json:
        payload = {
            "config": {
                "provider": args.provider,
                "base_url": args.base_url,
                "model": args.model,
                "concurrency": concurrencies,
                "rounds": args.rounds,
                "warmup_runs": args.warmup_runs,
                "thinking_levels": thinking_levels,
                "max_tokens": effective_max_tokens,
                "temperature": effective_temperature,
                "ctx_size": args.ctx_size,
                "timeout": args.timeout,
            },
            "summaries": [asdict(s) for s in summaries],
            "requests": [asdict(r) for r in all_results],
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote JSON results to %s", args.output_json)

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
