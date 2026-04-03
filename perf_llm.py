#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The perf-llm Project Authors

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class RequestResult:
    ok: bool
    provider: str
    concurrency: int
    round_id: int
    thinking_level: str | None
    latency_s: float
    first_response_latency_s: float | None
    first_content_latency_s: float | None
    output_tokens: int | None
    tokens_per_s: float | None
    stream_chunks: int | None
    status: int | None
    error: str | None


@dataclass
class PointSummary:
    provider: str
    api_variant: str
    model: str
    concurrency: int
    rounds: int
    thinking_level: str | None
    total_requests: int
    succeeded: int
    failed: int
    failure_rate_pct: float
    wall_time_s: float
    latency_min_s: float | None
    latency_mean_s: float | None
    latency_p50_s: float | None
    latency_max_s: float | None
    first_response_latency_min_s: float | None
    first_response_latency_mean_s: float | None
    first_response_latency_p50_s: float | None
    first_response_latency_max_s: float | None
    first_content_latency_min_s: float | None
    first_content_latency_mean_s: float | None
    first_content_latency_p50_s: float | None
    first_content_latency_max_s: float | None
    throughput_mean_tokens_s: float | None
    throughput_sum_tokens_s: float | None
    throughput_wall_tokens_s: float | None
    total_output_tokens: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str | None) -> list[str | None]:
    if value is None or not value.strip():
        return [None]
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_thinking_level(thinking_level: str | None) -> str | None:
    if thinking_level is None:
        return None
    lowered = thinking_level.strip().lower()
    if not lowered or lowered == "default":
        return None
    return thinking_level


def validate_positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise SystemExit(f"{name} must be >= 0")
    elif value <= 0:
        raise SystemExit(f"{name} must be > 0")


def validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be > 0")


def validate_args(args: argparse.Namespace) -> None:
    validate_positive_int("--rounds", args.rounds)
    validate_positive_int("--warmup-runs", args.warmup_runs, allow_zero=True)
    validate_positive_float("--timeout", args.timeout)
    if args.api_key and args.oauth_access_token:
        LOGGER.warning("Both --api-key and --oauth-access-token were provided; using OAuth token")
    if args.max_tokens is not None:
        validate_positive_int("--max-tokens", args.max_tokens)
    if args.ctx_size is not None:
        validate_positive_int("--ctx-size", args.ctx_size)
    if args.provider == "ollama" and args.api_variant != "default":
        raise SystemExit("--api-variant is only supported for provider=openai")
    try:
        concurrencies = parse_csv_ints(args.concurrency)
    except ValueError as exc:
        raise SystemExit(f"Invalid --concurrency value: {args.concurrency}") from exc
    if not concurrencies:
        raise SystemExit("--concurrency must contain at least one integer")
    for concurrency in concurrencies:
        validate_positive_int("--concurrency", concurrency)


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


DEFAULT_PROMPT = "ping"
DEFAULT_WARMUP_PROMPT = "ping"
LOGGER = logging.getLogger("perf_llm")


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def load_warmup_prompt(args: argparse.Namespace, prompt: str) -> str:
    if args.prompt_warmup is not None:
        return args.prompt_warmup
    return prompt


def merge_extra_body(base: dict[str, Any], extra_json: str | None) -> dict[str, Any]:
    if not extra_json:
        return base
    extra = json.loads(extra_json)
    if not isinstance(extra, dict):
        raise SystemExit("--extra-body-json must be a JSON object")
    merged = dict(base)
    merged.update(extra)
    return merged


def configure_logging(debug: bool, quiet: bool, log_file: str | None) -> None:
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    if quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        filename=log_file,
        filemode="a" if log_file else "w",
    )


def log_json_content(enabled: bool, label: str, payload: Any) -> None:
    if not enabled:
        return
    try:
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError:
        rendered = repr(payload)
    LOGGER.debug("%s: %s", label, rendered)


def make_headers(
    provider: str, api_key: str | None, oauth_access_token: str | None
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    bearer_token = oauth_access_token or api_key
    if provider == "openai" and bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
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
    api_variant: str,
    model: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
    thinking_level: str | None,
    thinking_key: str,
    ctx_size: int | None,
    extra_body_json: str | None,
    stream: bool = False,
) -> dict[str, Any]:
    normalized_thinking_level = normalize_thinking_level(thinking_level)

    if provider == "openai":
        if ctx_size is not None:
            LOGGER.warning(
                "Ignoring --ctx-size for provider=openai: no standard chat/completions field"
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if api_variant == "default":
            if normalized_thinking_level == "none":
                LOGGER.warning(
                    "Ignoring thinking_level=none for provider=openai api_variant=default: no standard disable-thinking field"
                )
            elif normalized_thinking_level is not None:
                body["reasoning_effort"] = normalized_thinking_level
        elif api_variant == "mlx":
            if normalized_thinking_level == "none":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            elif normalized_thinking_level is not None:
                body["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "reasoning_effort": normalized_thinking_level,
                }
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
            "stream": stream,
            "options": options,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if provider == "ollama" and normalized_thinking_level is not None:
        body[thinking_key] = normalized_thinking_level

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


def extract_text_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        text_parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return text_parts
    return []


def extract_stream_response_chunk(event_payload: dict[str, Any], provider: str) -> str:
    if provider == "openai":
        choices = event_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        return "".join(extract_text_parts(delta.get("content")))
    if provider == "ollama":
        response = event_payload.get("response")
        return response if isinstance(response, str) else ""
    return ""


def extract_stream_reasoning_chunk(event_payload: dict[str, Any], provider: str) -> str:
    if provider == "openai":
        choices = event_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        return "".join(extract_text_parts(delta.get("reasoning_content")))
    return ""


def extract_response(response_payload: dict[str, Any], provider: str) -> str:
    if provider == "openai":
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        message = choice.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return "".join(extract_text_parts(content))
    if provider == "ollama":
        response = response_payload.get("response")
        return response if isinstance(response, str) else ""
    return ""


async def collect_stream_response(
    resp: Any, provider: str, debug_content: bool, start_time: float
) -> tuple[str, dict[str, Any] | None, int, float | None, float | None]:
    response_parts: list[str] = []
    last_payload: dict[str, Any] | None = None
    chunk_count = 0
    first_response_latency_s: float | None = None
    first_content_latency_s: float | None = None

    async for raw_line in resp.content:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        if provider == "openai":
            if not line.startswith("data:"):
                continue
            line = line[5:].strip()
            if line == "[DONE]":
                continue
        try:
            event_payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event_payload, dict):
            continue
        last_payload = event_payload
        chunk_count += 1
        now = time.perf_counter()
        if first_response_latency_s is None:
            first_response_latency_s = now - start_time
        log_json_content(debug_content, "response_json", event_payload)
        response_part = extract_stream_response_chunk(event_payload, provider)
        reasoning_part = extract_stream_reasoning_chunk(event_payload, provider)
        if first_content_latency_s is None and (response_part or reasoning_part):
            first_content_latency_s = now - start_time
        if response_part:
            response_parts.append(response_part)

    return (
        "".join(response_parts),
        last_payload,
        chunk_count,
        first_response_latency_s,
        first_content_latency_s,
    )


async def collect_response(
    resp: Any, provider: str, stream: bool, debug_content: bool, start_time: float
) -> tuple[str, dict[str, Any] | None, str | None, int | None, float | None, float | None]:
    if stream:
        (
            response_text,
            response_payload,
            chunk_count,
            first_response_latency_s,
            first_content_latency_s,
        ) = await collect_stream_response(resp, provider, debug_content, start_time)
        return (
            response_text,
            response_payload,
            None,
            chunk_count,
            first_response_latency_s,
            first_content_latency_s,
        )

    text = await resp.text()
    try:
        response_payload = json.loads(text)
    except json.JSONDecodeError:
        log_json_content(debug_content, "response_json", text)
        return text, None, text, None, None, None

    if isinstance(response_payload, dict):
        log_json_content(debug_content, "response_json", response_payload)
        return (
            extract_response(response_payload, provider),
            response_payload,
            text,
            None,
            None,
            None,
        )

    log_json_content(debug_content, "response_json", response_payload)
    return text, None, text, None, None, None


async def list_models(
    *,
    provider: str,
    base_url: str,
    api_key: str | None,
    oauth_access_token: str | None,
    timeout_s: float,
) -> list[str]:
    headers = make_headers(provider, api_key, oauth_access_token)
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


async def test_request(
    *,
    provider: str,
    api_variant: str,
    base_url: str,
    model: str,
    api_key: str | None,
    oauth_access_token: str | None,
    prompt: str,
    thinking_level: str | None,
    thinking_key: str,
    max_tokens: int | None,
    temperature: float | None,
    ctx_size: int | None,
    timeout_s: float,
    extra_body_json: str | None,
    debug_content: bool,
    stream: bool,
) -> int:
    headers = make_headers(provider, api_key, oauth_access_token)
    url = endpoint_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    payload = make_request_payload(
        provider=provider,
        api_variant=api_variant,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        thinking_key=thinking_key,
        ctx_size=ctx_size,
        extra_body_json=extra_body_json,
        stream=stream,
    )

    print(f"prompt: {prompt}")
    log_json_content(debug_content, "request_json", payload)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        LOGGER.debug("POST %s", url)
        start = time.perf_counter()
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status >= 400:
                text = await resp.text()
                LOGGER.warning("test request failed with HTTP %s", resp.status)
                log_json_content(debug_content, "response_json", text)
                return 1

            print(f"status: {resp.status}")
            (
                response_text,
                _,
                _,
                stream_chunks,
                first_response_latency_s,
                first_content_latency_s,
            ) = await collect_response(resp, provider, stream, debug_content, start)
            if stream_chunks is not None:
                LOGGER.info(
                    "stream chunks received=%s first_response_latency_s=%.4f first_content_latency_s=%.4f",
                    stream_chunks,
                    first_response_latency_s if first_response_latency_s is not None else -1.0,
                    first_content_latency_s if first_content_latency_s is not None else -1.0,
                )
            print(f"response: {response_text}")
    return 0


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
    debug_content: bool,
    stream: bool,
) -> RequestResult:
    start = time.perf_counter()
    try:
        LOGGER.debug("POST %s", url)
        log_json_content(debug_content, "request_json", payload)
        async with session.post(url, headers=headers, json=payload) as resp:
            status = resp.status
            if status >= 400:
                text = await resp.text()
                latency_s = time.perf_counter() - start
                return RequestResult(
                    ok=False,
                    provider=provider,
                    concurrency=concurrency,
                    round_id=round_id,
                    thinking_level=thinking_level,
                    latency_s=latency_s,
                    first_response_latency_s=None,
                    first_content_latency_s=None,
                    output_tokens=None,
                    tokens_per_s=None,
                    stream_chunks=None,
                    status=status,
                    error=text[:500],
                )

            (
                response_text,
                response_payload,
                raw_text,
                stream_chunks,
                first_response_latency_s,
                first_content_latency_s,
            ) = await collect_response(resp, provider, stream, debug_content, start)
            latency_s = time.perf_counter() - start
            if response_payload is None:
                return RequestResult(
                    ok=False,
                    provider=provider,
                    concurrency=concurrency,
                    round_id=round_id,
                    thinking_level=thinking_level,
                    latency_s=latency_s,
                    first_response_latency_s=first_response_latency_s,
                    first_content_latency_s=first_content_latency_s,
                    output_tokens=None,
                    tokens_per_s=None,
                    stream_chunks=stream_chunks,
                    status=status,
                    error="Invalid JSON response" if raw_text is not None else None,
                )

            output_tokens = extract_output_tokens(provider, response_payload)
            if output_tokens is None and response_text:
                output_tokens = None
            tokens_per_s = (
                (output_tokens / latency_s) if output_tokens is not None and latency_s > 0 else None
            )
            if stream_chunks is not None:
                LOGGER.info(
                    "stream chunks received=%s first_response_latency_s=%.4f first_content_latency_s=%.4f",
                    stream_chunks,
                    first_response_latency_s if first_response_latency_s is not None else -1.0,
                    first_content_latency_s if first_content_latency_s is not None else -1.0,
                )
            return RequestResult(
                ok=True,
                provider=provider,
                concurrency=concurrency,
                round_id=round_id,
                thinking_level=thinking_level,
                latency_s=latency_s,
                first_response_latency_s=first_response_latency_s,
                first_content_latency_s=first_content_latency_s,
                output_tokens=output_tokens,
                tokens_per_s=tokens_per_s,
                stream_chunks=stream_chunks,
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
            first_response_latency_s=None,
            first_content_latency_s=None,
            output_tokens=None,
            tokens_per_s=None,
            stream_chunks=None,
            status=None,
            error=str(exc),
        )


async def run_point(
    *,
    provider: str,
    api_variant: str,
    base_url: str,
    model: str,
    api_key: str | None,
    oauth_access_token: str | None,
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
    debug_content: bool,
    stream: bool,
) -> tuple[PointSummary, list[RequestResult]]:
    headers = make_headers(provider, api_key, oauth_access_token)
    url = endpoint_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    payload = make_request_payload(
        provider=provider,
        api_variant=api_variant,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        thinking_key=thinking_key,
        ctx_size=ctx_size,
        extra_body_json=extra_body_json,
        stream=stream,
    )

    results: list[RequestResult] = []
    point_start = time.perf_counter()
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
                    debug_content=debug_content,
                    stream=stream,
                )
                for _ in range(concurrency)
            ]
            batch = await asyncio.gather(*tasks)
            results.extend(batch)
            if verbose:
                for item in batch:
                    if not item.ok:
                        LOGGER.warning(
                            "provider=%s conc=%s round=%s thinking=%r status=%s chunks=%s error=%s",
                            provider,
                            concurrency,
                            round_id,
                            thinking_level,
                            item.status,
                            item.stream_chunks,
                            item.error,
                        )

    wall_time_s = time.perf_counter() - point_start
    summary = summarize_point(
        provider,
        api_variant,
        model,
        concurrency,
        rounds,
        thinking_level,
        results,
        wall_time_s,
    )
    return summary, results


def summarize_point(
    provider: str,
    api_variant: str,
    model: str,
    concurrency: int,
    rounds: int,
    thinking_level: str | None,
    results: list[RequestResult],
    wall_time_s: float,
) -> PointSummary:
    latencies = [r.latency_s for r in results]
    first_response_latencies = [
        r.first_response_latency_s for r in results if r.first_response_latency_s is not None
    ]
    first_content_latencies = [
        r.first_content_latency_s for r in results if r.first_content_latency_s is not None
    ]
    tps_values = [r.tokens_per_s for r in results if r.tokens_per_s is not None]
    output_tokens = [r.output_tokens for r in results if r.output_tokens is not None]
    succeeded = sum(1 for r in results if r.ok)
    failed = len(results) - succeeded
    failure_rate_pct = (failed / len(results) * 100.0) if results else 0.0

    return PointSummary(
        provider=provider,
        api_variant=api_variant,
        model=model,
        concurrency=concurrency,
        rounds=rounds,
        thinking_level=thinking_level,
        total_requests=len(results),
        succeeded=succeeded,
        failed=failed,
        failure_rate_pct=failure_rate_pct,
        wall_time_s=wall_time_s,
        latency_min_s=min(latencies) if latencies else None,
        latency_mean_s=statistics.fmean(latencies) if latencies else None,
        latency_p50_s=percentile(latencies, 0.50),
        latency_max_s=max(latencies) if latencies else None,
        first_response_latency_min_s=min(first_response_latencies)
        if first_response_latencies
        else None,
        first_response_latency_mean_s=(
            statistics.fmean(first_response_latencies) if first_response_latencies else None
        ),
        first_response_latency_p50_s=percentile(first_response_latencies, 0.50),
        first_response_latency_max_s=max(first_response_latencies)
        if first_response_latencies
        else None,
        first_content_latency_min_s=min(first_content_latencies)
        if first_content_latencies
        else None,
        first_content_latency_mean_s=(
            statistics.fmean(first_content_latencies) if first_content_latencies else None
        ),
        first_content_latency_p50_s=percentile(first_content_latencies, 0.50),
        first_content_latency_max_s=max(first_content_latencies)
        if first_content_latencies
        else None,
        throughput_mean_tokens_s=statistics.fmean(tps_values) if tps_values else None,
        throughput_sum_tokens_s=sum(tps_values) if tps_values else None,
        throughput_wall_tokens_s=(sum(output_tokens) / wall_time_s)
        if output_tokens and wall_time_s > 0
        else None,
        total_output_tokens=sum(output_tokens) if output_tokens else 0,
    )


def print_summary_table(summaries: list[PointSummary]) -> None:
    headers = [
        "provider",
        "variant",
        "model",
        "thinking",
        "conc",
        "ok",
        "fail%",
        "lat_min",
        "lat_mean",
        "lat_p50",
        "lat_max",
        "ttfb_min",
        "ttfb_mean",
        "ttfb_p50",
        "ttfb_max",
        "ttfc_min",
        "ttfc_mean",
        "ttfc_p50",
        "ttfc_max",
        "tps_mean",
        "tps_wall",
        "tokens",
    ]
    rows = []
    for s in summaries:
        rows.append(
            [
                s.provider,
                s.api_variant,
                s.model,
                s.thinking_level or "-",
                str(s.concurrency),
                str(s.succeeded),
                f"{s.failure_rate_pct:.1f}",
                fmt_float(s.latency_min_s),
                fmt_float(s.latency_mean_s),
                fmt_float(s.latency_p50_s),
                fmt_float(s.latency_max_s),
                fmt_float(s.first_response_latency_min_s),
                fmt_float(s.first_response_latency_mean_s),
                fmt_float(s.first_response_latency_p50_s),
                fmt_float(s.first_response_latency_max_s),
                fmt_float(s.first_content_latency_min_s),
                fmt_float(s.first_content_latency_mean_s),
                fmt_float(s.first_content_latency_p50_s),
                fmt_float(s.first_content_latency_max_s),
                fmt_float(s.throughput_mean_tokens_s),
                fmt_float(s.throughput_wall_tokens_s),
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


def default_csv_path(now: datetime) -> Path:
    return Path("csv") / f"bench-{now.strftime('%Y%m%d-%H%M%S')}.csv"


def write_csv_results(path: Path, timestamp: str, summaries: list[PointSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", *asdict(summaries[0]).keys()] if summaries else ["date"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {"date": timestamp}
            row.update(asdict(summary))
            writer.writerow(row)
    LOGGER.info("Wrote CSV results to %s", path)


async def run_warmup(
    *,
    provider: str,
    api_variant: str,
    base_url: str,
    model: str,
    api_key: str | None,
    oauth_access_token: str | None,
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
    debug_content: bool,
    stream: bool,
) -> None:
    if warmup_runs <= 0:
        return

    headers = make_headers(provider, api_key, oauth_access_token)
    url = endpoint_url(provider, base_url)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    payload = make_request_payload(
        provider=provider,
        api_variant=api_variant,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_level=thinking_level,
        thinking_key=thinking_key,
        ctx_size=ctx_size,
        extra_body_json=extra_body_json,
        stream=stream,
    )

    LOGGER.info(
        "warmup provider=%s api_variant=%s model=%s thinking=%r runs=%s stream=%s",
        provider,
        api_variant,
        model,
        thinking_level,
        warmup_runs,
        stream,
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
                debug_content=debug_content,
                stream=stream,
            )
            if verbose or not result.ok:
                LOGGER.warning(
                    "warmup-result run=%s/%s ok=%s status=%s chunks=%s latency_s=%.4f error=%s",
                    warmup_id + 1,
                    warmup_runs,
                    result.ok,
                    result.status,
                    result.stream_chunks,
                    result.latency_s,
                    result.error,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible and Ollama text completion servers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        required=True,
        help="Target API provider",
    )
    parser.add_argument(
        "--api-variant",
        choices=["default", "mlx"],
        default="default",
        help="Provider-specific API flavor",
    )
    parser.add_argument("--base-url", required=True, help="Server base URL")
    parser.add_argument("--model", help="Model name to query")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available on the base URL and exit",
    )
    parser.add_argument(
        "--test-request",
        action="store_true",
        help="Send one request payload and print the raw response, without benchmarking",
    )
    parser.add_argument("--api-key", help="API key bearer token for OpenAI-compatible APIs")
    parser.add_argument(
        "--oauth-access-token",
        help="OAuth bearer access token for OpenAI-compatible APIs",
    )
    parser.add_argument("--prompt", help="Prompt text for benchmark or test request")
    parser.add_argument("--prompt-file", help="Read prompt text from file")
    parser.add_argument(
        "--prompt-warmup",
        default=DEFAULT_WARMUP_PROMPT,
        help="Prompt text used for warmup requests",
    )
    parser.add_argument(
        "--concurrency",
        default="1",
        help="Comma-separated concurrency levels, e.g. 1,2,4",
    )
    parser.add_argument("--rounds", type=int, default=1, help="Rounds per benchmark point")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup requests")
    parser.add_argument("--thinking-level", default=None, help="Comma-separated thinking levels")
    parser.add_argument(
        "--thinking-key", default="thinking_level", help="Request field used for thinking level"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum output tokens to request"
    )
    parser.add_argument(
        "--no-max-tokens", action="store_true", help="Omit max_tokens from the request"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--no-temperature", action="store_true", help="Omit temperature from the request"
    )
    parser.add_argument("--ctx-size", type=int, help="Requested context size when supported")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument(
        "--extra-body-json", default=None, help="Extra JSON object merged into the request body"
    )
    parser.add_argument("--output-json", help="Write detailed results to a JSON file")
    parser.add_argument("--csv-file", help="Write summary CSV to this file")
    parser.add_argument("--no-csv", action="store_true", help="Disable default CSV output")
    parser.add_argument("--verbose", action="store_true", help="Log per-request failures")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-content", action="store_true", help="Log request and response JSON payloads"
    )
    parser.add_argument("--log-file", help="Write logs to this file instead of stderr")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable streaming mode")
    parser.add_argument(
        "--no-stream", action="store_false", dest="stream", help="Disable streaming mode"
    )
    parser.add_argument("--quiet", action="store_true", help="Only log warnings and errors")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    validate_args(args)
    configure_logging(args.debug, args.quiet, args.log_file)
    run_timestamp = datetime.now()
    run_timestamp_iso = run_timestamp.isoformat(timespec="seconds")

    if args.list_models:
        models = await list_models(
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            oauth_access_token=args.oauth_access_token,
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
    warmup_prompt = load_warmup_prompt(args, prompt)
    thinking_levels = parse_csv_strings(args.thinking_level)

    if args.test_request:
        return await test_request(
            provider=args.provider,
            api_variant=args.api_variant,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            oauth_access_token=args.oauth_access_token,
            prompt=prompt,
            thinking_level=thinking_levels[0],
            thinking_key=args.thinking_key,
            max_tokens=effective_max_tokens,
            temperature=effective_temperature,
            ctx_size=args.ctx_size,
            timeout_s=args.timeout,
            extra_body_json=args.extra_body_json,
            debug_content=args.debug_content,
            stream=args.stream,
        )

    concurrencies = parse_csv_ints(args.concurrency)

    await run_warmup(
        provider=args.provider,
        api_variant=args.api_variant,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        oauth_access_token=args.oauth_access_token,
        prompt=warmup_prompt,
        thinking_level=thinking_levels[0],
        thinking_key=args.thinking_key,
        max_tokens=effective_max_tokens,
        temperature=effective_temperature,
        ctx_size=args.ctx_size,
        timeout_s=args.timeout,
        extra_body_json=args.extra_body_json,
        warmup_runs=args.warmup_runs,
        verbose=args.verbose,
        debug_content=args.debug_content,
        stream=args.stream,
    )

    summaries: list[PointSummary] = []
    all_results: list[RequestResult] = []

    for thinking_level in thinking_levels:
        for concurrency in concurrencies:
            LOGGER.info(
                "run provider=%s api_variant=%s model=%s thinking=%r concurrency=%s rounds=%s stream=%s",
                args.provider,
                args.api_variant,
                args.model,
                thinking_level,
                concurrency,
                args.rounds,
                args.stream,
            )
            summary, results = await run_point(
                provider=args.provider,
                api_variant=args.api_variant,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                oauth_access_token=args.oauth_access_token,
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
                debug_content=args.debug_content,
                stream=args.stream,
            )
            summaries.append(summary)
            all_results.extend(results)

    print_summary_table(summaries)

    csv_path = (
        None
        if args.no_csv
        else Path(args.csv_file)
        if args.csv_file
        else default_csv_path(run_timestamp)
    )
    if csv_path is not None:
        write_csv_results(csv_path, run_timestamp_iso, summaries)

    if args.output_json:
        payload = {
            "config": {
                "provider": args.provider,
                "api_variant": args.api_variant,
                "base_url": args.base_url,
                "model": args.model,
                "concurrency": concurrencies,
                "rounds": args.rounds,
                "warmup_runs": args.warmup_runs,
                "thinking_levels": thinking_levels,
                "max_tokens": effective_max_tokens,
                "temperature": effective_temperature,
                "ctx_size": args.ctx_size,
                "stream": args.stream,
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
