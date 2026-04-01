# perf-llm

`perf-llm` benchmarks LLM servers exposed through:

- OpenAI-compatible `v1/chat/completions`
- Ollama HTTP API

It varies:

- concurrency
- thinking level

And reports:

- latency statistics: min, mean, median, max
- streaming first-block latency (ttfb): min, mean, median, max
- streaming first-content latency (ttfc): min, mean, median, max
- throughput in tokens/sec, including wall-clock throughput

## Install

Python 3.10+ is required.

Install the project:

```bash
pip install .
```

Install with development tools:

```bash
pip install '.[dev]'
```

## Quick start

Default benchmark/test prompt:

```text
ping
```

Default warmup prompt:

```text
ping
```

Benchmark an OpenAI-compatible endpoint:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --thinking-level low,medium,high \
  --warmup-runs 1 \
  --prompt-warmup ping
```

Streaming is enabled by default in benchmark and test mode.
Disable it when needed:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --no-stream
```

Omit a setting from requests when needed:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --no-temperature \
  --no-max-tokens
```

Benchmark an Ollama endpoint:

```bash
python bench.py \
  --provider ollama \
  --base-url http://localhost:11434 \
  --model llama3 \
  --concurrency 1,2,4 \
  --warmup-runs 1 \
  --ctx-size 8192
```

List models:

```bash
python bench.py --provider openai --base-url http://localhost:8000 --list-models
python bench.py --provider ollama --base-url http://localhost:11434 --list-models
```

Send one test request without benchmarking:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --test-request
```

Streaming is also enabled by default for test mode:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --test-request
```

This prints only:

- `status: <code>`
- `response: <raw response text>` on success

Enable debug logs:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --list-models --debug
```

Write logs to a file instead of stderr:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --model llama3 --log-file perf-llm.log
```

Reduce log output to warnings only:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --model llama3 --quiet
```

Notes:

- Logging uses the standard Python logging interface.
- Default log level is `INFO`, `--debug` sets `DEBUG`, and `--quiet` sets `WARNING`.
- `--log-file` writes logs to a file instead of stderr.
- `--debug-content` logs request and response JSON content as one-line JSON at debug level.
- `--prompt-warmup` sets the warmup prompt and defaults to `ping`.
- Benchmark/test prompt defaults to `ping`.
- `--max-tokens` and `--temperature` default to `1024` and `1.0`.
- Use `--no-max-tokens` or `--no-temperature` to omit them from requests.
- `--test-request` sends one payload and prints the response without running a benchmark.
- Streaming is enabled by default for benchmark and test requests.
- Use `--no-stream` to disable streaming.
- `--ctx-size` is applied when supported by the target API.
- For Ollama, it is sent as `options.num_ctx`.
- If a setting is ignored by a provider, the tool prints a warning on stderr.

## Output fields

- `provider`: backend family used for the benchmark point, either `openai` or `ollama`.
- `model`: model identifier sent in the request body for that benchmark point.
- `thinking`: thinking level value sent with the request, or `-` when no thinking level is set.
- `conc`: number of requests issued concurrently in each round for that benchmark point.
- `ok`: number of requests that completed successfully for the benchmark point.
- `fail%`: percentage of failed requests over all requests in the benchmark point.
- `lat_min`: minimum end-to-end request latency observed across requests in the benchmark point.
- `lat_mean`: arithmetic mean end-to-end request latency across requests in the benchmark point.
- `lat_p50`: median end-to-end request latency across requests in the benchmark point.
- `lat_max`: maximum end-to-end request latency observed across requests in the benchmark point.
- `ttfb_min`: minimum streaming time-to-first-block observed across requests in the benchmark point.
- `ttfb_mean`: arithmetic mean streaming time-to-first-block across requests in the benchmark point.
- `ttfb_p50`: median streaming time-to-first-block across requests in the benchmark point.
- `ttfb_max`: maximum streaming time-to-first-block observed across requests in the benchmark point.
- `ttfc_min`: minimum streaming time-to-first-content observed across requests in the benchmark point.
- `ttfc_mean`: arithmetic mean streaming time-to-first-content across requests in the benchmark point.
- `ttfc_p50`: median streaming time-to-first-content across requests in the benchmark point.
- `ttfc_max`: maximum streaming time-to-first-content observed across requests in the benchmark point.
- `tps_mean`: arithmetic mean per-request output throughput in tokens per second across requests with token counts.
- `tps_wall`: aggregate output throughput computed as total output tokens divided by measured wall-clock time for the benchmark point.
- `tokens`: total number of output tokens reported by the backend across all successful requests in the benchmark point.

## Development

Install dev dependencies:

```bash
pip install '.[dev]'
```

Or with the Makefile:

```bash
make install-dev
```

## Test

Run all required checks:

```bash
PYTHON=.venv/bin/python PATH="$(pwd)/.venv/bin:$PATH" make test
```

This runs:

- Python compile check
- Ruff lint
- Ruff format check
- Pyright
