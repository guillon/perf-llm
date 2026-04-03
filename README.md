# perf-llm

`perf-llm` benchmarks LLM servers exposed through:

- OpenAI-compatible `v1/chat/completions`
- OpenAI Codex Responses API `/codex/responses`
- Ollama HTTP API `/api/generate`

It varies:

- concurrency
- thinking level

And reports:

- latency statistics: min, mean, median, max
- streaming first-block latency (ttfb): min, mean, median, max
- streaming first-token latency (ttft): min, mean, median, max
- throughput in tokens/sec, average and wall-clock throughput
- CSV summary output by default

## Install

Python 3.10+ is required.

Install the project:

```bash
pip install .
```

See the [Development](#development) section for development mode.

## Quick start

Default benchmark/test prompt:

```text
Generate a 256 words text.
```

Default warmup prompt:

```text
ping
```

Benchmark an OpenAI-compatible endpoint:

```bash
perf-llm bench \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --thinking-level low,medium,high \
  --warmup-runs 1 \
  --prompt-warmup ping
```

By default, a CSV summary is written to:

```text
csv/bench-YYYYMMDD-HHMMSS.csv
```

Streaming is enabled by default in benchmark and test mode.
Disable it when needed:

```bash
perf-llm bench \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --no-stream
```

Omit a setting from requests when needed:

```bash
perf-llm bench \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --no-temperature \
  --no-max-tokens
```

Use the MLX-compatible OpenAI variant when needed:

```bash
perf-llm bench \
  --provider openai \
  --api-variant mlx \
  --base-url http://localhost:8000 \
  --model my-model \
  --thinking-level high
```

Benchmark the OpenAI Codex Responses API with Pi authentication:

```bash
perf-llm bench \
  --provider openai-codex \
  --model gpt-5.4 \
  --auth-with pi
```

Benchmark an Ollama endpoint:

Example against an Ollama server:

```bash
perf-llm bench \
  --provider ollama \
  --base-url http://localhost:11434 \
  --model llama3 \
  --concurrency 1,2,4 \
  --warmup-runs 1 \
  --ctx-size 8192
```

List models:

```bash
perf-llm list --provider openai --base-url http://localhost:8000
perf-llm list --provider ollama --base-url http://localhost:11434
```

Authenticate to an OpenAI-compatible endpoint with an OAuth access token:

```bash
perf-llm test \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --oauth-access-token "$ACCESS_TOKEN"
```

Send one test request without benchmarking:

```bash
perf-llm test \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model
```

Streaming is also enabled by default for test mode:

```bash
perf-llm test \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model
```

This prints only:

- `status: <code>`
- `response: <raw response text>` on success

Enable debug logs:

```bash
perf-llm list --provider ollama --base-url http://localhost:11434 --debug
```

Write logs to a file instead of stderr:

```bash
perf-llm bench --provider ollama --base-url http://localhost:11434 --model llama3 --log-file perf-llm.log
```

Reduce log output to warnings only:

```bash
perf-llm bench --provider ollama --base-url http://localhost:11434 --model llama3 --quiet
```

Notes:

- `--api-variant default` uses OpenAI-style `reasoning_effort` for `provider=openai` when a thinking level is set.
- `--api-variant mlx` uses `chat_template_kwargs.enable_thinking` and `chat_template_kwargs.reasoning_effort` for `provider=openai`.
- For `provider=ollama`, `--api-variant` must stay `default`.
- If thinking level is omitted or set to `default`, it is not sent in the request.
- If thinking level is set to `none`, mlx disables thinking explicitly and Ollama sends the value through `--thinking-key`.
- OpenAI-compatible authentication supports either `--api-key` or `--oauth-access-token`.
- Use `--auth-with pi` to load a token from `~/.pi/agent/auth.json`.
- If both are provided, `--oauth-access-token` takes precedence.
- `openai-codex` uses the Codex Responses API, requires streaming mode, and defaults `--base-url` to `https://chatgpt.com/backend-api`.
- `openai-codex` does not support the `list` subcommand.
- `openai-codex` does not support `--max-tokens`; use `--no-max-tokens` or leave it unset.
- `openai-codex` does not support `--temperature`; use `--no-temperature` or leave it unset.
- Logging uses the standard Python logging interface.
- Default log level is `INFO`, `--debug` sets `DEBUG`, and `--quiet` sets `WARNING`.
- `--log-file` writes logs to a file instead of stderr.
- `--debug-content` logs request content and raw response content as one-line debug output.
- `--csv-file` writes the summary CSV to an explicit path.
- `--no-csv` disables the default summary CSV output.
- `--prompt-warmup` sets the warmup prompt and defaults to `ping`.
- Benchmark/test prompt defaults to `ping`.
- `--max-tokens` and `--temperature` default to `1024` and `1.0`.
- Use `--no-max-tokens` or `--no-temperature` to omit them from requests.
- Use the `test` subcommand to send one payload and print the response without running a benchmark.
- Streaming is enabled by default for benchmark and test requests.
- Use `--no-stream` to disable streaming.
- `--ctx-size` is applied when supported by the target API.
- For Ollama, it is sent as `options.num_ctx`.
- If a setting is ignored by a provider, the tool prints a warning on stderr.

## Output fields

- `provider`: backend family used for the benchmark point, either `openai` or `ollama`.
- `variant`: provider-specific API flavor used for the benchmark point, such as `default` or `mlx`.
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
- `ttft_min`: minimum streaming time-to-first-token observed across requests in the benchmark point.
- `ttft_mean`: arithmetic mean streaming time-to-first-token across requests in the benchmark point.
- `ttft_p50`: median streaming time-to-first-token across requests in the benchmark point.
- `ttft_max`: maximum streaming time-to-first-token observed across requests in the benchmark point.
- `tps_mean`: arithmetic mean per-request output throughput in tokens per second across requests with token counts.
- `tps_wall`: aggregate output throughput computed as total output tokens divided by measured wall-clock time for the benchmark point.
- `tokens`: total number of output tokens reported by the backend across all successful requests in the benchmark point.

## Development

Install dev dependencies in editable mode, for instance:

```bash
pip install -e '.[dev]'
```

Or with the Makefile:

```bash
make install-dev
```

## Test

Run all required checks:

```bash
PYTHON=.venv/bin/python3 PATH="$(pwd)/.venv/bin:$PATH" make test
```

This runs:

- Python compile check
- Ruff lint
- Ruff format check
- Pyright
