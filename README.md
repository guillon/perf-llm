# perf-llm

`perf-llm` benchmarks LLM servers exposed through:

- OpenAI-compatible `v1/chat/completions`
- Ollama HTTP API

It varies:

- concurrency
- thinking level

And reports:

- latency statistics
- throughput in tokens/sec

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

Enable streaming in benchmark or test mode:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --stream
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

Enable streaming for test mode only:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --test-request \
  --stream
```

This prints only:

- `status: <code>`
- `response: <raw response text>` on success

Enable debug logs:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --list-models --debug
```

Reduce log output to warnings only:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --model llama3 --quiet
```

Notes:

- Logging uses the standard Python logging interface.
- Default log level is `INFO`, `--debug` sets `DEBUG`, and `--quiet` sets `WARNING`.
- `--debug-content` logs request and response JSON content at debug level.
- `--prompt-warmup` sets the warmup prompt and defaults to `ping`.
- Benchmark/test prompt defaults to `ping`.
- `--max-tokens` and `--temperature` default to `1024` and `1.0`.
- Use `--no-max-tokens` or `--no-temperature` to omit them from requests.
- `--test-request` sends one payload and prints the response without running a benchmark.
- `--stream` enables streaming mode for both benchmark and test requests.
- `--ctx-size` is applied when supported by the target API.
- For Ollama, it is sent as `options.num_ctx`.
- If a setting is ignored by a provider, the tool prints a warning on stderr.

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
