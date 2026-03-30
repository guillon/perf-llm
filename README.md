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

Default prompt:

```text
Hello, tell me a joke.
```

Benchmark an OpenAI-compatible endpoint:

```bash
python bench.py \
  --provider openai \
  --base-url http://localhost:8000 \
  --model my-model \
  --concurrency 1,2,4 \
  --thinking-level low,medium,high \
  --warmup-runs 1
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

Enable debug logs:

```bash
python bench.py --provider ollama --base-url http://localhost:11434 --list-models --debug
```

Notes:

- `--max-tokens` and `--temperature` default to `1024` and `1.0`.
- Use `--no-max-tokens` or `--no-temperature` to omit them from requests.
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
