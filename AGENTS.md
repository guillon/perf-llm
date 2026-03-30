# AGENTS.md

## Purpose

This repository contains `perf-llm`, a small benchmark tool for LLM servers exposed through:

- OpenAI-compatible `v1/chat/completions`
- Ollama HTTP API

Main script:

- `bench.py`

## Development rules

- Keep changes small and focused.
- Prefer minimal dependencies.
- Maintain Python 3.10+ compatibility.
- Update `README.md` when user-facing behavior changes.
- Use the existing CLI style in `bench.py`.
- Preserve support for both providers: `openai` and `ollama`.
- Keep debug output on stderr.
- Do not leave code unformatted.

## Testing

After every change, run:

```bash
PYTHON=.venv/bin/python PATH="$(pwd)/.venv/bin:$PATH" make test
```

This must pass before considering the work complete.

## Commit rules

Use clear commit messages.

Every commit message should include an `Assisted-By` trailer like this:

```text
Assisted-By: gpt-5.4 [agent=pi] [thinking=medium]
```

Replace values as appropriate for the active coding agent/model/session.

## Notes for coding agents

- If `.venv` exists, prefer its tools.
- Use `make test` as the required validation command.
- For OpenAI requests, use `/v1/chat/completions`.
- For model listing:
  - OpenAI-compatible: `/v1/models`
  - Ollama: `/api/tags`
