# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The perf-llm Project Authors

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help install install-dev install-viz lint format format-check typecheck test

help:
	@echo "Targets:"
	@echo "  install       Install project"
	@echo "  install-dev   Install project with dev dependencies"
	@echo "  install-viz   Install project with visualization dependencies"
	@echo "  lint          Run Ruff checks"
	@echo "  format        Format code with Ruff"
	@echo "  format-check  Check formatting with Ruff"
	@echo "  typecheck     Run Pyright"
	@echo "  test          Run local validation checks"

install:
	$(PIP) install .

install-dev:
	$(PIP) install '.[dev]'

install-viz:
	$(PIP) install '.[viz]'

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	pyright

test:
	$(PYTHON) -m py_compile bench.py
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	pyright
