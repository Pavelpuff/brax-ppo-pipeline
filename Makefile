.PHONY: help setup format lint train clean

PYTHON := uv run python

help:
	@echo "Available commands:"
	@echo "  make setup   - Install project dependencies via uv"
	@echo "  make format  - Auto-format code and fix linting errors (ruff)"
	@echo "  make lint    - Check code for linting and formatting issues"
	@echo "  make train   - Train the PPO agent on Brax environments"
	@echo "  make clean   - Remove cache, temporary files, and wandb logs"

setup:
	uv sync --all-extras

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

train:
	$(PYTHON) -m src.train --env_name ant --ent_coef 0.01 --seed 42

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf wandb/