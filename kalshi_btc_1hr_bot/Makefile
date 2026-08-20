.PHONY: setup backtest paper paper-once test clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export PYTHONPATH := $(abspath ..)

setup:
	python3 -m venv $(VENV) || true
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt pytest
	@echo "Run: source .venv/bin/activate"

backtest:
	$(PYTHON) backtest.py --synthetic --n-markets 100

paper-once:
	$(PYTHON) bot.py --paper --once

paper:
	$(PYTHON) bot.py --paper

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(VENV) __pycache__ */__pycache__
