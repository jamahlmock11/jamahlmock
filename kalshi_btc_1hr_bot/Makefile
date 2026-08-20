.PHONY: setup backtest paper paper-once clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Run: source .venv/bin/activate"

backtest:
	$(PYTHON) backtest.py --synthetic --n-markets 100

paper-once:
	$(PYTHON) bot.py --paper --once

paper:
	$(PYTHON) bot.py --paper

clean:
	rm -rf $(VENV) __pycache__ */__pycache__
