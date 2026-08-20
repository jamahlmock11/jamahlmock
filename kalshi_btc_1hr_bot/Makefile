.PHONY: setup backtest paper paper-once test clean dashboard

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export PYTHONPATH := $(abspath ..)
DASHBOARD_PORT ?= 8090

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

live:
	$(PYTHON) bot.py --live --max-trade-usd 1

live-once:
	$(PYTHON) bot.py --live --max-trade-usd 1 --once

live-loop:
	$(PYTHON) bot.py --live --max-trade-usd 1

dashboard:
	DASHBOARD_PORT=$(DASHBOARD_PORT) $(PYTHON) dashboard.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__
