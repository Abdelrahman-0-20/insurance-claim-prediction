PYTHON := .venv/bin/python
PIP := .venv/bin/pip
CONFIG := configs/config.yaml
export MPLCONFIGDIR := .mplconfig

.PHONY: setup train evaluate test app clean

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	mkdir -p data/raw artifacts/models artifacts/reports artifacts/plots artifacts/explanations
	@test -f data/raw/claims.csv || (test -f "Insurance claims data.csv" && cp "Insurance claims data.csv" data/raw/claims.csv) || true
	@echo "Setup complete."

train:
	$(PYTHON) -m src.models.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluation.metrics --config $(CONFIG)
	$(PYTHON) -m src.evaluation.explain --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

app:
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf artifacts/models/* artifacts/reports/* artifacts/plots/* artifacts/explanations/*
