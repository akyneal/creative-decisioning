.PHONY: install run test clean

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
JUPYTER := $(VENV)/bin/jupyter
PYTEST := $(VENV)/bin/pytest

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(JUPYTER) nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		--output creative_decisioning_case_study.ipynb \
		notebooks/creative_decisioning_case_study.ipynb

test:
	$(PYTEST) tests/ -v

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
