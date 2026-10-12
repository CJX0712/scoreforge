.PHONY: install lint format test demo ci clean

install:
	python -m pip install -r requirements.txt -r requirements.lock.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	python -m pytest tests -q

demo:
	python -m scoreforge.examples.run_demo --fast

ci: lint test demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
