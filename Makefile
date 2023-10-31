.PHONY: all test clean

setup: env.yml
	conda env create -f env.yml
	conda activate FLIGHTED

format:
	black .
	isort .

test:
	python -m pytest -m 'not slow' .

test-all:
	python -m pytest .

test-new:
	python -m pytest -m new .
