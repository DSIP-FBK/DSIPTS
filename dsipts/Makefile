.PHONY: build
build:
	uv run sphinx-build -b html docs/source docs/build/html
	python generate_long_description.py
	uv build