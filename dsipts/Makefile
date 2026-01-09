.PHONY: build
build:
	bumpver update --patch 
	uv run sphinx-apidoc -o docs/source src/dsipts
	uv run sphinx-build -b html docs/source ../docs
	python generate_long_description.py
	uv build