
.PHONY: build
build:
	bumpver update --patch
	uv pip install -e .
	NEW_VERSION=$$(python -c "from dsipts import __version__ as v; print(v)"); \
	echo "New version: $$NEW_VERSION"; \
	uv run sphinx-apidoc -o docs/source src/dsipts; \
	uv run sphinx-build -b html docs/source ../docs; \
	python generate_long_description.py; \
	uv build; \
	twine upload dist/*; \
	git commit -am "chore: bump version $$NEW_VERSION"; \
	git tag v$$NEW_VERSION
	