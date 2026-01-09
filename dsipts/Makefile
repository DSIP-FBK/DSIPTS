
.PHONY: build
build:
	bumpver update --patch
	uv pip install -e .
	NEW_VERSION=$$(python -c "from dsipts import __version__ as v; print(v)"); \
	echo "New version: $$NEW_VERSION"; \

	git commit -am "chore: bump version $$NEW_VERSION"
	git tag v$$NEW_VERSION
	