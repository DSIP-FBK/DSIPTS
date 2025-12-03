from pathlib import Path

root = Path(__file__).parent
out = root / "LONG_DESCRIPTION.md"

out.write_text(
    (root / "README.md").read_text()
    + "\n\n"
    + (root / "../CHANGELOG.md").read_text()
)