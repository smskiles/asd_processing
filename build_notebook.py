"""Assemble notebooks/asd_processing.ipynb from the cell sources in nbcells/."""
import json
from pathlib import Path

cells = []
for path in sorted(Path("nbcells").iterdir()):
    source = path.read_text().rstrip("\n").splitlines(keepends=True)
    if path.suffix == ".md":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
    else:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": source})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path("notebooks/asd_processing.ipynb")
out.write_text(json.dumps(notebook, indent=1))
print(f"{out}: {len(cells)} cells")
