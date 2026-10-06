# Environment setup. Runs as-is in Colab; falls back to a local clone otherwise.
import sys
from pathlib import Path

REQUIRED_ASDSPEC = "1.4.0"          # the version this notebook is written against
REPO = "smskiles/asd_processing"

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # --force-reinstall defeats pip's cache, which otherwise keeps serving the
    # build from the first install of the session and hides a pushed update.
    !pip install -q --force-reinstall --no-deps git+https://github.com/{REPO}.git
    from google.colab import drive
    drive.mount("/content/drive")
else:
    # Running from a clone: make the package importable without installing it.
    # Assumes this notebook sits in notebooks/ inside the repository.
    sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import asdspec as asd
from asdspec import plots
from asdspec.irradiance import TRAPZ


def _older(found, required):
    return tuple(int(p) for p in found.split(".")) < tuple(int(p) for p in required.split("."))


if _older(asd.__version__, REQUIRED_ASDSPEC):
    raise ImportError(
        f"This notebook needs asdspec {REQUIRED_ASDSPEC} or newer, but {asd.__version__} "
        f"is loaded, so calls below would fail with a confusing TypeError instead.\n\n"
        f"In Colab: push the latest package to GitHub, then run\n"
        f"    !pip install -q --force-reinstall --no-deps git+https://github.com/{REPO}.git\n"
        f"and use Runtime > Restart session. Python keeps the already-imported module in\n"
        f"memory, so reinstalling without restarting changes nothing.\n\n"
        f"From a clone: git pull, then restart the kernel.\n"
        f"Loaded from: {Path(asd.__file__).parent}"
    )

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 400)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.facecolor": "white"})
print(f"asdspec {asd.__version__}   ({'Colab' if IN_COLAB else 'local'})")
