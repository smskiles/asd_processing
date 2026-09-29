# Environment setup. Runs as-is in Colab; falls back to a local clone otherwise.
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !pip install -q git+https://github.com/smskiles/asd_processing.git
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

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 400)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.facecolor": "white"})
print(f"asdspec {asd.__version__}   ({'Colab' if IN_COLAB else 'local'})")
