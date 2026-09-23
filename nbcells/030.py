# In Colab, install the package and skip the sys.path line:
#   !pip install -q git+https://github.com/<user>/<repo>.git
# Running from a clone, make the package importable instead:
import sys
from pathlib import Path

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
print(f"asdspec {asd.__version__}")
