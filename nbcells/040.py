# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATA_DIR = "../example_data"        # folder of ASD files, searched recursively
OUTPUT_DIR = "../output"

# Processing
SPLICE_MODE = "swir1_anchor"        # "swir1_anchor" | "vnir_anchor" | "none"
CORRECT_SWIR2 = False               # 1800 nm splice; see README
MANUAL_IT_FACTOR = None             # override the integration-time factor
Z_THRESHOLD = 3.5                   # outlier flag level
SNR_MIN = 5.0                       # 0 disables the data-driven noise mask

# Exclusions: block_id -> file indices to drop, e.g. {3: [7], 4: [0, 9]}
EXCLUDE = {}
AUTO_EXCLUDE_FLAGGED = False

# Irradiance weighting for broadband albedo
IRRADIANCE_SOURCE = "auto"          # "auto" | "measured" | "modeled"
MODELED_IRRADIANCE_FILE = None      # None looks for a *irrad*.csv near the data
MANUAL_SZA = None                   # degrees; or give the site below instead
SITE_LAT = None                     # e.g. 40.60
SITE_LON = None                     # e.g. -111.60, west negative
UTC_OFFSET_HOURS = None             # e.g. -6. ASD headers store local wall-clock time.

# Reflectance
PANEL_REFLECTANCE = 0.99            # flat placeholder; see README
PANEL_FILE = None                   # CSV: wavelength_nm, reflectance
REFLECTANCE_STEMS = []              # empty means auto-detect
# ----------------------------------------------------------------------

DATA_DIR, OUTPUT_DIR = Path(DATA_DIR), Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In Colab, upload a zip instead:
#   from google.colab import files, zipfile, io
#   up = files.upload(); DATA_DIR = Path('/content/asd_data'); DATA_DIR.mkdir(exist_ok=True)
#   for name, blob in up.items():
#       if name.endswith('.zip'):
#           zipfile.ZipFile(io.BytesIO(blob)).extractall(DATA_DIR)
#       else:
#           (DATA_DIR / name).write_bytes(blob)

data = asd.scan_folder(DATA_DIR)
WL = data.wavelength
print(f"{len(data)} spectra, {WL[0]:.0f}-{WL[-1]:.0f} nm at {WL[1] - WL[0]:.0f} nm")
