# ======================================================================
# Settings
#
# Paths are plain strings. Do not wrap them in Path(...): that happens
# below, and doing it twice creates a folder name containing the literal
# text "Path(...)".
# ======================================================================

DATA_DIR = "/content/drive/MyDrive/ASD_Spectra/Atwater_2024"   # searched recursively
OUTPUT_DIR = "/content/output"                                 # CSVs written here

# Processing
SPLICE_MODE = "swir1_anchor"        # "swir1_anchor" | "vnir_anchor" | "none"
CORRECT_SWIR2 = False               # 1800 nm splice; see README
MANUAL_IT_FACTOR = None             # override the integration-time factor
Z_THRESHOLD = 3.5                   # outlier flag level
SNR_MIN = 5.0                       # 0 disables the data-driven noise mask
PAIR_GROUP_BY = "date"              # "date" pairs across stems such as Au and Ad

# Exclusions: file indices to drop, keyed by block id or by the stable block key
# shown in the block table, e.g. {"240411Au:10-19": [7], 3: [0, 9]}. Prefer the key
# when scanning several date folders at once: block ids shift when folders are added.
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
# Notes on the two paths
#
# DATA_DIR in Colab: use the file browser on the left, right-click the
#   folder and choose "Copy path" rather than typing it. A folder shared
#   with you does not appear under MyDrive until you add a shortcut to
#   your own Drive.
# DATA_DIR locally: any path, including one starting with ~, for example
#   "~/Documents/data/spectra_asd/Atwater_2024"
#
# OUTPUT_DIR: keep it off Drive while iterating, since writing there is
#   slow. Copy the results across once at the end:
#     !cp -r /content/output "/content/drive/MyDrive/ASD_Spectra/output"
#
# Speed: reading hundreds of small files over the Drive mount is slow.
#   For a whole season, copy to the runtime's local disk first and point
#   DATA_DIR at the copy:
#     !cp -r "/content/drive/MyDrive/ASD_Spectra/Atwater_2024" /content/data
# ----------------------------------------------------------------------

DATA_DIR = Path(DATA_DIR).expanduser()
OUTPUT_DIR = Path(OUTPUT_DIR).expanduser()

if not DATA_DIR.exists():
    parent = DATA_DIR.parent
    nearby = (sorted(p.name for p in parent.iterdir())[:30] if parent.exists()
              else f"(the parent {parent} does not exist either)")
    raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}\n"
                            f"Contents of {parent}: {nearby}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = asd.scan_folder(DATA_DIR)
WL = data.wavelength
print(f"{len(data)} spectra, {WL[0]:.0f}-{WL[-1]:.0f} nm at {WL[1] - WL[0]:.0f} nm")
