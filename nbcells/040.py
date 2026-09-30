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

# Manual overrides. All three are keyed by the stable block key shown in the
# block table (folder/stem:first-last), or by block id. Prefer the key: block ids
# shift when folders are added or removed. See the notes at the end of the cell.
ROLE_OVERRIDES = {}                 # {"240401A:0-9": "reference", "240401A:10-19": "target"}
MANUAL_PAIRS = []                   # [("240509A2:0-8", "240509A3:0-19")]
DROP_PAIRS = []                     # [("240522a5:0-9", "240522a5:20-29")]

# Exclusions: file indices to drop, keyed the same way,
# e.g. {"240411Au:10-19": [7], 3: [0, 9]}
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
#
# ----------------------------------------------------------------------
# Fixing blocks the automatic rules get wrong
#
# Run the notebook once, read the block table in section 2, then come back
# here. Work in this order, since each step changes what the next one sees.
#
# 1. ROLE_OVERRIDES, when a block has the wrong role, or when one stem was
#    split into ragged pieces because its SWIR-to-visible ratio straddles the
#    threshold. Role is one of the fields blocks are cut on, so an override
#    also re-cuts the blocks. Give the index range you want each role to span:
#        ROLE_OVERRIDES = {"240401A:0-9": "reference",
#                          "240401A:10-19": "target"}
#    A bare stem sets the whole stem: {"240509i": "reference"}
#    A folder prefix disambiguates a stem that appears twice:
#        {"morning/240411Au:0-9": "reference"}
#
# 2. MANUAL_PAIRS, when two blocks are correctly labelled but were not paired,
#    for example a reference and target that carry different stems or sit in
#    different folders:
#        MANUAL_PAIRS = [("240509A2:0-8", "240509A3:0-19")]
#    Each entry is (reference_key, target_key). A manual pair replaces any
#    automatic pair on the same target.
#
# 3. DROP_PAIRS, to remove an automatic pair you do not want:
#        DROP_PAIRS = [("240522a5:0-9", "240522a5:20-29")]
#
# Copy keys verbatim from the block table. A key that matches nothing raises
# an error naming the problem rather than failing silently later.
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

data = asd.scan_folder(DATA_DIR, role_overrides=ROLE_OVERRIDES)
WL = data.wavelength
print(f"{len(data)} spectra, {WL[0]:.0f}-{WL[-1]:.0f} nm at {WL[1] - WL[0]:.0f} nm")
