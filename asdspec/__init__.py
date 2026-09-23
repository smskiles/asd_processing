"""Processing of ASD FieldSpec spectra into albedo and reflectance.

Typical use::

    import asdspec as asd

    data = asd.scan_folder("path/to/spectra")
    blocks = asd.find_blocks(data)
    pairs, orphans = asd.auto_pairs(blocks, asd.detect_reflectance_stems(blocks))
    results = [asd.compute_albedo(data, blocks[r], blocks[t]) for r, t in pairs]
"""

from .albedo import BANDS, broadband_albedo, compute_albedo, splice_qc_table
from .corrections import (DEFAULT_MASK, NOMINAL_SPLICE_FACTOR, SPLICE_FACTOR_OK,
                          integration_time_scale, mask_fill, snr_mask,
                          splice_correct, splice_step)
from .dataset import (Dataset, auto_pairs, block_table, detect_reflectance_stems,
                      find_blocks, scan_folder)
from .io import read_asd
from .irradiance import ModeledIrradiance, solar_zenith
from .qc import outlier_scores, qc_all, qc_block
from .reflectance import choose_panel, compute_reflectance, panel_drift

__version__ = "1.0.0"

__all__ = [
    "BANDS", "DEFAULT_MASK", "Dataset", "ModeledIrradiance", "NOMINAL_SPLICE_FACTOR",
    "SPLICE_FACTOR_OK", "auto_pairs", "block_table", "broadband_albedo", "choose_panel",
    "compute_albedo", "compute_reflectance", "detect_reflectance_stems", "find_blocks",
    "integration_time_scale", "mask_fill", "outlier_scores", "panel_drift", "qc_all",
    "qc_block", "read_asd", "scan_folder", "snr_mask", "solar_zenith", "splice_correct",
    "splice_qc_table", "splice_step",
]
