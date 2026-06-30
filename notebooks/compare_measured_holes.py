# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare calibrated-probe hole measurements to the implant fit
#
# Ingests a measured-holes spreadsheet (calibrated probe, manipulator µm) and
# compares each hole to the shaft bottom-opening location predicted by the
# implant fitting process. Both are expressed in bregma-relative RAS (mm).
#
# Predicted = holes YAML (CAD LPS) -> `*_implant_fit.h5` -> `*_com_plane.h5`.
# Measured  = probe µm -> reticle calibration -> bregma.
#
# No chemical-shift correction is applied: the implant segmentation and the
# headframe fiducials are both referenced to vaseline.

# %%
# Use the ipympl widget backend for interactive (rotatable) figures.
# %matplotlib widget
from pathlib import Path

import pandas as pd

from aind_mri_targeting.measured_holes import (
    compare_holes,
    load_measured_holes,
    measured_holes_bregma,
    plot_hole_errors,
    plot_hole_errors_3d,
    predicted_hole_bottoms_bregma,
)

# %%
# Configuration
mouse_id = "837229"
# NOTE: set the reticle actually used for this calibration. The reticle
# offset/rotation only matter for parallax-only calibrations (manual
# calibrations carry their own reticle metadata).
reticle = "H"

base_dir = Path("/mnt/vast/scratch/")
mouse_folder = base_dir / "ephys/persist/data/MRI/processed" / mouse_id
calibration_dir = base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations"

# Calibrations are layered to cover the full set of probes; later entries take
# priority, manual beats parallax. Bare names resolve against calibration_dir.
manual_files = []
parallax_dirs = [
    "log_20260610_180434",
    "log_20260611_150416",
    "log_20260611_172932",
]
# Also fold in the calibration named in the spreadsheet's Calibration column?
include_calibration_column = False
probes_to_ignore_manual = []

measured_holes_file = mouse_folder / f"{mouse_id}_measured_holes_CK_2026-06-12.xlsx"
implant_fit_h5 = mouse_folder / f"{mouse_id}_implant_fit.h5"
com_plane_h5 = mouse_folder / f"{mouse_id}_com_plane.h5"
holes_yml = Path("/home/galen.lynch/Documents/Code/aind-low-point/scratch/0283-300-04.holes.yml")

# %% [markdown]
# ## Predicted hole shaft bottoms (bregma RAS)

# %%
predicted = predicted_hole_bottoms_bregma(holes_yml, implant_fit_h5, com_plane_h5)
pd.DataFrame(predicted, index=["ML", "AP", "DV"]).T.sort_index()

# %% [markdown]
# ## Measured hole locations (bregma RAS)

# %%
measured_df = load_measured_holes(measured_holes_file)
measured = measured_holes_bregma(
    measured_df,
    calibration_dir,
    manual_files=manual_files,
    parallax_dirs=parallax_dirs,
    include_calibration_column=include_calibration_column,
    probes_to_ignore_manual=probes_to_ignore_manual,
    reticle=reticle,
)
pd.DataFrame(measured, index=["ML", "AP", "DV"]).T.sort_index()

# %% [markdown]
# ## Comparison
# Signed error components are measured - predicted; `error_mm` is the 3D
# magnitude.

# %%
comparison = compare_holes(measured, predicted)
comparison

# %%
print(f"Mean 3D error:   {comparison['error_mm'].mean():.3f} mm")
print(f"Median 3D error: {comparison['error_mm'].median():.3f} mm")
print(f"Max 3D error:    {comparison['error_mm'].max():.3f} mm")
for d in ["ML", "AP", "DV"]:
    print(f"  {d} bias (mean signed): {comparison[f'err_{d}'].mean():+.3f} mm")

# %% [markdown]
# ## Error plot (summary)

# %%
fig = plot_hole_errors(comparison, title=f"{mouse_id}: measured vs implant-fit holes")
fig

# %% [markdown]
# ## Interactive 3D error plot
# Black circles are implant-fit predictions, triangles are probe measurements,
# segments are the error (colored by 3D magnitude). Drag to rotate. Set
# `arrow_scale > 1` to exaggerate the sub-mm offsets.

# %%
fig3d = plot_hole_errors_3d(comparison, arrow_scale=1.0, title=f"{mouse_id}: hole errors (bregma RAS)")
fig3d

# %%
# Optionally save the comparison table
save_path = None  # mouse_folder / f"{mouse_id}_hole_comparison.csv"
if save_path is not None:
    comparison.to_csv(save_path)
