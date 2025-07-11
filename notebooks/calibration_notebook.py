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
# # Probe calibration and target transformation notebook
#
# This notebook is used to calibrate probes and transform targets from
# bregma-relative mm to manipulator coordinates
#
# # How to use this notebook
# 1. Set the mouse ID in the cell two below.
# 2. Set the path to the calibration files with the probe data.
# 3. Set the path to the target file.
# 4. Optionally set `fit_scale` to `True` if you want to fit the scale
# parameters as well. This is not recommended unless you have a good reason to
# do so. It does not guarantee that the error will be lower.
# 5. Run the next three cells to get the transformed targets, and see which
# targets are available
# 6. Configure the experiment by assigning each probe that you want to use to a
# target in the target file and specify the overshoot in µm. If you have
# targets that are not in the target file, you can specify them manually.
# 7. Run the next cell to fit the rotation parameters. If `verbose` is set to
# `True`, the mean and maximum error for each probe will be printed, as well as
# the predicted probe coordinates for each reticle coordinate with error for
# that coordinate.
# 8. Run the last cell to get the transformed targets in manipulator
# coordinates

# %%
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# %matplotlib inline
from aind_mri_utils.reticle_calibrations import (
    debug_parallax_and_manual_calibrations,
    debug_parallax_calibration,
    transform_bregma_to_probe,
    transform_probe_to_bregma,
)

logger = logging.getLogger(__name__)

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
# %%
# Set file paths and mouse ID here

# Calibration File with probe data
mouse_id = "786866"
reticle_used = "H"
basepath = Path("/mnt/vast/scratch/")
calibration_dir = basepath / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
# Target file with transformed targets
target_dir = basepath / f"ephys/persist/data/MRI/processed/{mouse_id}/"
# Calibration directories to use for parallax, with the latter ones taking
# priority
parallax_calibration_directories = []
# Calibration files to use for manual calibration, with the latter ones taking
# priority. All manual calibrations will take priority over parallax
# calibrations
manual_calibration_filenames = ["calibration_info_np2_2025_03_18T08_39_00.xlsx"]
# List of probes to ignore manual calibrations from
probes_to_ignore_manual = []
target_file = None  # target_dir / f"{mouse_id}_TransformedTargets.csv"

parallax_calibration_paths = [calibration_dir / f for f in parallax_calibration_directories]
manual_calibration_paths = [
    basepath / "ephys/persist/data/MRI/processed/786866/calibration_info_348_sws1_2025_07_10T11_00_00.xlsx"
]  # [calibration_dir / f for f in manual_calibration_filenames]

# Whether to fit the scale parameters as well. Does not guarantee that the
# error will be lower. If you have only a few data points, try setting this to
# True and False and compare.
fit_scale = False

# Whether to save the targets to a CSV file. If not None, the targets will be
# saved
save_path = None


# %%
# Reticle offsets and rotations
reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}
reticle_rotations = {"H": 0}


# %%
def _round_targets(target, probe_target):
    target_rnd = np.round(target, decimals=2)
    probe_target_and_overshoot_rnd = np.round(2000 * probe_target) / 2
    return target_rnd, probe_target_and_overshoot_rnd


# %%
if target_file:
    target_df = pd.read_csv(target_file)
    target_df = target_df.set_index("point")
else:
    target_df = None
# %% [markdown]
# ## Transformed targets
# print the transformed targets to see which targets are available
# %%
target_df

# %% [markdown]
# ## Configure experiment
# Assign each probe that you want to use to a target in the target file and
# specify the overshoot in µm. The format should be
# ```python
# targets_and_overshoots_by_probe = {
#     probe_id: (target_name, overshoot), # overshoot in µm
#     ...
# }
# ```
# Where each `probe_id` is the ID of a probe in the calibration file,
# `target_name` is the name of the target in the target file, and `overshoot`
# is the overshoot in µm.
#
# If you have targets that are not in the target file, you can specify them
# manually. The format should be
#
# ```python manual_bregma_targets_by_probe = { probe_id: [x, y, z], ...  } ```
# where `[x, y, z]` are the coordinates in mm.

# %%
# Set experiment configuration # here
#
# Names of targets in the target file and overshoots
# targets_and_overshoots_by_probe = {probe_id: (target_name, overshoot), ...}
# overshoot in µm
targets_and_overshoots_by_probe = {
    46110: ("Hole 1", 500),
    46100: ("Hole 2", 500),
}
# Targets in bregma-relative coordinates not in the target file
# manual_bregma_targets_by_probe = {probe_id: [x, y, z], ...}
# x y z in mm
manual_bregma_targets_by_probe = {
    "pipette": [1, 1, -1],  # in mm!
}


# %% [markdown]
# ## Fit rotation parameters
# Fit the rotation parameters and optionally the scale parameters. If `verbose`
# is set to `True`, the mean and maximum error for each probe will be printed,
# as well as the predicted probe coordinates for each reticle coordinate with
# error for that coordinate.
#
# Note: the reticle coordinates are in mm, as are the probe coordinates. The
# errors are in µm.
#
# The reticle coordinate displayed will NOT have the global offset applied.
# However, the scaling factor will have been applied.
# %%
# Calculate the rotation parameters
reticle_offset = reticle_offsets[reticle_used]
reticle_rotation = reticle_rotations[reticle_used]
if len(manual_calibration_paths) > 0:
    (
        combined_cal_by_probe,
        R_reticle_to_bregma,
        t_reticle_to_bregma,
        combined_pairs_by_probe,
        errs_by_probe,
    ) = debug_parallax_and_manual_calibrations(
        manual_calibration_paths,
        parallax_calibration_paths,
        probes_to_ignore_manual,
        find_scaling=fit_scale,
    )
else:
    (
        combined_cal_by_probe,
        R_reticle_to_bregma,
        combined_pairs_by_probe,
        errs_by_probe,
    ) = debug_parallax_calibration(
        parallax_calibration_paths[0],
        reticle_offset,
        reticle_rotation,
        find_scaling=fit_scale,
    )
    t_reticle_to_bregma = reticle_offset
# %% [markdown]
# ## Probe targets in manipulator coordinates
# Get the transformed targets in manipulator coordinates using the fitted
# calibration parameters and the experiment configuration set in the previous
# cells.


# %%
# Print the transformed targets in manipulator coordinates

# Combine the targets and overshoots with the manual targets

dims = ["ML (mm)", "AP (mm)", "DV (mm)"]
combined_targets_and_overshoots_by_probe = {}
for probe, (target_name, overshoot) in targets_and_overshoots_by_probe.items():
    if target_df is None or target_name not in target_df.index:
        logger.warning(f"Target {target_name} not found in target file")
        continue
    target = target_df.loc[target_name, dims].to_numpy().astype(np.float64)
    overshoot_arr = np.array([0, 0, overshoot / 1000])
    combined_targets_and_overshoots_by_probe[probe] = (
        target_name,
        target,
        overshoot_arr,
    )
for probe, target in manual_bregma_targets_by_probe.items():
    target_arr = np.array(target)
    combined_targets_and_overshoots_by_probe[probe] = (
        "Manual",
        target_arr,
        np.array([0, 0, 0]),
    )

zaber_dims = ["X (µm)", "Y (µm)", "Z (µm)"]
output_dims = ["Probe", "Target", "Overshoot (µm)"] + dims + zaber_dims
cols = {}
for probe, (
    target_name,
    target,
    overshoot,
) in combined_targets_and_overshoots_by_probe.items():
    if probe not in combined_cal_by_probe:
        logger.warning(f"Probe {probe} not in calibration files")
        continue
    cols.setdefault("Probe", []).append(probe)
    cols.setdefault("Target", []).append(target_name)
    cols.setdefault("Overshoot (µm)", []).append(1000 * overshoot[2])
    R, t, _ = combined_cal_by_probe[probe]
    probe_target = transform_bregma_to_probe(target, R, t)
    probe_target_and_overshoot = probe_target + overshoot
    target_rnd, probe_target_and_overshoot_rnd = _round_targets(target, probe_target_and_overshoot)
    final_target_bregma = np.round(transform_probe_to_bregma(probe_target_and_overshoot, R, t), 3)
    for dim, dim_val in zip(dims, final_target_bregma):
        cols.setdefault(dim, []).append(np.round(dim_val, 3))
    for dim, dim_val in zip(zaber_dims, probe_target_and_overshoot_rnd):
        cols.setdefault(dim, []).append(dim_val)

df_transformed = pd.DataFrame.from_dict(cols).set_index("Probe").sort_index()
# %%
# Print the targets (AP, ML, DV) and their transformed newscale coordinates (X Y Z)
df_transformed
# %%
if save_path is not None:
    df_transformed.to_csv(save_path)
# %%
