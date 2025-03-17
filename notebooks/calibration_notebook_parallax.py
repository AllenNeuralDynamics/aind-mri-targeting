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
# 2. Set the path to the calibration file with the probe data.
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
# the predicted probe coordinates for each reticle coordinate with error for that coordinate.
# 8. Run the last cell to get the transformed targets in manipulator coordinates

import os

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %matplotlib inline
from aind_mri_utils import reticle_calibrations as rc
from aind_mri_utils.reticle_calibrations import (
    debug_manual_calibration,
    debug_parallax_calibration,
    read_parallax_calibration,
    read_reticle_calibration,
)

# %%
# Set file paths and mouse ID here

# Calibration File with probe data
mouse_id = "760332"
calib_date = "20241203"
reticle_used = "H"
basepath = Path(r"Z:")
parallax_debug_dir = Path(
    r"C:\Users\svc_aind_ephys\Documents\Code\parallax\debug"
)
calib_folder = [
    f
    for f in os.listdir(parallax_debug_dir)
    if f.startswith("log_" + str(calib_date))
]
calib_dir = Path(
    r"C:\Users\svc_aind_ephys\Documents\Code\parallax\debug\\"
    + str(calib_folder[-1])
)

manual_calibration_file = r"path/to/file.xlsx"

# Target file with transformed targets
target_dir = basepath / f"ephys/persist/data/MRI/processed/{mouse_id}"
# target_file = target_dir / f"{mouse_id}_TransformedTargets.csv"
target_file = target_dir / f"{mouse_id}_TransformedTargets.csv"

# Whether to fit the scale parameters as well. This is not recommended unless
# you have a good reason to do so.  Does not guarantee that the error will be
# lower
fit_scale = True

# Whether to print the mean and maximum error for each probe and the predicted
# probe coordinates for each reticle coordinate with error for that coordinate
verbose = True

reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}

# List of probes where the calibration should be taken from parallax instead of
# manual. If empty, probes calibrated both by parallax and manual will use the
# manual calibration
probes_to_ignore_manual_calibration = []


# %%
def _round_targets(target, probe_target):
    target_rnd = np.round(target, decimals=2)
    probe_target_and_overshoot_rnd = np.round(2000 * probe_target) / 2
    return target_rnd, probe_target_and_overshoot_rnd


# %%
target_df = pd.read_csv(target_file)
target_df = target_df.set_index("point")
# %% [markdown]
# ## Transformed targets
# print the transformed targets to see which targets are available
# %%
print(target_df)

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
# Where each `probe_id` is the ID of a probe in the calibration file, `target_name`
# is the name of the target in the target file, and `overshoot` is the overshoot
# in µm.
#
# If you have targets that are not in the target file, you can specify them
# manually. The format should be
#
# ```python
# manual_bregma_targets_by_probe = {
#     probe_id: [x, y, z],
#     ...
# }
# ```
# where `[x, y, z]` are the coordinates in mm.
# %%
# Set experiment configuration here

# Names of targets in the target file and overshoots
# targets_and_overshoots_by_probe = {probe_id: (target_name, overshoot), ...}
# overshoot in µm
targets_and_overshoots_by_probe = {
    50209: ("CCpst", 0),
    50205: ("GenFacCran2", 0),
    50197: ("GenFacCran2", 0),
}
# Targets in bregma-relative coordinates not in the target file
# manual_bregma_targets_by_probe = {probe_id: [x, y, z], ...}
# x y z in mm
manual_bregma_targets_by_probe = {
    # 46110: [0, 0, 0], # in mm!
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
manips_used = list(
    set(targets_and_overshoots_by_probe.keys()).union(
        manual_bregma_targets_by_probe.keys()
    )
)

print("Calibration Directory: " + str(calib_dir))

adjusted_pairs_by_probe = dict()
global_offset = reticle_offsets[reticle_used]
global_rotation_degrees = 0

print("Loading parallax calibrations...")
(
    cal_by_probe_parallax,
    adjusted_pairs_by_probe_parallax,
    errs_by_probe_parallax,
) = debug_parallax_calibration(
    calib_dir,
    global_offset,
    global_rotation_degrees,
    verbose=verbose,
    find_scaling=fit_scale,
)

print("Loading manual calibrations...")
(
    cal_by_probe_manual,
    adjusted_pairs_by_probe_manual,
    global_offset_manual,
    global_rotation_degrees_manual,
    errs_by_probe_manual,
) = debug_manual_calibration(
    manual_calibration_file, verbose=verbose, find_scaling=fit_scale
)

cal_by_probe = cal_by_probe_parallax.copy()
cal_by_probe.update(cal_by_probe_manual)

# %% [markdown]
# ## Probe targets in manipulator coordinates
# Get the transformed targets in manipulator coordinates using the fitted
# calibration parameters and the experiment configuration set in the previous
# cells.


# %%
# Print the transformed targets in manipulator coordinates

dims = ["ML (mm)", "AP (mm)", "DV (mm)"]
for probe, (target_name, overshoot) in targets_and_overshoots_by_probe.items():
    if probe not in rotations:
        print(f"Probe {probe} not in calibration file")
        continue
    target = target_df.loc[target_name, dims].to_numpy().astype(np.float64)
    overshoot_arr = np.array([0, 0, overshoot / 1000])
    if fit_scale:
        scale = scale_vecs[probe]
    else:
        scale = None
    probe_target = rc.transform_bregma_to_probe(
        target, rotations[probe], translations[probe], scale
    )
    probe_target_and_overshoot = probe_target + overshoot_arr
    target_rnd, probe_target_and_overshoot_rnd = _round_targets(
        target, probe_target_and_overshoot
    )
    print(
        f"Probe {probe}: Target {target_name} {target_rnd} (mm) -> manipulator coord. {probe_target_and_overshoot_rnd} (µm) w/ {overshoot} µm overshoot"
    )
for probe, target in manual_bregma_targets_by_probe.items():
    if probe not in rotations:
        print(f"Probe {probe} not in calibration file")
        continue
    target_arr = np.array(target)
    if fit_scale:
        scale = scale_vecs[probe]
    else:
        scale = None
    probe_target = rc.transform_bregma_to_probe(
        target_arr, rotations[probe], translations[probe], scale
    )
    target_rnd, probe_target_rnd = _round_targets(target_arr, probe_target)
    print(
        f"Probe {probe}: Manual target {target_rnd} (mm) -> manipulator coord. {probe_target_rnd} (µm)"
    )

# %%
