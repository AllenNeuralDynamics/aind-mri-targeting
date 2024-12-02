# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3.9.12 ('base')
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

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# %matplotlib inline
from aind_mri_utils import reticle_calibrations as rc

# %%
# Set file paths and mouse ID here

# Calibration File with probe data
mouse_id = "728537"
retcile_used = "H"
basepath = Path("/mnt/aind1-vast/scratch/")
parallax_debug_dir = Path("/home/galen.lynch/Downloads/debug")

calibration_dir = (
    basepath / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
)

# Target file with transformed targets
target_dir = basepath / f"ephys/persist/data/MRI/processed/{mouse_id}/UW"
target_file = target_dir / f"{mouse_id}_TransformedTargets.csv"

# Whether to fit the scale parameters as well. This is not recommended unless
# you have a good reason to do so.  Does not guarantee that the error will be
# lower
fit_scale = True

# Whether to print the mean and maximum error for each probe and the predicted
# probe coordinates for each reticle coordinate with error for that coordinate
verbose = True

reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}


# %%
def _round_targets(target, probe_target):
    target_rnd = np.round(target, decimals=2)
    probe_target_and_overshoot_rnd = (
        np.round(2000 * probe_target_and_overshoot) / 2
    )
    return target_rnd, probe_target_and_overshoot_rnd


def pairs_from_parallax_points_csv(parallax_points_filename):
    df = pd.read_csv(parallax_points_filename)
    pairs = []
    dims = ["x", "y", "z"]
    reticle_colnames = [f"global_{dim}" for dim in dims]
    manipulator_colnames = [f"local_{dim}" for dim in dims]
    for i, row in df.iterrows():
        manip_pt = row[manipulator_colnames].to_numpy().astype(np.float64)
        ret_pt = row[reticle_colnames].to_numpy().astype(np.float64)
        pairs.append((ret_pt, manip_pt))
    return pairs


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
    45881: ("GPe_anterior", 700),
}
# Targets in bregma-relative coordinates not in the target file
# manual_bregma_targets_by_probe = {probe_id: [x, y, z], ...}
# x y z in mm
manual_bregma_targets_by_probe = {
    # 46110: [0, 0, 0], # in mm!
}

# %%
manips_used = list(
    set(targets_and_overshoots_by_probe.keys()).union(
        manual_bregma_targets_by_probe.keys()
    )
)
adjusted_pairs_by_probe = dict()
global_offset = reticle_offsets[retcile_used]
global_roatation_degrees = 0
reticle_name = retcile_used
for manip in manips_used:
    fname = parallax_debug_dir / f"points_SN{manip}.csv"
    pairs = pairs_from_parallax_points_csv(fname)
    reticle_pts, manip_pts = rc._apply_metadata_to_pair_lists(
        pairs, 1 / 1000, global_roatation_degrees, global_offset, 1 / 1000
    )
    adjusted_pairs_by_probe[manip] = (reticle_pts, manip_pts)

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
# Calculate the rotation parameters and display errors if verbose is set to
# True

rotations = dict()
translations = dict()
if fit_scale:
    scale_vecs = dict()
    for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
        (rotation, scale, translation, _) = rc.fit_rotation_params(
            reticle_pts, probe_pts, find_scaling=True
        )
        rotations[probe] = rotation
        scale_vecs[probe] = scale
        translations[probe] = translation
else:
    for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
        rotation, translation, _ = rc.fit_rotation_params(
            reticle_pts, probe_pts, find_scaling=False
        )
        rotations[probe] = rotation
        translations[probe] = translation

for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
    if fit_scale:
        scale = scale_vecs[probe]
    else:
        scale = None
    predicted_probe_pts = rc.transform_reticle_to_probe(
        reticle_pts, rotations[probe], translations[probe], scale
    )
    # in µm
    errs = 1000 * np.linalg.norm(predicted_probe_pts - probe_pts, axis=1)
    if verbose:
        print(
            f"Probe {probe}: Mean error {errs.mean():.2f} µm, max error {errs.max():.2f} µm"
        )
        print(f"rotation: {rotations[probe]}")
        print(f"translation: {translations[probe]}")
        print(f"scale: {scale}")
        original_reticle_pts = reticle_pts - global_offset
        for i in range(len(errs)):
            rounded_pred = np.round(predicted_probe_pts[i], decimals=2)
            print(
                f"\tReticle {original_reticle_pts[i]} -> Probe {probe_pts[i]}: predicted {rounded_pred} error {errs[i]:.2f} µm"
            )

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
    probe_target = rc.transform_reticle_to_probe(
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
    probe_target = rc.transform_reticle_to_probe(
        target_arr, rotations[probe], translations[probe], scale
    )
    target_rnd, probe_target_rnd = _round_targets(target_arr, probe_target)
    print(
        f"Probe {probe}: Manual target {target_rnd} (mm) -> manipulator coord. {probe_target_rnd} (µm)"
    )

# %%