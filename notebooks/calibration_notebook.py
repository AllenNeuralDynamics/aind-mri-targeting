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
# *Problem:* After we measure the coordinates of each manipulator for known positions on the reticle, can we use this to determine the rotation and offset of the probe?

# %%
from pathlib import Path
import numpy as np

# %matplotlib inline
from aind_mri_utils import reticle_calibrations as rc

import pandas as pd


# %%
# Calibration File with probe data
mouse_id = "727354"
basepath = Path("/mnt/aind1-vast/scratch/")
calibration_dir = (
    basepath / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
)
target_dir = basepath / f"ephys/persist/data/MRI/processed/{mouse_id}/"
cal_file = calibration_dir / "calibration_info_np2_2024_08_05T14_34_00.xlsx"
target_file = target_dir / f"{mouse_id}_TransformedTargets.csv"
# Whether to fit the scale parameters as well. This is not recommended unless you have a good reason to do so.
# Does not guarantee that the error will be lower
fit_scale = False
verbose = True


# %%
def _round_targets(target, probe_target):
    target_rnd = np.round(target, decimals=2)
    probe_target_and_overshoot_rnd = (
        np.round(2000 * probe_target_and_overshoot) / 2
    )
    return target_rnd, probe_target_and_overshoot_rnd


# %%
target_df = pd.read_csv(target_file)
target_df = target_df.set_index("point")
print(target_df)


# %%
(
    adjusted_pairs_by_probe,
    global_offset,
    global_rotation_degrees,
    reticle_name,
) = rc.read_reticle_calibration(cal_file)

# %%

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
    errs = 1000 * np.linalg.norm(predicted_probe_pts - probe_pts, axis=1)
    if verbose:
        print(
            f"Probe {probe}: Mean error {errs.mean():.2f} µm, max error {errs.max():.2f} µm"
        )
        original_reticle_pts = reticle_pts - global_offset
        for i in range(len(errs)):
            rounded_pred = np.round(predicted_probe_pts[i], decimals=2)
            print(
                f"\tReticle {original_reticle_pts[i]} -> Probe {probe_pts[i]}: predicted {rounded_pred} error {errs[i]:.2f} µm"
            )

# %%
# Names of targets in the target file and overshoots
targets_and_overshoots_by_probe = {
    46110: ("CCant", 500),
    46100: ("GenFacCran2", 500),
}
# Targets in bregma-relative coordinates not in the target file
manual_bregma_targets_by_probe = {
    # 46110: np.array([0, 0, 0]), # in mm!
}


# %%

for probe, (target_name, overshoot) in targets_and_overshoots_by_probe.items():
    target = target_df.loc[target_name].to_numpy()
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
    if fit_scale:
        scale = scale_vecs[probe]
    else:
        scale = None
    probe_target = rc.transform_reticle_to_probe(
        target, rotations[probe], translations[probe], scale
    )
    target_rnd, probe_target_rnd = _round_targets(target, probe_target)
    print(
        f"Probe {probe}: Manual target {target_rnd} (mm) -> manipulator coord. {probe_target_rnd} (µm)"
    )

# %%
