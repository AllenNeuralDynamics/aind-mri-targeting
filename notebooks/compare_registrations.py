# %%
from pathlib import Path

import numpy as np
import pandas as pd
from aind_mri_utils.reticle_calibrations import (
    debug_manual_calibration,
    debug_parallax_calibration,
    find_probe_angle,
    transform_bregma_to_probe,
)

# %%
calibration_base = Path("/mnt/vast/scratch/ephys/persist/data/probe_calibrations/CSVCalibrations")

reference_manual_calibrations = [calibration_base / p for p in ["calibration_info_np2_2025_08_27T09_32_00.xlsx"]]
comparison_parallax_calibration_dirs = [calibration_base / p for p in ["log_20250819_144133"]]

reticle_used = "H"

reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}
reticle_rotations = {"H": 0}

# %%
cal_by_probe_ref = {}
err_by_probe_ref = {}
for manual_cal in reference_manual_calibrations:
    this_cal_by_probe_ref, R_reticle_bregma, t_reticle_bregma, adjusted_pairs_by_probe, this_err_by_probe_ref = (
        debug_manual_calibration(manual_cal)
    )
    cal_by_probe_ref.update(this_cal_by_probe_ref)
    err_by_probe_ref.update(this_err_by_probe_ref)

reticle_offset = reticle_offsets[reticle_used]
reticle_rotation = reticle_rotations[reticle_used]
cal_by_probe_comp = {}
err_by_probe_comp = {}
for parallax_cal in comparison_parallax_calibration_dirs:
    this_cal_by_probe_comp, _, _, this_err_by_probe_comp = debug_parallax_calibration(
        parallax_cal, reticle_offset, reticle_rotation
    )
    cal_by_probe_comp.update(this_cal_by_probe_comp)
    err_by_probe_comp.update(this_err_by_probe_comp)

# %%
df_vals = {}
test_points = np.array(
    [
        [0, 0, 0],
        [-2, -2, -5],
    ],
    dtype=float,
)
dims = ["X", "Y", "Z"]


def _dim_helper(val, suffix):
    for i, d in enumerate(dims):
        df_vals.setdefault(f"{d}_{suffix}", []).append(val[i])


for probe, cal in cal_by_probe_comp.items():
    comp_R, comp_t = cal
    med_comp_err = np.median(err_by_probe_comp.get(probe, [np.nan]))
    pred_points_comp = transform_bregma_to_probe(test_points, comp_R, comp_t)
    ref_cal = cal_by_probe_ref.get(probe, None)
    if not ref_cal:
        continue
    ref_R, ref_t = ref_cal
    med_ref_err = np.median(err_by_probe_ref.get(probe, [np.nan]))
    ap_angle_ref, ml_angle_ref = find_probe_angle(ref_R)
    ap_angle_comp, ml_angle_comp = find_probe_angle(comp_R)
    pred_points_ref = transform_bregma_to_probe(test_points, ref_R, ref_t)
    distance_pred = np.linalg.norm(pred_points_ref - pred_points_comp, axis=1)
    df_vals.setdefault("probe", []).append(probe)
    df_vals.setdefault("ref_ap_angle", []).append(ap_angle_ref)
    df_vals.setdefault("ref_ml_angle", []).append(ml_angle_ref)
    df_vals.setdefault("comp_ap_angle", []).append(ap_angle_comp)
    df_vals.setdefault("comp_ml_angle", []).append(ml_angle_comp)
    df_vals.setdefault("err_ref", []).append(med_ref_err)
    df_vals.setdefault("err_comp", []).append(med_comp_err)
    _dim_helper(pred_points_ref[0], "at_0_0_0_ref")
    _dim_helper(pred_points_comp[0], "at_0_0_0_comp")
    df_vals.setdefault("distance_at_0_0_0", []).append(distance_pred[0])
    _dim_helper(pred_points_ref[1], "at_2_2_5_ref")
    _dim_helper(pred_points_comp[1], "at_2_2_5_comp")
    df_vals.setdefault("distance_at_2_2_5", []).append(distance_pred[1])
    _dim_helper(ref_t, "t_ref")
    _dim_helper(comp_t, "t_comp")
# %%
df = pd.DataFrame.from_dict(df_vals)

# %%
df.to_csv("/home/galen.lynch/test_probe_calibration_comparison.csv", index=False)
# %%
