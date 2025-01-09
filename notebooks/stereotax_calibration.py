# %%
from pathlib import Path
from aind_mri_utils.reticle_calibrations import (
    fit_rotation_params_from_excel,
    transform_probe_to_reticle,
    transform_reticle_to_probe,
)
import pandas as pd
import numpy as np

# %%
base_path = Path("/mnt/aind1-vast/scratch/ephys/persist/data/")
fidicual_calibration_file = base_path / "MRI/processed/754372/fiducials.xlsx"

calibration_dir = base_path / "probe_calibrations/CSVCalibrations"
stereotax_calibration_file = (
    calibration_dir / "calibration_info_stereotax_2025_01_09T11_59_00.xlsx"
)

targets_ras = {
    "PL": np.array([-1.212, 2.736, -0.112]),
}

# %%
df = pd.read_excel(fidicual_calibration_file, sheet_name="Sheet1")
# %%
calibration_file = (
    calibration_dir / "calibration_info_np2_2024_12_23T11_36_00.xlsx"
)

# %%
cal_by_probe = fit_rotation_params_from_excel(
    calibration_file, find_scaling=False
)
# %%
rotation, translation, _ = cal_by_probe[45883]
probe_pts = df.loc[df["probe"] == 45883, ["X", "Y", "Z"]].values / 1000
# %%
reticle_pts = transform_probe_to_reticle(probe_pts, rotation, translation)

# %%
cal_by_probe_stereotax = fit_rotation_params_from_excel(
    stereotax_calibration_file, find_scaling=False
)

rotation_stereo, translation_stereo, _ = cal_by_probe_stereotax[1]
pl_pt = transform_probe_to_reticle(
    targets_ras["PL"].reshape(1, 3), rotation_stereo, translation_stereo
)
