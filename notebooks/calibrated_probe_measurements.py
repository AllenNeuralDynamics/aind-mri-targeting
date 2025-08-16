# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.implant import make_hole_seg_dict
from aind_mri_utils.plots import make_3d_ax_look_normal
from aind_mri_utils.reticle_calibrations import fit_rotation_params_from_manual_calibration, transform_probe_to_bregma
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    invert_rotate_translate,
)
from matplotlib import pyplot as plt

# %%
# %matplotlib ipympl
# %%
mouse = "786864"
WHOAMI = "Galen"
if WHOAMI == "Galen":
    vast_path = Path("/mnt/vast")
calibration_dir = vast_path / "scratch" / "ephys" / "persist" / "data" / "probe_calibrations"
manual_calibration_dir = calibration_dir / "CSVCalibrations"
measured_hole_location_file = (
    vast_path
    / "scratch"
    / "ephys"
    / "persist"
    / "data"
    / "MRI"
    / "processed"
    / "786864"
    / "recorded-hole-locations-2025-07-23T09-46-00.xlsx"
)

implant_annotation_path = vast_path / "scratch/ephys/persist/data/MRI/processed/{}".format(mouse)
headframe_transform_file = implant_annotation_path / "com_plane.h5"
implant_file = implant_annotation_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
implant_fit_transform_file = implant_annotation_path / "{}_implant_fit.h5".format(mouse)
model_path = vast_path / "scratch/ephys/persist/data/MRI/HeadframeModels"
hole_model_path = model_path / "HoleOBJs"
# %%
hole_measurements = pd.read_excel(measured_hole_location_file, sheet_name="Sheet1")
# %%
# Get unique probe name and calibration file pairs
probe_calibration_pairs = hole_measurements[["Probe #", "Calibration"]].drop_duplicates()
# Find unique calibration files
calibration_files = probe_calibration_pairs["Calibration"].unique().tolist()
# Build a mapping of calibrations
calibrations = {}
for calibration_file in calibration_files:
    calibration_file_path = manual_calibration_dir / calibration_file
    cal_by_probe = fit_rotation_params_from_manual_calibration(calibration_file_path)[0]
    calibrations[calibration_file] = cal_by_probe

# %%
calibration_lookup = {}
for row in probe_calibration_pairs.iterrows():
    probe_name = row[1]["Probe #"]
    calibration_file = row[1]["Calibration"]
    calibration_lookup[(probe_name, calibration_file)] = calibrations[calibration_file][probe_name]
# %%
hole_locations_bregma = {}
for row in hole_measurements.iterrows():
    hole_name = row[1]["Hole #"]
    probe_name = row[1]["Probe #"]
    calibration_file = row[1]["Calibration"]
    hole_location = row[1][["X", "Y", "Z"]].values.astype(float) / 1000
    if np.isnan(hole_location).any():
        continue
    if (probe_name, calibration_file) in calibration_lookup:
        rotation_params = calibration_lookup[(probe_name, calibration_file)]
        hole_locations_bregma[hole_name] = transform_probe_to_bregma(hole_location, *rotation_params)

implant_seg_vol = sitk.ReadImage(str(implant_file))
implant_targets_by_hole = make_hole_seg_dict(implant_seg_vol, fun=lambda x: np.mean(x, axis=0))
implant_names = list(implant_targets_by_hole.keys())
implant_targets = np.vstack(list(implant_targets_by_hole.values()))
headframe_R, headframe_t, headframe_c = load_sitk_transform(str(headframe_transform_file))

transformed_implant_targets_lps = apply_rotate_translate(
    implant_targets, *invert_rotate_translate(headframe_R, headframe_t)
)
transformed_implant_targets_by_hole_lps = {
    h: transformed_implant_targets_lps[i, :] for i, h in enumerate(implant_names)
}

# %%
transformed_model_implant_targets_lps = {}
implant_R, implant_t, implant_c = load_sitk_transform(implant_fit_transform_file)
hole_files = [x for x in os.listdir(hole_model_path) if ".obj" in x and "Hole" in x]
hole_dict = {}
for i, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_model_path, flname))
    hole_dict[hole_num].vertices = convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_model_path, "LowerFace.obj"))
hole_dict[-1].vertices = convert_coordinate_system(hole_dict[-1].vertices, "ASR", "LPS")  # Preserves shape!

model_implant_targets = {}
for i, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

for i, key in enumerate(model_implant_targets.keys()):
    implant_tgt = model_implant_targets[key]
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(implant_R, implant_t))
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(headframe_R, headframe_t))
    transformed_model_implant_targets_lps[key] = implant_tgt


# %%
lps_to_ras = np.diag(np.array([-1, -1, 1]))
transformed_model_implant_targets_ras = np.vstack(list(transformed_model_implant_targets_lps.values())) @ lps_to_ras
transformed_implant_targets_ras = transformed_implant_targets_lps @ lps_to_ras
measured_hole_locations_ras = np.vstack(list(hole_locations_bregma.values()))

plot_pts = {
    "segmentation": transformed_implant_targets_ras,
    "model": transformed_model_implant_targets_ras,
    "measured": measured_hole_locations_ras,
}

# %%
f, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
ax.set_title("Transformed Implant Targets in RAS Coordinates")
for key, pts in plot_pts.items():
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=key)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()
make_3d_ax_look_normal(ax)

# %%
