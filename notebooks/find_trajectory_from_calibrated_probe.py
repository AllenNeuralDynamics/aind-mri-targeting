# %%
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils import slicer as sf
from aind_mri_utils import reticle_calibrations as rc
from aind_mri_utils import rotations as rot
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io import simpleitk as mr_sitk
from aind_mri_utils.planning import candidate_insertions

# %%
mouse = "721685"
whoami = "galen"
if whoami == "galen":
    base_dir = Path("/mnt/aind1-vast/scratch/")
    base_save_dir = Path("/home/galen.lynch/")
elif whoami == "yoni":
    base_dir = Path("Y:/")
    base_save_dir = Path("C:/Users/yoni.browning/OneDrive - Allen Institute/Desktop/")
else:
    raise ValueError("Who are you again?")

headframe_model_dir = base_dir / "ephys/persist/data/MRI/HeadframeModels/"
probe_model_file = (
    headframe_model_dir / "dovetailtweezer_oneShank_centered_corrected.obj"
)  # "modified_probe_holder.obj"
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}/HF".format(mouse)

headframe_path = headframe_model_dir / "TenRunHeadframe.obj"
holes_path = headframe_model_dir / "OneOff_HolesOnly.obj"


implant_holes_path = str(annotations_path / "{}_ImplantHoles.seg.nrrd".format(mouse))

image_path = str(annotations_path / "{}_100.nii.gz".format(mouse))  # '_100.nii.gz'))
labels_path = str(annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse))  # 'Segmentation.seg.nrrd')#
brain_mask_path = str(annotations_path / ("{}_auto_skull_strip.nrrd".format(mouse)))
manual_annotation_path = str(annotations_path / (f"{mouse}_ManualAnnotations.fcsv"))
cone_path = base_dir / "ephys/persist/Software/PinpointBuilds/WavefrontFiles/Cone_0160-200-53.obj"

uw_yoni_annotation_path = annotations_path / f"targets-{mouse}-transformed.fcsv"

newscale_file_name = headframe_path / "Centered_Newscale_2pt0.obj"
#


calibration_filename = "calibration_info_np2_2024_08_13T09_57_00.xlsx"
calibration_dir = base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
calibration_file = calibration_dir / calibration_filename
measured_hole_centers = annotations_path / "measured_hole_centers_conflict.xlsx"


# manual_hole_centers_file = annotations_path / 'hole_centers.mrk.json'

transformed_targets_save_path = annotations_path / (mouse + "TransformedTargets.csv")
test_probe_translation_save_path = str(base_save_dir / "test_probe_translation.h5")
transform_filename = str(annotations_path / (mouse + "_com_plane.h5"))

# %%
# measurement_df = pd.read_excel(measured_hole_centers)

# %%
target_structure_pair = [("4", "CCant"), ("3", "CCant")]

# %%

(
    adjusted_pairs_by_probe,
    global_offset,
    global_rotation_degrees,
    reticle_name,
) = rc.read_reticle_calibration(calibration_file)


# %%
measurements = {
    46110: {
        "3": np.array(
            [
                [8301, 7747, 8700],
            ],
            dtype=float,
        ),
    },
}
probe_scales = {46100: False, 46110: False}
# %%
transform_rs = dict()
transform_offsets = dict()
scaling_vecs = dict()
for probe in measurements:
    reticle_coords, probe_coords = adjusted_pairs_by_probe[probe]
    if probe_scales[probe]:
        (
            transform_rs[probe],
            scaling_vecs[probe],
            transform_offsets[probe],
            _,
        ) = rc.fit_rotation_params(
            reticle_pts=reticle_coords,
            probe_pts=probe_coords,
            find_scaling=True,
        )
    else:
        transform_rs[probe], transform_offsets[probe], _ = rc.fit_rotation_params(
            reticle_pts=reticle_coords,
            probe_pts=probe_coords,
            find_scaling=False,
        )
        scaling_vecs[probe] = None

# %%
transformed_global_points = dict()
for probe in measurements:
    transformed_global_points[probe] = dict()
    for name, coord in measurements[probe].items():
        transformed_global_points[probe][name] = rc.transform_probe_to_bregma(
            measurements[probe][name] / 1000,
            transform_rs[probe],
            transform_offsets[probe],
            scale=scaling_vecs[probe],
        )
# %%
manual_annotation = sf.read_slicer_fcsv(manual_annotation_path)
image = sitk.ReadImage(image_path)
trans = mr_sitk.load_sitk_transform(transform_filename, homogeneous=True, invert=True)[0]


# %%
chem_shift = compute_chemical_shift(image)
chem_shift_trans = chemical_shift_transform(chem_shift, readout="HF")
# -

# List targeted locations
preferred_pts = {k[1]: manual_annotation[k[1]] for k in target_structure_pair}

hmg_pts = rot.prepare_data_for_homogeneous_transform(np.array(tuple(preferred_pts.values())))
chem_shift_annotation = hmg_pts @ trans.T @ chem_shift_trans.T
transformed_annotation = rot.extract_data_for_homogeneous_transform(chem_shift_annotation)
target_names = tuple(preferred_pts.keys())

# %%
transformed_global_points[46110]
df = candidate_insertions(
    transformed_annotation,
    transformed_global_points[46110]["3"] * np.array([-1, -1, 1]),
    target_names,
    ["3"],
)

# %%
