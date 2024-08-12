# %%
from aind_mri_targeting.planning import (
    candidate_insertions,
    compatible_insertion_pairs,
    transform_matrix_from_angles_and_target,
    get_implant_targets,
    make_final_insertion_scene,
    make_scene_for_insertion,
    test_for_collisions,
    plan_insertion,
)
from aind_mri_utils.file_io import slicer_files as sf

from pathlib import Path
import trimesh
import SimpleITK as sitk
from aind_mri_utils import rotations as rot
from aind_mri_utils.file_io import slicer_files as sf
from aind_mri_utils.file_io import simpleitk as mr_sitk
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.optimization import (
    get_headframe_hole_lines,
    append_ones_column,
)
from aind_mri_utils.meshes import load_newscale_trimesh
from aind_mri_utils.chemical_shift import (
    compute_chemical_shift,
    chemical_shift_transform,
)
from aind_mri_utils.arc_angles import calculate_arc_angles

import numpy as np
import pandas as pd

# %%
mouse = "727354"
whoami = "galen"
if whoami == "galen":
    base_dir = Path("/mnt/aind1-vast/scratch/")
    base_save_dir = Path("/home/galen.lynch/")
elif whoami == "yoni":
    base_dir = Path("Y:/")
    base_save_dir = Path(
        "C:/Users/yoni.browning/OneDrive - Allen Institute/Desktop/"
    )
else:
    raise ValueError("Who are you again?")

headframe_model_dir = base_dir / "ephys/persist/data/MRI/HeadframeModels/"
probe_model_file = (
    headframe_model_dir / "dovetailtweezer_oneShank_centered_corrected.obj"
)  # "modified_probe_holder.obj"
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}/".format(
    mouse
)

headframe_path = headframe_model_dir / "TenRunHeadframe.obj"
holes_path = headframe_model_dir / "OneOff_HolesOnly.obj"


implant_holes_path = str(
    annotations_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
)

image_path = str(
    annotations_path / "{}_100.nii.gz".format(mouse)
)  # '_100.nii.gz'))
labels_path = str(
    annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse)
)  # 'Segmentation.seg.nrrd')#
brain_mask_path = str(
    annotations_path / ("{}_auto_skull_strip.nrrd".format(mouse))
)
manual_annotation_path = str(
    annotations_path / (f"{mouse}_ManualAnnotations.fcsv")
)
cone_path = (
    base_dir
    / "ephys/persist/Software/PinpointBuilds/WavefrontFiles/Cone_0160-200-53.obj"
)

uw_yoni_annotation_path = (
    annotations_path / f"targets-{mouse}-transformed.fcsv"
)

newscale_file_name = headframe_path / "Centered_Newscale_2pt0.obj"
#


calibration_filename = "calibration_info_np2_2024_04_22T11_15_00.xlsx"
calibration_dir = (
    base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
)
calibration_file = calibration_dir / calibration_filename
measured_hole_centers = (
    annotations_path / "measured_hole_centers_conflict.xlsx"
)


# manual_hole_centers_file = annotations_path / 'hole_centers.mrk.json'

transformed_targets_save_path = annotations_path / (
    f"{mouse}_TransformedTargets.csv"
)
test_probe_translation_save_path = str(
    base_save_dir / "test_probe_translation.h5"
)
transform_filename = str(annotations_path / (mouse + "_com_plane.h5"))

# %%
target_structures = ["CCant", "CCpst", "AntComMid", "GenFacCran2"]

# %%
image = sitk.ReadImage(image_path)
# Read points
manual_annotation = sf.read_slicer_fcsv(manual_annotation_path)

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_path)
headframe_lps = cs.convert_coordinate_system(
    headframe, "ASR", "LPS"
)  # Preserves shape!

# Load the computed transform
trans = mr_sitk.load_sitk_transform(
    transform_filename, homogeneous=True, invert=True
)[0]

cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")

probe_mesh = load_newscale_trimesh(probe_model_file, move_down=0.5)

# Get chemical shift from MRI image.
# Defaults are standard UW scans- set params for anything else.
chem_shift = compute_chemical_shift(image)
chem_shift_trans = chemical_shift_transform(chem_shift, readout="HF")
# -

# List targeted locations
preferred_pts = {k: manual_annotation[k] for k in target_structures}

hmg_pts = rot.prepare_data_for_homogeneous_transform(
    np.array(tuple(preferred_pts.values()))
)
chem_shift_annotation = hmg_pts @ chem_shift_trans.T @ trans.T
transformed_annotation = rot.extract_data_for_homogeneous_transform(
    chem_shift_annotation
)
target_names = tuple(preferred_pts.keys())


# %%

implant_vol = sitk.ReadImage(implant_holes_path)

implant_targets, implant_names = get_implant_targets(implant_vol)


# Visualize Holes, list locations

transformed_implant_homog = (
    rot.prepare_data_for_homogeneous_transform(implant_targets) @ trans.T
)
transformed_implant = rot.extract_data_for_homogeneous_transform(
    transformed_implant_homog
)

# %%
dim_names = ["ML (mm)", "AP (mm)", "DV (mm)"]
transformed_annotation_ras = np.array([-1, -1, 1]) * transformed_annotation
target_df = pd.DataFrame(
    data={
        "point": target_names,
        **{
            d: transformed_annotation_ras[:, i]
            for i, d in enumerate(dim_names)
        },
    }
)
sp = np.argsort(implant_names)
implant_names_sorted = np.array(implant_names)[sp]
transformed_implant_sorted_ras = (
    np.array([-1, -1, 1]) * transformed_implant[sp, :]
)
implant_df = pd.DataFrame(
    data={
        "point": [f"Hole {n}" for n in implant_names_sorted],
        **{
            d: transformed_implant_sorted_ras[:, i]
            for i, d in enumerate(dim_names)
        },
    }
)
df_joined = pd.concat((target_df, implant_df), ignore_index=True)
df_joined.to_csv(transformed_targets_save_path, index=False)
# %%
df = candidate_insertions(
    transformed_annotation,
    transformed_implant,
    target_names,
    implant_names,
)
valid = compatible_insertion_pairs(df)


# %%
