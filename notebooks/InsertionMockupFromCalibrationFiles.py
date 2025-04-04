# jupyter:
#   jupytext:
#     formats: py:percent
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

# %%
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.arc_angles import calculate_arc_angles, transform_matrix_from_angles
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.implant import make_hole_seg_dict
from aind_mri_utils.meshes import (
    apply_transform_to_trimesh,
    load_newscale_trimesh,
    mask_to_trimesh,
)
from aind_mri_utils.reticle_calibrations import (
    combine_parallax_and_manual_calibrations,
    find_probe_angle,
    fit_rotation_params_from_parallax,
    reticle_metadata_transform,
)
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    compose_transforms,
    invert_rotate_translate,
)
from aind_mri_utils.sitk_volume import find_points_equal_to

# %%
# %matplotlib ipympl

# %%
# Set the log verbosity to get debug statements
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.DEBUG)
# %%
# File Paths
mouse = "781369"
reticle_used = "H"
target_structures = ["PL", "CLA", "MD", "CA1", "VM", "BLA", "RSP"]

WHOAMI = "Galen"

if WHOAMI == "Galen":
    base_path = Path("/mnt/aind1-vast/scratch")
elif WHOAMI == "Yoni":
    base_path = Path(r"Y:/")
else:
    raise ValueError("Who are you again?")

# File Paths
# Image and image annotations.
annotations_path = base_path / "ephys/persist/data/MRI/processed/{}".format(mouse)
image_path = annotations_path / "{}_100.nii.gz".format(mouse)
labels_path = annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse)
brain_mask_path = annotations_path / "{}_100_auto_skull_strip.nrrd".format(mouse)
image_transform_file = annotations_path / "{}_com_plane.h5".format(mouse)
structure_mask_path = annotations_path / "Masks"
structure_files = {
    structure: structure_mask_path / f"{mouse}-{structure}-Mask.nrrd"
    for structure in target_structures
}

# Implant annotation
# Note that this can be different than the image annotation,
# this is in the event that an insertion is planned with data from multiple scans
# (see 750107 for example).
implant_annotation_path = base_path / "ephys/persist/data/MRI/processed/{}".format(
    mouse
)
headframe_transform_file = image_transform_file
implant_file = implant_annotation_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
implant_mesh_file = implant_annotation_path / "{}_ImplantHoles.obj".format(mouse)
implant_fit_transform_file = implant_annotation_path / "{}_implant_fit.h5".format(mouse)


# OBJ files
model_path = base_path / "ephys/persist/data/MRI/HeadframeModels"
hole_model_path = model_path / "HoleOBJs"

probe_model_files = {
    "2.1-alpha": model_path / "Centered_Newscale_2pt0.obj",
    "2.1": model_path / "dovetailtweezer_oneShank_centered_corrected.obj",
    "quadbase": model_path / "Quadbase_customHolder_centeredOnShank0.obj",
    "2.4": model_path / "dovetailwtweezer_fourShank_centeredOnShank0.obj",
}

headframe_file = model_path / "TenRunHeadframe.obj"
cone_file = model_path / "TacoForBehavior" / "0160-200-72_X06.obj"
well_file = model_path / "WHC_Well" / "0274-400-07_X02.obj"
implant_model_file = model_path / "0283-300-04.obj"

calibration_path = base_path / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
calibration_files = [calibration_path / "calibration_info_np2_2025_04_08T14_15_00.xlsx"]
parallax_calibration_dir = []  # calibration_path / "log_20250303_122136"
iso_time = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
plan_save_path = annotations_path / f"{mouse}_InsertionPlan_{iso_time}.csv"
target_save_path = annotations_path / f"{mouse}_InsertionPlan_Targets_{iso_time}.csv"
scene_save_path = annotations_path / f"{mouse}_InsertionPlan_{iso_time}.obj"

# If None, use the files provided to determine if the model should be based on
# calibration data. Otherwise, set to True or False to set `from_calibration`
# to that value.
calibration_override = None
legacy_print = False
# %%
# Reticle offsets and rotations
reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}
reticle_rotations = {"H": 0}

# %%
if calibration_override is not None:
    from_calibration = len(calibration_files) > 0 or len(parallax_calibration_dir) > 0
else:
    from_calibration = calibration_override

image_R, image_t, image_c = load_sitk_transform(str(image_transform_file))
headframe_R, headframe_t, headframe_c = load_sitk_transform(
    str(headframe_transform_file)
)

image = sitk.ReadImage(str(image_path))

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_file)
headframe_lps = cs.convert_coordinate_system(
    headframe, "ASR", "LPS"
)  # Preserves shape!

# Load the headframe
cone, cone_faces = get_vertices_and_faces(cone_file)
cone_lps = cs.convert_coordinate_system(cone, "ASR", "LPS")  # Preserves shape!

well, well_faces = get_vertices_and_faces(well_file)
well_lps = cs.convert_coordinate_system(well, "ASR", "LPS")  # Preserves shape!

implant_model, implant_faces = get_vertices_and_faces(implant_model_file)
implant_model_lps = cs.convert_coordinate_system(
    implant_model, "ASR", "LPS"
)  # Preserves shape!

# Load the brain mask
brain_mask = sitk.ReadImage(str(brain_mask_path))
brain_mask_pos = find_points_equal_to(brain_mask)
# Downsample the brain position data to reduce the number of points for
# visualization or processing.
brain_mask_pos_ds = brain_mask_pos[
    np.arange(0, brain_mask_pos.shape[0], max(1, brain_mask_pos.shape[0] // 1000))
]

# Load the brain mesh
brain_mesh = mask_to_trimesh(brain_mask)
implant_seg_vol = sitk.ReadImage(str(implant_file))

probe_models = {k: load_newscale_trimesh(v, 0) for k, v in probe_model_files.items()}

# %%

# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = [x for x in os.listdir(hole_model_path) if ".obj" in x and "Hole" in x]
hole_dict = {}
for i, filename in enumerate(hole_files):
    hole_num = int(filename.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_model_path, filename))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_model_path, "LowerFace.obj"))
hole_dict[-1].vertices = cs.convert_coordinate_system(
    hole_dict[-1].vertices, "ASR", "LPS"
)  # Preserves shape!

# load and transform the target structure
structure_meshes = {}
for structure, structure_file in structure_files.items():
    mesh = mask_to_trimesh(
        sitk.ReadImage(str(structure_file)),
    )
    structure_meshes[structure] = mesh


# %%
model_implant_targets = {}
for i, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

# %%
# If implant has holes that are segmented.

implant_targets_by_hole = make_hole_seg_dict(
    implant_seg_vol, fun=lambda x: np.mean(x, axis=0)
)
implant_names = list(implant_targets_by_hole.keys())
implant_targets = np.vstack(list(implant_targets_by_hole.values()))

# %%
chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(
    compute_chemical_shift(image, ppm=3.7)
)
chem_shift_image_R, chem_shift_image_t = invert_rotate_translate(
    chem_shift_pt_R, chem_shift_pt_t
)
chem_image_R, chem_image_t = compose_transforms(
    chem_shift_image_R, chem_shift_image_t, image_R, image_t
)

# %%
transformed_brain = apply_rotate_translate(
    brain_mask_pos_ds, *invert_rotate_translate(chem_image_R, chem_image_t)
)
transformed_brain_mesh = apply_transform_to_trimesh(
    brain_mesh.copy(), *invert_rotate_translate(chem_image_R, chem_image_t)
)
transformed_implant_targets = apply_rotate_translate(
    implant_targets, *invert_rotate_translate(headframe_R, headframe_t)
)
transformed_structure_meshes = {}
for structure, structure_mesh in structure_meshes.items():
    transformed_structure_mesh = apply_transform_to_trimesh(
        structure_mesh.copy(), *invert_rotate_translate(chem_image_R, chem_image_t)
    )
    transformed_structure_meshes[structure] = transformed_structure_mesh
transformed_structure_centroids = {}
# Get the centroids of the left component of each mesh
for structure, mesh in transformed_structure_meshes.items():
    components = mesh.split()
    centroids = [component.centroid for component in components]
    # Find the leftmost component
    leftmost_component = max(centroids, key=lambda x: x[0])
    transformed_structure_centroids[structure] = leftmost_component
transformed_structure_centroids = {
    structure: mesh.centroid for structure, mesh in transformed_structure_meshes.items()
}
# %%
# Find calibrated probes

# Read the calibrations
reticle_offset = reticle_offsets[reticle_used]
reticle_rotation = reticle_rotations[reticle_used]
if from_calibration:
    if len(calibration_files) == 0:
        cal_by_probe_combined, R_reticle_to_bregma = fit_rotation_params_from_parallax(
            parallax_calibration_dir,
            reticle_offset,
            reticle_rotation,
            find_scaling=True,
        )
        global_offset = reticle_offset
    else:
        cal_by_probe_combined, R_reticle_to_bregma, global_offset = (
            combine_parallax_and_manual_calibrations(
                manual_calibration_files=calibration_files,
                parallax_directories=parallax_calibration_dir,
            )
        )
else:
    cal_by_probe_combined = {}
    R_reticle_to_bregma = reticle_metadata_transform(reticle_rotation)
    global_offset = reticle_offset


def find_hole_angles(centroid, hole_center_dict):
    """
    Find the angles of the holes in the mesh.
    """
    hole_angles = {}
    ras_to_lps = np.diag([-1, -1, 1])
    for i, hole_center in hole_center_dict.items():
        hole_vector = centroid - hole_center
        rx, ry = calculate_arc_angles(ras_to_lps @ hole_vector)
        hole_angles[i] = (rx, ry)
    return hole_angles


# %%
# Only use this if you are planning based on calibration files
probe_to_target_mapping = {
    "PL": 45883,
    "CLA": 46110,
    "MD": 46116,
    "VM": 46100,
    "CA1": 46113,
    "BLA": 46122,
    "RSP": 50209,
}

# %%
arcs = {
    "a": -8,
    "b": 16,
    "c": -42.5,
    "d": -24,
}
probe_by_struct = {
    "PL": "quadbase",
    "CLA": "2.1",
    "MD": "quadbase",
    "CA1": "2.1",
    "VM": "2.1",
    "BLA": "2.1",
    "RSP": "2.1",
}
arc_by_struct = {
    "PL": "a",
    "CLA": "c",
    "MD": "b",
    "CA1": "d",
    "VM": "a",
    "BLA": "a",
    "RSP": "d",
}
arc_ap_by_struct = {k: arcs[v] for k, v in arc_by_struct.items()}

hole_by_struct = {
    "PL": 1,
    "CLA": 12,
    "MD": 3,
    "CA1": 10,
    "VM": 6,
    "BLA": 4,
    "RSP": 5,
}

slider_ml_by_struct = {
    "PL": -40,
    "CLA": -15,  # 17.5 for A2
    "MD": -9,
    "CA1": 6,
    "VM": -13,
    "BLA": 27.7,
    "RSP": 22,
}

spin_by_struct = {
    "PL": 125,
    "CLA": -30,
    "MD": -55,
    "CA1": 190,
    "VM": 45,
    "BLA": 50,
    "RSP": -100,
}

# This is LP, whereas yoni uses LA
# So compared to Yoni's offsets, this should be the x offset and the inverse of
# the y offset
offsets_LP_by_struct = {
    "PL": [-0.2, 0.3],
    "CLA": [0.0, 0],
    "MD": [0.350, -0.350],
    "CA1": [0, 0],
    "VM": [0, 0],
    "BLA": [-0.1, -0.2],
    "RSP": [-0.1, 0],
}
offsets_LP_arr_by_struct = {k: np.array(v) for k, v in offsets_LP_by_struct.items()}

target_depth = np.array([2.4, 4.75, 4, 5.8, 4.9, 5.75, 2])  # Guesses, check
depth_by_struct = {
    "PL": 2.4,
    "CLA": 5,
    "MD": 5.5,
    "CA1": 5.9,
    "VM": 5.5,
    "BLA": 7,
    "RSP": 2,
}

insertion_trim_coordinates_struct = {
    "CLA": np.array([0.1, 0.1, 0.0]),
    "BLA": np.array([0.3, 0.0, 0.0]),
}
insertion_trim_angles_struct = {}


# %%
implant_R, implant_t, implant_c = load_sitk_transform(implant_fit_transform_file)

S = trimesh.Scene()
transformed_implant_targets = {}

for i, key in enumerate(model_implant_targets.keys()):
    implant_tgt = model_implant_targets[key]
    implant_tgt = apply_rotate_translate(
        implant_tgt, *invert_rotate_translate(implant_R, implant_t)
    )
    implant_tgt = apply_rotate_translate(
        implant_tgt, *invert_rotate_translate(headframe_R, headframe_t)
    )
    transformed_implant_targets[key] = implant_tgt

vertices = apply_rotate_translate(
    implant_model_lps, *invert_rotate_translate(implant_R, implant_t)
)
vertices = apply_rotate_translate(
    vertices, *invert_rotate_translate(headframe_R, headframe_t)
)
implant_mesh = trimesh.Trimesh(vertices=vertices, faces=implant_faces[0])


# holeCM = trimesh.collision.CollisionManager()
implantCM = trimesh.collision.CollisionManager()
probeCM = trimesh.collision.CollisionManager()
coneCM = trimesh.collision.CollisionManager()
wellCM = trimesh.collision.CollisionManager()

ras_to_lps = np.diag([-1, -1, 1])
probe_tip_pts_by_struct = {}
probe_insertion_pts_by_struct = {}
for i, structure in enumerate(target_structures):
    if structure not in [
        "PL",
        "CLA",
        "MD",
        "CA1",
        "VM",
        "BLA",
        "RSP",
    ]:  # target_structures:
        continue

    structureCM = trimesh.collision.CollisionManager()

    probe_type = probe_by_struct[structure]
    ap_angle = arcs[arc_by_struct[structure]]
    ml_angle = slider_ml_by_struct[structure]
    LP_offset = offsets_LP_arr_by_struct[structure]
    depth = depth_by_struct[structure]
    probe_model = probe_models[probe_type].copy()
    hole_number = hole_by_struct[structure]
    spin = spin_by_struct[structure]

    # Generate a single random color for both probe and structure
    this_color = trimesh.visual.random_color()

    # Assign the same color to the probe
    probe_model.visual.face_colors = this_color

    # Apply transformations to the probe
    implant_target = transformed_implant_targets[hole_number]

    this_pt = trimesh.creation.uv_sphere(radius=0.25)
    this_pt.apply_translation(implant_target)
    this_pt.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(this_pt)

    offset = np.zeros(3)
    offset[:2] = LP_offset
    adjusted_insertion_pt = implant_target + offset
    if from_calibration:
        this_probe = probe_to_target_mapping[structure]
        this_affine, this_translation, this_scaling = cal_by_probe_combined[this_probe]
        insertion_vector = (
            ras_to_lps @ np.linalg.inv(this_affine) @ np.array([0, 0, depth])
        )
        insertion_trims = insertion_trim_coordinates_struct.get(
            structure, np.array([0, 0, 0])
        )
        final_insertion_pt = adjusted_insertion_pt + insertion_trims
        probe_tip_location = final_insertion_pt + insertion_vector
        scaling_inv = np.diag(1 / this_scaling)
        rigid_affine = scaling_inv @ this_affine
        this_ap, this_ml = find_probe_angle(rigid_affine)
        logger.debug(f"Probe {this_probe} has angles: {this_ap}, {this_ml}")
        R_probe_mesh = transform_matrix_from_angles(this_ap, this_ml, spin)

        probe_model = apply_transform_to_trimesh(
            probe_model, R_probe_mesh, probe_tip_location
        )
    else:
        final_insertion_pt = adjusted_insertion_pt
        R_probe_mesh = transform_matrix_from_angles(ap_angle, ml_angle, spin)
        insertion_vector = R_probe_mesh @ np.array([0, 0, -depth])
        probe_tip_location = adjusted_insertion_pt + insertion_vector
        probe_model = apply_transform_to_trimesh(
            probe_model, R_probe_mesh, probe_tip_location
        )

    adjusted_insertion_pt = ras_to_lps @ adjusted_insertion_pt
    probe_tip_pts_by_struct[structure] = ras_to_lps @ probe_tip_location

    S.add_geometry(probe_model)
    probeCM.add_object(structure, probe_model)
    implantCM.add_object(structure, probe_model)
    coneCM.add_object(structure, probe_model)
    wellCM.add_object(structure, probe_model)

    structureCM.add_object("probe", probe_model)

    # load and transform the target structure
    structure_mesh = transformed_structure_meshes[structure].copy()

    # Assign the same color to the structure
    structure_mesh.visual.face_colors = this_color
    structure_mesh.visual.vertex_colors = this_color
    # structure_mesh.visual.material.main_color[:] = this_color

    # Add structure to the scene and collision manager
    S.add_geometry(structure_mesh)
    structureCM.add_object("structure", structure_mesh)

    # Check collisions
    if structureCM.in_collision_internal(False, False):
        print(f"Probe for {structure} is a hit :)")
    else:
        print(f"Probe for {structure} is a miss! :(")
    print(i)


# S.add_geometry(transformed_brain_mesh)
headframe_mesh = trimesh.Trimesh(vertices=headframe_lps, faces=headframe_faces[0])
cone_mesh = trimesh.Trimesh(vertices=cone_lps, faces=cone_faces[0])
coneCM.add_object("cone", headframe_mesh)

well_mesh = trimesh.Trimesh(vertices=well_lps, faces=well_faces[0])
wellCM.add_object("well", well_mesh)

implantCM.add_object("implant", implant_mesh)

# Optionally assign unique colors to headframe, cone, and well if desired:
headframe_color = trimesh.visual.random_color()
cone_color = trimesh.visual.random_color()
well_color = trimesh.visual.random_color()

headframe_mesh.visual.face_colors = headframe_color
headframe_mesh.vertices
cone_mesh.visual.face_colors = cone_color
well_mesh.visual.face_colors = well_color

S.add_geometry(headframe_mesh)
# S.add_geometry(cone_mesh)
S.add_geometry(well_mesh)

probe_fail, fail_names = probeCM.in_collision_internal(return_names=True)
if probe_fail:
    print("Probes are colliding :(")
    print(f"Problems: {list(fail_names)}")
else:
    print("Probes are clear! :)")

    if coneCM.in_collision_internal(False, False):
        print("Probes are hitting cone! :(")
    else:
        print("Probes are clearing cone :)")

    if wellCM.in_collision_internal(False, False):
        print("Probes are hitting well! :(")
    else:
        print("Probes are clearing well :)")

probe_fail, fail_names_2 = implantCM.in_collision_internal(return_names=True)
if probe_fail:
    print("Probes are striking implant! :(")
    print(f"problems: {list(fail_names_2)}")
else:
    print("Probes clear implant! :)")
S.add_geometry(implant_mesh)
with open(scene_save_path, "w") as f:
    S.export(f, file_type="obj")
S.show()

# %%
mouse_to_rig_ap = 14
ap_angle_rig_by_struct = {k: v + mouse_to_rig_ap for k, v in arc_ap_by_struct.items()}
bregma_dims = ["ML (mm)", "AP (mm)", "DV (mm)"]
plan_cols = {}
target_cols = {}
for structure in target_structures:
    if structure not in probe_tip_pts_by_struct:
        continue
    plan_cols.setdefault("Structure", []).append(structure)
    target_cols.setdefault("point", []).append(structure)
    plan_cols.setdefault("Probe type", []).append(probe_by_struct[structure])
    plan_cols.setdefault("Arc", []).append(arc_by_struct[structure])
    plan_cols.setdefault("AP angle", []).append(ap_angle_rig_by_struct[structure])
    plan_cols.setdefault("ML angle", []).append(slider_ml_by_struct[structure])
    plan_cols.setdefault("Spin", []).append(spin_by_struct[structure])
    plan_cols.setdefault("Hole", []).append("Hole {}".format(hole_by_struct[structure]))
    plan_cols.setdefault("Approx. depth", []).append(depth_by_struct[structure])

    target = probe_tip_pts_by_struct[structure]
    for dim, dim_val in zip(bregma_dims, target):
        plan_cols.setdefault(dim, []).append(np.round(dim_val, 3))
        target_cols.setdefault(dim, []).append(np.round(dim_val, 3))
    target_cols.setdefault("Source", []).append("insertion plan")

plan_df = (
    pd.DataFrame.from_dict(plan_cols).set_index("Structure").sort_values(by=["Arc"])
)
target_df = (
    pd.DataFrame.from_dict(target_cols).set_index("point").sort_values(by=["point"])
)
if plan_save_path is not None:
    plan_df.to_csv(plan_save_path)
plan_df
# %%
if target_save_path is not None:
    target_df.to_csv(target_save_path)
target_df
# %%
tgt_structure = "CLA"
this_probe = probe_to_target_mapping[tgt_structure]
this_affine = cal_by_probe_combined[this_probe][0]
this_ap, this_ml = find_probe_angle(this_affine)
R_probe_mesh = transform_matrix_from_angles(this_ap, this_ml)
R_probe_mesh[:3, :3]

# %%
name = []
ML = []
AP = []
DV = []
source = []
for i, structure in enumerate(target_structures):
    probe_type = probe_by_struct[structure]
    ap_angle = arcs[arc_by_struct[structure]]
    ml_angle = slider_ml_by_struct[structure]
    LP_offset = offsets_LP_arr_by_struct[structure]
    depth = depth_by_struct[structure]
    probe_model = probe_models[probe_type].copy()
    hole_number = hole_by_struct[structure]
    spin = spin_by_struct[structure]

    print(structure)
    print(f"AP: {ap_angle + 14}; ML: {ml_angle}; Spin: {spin}")
    print(f"Hole: {hole_number}")
    hole_coord = transformed_implant_targets[hole_number]
    offset = np.zeros(3)
    offset[:2] = LP_offset
    adjusted_insertion_pt = implant_target + offset

    hole_coord_ras = ras_to_lps @ hole_coord
    hole_ml, hole_ap, hole_dv = hole_coord_ras
    name.append(hole_number)
    print(f"Hole Target: ML: {hole_ml}  AP: {hole_ap} DV: {hole_dv}")
    ML.append(hole_ml)
    AP.append(hole_ap)
    DV.append(hole_dv)
    source.append("insertion plan")
    print(f"Distance past target: {depth}")
    print(f"Needs probe: {probe_type}")
    print("\n")
this_affine.T @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
