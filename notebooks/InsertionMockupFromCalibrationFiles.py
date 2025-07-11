# ---
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

import datetime
import logging

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils import coordinate_systems as cs
from aind_mri_utils.arc_angles import arc_angles_to_affine
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
)
from aind_mri_utils.reticle_calibrations import (
    combine_parallax_and_manual_calibrations,
    find_probe_angle,
    fit_rotation_params_from_parallax,
)
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    compose_transforms,
    invert_rotate_translate,
)
from scipy.spatial.transform import Rotation

# %%
# %matplotlib ipympl

# %%
# Set the log verbosity to get debug statements
logging.basicConfig(format="%(message)s", level=logging.DEBUG)
# %%
# File Paths
mouse = "786866"
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
brain_mask_path = annotations_path / "{}_manual_skull_strip.nrrd".format(mouse)
image_transform_file = annotations_path / "com_plane.h5"
structure_mask_path = annotations_path / "Masks"
structure_files = {structure: structure_mask_path / f"{mouse}{structure}Mask.obj" for structure in target_structures}
brain_mesh_path = structure_mask_path / "{}_manual_skull_strip.obj".format(mouse)

# Implant annotation
# Note that this can be different than the image annotation,
# this is in the event that an instion is planned with data from multiple scans
# (see 750107 for example).
implant_annoation_path = base_path / "ephys/persist/data/MRI/processed/{}".format(mouse)
headframe_transform_file = implant_annoation_path / "com_plane.h5"
implant_file = implant_annoation_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
implant_mesh_file = implant_annoation_path / "{}_ImplantHoles.obj".format(mouse)
implant_fit_transform_file = implant_annoation_path / "{}_implant_fit.h5".format(mouse)


# OBJ files
model_path = base_path / "ephys/persist/data/MRI/HeadframeModels"
hole_model_path = model_path / "HoleOBJs"
modified_probe_mesh_file = model_path / "modified_probe_holder.obj"


probe_model_files = {
    "2.1-alpha": model_path / "Centered_Newscale_2pt0.obj",
    "2.1": model_path / "dovetailtweezer_oneShank_centered_corrected.obj",
    "quadbase": model_path / "Quadbase_customHolder_centeredOnShank0.obj",
    "2.4": model_path / "dovetailwtweezer_fourShank_centeredOnShank0.obj",
    "pipette": model_path / "injection_pipette.obj",
}

headframe_file = model_path / "TenRunHeadframe.obj"
holes_file = model_path / "OneOff_HolesOnly.obj"
cone_file = model_path / "TacoForBehavior" / "0160-200-72_X06.obj"
well_file = model_path / "WHC_Well" / "0274-400-07_X02.obj"
implant_model_file = model_path / "0283-300-04.obj"

calibration_path = base_path / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
calibration_file = calibration_path / "calibration_info_np2_2025_03_18T08_39_00.xlsx"
parallax_calibration_dir = calibration_path / "log_20250303_122136"
iso_time = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
plan_save_path = annotations_path / f"{mouse}_InsertionPlan_{iso_time}.csv"

# %%
# Reticle offsets and rotations
reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}
reticle_rotations = {"H": 0}

# %%
image_R, image_t, image_c = load_sitk_transform(str(image_transform_file))
headframe_R, headframe_t, headframe_c = load_sitk_transform(str(headframe_transform_file))

image = sitk.ReadImage(str(image_path))

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_file)
headframe_lps = cs.convert_coordinate_system(headframe, "ASR", "LPS")  # Preserves shape!

# Load the headframe
cone, cone_faces = get_vertices_and_faces(cone_file)
cone_lps = cs.convert_coordinate_system(cone, "ASR", "LPS")  # Preserves shape!

well, well_faces = get_vertices_and_faces(well_file)
well_lps = cs.convert_coordinate_system(well, "ASR", "LPS")  # Preserves shape!

implant_model, implant_faces = get_vertices_and_faces(implant_model_file)
implant_model_lps = cs.convert_coordinate_system(implant_model, "ASR", "LPS")  # Preserves shape!

# Load the brain mask
mask = sitk.ReadImage(str(brain_mask_path))
idxx = np.where(sitk.GetArrayViewFromImage(mask))
idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
brain_pos = np.zeros(idx.shape)
brain_pos = np.vstack([mask.TransformIndexToPhysicalPoint(idx[ii, :].tolist()) for ii in range(idx.shape[0])])
brain_pos = brain_pos[np.arange(0, brain_pos.shape[0], brain_pos.shape[0] // 1000)]

# Load the brain mesh
brain_mesh = trimesh.load(
    str(brain_mesh_path),
    force="Mesh",
)

implant_seg_vol = sitk.ReadImage(str(implant_file))

probe_models = {k: load_newscale_trimesh(v, 0) for k, v in probe_model_files.items()}

# %%

# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = [x for x in os.listdir(hole_model_path) if ".obj" in x and "Hole" in x]
hole_dict = {}
for i, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_model_path, flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_model_path, "LowerFace.obj"))
hole_dict[-1].vertices = cs.convert_coordinate_system(hole_dict[-1].vertices, "ASR", "LPS")  # Preserves shape!


# %%
model_implant_targets = {}
for i, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

# %%
# If implant has holes that are segmented.

implant_targets_by_hole = make_hole_seg_dict(implant_seg_vol, fun=lambda x: np.mean(x, axis=0))
implant_names = list(implant_targets_by_hole.keys())
implant_targets = np.vstack(list(implant_targets_by_hole.values()))

# %%
chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(compute_chemical_shift(image, ppm=3.7))
chem_shift_image_R, chem_shift_image_t = invert_rotate_translate(chem_shift_pt_R, chem_shift_pt_t)
chem_image_R, chem_image_t = compose_transforms(chem_shift_image_R, chem_shift_image_t, image_R, image_t)

# %%

transformed_brain = apply_rotate_translate(brain_pos, *invert_rotate_translate(chem_image_R, chem_image_t))
transformed_brain_mesh = apply_transform_to_trimesh(
    brain_mesh.copy(), *invert_rotate_translate(chem_image_R, chem_image_t)
)
transformed_implant_targets = apply_rotate_translate(
    implant_targets, *invert_rotate_translate(headframe_R, headframe_t)
)

# %%
# Find calibrated probes

# Read the calibrations
reticle_offset = reticle_offsets[reticle_used]
reticle_rotation = reticle_rotations[reticle_used]
if calibration_file is None:
    cal_by_probe_combined, R_reticle_to_bregma = fit_rotation_params_from_parallax(
        parallax_calibration_dir,
        reticle_offset,
        reticle_rotation,
        find_scaling=True,
    )
    global_offset = reticle_offset
else:
    cal_by_probe_combined, R_reticle_to_bregma, global_offset = combine_parallax_and_manual_calibrations(
        manual_calibration_files=calibration_file,
        parallax_directories=parallax_calibration_dir,
    )

probes_used = list(cal_by_probe_combined.keys())

# %%
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
    "a": 18,
    "b": 2,
    "c": -12.745,
    "d": -30,
}
probe_by_struct = {
    "PL": "quadbase",
    "CLA": "2.1-alpha",
    "MD": "2.4",
    "CA1": "2.1-alpha",
    "VM": "2.1-alpha",
    "BLA": "2.1",
    "RSP": "2.1",
}
arc_by_struct = {
    "PL": "a",
    "CLA": "a",
    "MD": "c",
    "CA1": "b",
    "VM": "b",
    "BLA": "d",
    "RSP": "d",
}
arc_ap_by_struct = {k: arcs[v] for k, v in arc_by_struct.items()}
slider_ml_by_struct = {
    "PL": -20,
    "CLA": 29.5,
    "MD": -23,
    "CA1": 25,
    "VM": -19,
    "BLA": 3,
    "RSP": 19,
}
spin_by_struct = {
    "PL": -45,
    "CLA": 145,
    "MD": 90,
    "CA1": 90,
    "VM": -90,
    "BLA": 110,
    "RSP": 135,
}

# This should be the x offset and the inverse of the y offset
offsets_LP_by_struct = {
    "PL": [-0.25, 0.25],
    "CLA": [0.0, -0.4],
    "MD": [0, -0.35],
    "CA1": [-0.1, 0.4],
    "VM": [0, 0.2],
    "BLA": [0, 0.1],
    "RSP": [0, 0],
}
offsets_LP_arr_by_struct = {k: np.array(v) for k, v in offsets_LP_by_struct.items()}

target_depth = np.array([2.4, 4.75, 4, 5.8, 4.9, 5.75, 2])  # Guesses, check
depth_by_struct = {
    "PL": 2.4,
    "CLA": 5.5,
    "MD": 4,
    "CA1": 6.5,
    "VM": 4.9,
    "BLA": 5.75,
    "RSP": 2,
}

hole_by_struct = {
    "PL": 1,
    "CLA": 2,
    "MD": 6,
    "CA1": 4,
    "VM": 8,
    "BLA": 10,
    "RSP": 5,
}

insertion_trim_coordinates_struct = {}

# %%
implant_R, implant_t, implant_c = load_sitk_transform(implant_fit_transform_file)

S = trimesh.Scene()
transformed_implant_targets = {}

for i, key in enumerate(model_implant_targets.keys()):
    implant_tgt = model_implant_targets[key]
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(implant_R, implant_t))
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(headframe_R, headframe_t))
    transformed_implant_targets[key] = implant_tgt

vertices = apply_rotate_translate(implant_model_lps, *invert_rotate_translate(implant_R, implant_t))
vertices = apply_rotate_translate(vertices, *invert_rotate_translate(headframe_R, headframe_t))
implant_mesh = trimesh.Trimesh(vertices=vertices, faces=implant_faces[0])


# holeCM = trimesh.collision.CollisionManager()
implantCM = trimesh.collision.CollisionManager()
probeCM = trimesh.collision.CollisionManager()
coneCM = trimesh.collision.CollisionManager()
wellCM = trimesh.collision.CollisionManager()

from_calibration = True
ras_to_lps = np.diag([-1, -1, 1])
final_target_by_struct = {}
for i, structure in enumerate(target_structures):
    if structure not in [
        # "PL",
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
        insertion_vector = ras_to_lps @ np.linalg.inv(this_affine) @ np.array([0, 0, depth])
        insertion_trims = insertion_trim_coordinates_struct.get(structure, np.array([0, 0, 0]))
        combined_bregma_vector = adjusted_insertion_pt + insertion_vector + insertion_trims
        scaling_inv = np.diag(1 / this_scaling)
        rigid_affine = scaling_inv @ this_affine
        this_ap, this_ml = find_probe_angle(rigid_affine)

        R_probe_mesh = arc_angles_to_affine(this_ap, this_ml, spin)

        probe_model = apply_transform_to_trimesh(probe_model, R_probe_mesh, combined_bregma_vector)
    else:
        R_probe_mesh = arc_angles_to_affine(ap_angle, ml_angle, spin)
        insertion_vector = R_probe_mesh @ np.array([0, 0, -depth])
        combined_bregma_vector = adjusted_insertion_pt + insertion_vector
        probe_model = apply_transform_to_trimesh(probe_model, R_probe_mesh, combined_bregma_vector)

    final_target_by_struct[structure] = ras_to_lps @ combined_bregma_vector

    S.add_geometry(probe_model)
    probeCM.add_object(structure, probe_model)
    implantCM.add_object(structure, probe_model)
    coneCM.add_object(structure, probe_model)
    wellCM.add_object(structure, probe_model)

    structureCM.add_object("probe", probe_model)

    # Load and transform the target structure
    this_target_mesh = trimesh.load(
        str(structure_files[structure]),
        force="Mesh",
    )
    trimesh.repair.fix_normals(this_target_mesh)
    trimesh.repair.fix_inversion(this_target_mesh)

    vertices = this_target_mesh.vertices
    vertices = apply_rotate_translate(vertices, *invert_rotate_translate(chem_image_R, chem_image_t))
    this_target_mesh.vertices = vertices

    # Assign the same color to the structure
    this_target_mesh.visual.face_colors = this_color
    this_target_mesh.visual.vertex_colors = this_color
    this_target_mesh.visual.material.main_color[:] = this_color

    # Add structure to the scene and collision manager
    S.add_geometry(this_target_mesh)
    structureCM.add_object("structure", this_target_mesh)

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
S.show(viewer="gl")

# %%
mouse_to_rig_ap = 14
ap_angle_rig_by_struct = {k: v + mouse_to_rig_ap for k, v in arc_ap_by_struct.items()}
bregma_dims = ["ML", "AP", "DV"]
cols = {}
for structure in target_structures:
    if structure not in final_target_by_struct:
        continue
    cols.setdefault("Structure", []).append(structure)
    cols.setdefault("Probe type", []).append(probe_by_struct[structure])
    cols.setdefault("Arc", []).append(arc_by_struct[structure])
    cols.setdefault("AP angle", []).append(ap_angle_rig_by_struct[structure])
    cols.setdefault("ML angle", []).append(slider_ml_by_struct[structure])
    cols.setdefault("Spin", []).append(spin_by_struct[structure])
    cols.setdefault("Hole", []).append("Hole {}".format(hole_by_struct[structure]))
    cols.setdefault("Approx. depth", []).append(depth_by_struct[structure])

    target = final_target_by_struct[structure]
    for dim, dim_val in zip(bregma_dims, target):
        cols.setdefault(dim, []).append(np.round(dim_val, 3))

plan_df = pd.DataFrame.from_dict(cols).set_index("Structure").sort_values(by=["Arc"])
if plan_save_path is not None:
    plan_df.to_csv(plan_save_path)
plan_df
# %%
tgt_structure = "CLA"
this_probe = probe_to_target_mapping[tgt_structure]
this_affine = cal_by_probe_combined[this_probe][0]
this_ap, this_ml = find_probe_angle(this_affine)
R_probe_mesh = arc_angles_to_affine(this_ap, this_ml)
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

# %%

R = Rotation.from_matrix(this_affine.T @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])).as_euler("xyz")
R

# %%
R = Rotation.from_matrix(R_probe_mesh[:3, :3]).as_euler("xyz")
R

# %%
