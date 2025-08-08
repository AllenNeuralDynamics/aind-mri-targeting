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

# %%
# Imports
import datetime
import logging
from pathlib import Path
from typing import Dict, List

import k3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_anatomical_utils.sitk_volume import find_points_equal_to
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
    mask_to_trimesh,
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
from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator
from scipy.spatial.transform import Rotation

# %%
# Uncomment the following line to enable interactive plotting in Jupyter
# notebooks
# %matplotlib ipympl

# %%
# Set the log verbosity to get debug statements
logging.basicConfig(format="%(message)s", level=logging.DEBUG)


# %%
# Attempt to use yml
class AppConfig(BaseModel):
    mouse: str
    from_calibration: bool
    target_structures: List[str]
    reticle_offset: List[float]
    reticle_rotation: float

    base_path: Path

    annotations_path: Path
    image_path: Path
    labels_path: Path
    brain_mask_path: Path
    image_transform_file: Path
    structure_mask_path: Path
    structure_files: Dict[str, Path]  # filled in dynamically
    brain_mesh_path: Path

    implant_annotation_path: Path
    headframe_transform_file: Path
    implant_file: Path
    implant_fit_transform_file: Path

    model_path: Path
    hole_model_path: Path
    modified_probe_mesh_file: Path
    probe_model_files: Dict[str, Path]

    headframe_file: Path
    cone_file: Path
    well_file: Path
    implant_model_file: Path

    calibration_path: Path
    calibration_file: Path
    parallax_calibration_dir: List[Path]

    plan_save_path: Path

    @field_validator("probe_model_files", mode="before")
    def ensure_string_keys(cls, v):
        return {str(k): v[k] for k in v}


# Load YAML with OmegaConf and resolve all ${...}
cfg = OmegaConf.load("/home/galen.lynch/786864-planning-config.yml")
resolved = OmegaConf.to_container(cfg, resolve=True)

# Build structure_files dynamically from target_structures
structure_mask_path = Path(resolved["structure_mask_path"])
mouse = resolved["mouse"]
resolved["structure_files"] = {
    struct: structure_mask_path / f"{mouse}-{struct}-Mask.nrrd" for struct in resolved["target_structures"]
}

# Create validated Pydantic model
app_config = AppConfig(**resolved)

print(app_config.structure_files["PL"])

# %%
# Global configuration

# Reticle offsets and rotations
# File Paths
# Image and image annotations.
structure_files = app_config.structure_files
brain_mesh_path = app_config.brain_mesh_path

# Implant annotation
# Note that this can be different than the image annotation,
# this is in the event that an instion is planned with data from multiple scans
# (see 750107 for example).
implant_annotation_path = app_config.implant_annotation_path
headframe_transform_file = app_config.headframe_transform_file
implant_file = app_config.implant_file
implant_fit_transform_file = app_config.implant_fit_transform_file


# OBJ files
model_path = app_config.model_path
hole_model_path = app_config.hole_model_path
modified_probe_mesh_file = app_config.modified_probe_mesh_file


probe_model_files = app_config.probe_model_files

headframe_file = app_config.headframe_file
cone_file = app_config.cone_file
well_file = app_config.well_file
implant_model_file = app_config.implant_model_file

calibration_path = app_config.calibration_path
calibration_file = app_config.calibration_file
parallax_calibration_dir = app_config.parallax_calibration_dir
iso_time = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
plan_save_path = app_config.plan_save_path

# %%
# Load the transforms
image_R, image_t, image_c = load_sitk_transform(str(app_config.image_transform_file))
headframe_R, headframe_t, headframe_c = load_sitk_transform(str(app_config.headframe_transform_file))
implant_R, implant_t, implant_c = load_sitk_transform(str(app_config.implant_fit_transform_file))

image = sitk.ReadImage(str(app_config.image_path))

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_file)
headframe_lps = convert_coordinate_system(headframe, "ASR", "LPS")  # Preserves shape!

# Load the headframe
cone, cone_faces = get_vertices_and_faces(cone_file)
cone_lps = convert_coordinate_system(cone, "ASR", "LPS")  # Preserves shape!

well, well_faces = get_vertices_and_faces(well_file)
well_lps = convert_coordinate_system(well, "ASR", "LPS")  # Preserves shape!

implant_model, implant_faces = get_vertices_and_faces(implant_model_file)
implant_model_lps = convert_coordinate_system(implant_model, "ASR", "LPS")  # Preserves shape!

# Load the brain mask
brain_mask_img = sitk.ReadImage(str(app_config.brain_mask_path))
brain_pos = find_points_equal_to(brain_mask_img)  # Get the points where the brain mask is equal to 1
# Downsample the brain mask to 1000 points for visualization
brain_pos = brain_pos[np.arange(0, brain_pos.shape[0], brain_pos.shape[0] // 1000)]


implant_seg_vol = sitk.ReadImage(str(implant_file))

probe_models = {k: load_newscale_trimesh(v, 0) for k, v in probe_model_files.items()}

# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = hole_model_path.glob("Hole*.obj")
hole_dict = {}
for f in hole_files:
    hole_num = int(f.stem.split("Hole")[-1])
    hole_mesh = trimesh.load(str(f))
    hole_mesh.vertices = convert_coordinate_system(hole_mesh.vertices, "ASR", "LPS")  # Preserves shape!
    hole_dict[hole_num] = hole_mesh

# Get the lower face, store with key -1
hole_mesh = trimesh.load(str(hole_model_path / "LowerFace.obj"))
hole_mesh.vertices = convert_coordinate_system(hole_mesh.vertices, "ASR", "LPS")  # Preserves shape!
hole_dict[-1] = hole_mesh


# %%
model_implant_targets = {}
for hole_id in hole_dict.keys():
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

# %%
# If implant has holes that are segmented.

implant_targets_by_hole = make_hole_seg_dict(implant_seg_vol, fun=lambda x: np.mean(x, axis=0))
implant_names = list(implant_targets_by_hole.keys())
implant_targets = np.vstack(list(implant_targets_by_hole.values()))

# %%
# Apply the chemical shift correction to the image
chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(compute_chemical_shift(image, ppm=3.7))
chem_shift_image_R, chem_shift_image_t = invert_rotate_translate(chem_shift_pt_R, chem_shift_pt_t)
chem_image_R, chem_image_t = compose_transforms(chem_shift_image_R, chem_shift_image_t, image_R, image_t)

# %%
# Apply the image transform to the brain mask
transformed_brain = apply_rotate_translate(brain_pos, *invert_rotate_translate(chem_image_R, chem_image_t))
transformed_implant_targets = apply_rotate_translate(
    implant_targets, *invert_rotate_translate(headframe_R, headframe_t)
)

# %%
# Find calibrated probes

# Read the calibrations
if calibration_file is None:
    cal_by_probe_combined, R_reticle_to_bregma = fit_rotation_params_from_parallax(
        parallax_calibration_dir,
        app_config.reticle_offset,
        app_config.reticle_rotation,
    )
    global_offset = app_config.reticle_offset
else:
    cal_by_probe_combined, R_reticle_to_bregma, global_offset = combine_parallax_and_manual_calibrations(
        manual_calibration_files=calibration_file,
        parallax_directories=parallax_calibration_dir,
    )

probes_used = list(cal_by_probe_combined.keys())

# %%
probe_to_target_mapping = {
    "PL": 46811,
    "CLA": 46810,
    "MD": 50213,
    "VM": 50210,
    "CA1": 50720,
    "BLA": 50197,
    "RSP": 46101,
}

# %%
# Recording configuration
arcs = {
    "a": 17,
    "b": -8,
    "c": -24,
    "d": -42,
}
probe_info_by_struct = {
    "PL": {"type": "quadbase", "arc": "b", "slider_ml": -30, "spin": 135},
    "CLA": {"type": "2.1", "arc": "d", "slider_ml": -27, "spin": 0},
    "MD": {"type": "quadbase", "arc": "a", "slider_ml": -10, "spin": -45},
    "CA1": {"type": "2.1", "arc": "c", "slider_ml": 6, "spin": -180},
    "VM": {"type": "2.1", "arc": "c", "slider_ml": -20, "spin": 0},
    "BLA": {"type": "2.1", "arc": "b", "slider_ml": 27, "spin": 50},
    "RSP": {"type": "2.1", "arc": "c", "slider_ml": 22, "spin": -100},
}
target_info_by_struct = {
    "PL": {"offsets": np.array([-0.25, 0.25]), "depth": 2.4, "hole": 1},
    "CLA": {"offsets": np.array([0.0, -0.4]), "depth": 5.5, "hole": 12},
    "MD": {"offsets": np.array([0, -0.35]), "depth": 4, "hole": 3},
    "CA1": {"offsets": np.array([-0.1, 0.4]), "depth": 6.5, "hole": 10},
    "VM": {"offsets": np.array([0, 0.2]), "depth": 4.9, "hole": 6},
    "BLA": {"offsets": np.array([0, 0.1]), "depth": 5.75, "hole": 4},
    "RSP": {"offsets": np.array([0, 0]), "depth": 2, "hole": 5},
}

insertion_trim_coordinates_struct = {}

# %%
plot = k3d.plot()
plot.display()
plot.grid_visible = False

S = trimesh.Scene()
transformed_model_implant_targets = {}

for key in model_implant_targets.keys():
    implant_tgt = model_implant_targets[key]
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(implant_R, implant_t))
    implant_tgt = apply_rotate_translate(implant_tgt, *invert_rotate_translate(headframe_R, headframe_t))
    transformed_model_implant_targets[key] = implant_tgt

vertices = apply_rotate_translate(implant_model_lps, *invert_rotate_translate(implant_R, implant_t))
vertices = apply_rotate_translate(vertices, *invert_rotate_translate(headframe_R, headframe_t))
implant_mesh = trimesh.Trimesh(vertices=vertices, faces=implant_faces[0])


# holeCM = trimesh.collision.CollisionManager()
implantCM = trimesh.collision.CollisionManager()
probeCM = trimesh.collision.CollisionManager()
coneCM = trimesh.collision.CollisionManager()
wellCM = trimesh.collision.CollisionManager()

ras_to_lps = np.diag([-1, -1, 1])
final_target_by_struct = {}
original_structure_meshes = {}
structure_calibration_angles = {}

for structure in app_config.target_structures:
    structureCM = trimesh.collision.CollisionManager()

    probe_type = probe_info_by_struct["type"][structure]
    ap_angle = arcs[probe_info_by_struct["arc"][structure]]
    ml_angle = probe_info_by_struct["slider_ml"][structure]
    LP_offset = target_info_by_struct["offsets"][structure]
    depth = target_info_by_struct["depth"][structure]
    probe_model = probe_models[probe_type].copy()
    hole_number = target_info_by_struct["hole"][structure]
    spin = probe_info_by_struct["spin"][structure]

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

    if app_config.from_calibration:
        this_probe = probe_to_target_mapping[structure]
        this_affine, this_translation = cal_by_probe_combined[this_probe]
        insertion_vector = ras_to_lps @ np.linalg.inv(this_affine) @ np.array([0, 0, depth])
        insertion_trims = insertion_trim_coordinates_struct.get(structure, np.array([0, 0, 0]))
        combined_bregma_vector = adjusted_insertion_pt + insertion_vector + insertion_trims
        this_ap, this_ml = find_probe_angle(this_affine)
        structure_calibration_angles[structure] = (this_ap, this_ml)
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
    if structure in original_structure_meshes:
        this_target_mesh = original_structure_meshes[structure].copy()
    else:
        this_target_mesh = mask_to_trimesh(sitk.ReadImage(str(structure_files[structure])))
        trimesh.repair.fix_normals(this_target_mesh)
        trimesh.repair.fix_inversion(this_target_mesh)
        original_structure_meshes[structure] = this_target_mesh.copy()

    vertices = this_target_mesh.vertices
    vertices = apply_rotate_translate(vertices, *invert_rotate_translate(chem_image_R, chem_image_t))
    this_target_mesh.vertices = vertices

    # Assign the same color to the structure
    this_target_mesh.visual.face_colors = this_color
    this_target_mesh.visual.vertex_colors = this_color
    this_target_mesh.visual.main_color[:] = this_color

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
for structure in app_config.target_structures:
    if structure not in final_target_by_struct:
        continue
    cols.setdefault("Structure", []).append(structure)
    cols.setdefault("Probe type", []).append(probe_info_by_struct["type"][structure])
    cols.setdefault("Arc", []).append(probe_info_by_struct["arc"][structure])
    cols.setdefault("AP angle", []).append(ap_angle_rig_by_struct[structure])
    cols.setdefault("ML angle", []).append(probe_info_by_struct["slider_ml"][structure])
    cols.setdefault("Spin", []).append(probe_info_by_struct["spin"][structure])
    cols.setdefault("Hole", []).append("Hole {}".format(target_info_by_struct["hole"][structure]))
    cols.setdefault("Approx. depth", []).append(target_info_by_struct["depth"][structure])

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
for structure in app_config.target_structures:
    probe_type = probe_info_by_struct["type"][structure]
    ap_angle = arcs[probe_info_by_struct["arc"][structure]]
    ml_angle = probe_info_by_struct["slider_ml"][structure]
    LP_offset = offsets_LP_arr_by_struct[structure]
    depth = target_info_by_struct["depth"][structure]
    probe_model = probe_models[probe_type].copy()
    hole_number = target_info_by_struct["hole"][structure]
    spin = probe_info_by_struct["spin"][structure]

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
