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

import aind_mri_utils.reticle_calibrations as rc
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3

# %%
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

implant_cmap = matplotlib.cm.get_cmap("rainbow")

import os
from pathlib import Path

import pandas as pd
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils import rotations as rot
from aind_mri_utils.arc_angles import transform_matrix_from_angles
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.file_io.slicer_files import (
    find_seg_nrrd_header_segment_info,
    load_segmentation_points,
)
from aind_mri_utils.meshes import (
    apply_transform_to_trimesh,
    load_newscale_trimesh,
)
from aind_mri_utils.optimization import get_headframe_hole_lines
from aind_mri_utils.plots import get_prop_cycle
from aind_mri_utils.reticle_calibrations import find_probe_angle
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    compose_transforms,
    invert_rotate_translate,
)
from scipy.spatial.transform import Rotation

colors = get_prop_cycle()


def create_single_colormap(
    colorname,
    N=256,
    saturation=0,
    start_color="white",
    is_transparent=True,
    is_reverse=False,
):
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    cmap = ListedColormap([start_color, colorname])
    start_color = np.array(cmap(0))
    if is_transparent:
        start_color[-1] = 0
    if not is_reverse:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.linspace(start_color, cmap(1), N),
                    np.tile(cmap(1), (int(saturation * N), 1)),
                )
            )
        )
    else:
        cmap = ListedColormap(
            np.vstack(
                (
                    np.tile(cmap(1), (int(saturation * N), 1)),
                    np.linspace(cmap(1), start_color, N),
                )
            )
        )
    return cmap


def define_transform(source_landmarks, target_landmarks):
    """
    Defines a non-linear warp between a set of source and target landmarks

    Parameters
    ==========
    source_landmarks - np.ndarray (N x 3)
    target_landmarks - np.ndarray (N x 3)

    Returns
    =======
    transform - vtkThinPlateSplineTransform

    """

    transform = vtk.vtkThinPlateSplineTransform()
    source_points = vtk.vtkPoints()
    target_points = vtk.vtkPoints()

    for i in range(source_landmarks.shape[0]):
        source_points.InsertNextPoint(source_landmarks[i, :])

    for i in range(target_landmarks.shape[0]):
        target_points.InsertNextPoint(target_landmarks[i, :])

    transform.SetBasisToR()  # for 3D transform
    transform.SetSourceLandmarks(source_points)
    transform.SetTargetLandmarks(target_points)
    transform.Update()

    return transform


import trimesh

# %%
# %matplotlib ipympl
# %%
# File Paths
mouse = "771432"
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
annotations_path = base_path / "ephys/persist/data/MRI/processed/{}".format(
    mouse
)
image_path = annotations_path / "{}_100.nii.gz".format(mouse)
labels_path = annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse)
brain_mask_path = annotations_path / "{}_auto_skull_strip.nrrd".format(mouse)
image_transform_file = annotations_path / "com_plane.h5".format(mouse)
structure_mask_path = annotations_path / "Masks"
structure_files = {
    structure: structure_mask_path / f"{mouse}{structure}Mask.obj"
    for structure in target_structures
}
brain_mesh_path = structure_mask_path / "{}_auto_skull_strip.obj".format(mouse)

# Implant annotation
# Note that this can be different than the image annotation,
# this is in the event that an instion is planned with data from multiple scans (see 750107 for example).
implant_annoation_path = (
    base_path / "ephys/persist/data/MRI/processed/{}".format(mouse)
)
headframe_transform_file = implant_annoation_path / "com_plane.h5".format(
    mouse
)
implant_file = implant_annoation_path / "{}_ImplantHoles.seg.nrrd".format(
    mouse
)
implant_mesh_file = implant_annoation_path / "{}_ImplantHoles.obj".format(
    mouse
)
implant_fit_transform_file = (
    implant_annoation_path / "{}_implant_fit.h5".format(mouse)
)


# OBJ files
model_path = base_path / "ephys/persist/data/MRI/HeadframeModels"
hole_model_path = model_path / "HoleOBJs"
modified_probe_mesh_file = model_path / "modified_probe_holder.obj"
dovetail_tweezer_file = (
    model_path / "dovetailtweezer_oneShank_centered_corrected.obj"
)
dovetail_tweezer_4shank_file = (
    model_path / "dovetailwtweezer_fourShank_centeredOnShank0.obj"
)
quadbase_file = model_path / "Quadbase_customHolder_centeredOnShank0.obj"

newscale_model_file = model_path / "Centered_Newscale_2pt0.obj"
headframe_file = model_path / "TenRunHeadframe.obj"
holes_file = model_path / "OneOff_HolesOnly.obj"
cone_file = model_path / "TacoForBehavior" / "0160-200-72_X06.obj"
well_file = model_path / "WHC_Well" / "0274-400-07_X02.obj"
implant_model_file = model_path / "0283-300-04.obj"

calibration_path = (
    base_path / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
)
calibration_file = (
    calibration_path / "calibration_info_np2_2025_03_03T14_14_00.xlsx"
)

# Save file paths
transform_save_file = annotations_path / "{}_test.h5".format(mouse)


# Magic numbers
resolution = 100

# %%
image_R, image_t, image_c = load_sitk_transform(
    str(image_transform_file)
)  #'_with_plane.h5')))
headframe_R, headframe_t, headframe_c = load_sitk_transform(
    str(headframe_transform_file)
)  #'_with_plane.h5')))

# Handle inconsistant labeling
label_vol = sitk.ReadImage(str(labels_path))
odict = {k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()}
insert_underscores = (
    "_" in list(find_seg_nrrd_header_segment_info(odict).keys())[0]
)

# Load the points on the headframe lines.
pts1, pts2, order = get_headframe_hole_lines(
    insert_underscores=insert_underscores, coordinate_system="LPS"
)

# order.remove('anterior_vertical')

image = sitk.ReadImage(str(image_path))
fiducial_positions, _, _ = load_segmentation_points(
    str(labels_path), order=order, image=image
)

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


# Load just the headframe holes
holes, holes_faces = get_vertices_and_faces(holes_file)
holes_faces = holes_faces[-1]
holes_lps = cs.convert_coordinate_system(holes, "ASR", "LPS")

# Load the brain mask
mask = sitk.ReadImage(str(brain_mask_path))
idxx = np.where(sitk.GetArrayViewFromImage(mask))
idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
brain_pos = np.zeros(idx.shape)
brain_pos = np.vstack(
    [
        mask.TransformIndexToPhysicalPoint(idx[ii, :].tolist())
        for ii in range(idx.shape[0])
    ]
)
brain_pos = brain_pos[
    np.arange(0, brain_pos.shape[0], brain_pos.shape[0] // 1000)
]

# Load the brain mesh
brain_mesh = trimesh.load(
    str(brain_mesh_path),
    force="Mesh",
)

# %%

hole_folder = hole_model_path
# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = [
    x for x in os.listdir(hole_folder) if ".obj" in x and "Hole" in x
]
hole_dict = {}
for ii, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder, flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_folder, "LowerFace.obj"))
hole_dict[-1].vertices = cs.convert_coordinate_system(
    hole_dict[-1].vertices, "ASR", "LPS"
)  # Preserves shape!


model_implant_targets = {}
for ii, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

# %%
# If implant has holes that are segmented.
implant_vol = sitk.ReadImage(str(implant_file))
odict = {k: implant_vol.GetMetaData(k) for k in implant_vol.GetMetaDataKeys()}
label_dict = find_seg_nrrd_header_segment_info(odict)

implant_names = []
implant_targets = []
implant_pts = []

for ii, key in enumerate(label_dict.keys()):
    filt = sitk.EqualImageFilter()
    is_label = filt.Execute(implant_vol, label_dict[key])
    idxx = np.where(sitk.GetArrayViewFromImage(is_label))
    idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
    implant_pos = np.vstack(
        [
            implant_vol.TransformIndexToPhysicalPoint(idx[ii, :].tolist())
            for ii in range(idx.shape[0])
        ]
    )
    implant_pts.append(implant_pos)
    implant_targets.append(np.mean(implant_pos, axis=0))
    this_key = key.split("-")[-1].split("_")[-1]
    implant_names.append(int(this_key))
implant_targets = np.vstack(implant_targets)

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
implant_model, implant_faces = get_vertices_and_faces(implant_model_file)
implant_model_lps = cs.convert_coordinate_system(
    implant_model, "ASR", "LPS"
)  # Preserves shape!

transformed_brain = apply_rotate_translate(
    brain_pos, *invert_rotate_translate(chem_image_R, chem_image_t)
)
transformed_brain_mesh = apply_transform_to_trimesh(
    brain_mesh.copy(), *invert_rotate_translate(chem_image_R, chem_image_t)
)
transformed_implant_targets = apply_rotate_translate(
    implant_targets, *invert_rotate_translate(headframe_R, headframe_t)
)
transformed_fidicuals = apply_rotate_translate(
    fiducial_positions, *invert_rotate_translate(image_R, image_t)
)

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


probe_cal_file = calibration_file

fit_scale = False
verbose = False

rotations = dict()
translations = dict()

# First compute the rotation and translation for the probe_cal File
(
    adjusted_pairs_by_probe,
    global_offset,
    global_rotation_degrees,
    reticle_name,
) = rc.read_reticle_calibration(probe_cal_file)

# # Flip AP,ML convention for plotting
# for ii, probe in enumerate(adjusted_pairs_by_probe.keys()):
#     adjusted_pairs_by_probe[probe][0][:,:] = -adjusted_pairs_by_probe[probe][0][:,:]

if fit_scale:
    scale_vecs = dict()
    for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
        (rotation, scale, translation) = rc._fit_params_with_scaling(
            reticle_pts,
            probe_pts,
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
    predicted_probe_pts = rc.transform_bregma_to_probe(
        reticle_pts, rotations[probe], translations[probe], scale
    )
    # in µm
    errs = 1000 * np.linalg.norm(predicted_probe_pts - probe_pts, axis=1)
    if verbose:
        print(
            f"Probe {probe}: Mean error {errs.mean():.2f} µm, "
            f"max error {errs.max():.2f} µm"
        )
        original_reticle_pts = reticle_pts - global_offset
        for i in range(len(errs)):
            rounded_pred = np.round(predicted_probe_pts[i], decimals=2)
            print(
                f"\tReticle {original_reticle_pts[i]} -> "
                f"Probe {probe_pts[i]}: predicted {rounded_pred} "
                f"error {errs[i]:.2f} µm"
            )

# %%
calib_dir = calibration_path / "log_20250303_122136"
reticle_offsets = {"H": np.array([0.076, 0.062, 0.311])}
reticle_used = "H"
find_scaling = False
# Find calibrated probes
pts_df = pd.read_csv(os.path.join(calib_dir, "points.csv"))
manips_used = [x.split("SN")[-1] for x in np.unique(pts_df.sn)]

# Read the calibrations
adjusted_pairs_by_probe = dict()
global_offset = reticle_offsets[reticle_used]
global_rotation_degrees = 0
reticle_name = reticle_used
for manip in manips_used:
    manip_f = [
        f
        for f in os.listdir(calib_dir)
        if f.startswith("points_SN" + str(manip))
    ]
    fname = calib_dir / f"{manip_f[-1]}"
    pairs = pairs_from_parallax_points_csv(fname)
    reticle_pts, manip_pts = rc._apply_metadata_to_pair_lists(
        pairs, 1 / 1000, global_rotation_degrees, global_offset, 1 / 1000
    )
    adjusted_pairs_by_probe[manip] = (reticle_pts, manip_pts)

# # Flip AP,ML convention for plotting
# for ii, probe in enumerate(adjusted_pairs_by_probe.keys()):
#     adjusted_pairs_by_probe[probe][0][:,:] = -adjusted_pairs_by_probe[probe][0][:,:]

# Compute the matrices
if fit_scale:
    scale_vecs = dict()
    for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
        (rotation, scale, translation, _) = rc.fit_rotation_params(
            reticle_pts, probe_pts, find_scaling=True
        )
        probe = int(probe)
        rotations[probe] = rotation
        scale_vecs[probe] = scale
        translations[probe] = translation
else:
    for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
        probe = int(probe)

        rotation, translation, _ = rc.fit_rotation_params(
            reticle_pts, probe_pts, find_scaling=False
        )
        rotations[probe] = rotation
        translations[probe] = translation

for probe, (reticle_pts, probe_pts) in adjusted_pairs_by_probe.items():
    probe = int(probe)

    if fit_scale:
        scale = scale_vecs[probe]
    else:
        scale = None
    predicted_probe_pts = rc.transform_bregma_to_probe(
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
# Adapter rotation
lps_to_ras = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
# %%
# manipulator coordinate system is normally RPI, but now LAI
# These transforms go from reticle to probe coordinate system
# To determine the orientation of the probe relative to the reticle,
# we need to invert the rotation matrix and the translation vector
target_structure = "CA1"
probe = probe_to_target_mapping[target_structure]
this_rotation = rotations[probe]
this_translation = translations[probe]
probe_ndx = list(probe_to_target_mapping.keys()).index(target_structure)
implant_target = lps_to_ras @ transformed_implant_targets[probe_ndx].copy()
implant_target[0] += target_x_offset[probe_ndx]
implant_target[1] -= target_y_offset[probe_ndx]
np_target_pos = rc.transform_bregma_to_probe(
    implant_target, this_rotation, this_translation
)
np_z_axis_in_reticle_ras = rc.transform_probe_to_bregma(
    np.array([0, 0, 1]), this_rotation, np.zeros(3)
)


# %%
target_verts = len(target_structures) * [None]
target_faces = len(target_structures) * [None]
arc_angle = np.array(
    [
        15,
        -1,
        -17,
        -33,
    ]
)
arc_id = [0, 0, 2, 1, 1, 3, 3]  # CLA 0
probe_type = [3, 0, 4, 0, 0, 1, 1]

target_hole = [1, 2, 6, 4, 8, 10, 5]  # [1,  2, 3, 6 ,8 ,13,5] # Guesses, check
target_ml = [-20, 28.5, -28, +20, -26, 3, 3 + 16]  # CLA 30


target_ap = [arc_angle[x] for x in arc_id]
target_y_offset = [-0.25, 0.2, 0.4, -0.2, 0.3, 0, 0]
target_x_offset = [-0.25, -0.1, +0, 0, 0, -0, 0]
target_spin = np.array([-45, 180 - 45, 90, -90, -90, -45 + 180 - 25, 180 - 45])

target_depth = np.array([2.4, 4.75, 4, 5.8, 4.9, 5.75, 2])  # Guesses, check

# %%
implant_R, implant_t, implant_c = load_sitk_transform(
    implant_fit_transform_file
)

S = trimesh.Scene()
transformed_implant_targets = {}

for ii, key in enumerate(model_implant_targets.keys()):
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

hole_mesh = trimesh.load(str(implant_mesh_file), force="Mesh")
mesh_ids = list(hole_mesh.geometry.keys())

# holeCM = trimesh.collision.CollisionManager()
implantCM = trimesh.collision.CollisionManager()
probeCM = trimesh.collision.CollisionManager()
coneCM = trimesh.collision.CollisionManager()
wellCM = trimesh.collision.CollisionManager()


for ii, structure in enumerate(target_structures):
    structureCM = trimesh.collision.CollisionManager()

    if structure not in ["CLA", "MD", "CA1", "VM"]:  # target_structures:
        continue

    if probe_type[ii] == 0:
        this_probe_mesh = load_newscale_trimesh(
            modified_probe_mesh_file,
            target_depth[ii],
        )
    elif probe_type[ii] == 1:
        this_probe_mesh = load_newscale_trimesh(
            dovetail_tweezer_file,
            target_depth[ii],
        )
    elif probe_type[ii] == 3:
        this_probe_mesh = load_newscale_trimesh(
            quadbase_file,
            target_depth[ii],
        )
    elif probe_type[ii] == 4:
        this_probe_mesh = load_newscale_trimesh(
            dovetail_tweezer_4shank_file,
            target_depth[ii],
        )
    else:
        # Handle unexpected probe_type if necessary
        continue

    # Generate a single random color for both probe and structure
    this_color = trimesh.visual.random_color()

    # Assign the same color to the probe
    this_probe_mesh.visual.face_colors = this_color

    # Apply transformations to the probe
    implant_target = transformed_implant_targets[target_hole[ii]].copy()

    this_pt = trimesh.creation.uv_sphere(radius=0.25)
    this_pt.apply_translation(implant_target)
    this_pt.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(this_pt)

    implant_target[0] += target_x_offset[ii]
    implant_target[1] -= target_y_offset[ii]

    this_probe = probe_to_target_mapping[structure]
    this_rotation = rotations[this_probe]
    this_translation = translations[this_probe]
    this_ap, this_ml = find_probe_angle(this_rotation)
    if structure == "MD":
        this_ml += 4
    elif structure == "VM":
        this_ml += 4
    print(this_ap + 14)
    print(this_ml)
    T1 = transform_matrix_from_angles(this_ap, this_ml, target_spin[ii])
    # this_rotation = rotations[probe_to_target_mapping[structure]]
    # this_rotation = this_rotation@np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    # T1 = np.vstack([this_rotation,[0,0,0]])
    # T1 = np.hstack([T1,np.vstack([implant_target.reshape(3,1),0])])

    this_probe_mesh = apply_transform_to_trimesh(
        this_probe_mesh, T1, implant_target
    )

    S.add_geometry(this_probe_mesh)
    probeCM.add_object(structure, this_probe_mesh)
    implantCM.add_object(structure, this_probe_mesh)
    # holeCM.add_object(structure, this_probe_mesh)
    coneCM.add_object(structure, this_probe_mesh)
    wellCM.add_object(structure, this_probe_mesh)

    structureCM.add_object("probe", this_probe_mesh)

    # Load and transform the target structure
    this_target_mesh = trimesh.load(
        str(structure_files[structure]),
        force="Mesh",
    )
    vertices = this_target_mesh.vertices
    vertices = apply_rotate_translate(
        vertices, *invert_rotate_translate(chem_image_R, chem_image_t)
    )
    this_target_mesh.vertices = vertices
    trimesh.repair.fix_normals(this_target_mesh)
    trimesh.repair.fix_inversion(this_target_mesh)

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
    print(ii)

# S.add_geometry(hole_mesh_trans)

# S.add_geometry(transformed_brain_mesh)
headframe_mesh = trimesh.Trimesh(
    vertices=headframe_lps, faces=headframe_faces[0]
)
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
    print(f"Probes are colliding :(")
    print(f"Problems: {list(fail_names)}")
else:
    print(f"Probes are clear! :)")

    if coneCM.in_collision_internal(False, False):
        print(f"Probes are hitting cone! :(")
    else:
        print(f"Probes are clearing cone :)")

    if wellCM.in_collision_internal(False, False):
        print(f"Probes are hitting well! :(")
    else:
        print(f"Probes are clearing well :)")

probe_fail, fail_names = implantCM.in_collision_internal(return_names=True)
if probe_fail:
    print(f"Probes are striking implant! :(")
    print(f"problems: {list(fail_names)}")
else:
    print(f"Probes clear implant! :)")
S.add_geometry(implant_mesh)
S.show(viewer="gl")

# %%
this_rotation = rotations[probe_to_target_mapping["CLA"]].T
this_rotation

# %%
this_ap, this_ml = matrix_to_rig_angles(
    rotations[probe_to_target_mapping["CLA"]],
    translations[probe_to_target_mapping["CLA"]],
    mouse_ap_tilt=0,
)
T1 = transform_matrix_from_angles_and_target(this_ap, this_ml, [0, 0, 0])
T1[:3, :3]

# %%
name = []
ML = []
AP = []
DV = []
source = []
for ii, structure in enumerate(target_structures):
    print(structure)
    ss = target_spin[ii]
    if ss > 180:
        ss = ss - 360
    if ss < -180:
        ss = ss + 360
    print(f"AP: {target_ap[ii]+14}; ML: {target_ml[ii]}; Spin: {ss}")
    idx = implant_names.index(target_hole[ii])
    print(f"Hole: {target_hole[ii]}")
    name.append(target_hole[ii])
    print(
        f"Hole Target: ML: {-transformed_implant[target_hole[ii]][0]-target_x_offset[ii]}  AP: {-transformed_implant[target_hole[ii]][1]+target_y_offset[ii]} DV: {transformed_implant[target_hole[ii]][2]}"
    )
    ML.append(-transformed_implant[target_hole[ii]][0] - target_x_offset[ii])
    AP.append(-transformed_implant[target_hole[ii]][1] + target_y_offset[ii])
    DV.append(transformed_implant[target_hole[ii]][2])
    source.append("insertion plan")
    print(f"Distance past target: {target_depth[ii]}")
    print(f"Needs fancy probe: {probe_type[ii]}")
    print("\n")
this_rotation.T @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

# %%
from scipy.spatial.transform import Rotation

R = Rotation.from_matrix(
    this_rotation.T @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
).as_euler("xyz")
R

# %%
R = Rotation.from_matrix(T1[:3, :3]).as_euler("xyz")
R

# %%
