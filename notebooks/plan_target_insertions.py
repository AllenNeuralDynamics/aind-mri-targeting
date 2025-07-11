# %%
# Basic python functions that we need
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh

# Functions from aind_mri_utils
from aind_anatomical_utils import coordinate_systems as cs
from aind_anatomical_utils import slicer as sf
from aind_mri_utils import rotations as rot
from aind_mri_utils.arc_angles import arc_angles_to_affine, vector_to_arc_angles
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io import simpleitk as mr_sitk
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils.meshes import load_newscale_trimesh
from aind_mri_utils.optimization import get_headframe_hole_lines
from matplotlib import cm

# %%
# File Paths
mouse = "743700"
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
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}/".format(mouse)

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


calibration_filename = "calibration_info_np2_2024_04_22T11_15_00.xlsx"
calibration_dir = base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
calibration_file = calibration_dir / calibration_filename
measured_hole_centers = annotations_path / "measured_hole_centers.xlsx"


# manual_hole_centers_file = annotations_path / 'hole_centers.mrk.json'

transformed_targets_save_path = annotations_path / (mouse + "TransformedTargets.csv")
test_probe_translation_save_path = str(base_save_dir / "test_probe_translation.h5")
transform_filename = str(annotations_path / (mouse + "_com_plane.h5"))
# Magic numbers
resolution = 100


# %%
# Handle inconsistent labeling
label_vol = sitk.ReadImage(labels_path)
odict = {k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()}
insert_underscores = "_" in list(sf.find_seg_nrrd_header_segment_info(odict).keys())[0]

# Load the points on the headframe lines.
pts1, pts2, order = get_headframe_hole_lines(insert_underscores=insert_underscores, coordinate_system="LPS")

# order.remove('anterior_vertical')

image = sitk.ReadImage(image_path)


# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_path)
headframe_lps = cs.convert_coordinate_system(headframe, "ASR", "LPS")  # Preserves shape!

# Load just the headframe holes
holes, holes_faces = get_vertices_and_faces(holes_path)
holes_faces = holes_faces[-1]
holes_lps = cs.convert_coordinate_system(holes, "ASR", "LPS")

# Load the brain mask
mask = sitk.ReadImage(brain_mask_path)
idxx = np.where(sitk.GetArrayViewFromImage(mask))
idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
brain_pos = np.zeros(idx.shape)
brain_pos = np.vstack([mask.TransformIndexToPhysicalPoint(idx[ii, :].tolist()) for ii in range(idx.shape[0])])
brain_pos = brain_pos[np.arange(0, brain_pos.shape[0], brain_pos.shape[0] // 1000)]


# %%
# Load the computed transform
trans = mr_sitk.load_sitk_transform(transform_filename, homogeneous=True, invert=True)[0]

# Get chemical shift from MRI image.
# Defaults are standard UW scans- set params for anything else.
chem_shift = compute_chemical_shift(image)
chem_shift_trans = chemical_shift_transform(chem_shift, readout="HF")
# -

# Read points
manual_annotation = sf.read_slicer_fcsv(manual_annotation_path)

# %%
# List targeted locations
preferred_pts = {
    # "LGN": manual_annotation["LGN"],
    "CCant": manual_annotation["CCant"],
    "CCpst": manual_annotation["CCpst"],
    "AntComMid": manual_annotation["AntComMid"],
    "GenFacCran": manual_annotation["GenFacCran2"],
}

hmg_pts = rot.prepare_data_for_homogeneous_transform(np.array(tuple(preferred_pts.values())))
chem_shift_annotation = hmg_pts @ trans.T @ chem_shift_trans.T
transformed_annotation = rot.extract_data_for_homogeneous_transform(chem_shift_annotation)
target_nms = tuple(preferred_pts.keys())

for ii in range(transformed_annotation.shape[0]):
    print(
        "Target "
        + str(target_nms[ii])
        + " ML: "
        + str(-transformed_annotation[ii, 0])
        + " AP: "
        + str(-transformed_annotation[ii, 1])
        + " DV: "
        + str(transformed_annotation[ii, 2])
    )

ndf = pd.DataFrame(
    {
        "Target": target_nms,
        "ML": -transformed_annotation[:, 0],
        "AP": -transformed_annotation[:, 1],
        "DV": transformed_annotation[:, 2],
    }
)
ndf.to_csv(transformed_targets_save_path)

# %%
# If implant has holes that are segmented.
implant_vol = sitk.ReadImage(implant_holes_path)
odict = {k: implant_vol.GetMetaData(k) for k in implant_vol.GetMetaDataKeys()}
label_dict = sf.find_seg_nrrd_header_segment_info(odict)

implant_names = []
implant_targets = []
implant_pts = []

for ii, key in enumerate(label_dict.keys()):
    filt = sitk.EqualImageFilter()
    is_label = filt.Execute(implant_vol, label_dict[key])
    idxx = np.where(sitk.GetArrayViewFromImage(is_label))
    if len(idxx[0]) == 0:
        continue
    idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
    implant_pos = np.vstack(
        [implant_vol.TransformIndexToPhysicalPoint(idx[ii, :].tolist()) for ii in range(idx.shape[0])]
    )
    implant_pts.append(implant_pos)
    implant_targets.append(np.mean(implant_pos, axis=0))
    this_key = key.split("-")[-1].split("_")[-1]
    implant_names.append(int(this_key))
implant_targets = np.vstack(implant_targets)


# %%
# Visualize Holes, list locations
transformed_brain = rot.extract_data_for_homogeneous_transform(
    np.dot(
        np.dot(
            rot.prepare_data_for_homogeneous_transform(brain_pos),
            chem_shift_trans,
        ),
        trans,
    )
)
transformed_implant = rot.extract_data_for_homogeneous_transform(
    np.dot(rot.prepare_data_for_homogeneous_transform(implant_targets), trans)
)


for ii in range(transformed_implant.shape[0]):
    print(
        "MRI "
        + str(implant_names[ii])
        + " ML: "
        + str(-transformed_implant[ii, 0])
        + " AP: "
        + str(-transformed_implant[ii, 1])
        + " DV: "
        + str(transformed_implant[ii, 2])
    )


# %%
preferred_pts = ["LGN", "CCant", "CCpst", "AntComMid", "GenFacCran"]
targets = list(preferred_pts.keys())
transformed_preferred = transformed_annotation

TARGET = []
HOLE = []
AP = []
AP_Range = []
RIG_AP = []
ML = []
ML_Range = []
TARGET_LOC = []
for tt in range(transformed_preferred.shape[0]):
    for hh in range(transformed_implant.shape[0]):
        insertion_vector = transformed_implant[hh, :] - transformed_preferred[tt, :]
        ap, ml = vector_to_arc_angles(insertion_vector)

        radius = 0.3
        theta = np.deg2rad(np.arange(0, 360, 1))
        a = transformed_implant[hh, 0] + radius * np.cos(theta)
        b = transformed_implant[hh, 1] + radius * np.sin(theta)
        circle = np.vstack([a, b, np.ones(b.shape) * transformed_implant[hh, 2]]).T

        this_ML = []
        this_AP = []

        for jj in range(len(a)):
            insertion_vector = transformed_preferred[tt, :], circle[jj, :]
            this_ap, this_ml = vector_to_arc_angles(insertion_vector)
            this_ML.append(this_ml)
            this_AP.append(this_ap)

        ML_Range.append(np.abs(np.max(this_ML) - np.min(this_ML)) / 2)
        AP_Range.append(np.abs(np.max(this_AP) - np.min(this_AP)) / 2)

        if False:
            continue
        else:
            TARGET.append(targets[tt])
            HOLE.append(implant_names[hh])
            ML.append(ml)
            AP.append(ap)
            RIG_AP.append(ap + 14)
            TARGET_LOC.append(transformed_preferred[tt, :])


df = pd.DataFrame(
    {
        "target": TARGET,
        "hole": HOLE,
        "rig_ap": RIG_AP,
        "ml": ML,
        "ap": AP,
        "ML_range": ML_Range,
        "AP_range": AP_Range,
        "target_loc": TARGET_LOC,
    }
)


# %%
ap_wiggle = 1
ap_min = 16
ml_min = 16
rotation_inc = 45
valid = np.zeros([df.shape[0], df.shape[0]], dtype=bool)
# mesh = load_newscale_model(move_down=.5,filename=probe_model_file)
mesh = load_newscale_trimesh(probe_model_file, move_down=0.5)

cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")

hf = trimesh.Trimesh()
hf.vertices = headframe_lps
hf.faces = headframe_faces
CM = trimesh.collision.CollisionManager()
CM.add_object("cone", cone)
# check the validity of every insertions. This is order N**2, so may need some
# thought.

for ii, row1 in df.iterrows():
    for jj, row2 in df.iterrows():
        if ii == jj or ii < jj:
            continue
        # elif row1.target == row2.target:
        #    continue
        elif row1.hole == row2.hole:
            continue
        elif (np.abs(row1.ap - row2.ap) < ap_wiggle) and (np.abs(row1.ml - row2.ml) < ml_min):
            continue
        elif (np.abs(row1.ap - row2.ap) < ap_min) and (np.abs(row1.ap - row2.ap) > ap_wiggle):
            continue
        else:
            valid[ii, jj] = True
valid = np.bitwise_or(valid, valid.T)

# -

df[(df.target == "GenFacCran")]

# %%
match_insertions = [68, 59, 67]
works_for_all = set(np.where(valid[match_insertions[0], :])[0])

for ii in range(0, len(match_insertions)):
    works_for_all = (
        works_for_all & set(np.where(valid[match_insertions[ii], :])[0]) & set(np.where(df.target == "GenFacCran")[0])
    )

df.iloc[np.concatenate([match_insertions, list(works_for_all)])]


# %%
insertion_list = match_insertions


new_mesh = trimesh.Trimesh()
new_mesh.vertices = headframe_lps
new_mesh.faces = headframe_faces


cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")


S = trimesh.scene.Scene([new_mesh])
S.add_geometry(cone)

meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_implant))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant[i, :])
    m.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(m)

meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_preferred))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_preferred[i, :])
    m.visual.vertex_colors = trimesh.visual.random_color()
    S.add_geometry(m)


S.add_geometry(new_mesh)

for ii in match_insertions:
    T1 = arc_angles_to_affine(df.ap[ii], -df.ml[ii])
    S.add_geometry(rot.apply_transform_to_trimesh(mesh.copy(), T1))

S.set_camera([0, 0, 0], distance=150, center=[0, 0, 0])
S.show()
# -

#


# %%

# Tools for collision testings
mesh = load_newscale_trimesh(
    probe_model_file,
    -0.2,
)

angle_ranges = [np.arange(0, 360, 45)] * len(match_insertions)

angle_sets = [x for x in product(*angle_ranges)]

# Load the stuff that doesn't change on each iteration
cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")
CM = trimesh.collision.CollisionManager()
new_mesh = trimesh.Trimesh()
new_mesh.vertices = headframe_lps
new_mesh.faces = headframe_faces
# CM.add_object("cone", cone)

insert_list = match_insertions

for this_angle in angle_sets:
    for this_insertion in range(len(match_insertions)):
        # Mesh1
        this_mesh = mesh.copy()
        TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(this_angle[this_insertion]))
        TB = arc_angles_to_affine(
            df.ap[insert_list[this_insertion]],
            -df.ml[insert_list[this_insertion]],
        )  # my ml convention is backwards

        rot.apply_transform_to_trimesh(this_mesh, TA)
        rot.apply_transform_to_trimesh(this_mesh, TB)
        CM.add_object(f"mesh{this_insertion}", this_mesh)

    if not CM.in_collision_internal(False, False):
        print("pass :" + str(this_angle))
        break
    else:
        print("fail :" + str(this_angle))

    for this_insertion in range(len(match_insertions)):
        CM.remove_object(f"mesh{this_insertion}")


S = trimesh.scene.Scene([new_mesh])

S.add_geometry(new_mesh)


cstep = (256) // (len(match_insertions))

for this_insertion in range(len(match_insertions)):
    this_mesh = mesh.copy()
    TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(this_angle[this_insertion]))
    TB = arc_angles_to_affine(
        df.ap[insert_list[this_insertion]],
        -df.ml[insert_list[this_insertion]],
    )  # my ml convention is backwards

    rot.apply_transform_to_trimesh(this_mesh, TA)
    rot.apply_transform_to_trimesh(this_mesh, TB)
    this_mesh.visual.vertex_colors = (np.array(cm(this_insertion * cstep)) * 255).astype(int)
    S.add_geometry(this_mesh)


meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_implant))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant[i, :])
    m.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(m)

meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_preferred))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_preferred[i, :])
    m.visual.vertex_colors = trimesh.visual.random_color()
    S.add_geometry(m)
S.add_geometry(cone)

S.show()
