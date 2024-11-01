# %%
from aind_mri_utils.planning import (
    candidate_insertions,
    compatible_insertion_pairs,
    get_implant_targets,
    make_scene_for_insertion,
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
import os
from itertools import product
from scipy.spatial.transform import Rotation

import numpy as np
import pandas as pd

# %%
mouse = "754371"
whoami = "yoni"
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
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}".format(
    mouse
)

hole_folder = headframe_model_dir/'HoleOBJs'
implant_model_path = headframe_model_dir/'0283-300-04.obj'
implant_transform = annotations_path/f"{mouse}_implant_annotations_to_lps.h5"

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
    annotations_path / f"fiducials-{mouse}-transformed.fcsv"
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
target_structures = ["AntComMid","CCant","CCpst","GenFacCran2","LC","LGN"]

# %%
image = sitk.ReadImage(image_path)
# Read points
#manual_annotation = sf.read_slicer_fcsv(manual_annotation_path)
fiducial_annotation = sf.read_slicer_fcsv(str(uw_yoni_annotation_path))

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_path)
headframe_lps = cs.convert_coordinate_system(
    headframe, "ASR", "LPS"
)  # Preserves shape!

implant,implant_faces = get_vertices_and_faces(implant_model_path)
implant_lps =  cs.convert_coordinate_system(
    implant, "ASR", "LPS"
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
chem_shift = compute_chemical_shift(image,ppm = 3.67)
chem_shift_trans = chemical_shift_transform(chem_shift, readout="HF")
# -

# List targeted locations
preferred_pts = {k: fiducial_annotation[k] for k in target_structures}

hmg_pts = rot.prepare_data_for_homogeneous_transform(
    np.array(tuple(preferred_pts.values()))
)
chem_shift_annotation = hmg_pts @ chem_shift_trans.T @ trans.T
transformed_annotation = rot.extract_data_for_homogeneous_transform(
    chem_shift_annotation
)
target_names = tuple(preferred_pts.keys())


# %%
# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = [x for x in os.listdir(hole_folder) if '.obj' in x and 'Hole' in x]
hole_dict = {}
for ii,flname  in enumerate(hole_files):
    hole_num = int(flname.split('Hole')[-1].split('.')[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder,flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(hole_dict[hole_num].vertices, "ASR", "LPS") # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_folder,'LowerFace.obj'))
hole_dict[-1].vertices = cs.convert_coordinate_system(hole_dict[-1].vertices, "ASR", "LPS") # Preserves shape!


model_implant_targets  = {}
for ii,hole_id in enumerate(hole_dict.keys()):
    if hole_id<0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid


# %%
implant_model_trans = mr_sitk.load_sitk_transform(
    implant_transform, homogeneous=True, invert=False
)[0].T


# %%
implant_names = [*model_implant_targets]
model_targets  = np.vstack(list(model_implant_targets.values()))
implant_targets = rot.prepare_data_for_homogeneous_transform(model_targets)@implant_model_trans
transformed_implant = rot.extract_data_for_homogeneous_transform(implant_targets@trans.T)

# %%

# %%

implant_vol = sitk.ReadImage(implant_holes_path)

implant_targets_, implant_names_ = get_implant_targets(implant_vol)


# Visualize Holes, list locations

transformed_implant_homog_ = (
    rot.prepare_data_for_homogeneous_transform(implant_targets_) @ trans.T
)
transformed_implant_vol = rot.extract_data_for_homogeneous_transform(
    transformed_implant_homog_
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
            d: transformed_implant[:, i]
            for i, d in enumerate(dim_names)
        },
    }
)
df_joined = pd.concat((target_df, implant_df), ignore_index=True)
df_joined.to_csv(transformed_targets_save_path, index=False)
# %%
implant_df

# %%
df = candidate_insertions(
    transformed_annotation,
    transformed_implant,
    target_names,
    implant_names,
)
valid = compatible_insertion_pairs(df)
df[df.target=='AntComMid']


# %%
match_insertions = [8,9]# [37,17,48] # LGN,LC,ACT
# match_insertions = [17,39,34] # GVII,CC,RN
works_for_all = set(np.where(valid[match_insertions[0], :])[0])

for ii in range(0, len(match_insertions)):
    works_for_all = (
        works_for_all
        & set(np.where(valid[match_insertions[ii], :])[0])
        & set(np.where(df.target == "GenFacCran2")[0])
    )
works_for_all = [ii for ii in list(works_for_all) if ii not in match_insertions]

df.iloc[np.concatenate([match_insertions, list(works_for_all)])]

# %%
df.to_csv(annotations_path/'possible_insertions_10_29_2024.csv')


# %%
def transform_matrix_from_angles_and_target(AP, ML, Target, degrees=True):
    R = Rotation.from_euler(
        "XYZ", np.array([np.deg2rad(AP), np.deg2rad(ML), 0])
    ).as_matrix()
    T = np.zeros([4, 4])
    T[:3, :3] = R
    T[0:3, 3] = Target
    return T


def apply_transform_to_mesh(mesh,T):
    mesh.vertices = trimesh.transform_points(mesh.vertices,T)
    return mesh

import matplotlib
cm = matplotlib.colormaps['rainbow']

# Tools for collsion testings
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
new_mesh.faces = headframe_faces[0]

implant,implant_faces = get_vertices_and_faces(implant_model_path)
implant_lps =  cs.convert_coordinate_system(
    implant, "ASR", "LPS"
)  # Preserves shape!
implant_ = rot.prepare_data_for_homogeneous_transform(implant_lps)@implant_model_trans
implant_verts = rot.extract_data_for_homogeneous_transform(implant_@trans.T)
implant_mesh = trimesh.Trimesh()
implant_mesh.vertices = implant_verts
implant_mesh.faces = implant_faces[0]
implant_mesh.visual.vertex_colors = [0,0,255,255//3]

insert_list = match_insertions

for this_angle in angle_sets:

    for this_insertion in range(len(match_insertions)):
        # Mesh1
        this_mesh = mesh.copy()
        TA = trimesh.transformations.euler_matrix(
            0, 0, np.deg2rad(this_angle[this_insertion])
        )
        TB = transform_matrix_from_angles_and_target(
            df.ap[insert_list[this_insertion]],
            -df.ml[insert_list[this_insertion]],
            df.target_loc[insert_list[this_insertion]],
        )  # my ml convention is backwards

        apply_transform_to_mesh(this_mesh, TA)
        apply_transform_to_mesh(this_mesh, TB)
        CM.add_object(f"mesh{this_insertion}", this_mesh)

    if not CM.in_collision_internal(False, False):
        print("pass :" + str(this_angle))
        CM_implant =  trimesh.collision.CollisionManager()
        CM_implant.add_object('implant',implant_mesh)
        CM_headframe = trimesh.collision.CollisionManager()
        CM_headframe.add_object('headframe',new_mesh)
        CM_cone = trimesh.collision.CollisionManager()
        CM_cone.add_object('cone',cone)
        # Do secondary checks for collisons
        for this_insertion in range(len(match_insertions)):
            # Mesh1
            this_mesh = mesh.copy()
            TA = trimesh.transformations.euler_matrix(
                0, 0, np.deg2rad(this_angle[this_insertion])
            )
            TB = transform_matrix_from_angles_and_target(
                df.ap[insert_list[this_insertion]],
                -df.ml[insert_list[this_insertion]],
                df.target_loc[insert_list[this_insertion]],
            )  # my ml convention is backwards
    
            apply_transform_to_mesh(this_mesh, TA)
            apply_transform_to_mesh(this_mesh, TB)
            CM_cone.add_object(f"mesh{this_insertion}", this_mesh)
            CM_implant.add_object(f"mesh{this_insertion}", this_mesh)
            CM_headframe.add_object(f"mesh{this_insertion}", this_mesh)


        if not CM_cone.in_collision_internal(False, False):
            print('Cone is clear :)')
        else:
            print('Cone collision :(')
        if not CM_implant.in_collision_internal(False, False):
            print('Implant is clear :)')
        else:
            print('Implant collision :(')
        if not CM_headframe.in_collision_internal(False, False):
            print('Headframe is clear :)')
        else:
            print('Headframe collision :(')



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
    TA = trimesh.transformations.euler_matrix(
        0, 0, np.deg2rad(this_angle[this_insertion])
    )
    TB = transform_matrix_from_angles_and_target(
        df.ap[insert_list[this_insertion]],
        -df.ml[insert_list[this_insertion]],
        df.target_loc[insert_list[this_insertion]],
    )  # my ml convention is backwards

    apply_transform_to_mesh(this_mesh, TA)
    apply_transform_to_mesh(this_mesh, TB)
    this_mesh.visual.vertex_colors = (
        np.array(cm(this_insertion * cstep)) * 255
    ).astype(int)
    S.add_geometry(this_mesh)


meshes = [
    trimesh.creation.uv_sphere(radius=0.1)
    for i in range(len(transformed_implant))
]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant[i, :])
    m.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(m)

meshes = [
    trimesh.creation.uv_sphere(radius=0.25)
    for i in range(len(transformed_implant_vol))
]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant_vol[i, :])
    m.visual.vertex_colors = [0, 0, 255, 255]
    S.add_geometry(m)

meshes = [
    trimesh.creation.uv_sphere(radius=0.25)
    for i in range(len(transformed_annotation))
]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_annotation[i, :])
    m.visual.vertex_colors = trimesh.visual.random_color()
    S.add_geometry(m)
#S.add_geometry(cone)
S.add_geometry(implant_mesh)


S.show(viewer = 'gl')

# %%
angle_ranges = [np.arange(0, 360, 45)] * len(match_insertions)
angle_ranges

# %% [raw]
# reload(aind_mri_utils)

# %%
import aind_mri_utils

# %%

# %%
