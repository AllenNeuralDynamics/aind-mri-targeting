# %%
import os
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils import coordinate_systems as cs
from aind_anatomical_utils import slicer as sf
from aind_mri_utils import rotations as rot
from aind_mri_utils.arc_angles import arc_angles_to_affine
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.meshes import (
    apply_transform_to_trimesh,
    load_newscale_trimesh,
)
from aind_mri_utils.planning import (
    candidate_insertions,
    compatible_insertion_pairs,
    find_other_compatible_insertions,
    get_implant_targets,
)
from aind_mri_utils.reticle_calibrations import (
    fit_rotation_params,
    read_reticle_calibration,
    transform_probe_to_bregma,
)

# %%
mouse = "765860"
vaseline_ppm = 3.67  # (3.7 + 4.1) / 2 for previous
whoami = "yoni"
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
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}/UW_2025".format(mouse)

hole_folder = headframe_model_dir / "HoleOBJs"
implant_model_path = headframe_model_dir / "0283-300-04.obj"
implant_transform = annotations_path / f"{mouse}_implant_fit.h5"

headframe_path = headframe_model_dir / "TenRunHeadframe.obj"
holes_path = headframe_model_dir / "OneOff_HolesOnly.obj"


implant_holes_path = str(annotations_path / "{}_ImplantHoles.seg.nrrd".format(mouse))

image_path = str(annotations_path / "{}_100.nii.gz".format(mouse))  # '_100.nii.gz'))
labels_path = str(annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse))  # 'Segmentation.seg.nrrd')#
brain_mask_path = str(annotations_path / ("{}_auto_skull_strip.nrrd".format(mouse)))

template_annotation_path = str(annotations_path / ("fiducials-{}-transformed.fcsv".format(mouse)))
cone_path = base_dir / "ephys/persist/Software/PinpointBuilds/WavefrontFiles/Cone_0160-200-53.obj"

# uw_yoni_annotation_path = (
#     annotations_path / "fiducials-{}-transformed.fcsv".format(mouse)
# )

newscale_file_name = headframe_path / "Centered_Newscale_2pt0.obj"
#


calibration_dir = base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
measured_hole_centers = None
# (  annotations_path / f"{mouse}_measured_hole_centers.xlsx"  )


# manual_hole_centers_file = annotations_path / 'hole_centers.mrk.json'

transformed_targets_save_path = annotations_path / (f"{mouse}_TransformedTargets.csv")
test_probe_translation_save_path = str(base_save_dir / "test_probe_translation.h5")
transform_filename = str(annotations_path / ("com_plane.h5"))

# %%
target_structures = None  # ["CCant", "CCpst", "AntComMid", "GenFacCran2"]

# %%
image = sitk.ReadImage(image_path)
# Read points
template_annotation = sf.read_slicer_fcsv(template_annotation_path)

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_path)
headframe_lps = cs.convert_coordinate_system(headframe, "ASR", "LPS")  # Preserves shape!

implant, implant_faces = get_vertices_and_faces(implant_model_path)
implant_lps = cs.convert_coordinate_system(implant, "ASR", "LPS")  # Preserves shape!

# Load the computed transform
trans = load_sitk_transform(transform_filename, homogeneous=True, invert=True)[0]

cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")

probe_mesh = load_newscale_trimesh(probe_model_file, move_down=0.5)

# Get chemical shift from MRI image.
# Defaults are standard UW scans- set params for anything else.
chem_shift = compute_chemical_shift(image, ppm=vaseline_ppm)
chem_shift_trans = chemical_shift_transform(chem_shift, readout="HF")
# -

# List targeted locations
if target_structures is None:
    target_structures = list(template_annotation.keys())
preferred_pts = {k: template_annotation[k] for k in target_structures}

hmg_pts = rot.prepare_data_for_homogeneous_transform(np.array(tuple(preferred_pts.values())))
chem_shift_annotation = hmg_pts @ chem_shift_trans.T @ trans.T
transformed_annotation = rot.extract_data_for_homogeneous_transform(chem_shift_annotation)
target_names = tuple(preferred_pts.keys())


# %%
# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_files = [x for x in os.listdir(hole_folder) if ".obj" in x and "Hole" in x]
hole_dict = {}
for ii, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder, flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

# Get the lower face, store with key -1
hole_dict[-1] = trimesh.load(os.path.join(hole_folder, "LowerFace.obj"))
hole_dict[-1].vertices = cs.convert_coordinate_system(hole_dict[-1].vertices, "ASR", "LPS")  # Preserves shape!


model_implant_targets = {}
for ii, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid


# %%
# This is stored as inverse for the sake of slicer, and must be re-inverted
# to be used in python
implant_model_trans = load_sitk_transform(implant_transform, homogeneous=True, invert=True)[0]


# %%
implant_names = list(model_implant_targets.keys())
model_targets = np.vstack(list(model_implant_targets.values()))
implant_targets = rot.prepare_data_for_homogeneous_transform(model_targets) @ implant_model_trans.T
transformed_implant = rot.extract_data_for_homogeneous_transform(implant_targets @ trans.T)

# %%

implant_vol = sitk.ReadImage(implant_holes_path)

implant_targets_, implant_names_ = get_implant_targets(implant_vol)


# Visualize Holes, list locations

transformed_implant_homog = rot.prepare_data_for_homogeneous_transform(implant_targets_) @ trans.T
transformed_implant_vol = rot.extract_data_for_homogeneous_transform(transformed_implant_homog)

# %% Load measured hole centers
transforms = dict()
hole_measures_ras = dict()
# %%
if measured_hole_centers:
    measured_hole_centers_df = pd.read_excel(measured_hole_centers)
    for i, row in measured_hole_centers_df.iterrows():
        calfile = calibration_dir / row["Calibration"]
        if calfile not in transforms:
            (
                adjusted_pairs_by_probe,
                global_offset,
                global_rotation_degrees,
                reticle_name,
            ) = read_reticle_calibration(calfile)
            transforms[calfile] = dict()
            for probe in adjusted_pairs_by_probe:
                reticle_coords, probe_coords = adjusted_pairs_by_probe[probe]
                (
                    transform_rs,
                    scaling_vecs,
                    transform_offsets,
                    _,
                ) = fit_rotation_params(
                    reticle_pts=reticle_coords,
                    probe_pts=probe_coords,
                    find_scaling=True,
                )
                transforms[calfile][probe] = {
                    "scaling": {
                        "rotation": transform_rs,
                        "offset": transform_offsets,
                        "scale": scaling_vecs,
                    }
                }
                transform_rs, transform_offsets, _ = fit_rotation_params(
                    reticle_pts=reticle_coords,
                    probe_pts=probe_coords,
                    find_scaling=False,
                )
                transforms[calfile][probe]["no_scaling"] = {
                    "rotation": transform_rs,
                    "offset": transform_offsets,
                    "scale": None,
                }
        hole_num = row["Hole #"]
        probe_pt = np.array([row[dim] for dim in ["X", "Y", "Z"]]) / 1000
        thisprobe = row["Probe"]
        used_scaling = row["Scaling used?"] == "yes"
        fit_type = "scaling" if used_scaling else "no_scaling"
        hole_measures_ras[hole_num] = transform_probe_to_bregma(
            probe_pt,
            transforms[calfile][thisprobe][fit_type]["rotation"],
            transforms[calfile][thisprobe][fit_type]["offset"],
            transforms[calfile][thisprobe][fit_type]["scale"],
        )

# %%
dim_names = ["ML (mm)", "AP (mm)", "DV (mm)"]
transformed_annotation_ras = np.array([-1, -1, 1]) * transformed_annotation
annotation_data = []
for i, n in enumerate(target_names):
    target = {
        "point": n,
        **{d: transformed_annotation_ras[i, j] for j, d in enumerate(dim_names)},
        "point source": "Fiducial From Template",
    }
    annotation_data.append(target)

sp = np.argsort(implant_names)
implant_names_sorted = np.array(implant_names)[sp]
transformed_implant_sorted_ras = np.array([-1, -1, 1]) * transformed_implant[sp, :]
implant_data = []
for i, n in enumerate(implant_names_sorted):
    if n in hole_measures_ras:
        implant_point_source = "measured"
        hole_coord = hole_measures_ras[n]
    else:
        implant_point_source = "minimization"
        hole_coord = transformed_implant_sorted_ras[i]
    target = {
        "point": f"Hole {n}",
        **{d: hole_coord[j] for j, d in enumerate(dim_names)},
        "point source": implant_point_source,
    }
    implant_data.append(target)
df_joined = pd.DataFrame(annotation_data + implant_data)
df_joined.to_csv(transformed_targets_save_path, index=False)
# %%
implant_targets_las = np.array([[implant_data[i][d] for d in dim_names] for i in range(len(implant_data))]) * np.array(
    [-1, -1, 1]
)
# %%
df = candidate_insertions(
    transformed_annotation,
    implant_targets_las,
    target_names,
    implant_names_sorted,
)
compat_matrix = compatible_insertion_pairs(df)
df.to_csv(annotations_path / f"{mouse}_candidate_insertions.csv")

# %%
df[df.target == "GPe_anterior"]
# %%
seed_insertions = [
    74,
    76,
    77,
    78,
    79,
    80,
    82,
    83,
    88,
    90,
    91,
    92,
    93,
    96,
    97,
    104,
    105,
    106,
    107,
    108,
    110,
    111,
]  # [37,17,48] # LGN,LC,ACT
bad_holes = [0, 1, 2, 3]
good_holes_mask = np.logical_not(np.isin(df.hole.to_numpy(), bad_holes))
target_mask = np.logical_not(np.isin(df.target.to_numpy(), np.array(["AntComMid", "CCant", "GenFacCran2"])))
consider_mask = np.logical_and(good_holes_mask, target_mask)
considered_ndxs = np.nonzero(consider_mask)[0]
# match_insertions = [17,39,34] # GVII,CC,RN
other_insertions = find_other_compatible_insertions(compat_matrix, considered_ndxs, seed_insertions)

df.iloc[np.concatenate([seed_insertions, other_insertions])]


# %%
df.iloc[np.array(seed_insertions)]
# %%
insertion_df = df.iloc[np.array(seed_insertions)]
data = []
for i, row in insertion_df.iterrows():
    ml, ap, dv = np.array([-1, -1, 1]) * row["target_loc"]
    data.append(
        {
            "Target": row["target"],
            # "Annotation": None,
            # "Headframe Registration": None,
            # "Coordinate Sys": None,
            # "CCF ML (um)": None,
            # "CCF AP (um)": None,
            # "CCF DV (um)": None,
            "Target ML (mm)": ml,
            "Target AP (mm)": ap,
            "Target DV (mm)": dv,
            "Intended Hole": row["hole"],
            "Arc AP": row["rig_ap"],
            "Arc ML": row["ml"],
        }
    )
target_sheet_df = pd.DataFrame(data)

# %%

cm = matplotlib.colormaps["rainbow"]

# Tools for collision testings
mesh = load_newscale_trimesh(
    probe_model_file,
    -0.2,
)

angle_ranges = [np.arange(0, 360, 45)] * len(seed_insertions)

angle_sets = [x for x in product(*angle_ranges)]

# Load the stuff that doesn't change on each iteration
cone = trimesh.load_mesh(cone_path)
cone.vertices = cs.convert_coordinate_system(cone.vertices, "ASR", "LPS")


new_mesh = trimesh.Trimesh()
new_mesh.vertices = headframe_lps
new_mesh.faces = headframe_faces[0]
# %%
CM = trimesh.collision.CollisionManager()


implant, implant_faces = get_vertices_and_faces(implant_model_path)
implant_lps = cs.convert_coordinate_system(implant, "ASR", "LPS")  # Preserves shape!
implant_ = rot.prepare_data_for_homogeneous_transform(implant_lps) @ implant_model_trans
implant_verts = rot.extract_data_for_homogeneous_transform(implant_ @ trans.T)
implant_mesh = trimesh.Trimesh()
implant_mesh.vertices = implant_verts
implant_mesh.faces = implant_faces[0]
implant_mesh.visual.vertex_colors = [0, 0, 255, 255 // 3]

insert_list = seed_insertions

for this_angle in angle_sets:
    for this_insertion in range(len(seed_insertions)):
        # Mesh1
        this_mesh = mesh.copy()
        TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(this_angle[this_insertion]))
        TB = arc_angles_to_affine(
            df.ap[insert_list[this_insertion]],
            -df.ml[insert_list[this_insertion]],
        )  # my ml convention is backwards

        apply_transform_to_trimesh(this_mesh, TA)
        apply_transform_to_trimesh(this_mesh, TB)
        CM.add_object(f"mesh{this_insertion}", this_mesh)

    if not CM.in_collision_internal(False, False):
        print("pass :" + str(this_angle))
        CM_implant = trimesh.collision.CollisionManager()
        CM_implant.add_object("implant", implant_mesh)
        CM_headframe = trimesh.collision.CollisionManager()
        CM_headframe.add_object("headframe", new_mesh)
        CM_cone = trimesh.collision.CollisionManager()
        CM_cone.add_object("cone", cone)
        # Do secondary checks for collisions
        for this_insertion in range(len(seed_insertions)):
            # Mesh1
            this_mesh = mesh.copy()
            TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(this_angle[this_insertion]))
            TB = arc_angles_to_affine(
                df.ap[insert_list[this_insertion]],
                -df.ml[insert_list[this_insertion]],
            )  # my ml convention is backwards

            apply_transform_to_trimesh(this_mesh, TA)
            apply_transform_to_trimesh(this_mesh, TB)
            CM_cone.add_object(f"mesh{this_insertion}", this_mesh)
            CM_implant.add_object(f"mesh{this_insertion}", this_mesh)
            CM_headframe.add_object(f"mesh{this_insertion}", this_mesh)

        if not CM_cone.in_collision_internal(False, False):
            print("Cone is clear :)")
        else:
            print("Cone collision :(")
        if not CM_implant.in_collision_internal(False, False):
            print("Implant is clear :)")
        else:
            print("Implant collision :(")
        if not CM_headframe.in_collision_internal(False, False):
            print("Headframe is clear :)")
        else:
            print("Headframe collision :(")

        break
    else:
        print("fail :" + str(this_angle))

    for this_insertion in range(len(seed_insertions)):
        CM.remove_object(f"mesh{this_insertion}")

# %%
S = trimesh.scene.Scene([new_mesh])

S.add_geometry(new_mesh)


cstep = (256) // (len(seed_insertions))

for this_insertion in range(len(seed_insertions)):
    this_mesh = mesh.copy()
    TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(this_angle[this_insertion]))
    TB = arc_angles_to_affine(
        df.ap[insert_list[this_insertion]],
        -df.ml[insert_list[this_insertion]],
    )  # my ml convention is backwards

    apply_transform_to_trimesh(this_mesh, TA)
    apply_transform_to_trimesh(this_mesh, TB)
    this_mesh.visual.vertex_colors = (np.array(cm(this_insertion * cstep)) * 255).astype(int)
    S.add_geometry(this_mesh)


meshes = [trimesh.creation.uv_sphere(radius=0.1) for i in range(len(transformed_implant))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant[i, :])
    m.visual.vertex_colors = [255, 0, 255, 255]
    S.add_geometry(m)

meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_implant_vol))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_implant_vol[i, :])
    m.visual.vertex_colors = [0, 0, 255, 255]
    S.add_geometry(m)

meshes = [trimesh.creation.uv_sphere(radius=0.25) for i in range(len(transformed_annotation))]
for i, m in enumerate(meshes):
    m.apply_translation(transformed_annotation[i, :])
    m.visual.vertex_colors = trimesh.visual.random_color()
    S.add_geometry(m)
# S.add_geometry(cone)
S.add_geometry(implant_mesh)


S.show(viewer="gl")

# %%
angle_ranges = [np.arange(0, 360, 45)] * len(seed_insertions)
angle_ranges
