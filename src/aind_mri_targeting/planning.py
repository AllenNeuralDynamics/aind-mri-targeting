import numpy as np
import SimpleITK as sitk
import pandas as pd
import trimesh
from itertools import product
from scipy.spatial.transform import Rotation


# Functions from aind_mri_utils
from aind_mri_utils import rotations as rot
from aind_mri_utils.file_io import slicer_files as sf
from aind_mri_utils.file_io import simpleitk as mr_sitk
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.meshes import load_newscale_trimesh
from aind_mri_utils.chemical_shift import (
    compute_chemical_shift,
    chemical_shift_transform,
)
from aind_mri_utils.arc_angles import (
    calculate_arc_angles,
    transform_matrix_from_angles_and_target,
)
from aind_mri_utils.planning import (
    candidate_insertions,
    compatible_insertion_pairs,
    get_implant_targets,
    is_insertion_valid,
    find_other_compatible_insertions,
)


def find_targets(pts):
    pass


def load_files_to_plan_insertion():
    pass


def apply_shift_and_transform(
    pts, rotation, translation, chemical_shift, readout_direction
):
    pass


def test_for_collisions(insert_list, probe_mesh, df, rotations_to_test):
    angle_sets = [x for x in product(*rotations_to_test)]

    # Load the stuff that doesn't change on each iteration
    CM = trimesh.collision.CollisionManager()

    for this_angle in angle_sets:
        for this_insertion in range(len(insert_list)):
            this_mesh = probe_mesh.copy()
            TA = trimesh.transformations.euler_matrix(
                0, 0, np.deg2rad(this_angle[this_insertion])
            )
            TB = transform_matrix_from_angles_and_target(
                df.ap[insert_list[this_insertion]],
                -df.ml[insert_list[this_insertion]],
                df.target_loc[insert_list[this_insertion]],
            )  # my ml convention is backwards

            rot.apply_transform_to_trimesh(this_mesh, TA)
            rot.apply_transform_to_trimesh(this_mesh, TB)
            CM.add_object(f"mesh{this_insertion}", this_mesh)

        if not CM.in_collision_internal(False, False):
            print("pass :" + str(this_angle))
            break
        else:
            print("fail :" + str(this_angle))

        for this_insertion in range(len(insert_list)):
            CM.remove_object(f"mesh{this_insertion}")
    return this_angle


def make_final_insertion_scene(
    working_angle,
    headframe_mesh,
    probe_mesh,
    cone,
    transformed_implant,
    transformed_annotation,
    insert_list,
    df,
    cm,
):
    S = trimesh.scene.Scene([headframe_mesh])

    S.add_geometry(headframe_mesh)

    cstep = (256) // (len(insert_list))

    for this_insertion in range(len(insert_list)):
        this_mesh = probe_mesh.copy()
        TA = trimesh.transformations.euler_matrix(
            0, 0, np.deg2rad(working_angle[this_insertion])
        )
        TB = transform_matrix_from_angles_and_target(
            df.ap[insert_list[this_insertion]],
            -df.ml[insert_list[this_insertion]],
            df.target_loc[insert_list[this_insertion]],
        )  # my ml convention is backwards

        rot.apply_transform_to_trimesh(this_mesh, TA)
        rot.apply_transform_to_trimesh(this_mesh, TB)
        this_mesh.visual.vertex_colors = (
            np.array(cm(this_insertion * cstep)) * 255
        ).astype(int)
        S.add_geometry(this_mesh)

    meshes = [
        trimesh.creation.uv_sphere(radius=0.25)
        for i in range(len(transformed_implant))
    ]
    for i, m in enumerate(meshes):
        m.apply_translation(transformed_implant[i, :])
        m.visual.vertex_colors = [255, 0, 255, 255]
        S.add_geometry(m)

    meshes = [
        trimesh.creation.uv_sphere(radius=0.25)
        for i in range(len(transformed_annotation))
    ]
    for i, m in enumerate(meshes):
        m.apply_translation(transformed_annotation[i, :])
        m.visual.vertex_colors = trimesh.visual.random_color()
        S.add_geometry(m)
    S.add_geometry(cone)
    return S


def make_scene_for_insertion(
    headframe_mesh,
    cone,
    transformed_implant,
    transformed_annotation,
    match_insertions,
    df,
    probe_mesh,
):
    S = trimesh.scene.Scene([headframe_mesh])
    S.add_geometry(cone)

    meshes = [
        trimesh.creation.uv_sphere(radius=0.25)
        for i in range(len(transformed_implant))
    ]
    for i, m in enumerate(meshes):
        m.apply_translation(transformed_implant[i, :])
        m.visual.vertex_colors = [255, 0, 255, 255]
        S.add_geometry(m)

    meshes = [
        trimesh.creation.uv_sphere(radius=0.25)
        for i in range(len(transformed_annotation))
    ]
    for i, m in enumerate(meshes):
        m.apply_translation(transformed_annotation[i, :])
        m.visual.vertex_colors = trimesh.visual.random_color()
        S.add_geometry(m)

    S.add_geometry(headframe_mesh)

    for ii in match_insertions:
        T1 = transform_matrix_from_angles_and_target(
            df.ap[ii], -df.ml[ii], df.target_loc[ii]
        )
        S.add_geometry(rot.apply_transform_to_trimesh(probe_mesh.copy(), T1))

    S.set_camera([0, 0, 0], distance=150, center=[0, 0, 0])
    return S


def _make_target_df(
    names,
    targets,
    sps,
    target_colname="point",
    dim_names=["ML (mm)", "AP (mm)", "DV (mm)"],
):
    ndim = len(dim_names)
    if targets.shape[1] != ndim:
        raise ValueError(
            "Targets must have the same number of dimensions as len(dim_names)"
        )
    if ndim != 3:
        raise NotImplementedError("Only 3D points are supported")
    lps_to_ras_arr = np.array([-1, -1, 1])
    names_sorted = []
    dims_sorted = [[]] * ndim  # This is a ndim long list of lists
    for n, t, sp in zip(names, targets, sps):
        t_lps = lps_to_ras_arr * t[sp, :]
        for dno in range(ndim):
            dims_sorted[dno].extend(t_lps[:, dno])
        for i in sp:
            names_sorted.append(n[i])
    return pd.DataFrame(
        data={
            target_colname: names_sorted,
            **{d: dims_sorted[i] for i, d in enumerate(dim_names)},
        }
    )


def plan_insertion(
    image_path,
    headframe_path,
    transform_filename,
    manual_annotation_path,
    transformed_targets_save_path,
    implant_holes_path,
    cone_path,
    probe_model_file,
    cm,
    target_structures=["LGN", "CCant", "CCpst", "AntComMid", "GenFacCran"],
):

    # order.remove('anterior_vertical')

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
    # If implant has holes that are segmented.
    implant_vol = sitk.ReadImage(implant_holes_path)

    # %%
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
    chem_shift_annotation = hmg_pts @ trans.T @ chem_shift_trans.T
    transformed_annotation = rot.extract_data_for_homogeneous_transform(
        chem_shift_annotation
    )
    target_names = list(preferred_pts.keys())

    implant_targets, implant_indices = get_implant_targets(implant_vol)

    transformed_implant = rot.extract_data_for_homogeneous_transform(
        np.dot(
            rot.prepare_data_for_homogeneous_transform(implant_targets), trans
        )
    )

    # %%
    # Visualize Holes, list locations

    hole_target_names = [f"Hole {i}" for i in implant_indices]
    names = [target_names, hole_target_names]
    targets = [transformed_annotation, transformed_implant]
    sps = [np.argsort(n) for n in [target_names, implant_indices]]
    target_df = _make_target_df(names, targets, sps)
    target_df.to_csv(transformed_targets_save_path, index=False)

    # %%

    df = candidate_insertions(
        transformed_annotation,
        transformed_implant,
        target_names,
        implant_indices,
    )

    compat_matrix = compatible_insertion_pairs(df)

    # %%
    seed_insertions = [68, 59, 67]
    target_ndxs = np.nonzero(df.target == "GenFacCran")[0]
    compatible_insertions = find_other_compatible_insertions(
        compat_matrix, target_ndxs, seed_insertions
    )
    df.iloc[np.concatenate([seed_insertions, compatible_insertions])]

    # %%
    headframe_mesh = trimesh.Trimesh()
    headframe_mesh.vertices = headframe_lps
    headframe_mesh.faces = headframe_faces

    S = make_scene_for_insertion(
        headframe_mesh,
        cone,
        transformed_implant,
        transformed_annotation,
        seed_insertions,
        df,
        probe_mesh,
    )

    S.show()
    # %%

    # Tools for collsion testings
    probe_mesh = load_newscale_trimesh(
        probe_model_file,
        move_down=-0.2,
    )

    rotations_to_test = [np.arange(0, 360, 45)] * len(seed_insertions)

    working_angle = test_for_collisions(
        seed_insertions, probe_mesh, df, rotations_to_test
    )
    S = make_final_insertion_scene(
        working_angle,
        headframe_mesh,
        probe_mesh,
        cone,
        transformed_implant,
        transformed_annotation,
        seed_insertions,
        df,
        cm,
    )
    S.show()
