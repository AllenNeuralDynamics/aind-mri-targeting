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
    test_for_collisions,
    make_scene_for_insertion,
    make_final_insertion_scene,
)


def find_targets(pts):
    pass


def load_files_to_plan_insertion():
    pass


def apply_shift_and_transform(
    pts, rotation, translation, chemical_shift, readout_direction
):
    pass


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
