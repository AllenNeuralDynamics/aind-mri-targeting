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
from aind_mri_utils.arc_angles import arc_angles_to_hit_two_points


def find_targets(pts):
    pass


def load_files_to_plan_insertion():
    pass


def candidate_insertions(
    transformed_annotation, transformed_implant, target_names, implant_names
):
    TARGET = []
    HOLE = []
    AP = []
    AP_Range = []
    RIG_AP = []
    ML = []
    ML_Range = []
    TARGET_LOC = []
    for tt in range(transformed_annotation.shape[0]):
        for hh in range(transformed_implant.shape[0]):
            ap, ml = arc_angles_to_hit_two_points(
                transformed_annotation[tt, :],
                transformed_implant[hh, :],
                ap_offset=0,
            )

            radius = 0.3
            theta = np.deg2rad(np.arange(0, 360, 1))
            a = transformed_implant[hh, 0] + radius * np.cos(theta)
            b = transformed_implant[hh, 1] + radius * np.sin(theta)
            circle = np.vstack(
                [a, b, np.ones(b.shape) * transformed_implant[hh, 2]]
            ).T

            this_ML = []
            this_AP = []

            for jj in range(len(a)):

                this_ap, this_ml = arc_angles_to_hit_two_points(
                    transformed_annotation[tt, :], circle[jj, :], ap_offset=0
                )
                this_ML.append(this_ml)
                this_AP.append(this_ap)

            ML_Range.append(np.abs(np.max(this_ML) - np.min(this_ML)) / 2)
            AP_Range.append(np.abs(np.max(this_AP) - np.min(this_AP)) / 2)

            if False:
                continue
            else:
                TARGET.append(target_names[tt])
                HOLE.append(implant_names[hh])
                ML.append(ml)
                AP.append(ap)
                RIG_AP.append(ap + 14)
                TARGET_LOC.append(transformed_annotation[tt, :])

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
    return df


def valid_insertion_pairs(df, ap_wiggle=1, ap_min=16, ml_min=16):
    valid = np.zeros([df.shape[0], df.shape[0]], dtype=bool)

    # check the validity of every insertions. This is order N**2, so may need
    # some thought.

    for ii, row1 in df.iterrows():
        for jj, row2 in df.iterrows():
            if ii == jj or ii < jj:
                continue
            # elif row1.target == row2.target:
            #    continue
            elif row1.hole == row2.hole:
                continue
            elif (np.abs(row1.ap - row2.ap) < ap_wiggle) and (
                np.abs(row1.ml - row2.ml) < ml_min
            ):
                continue
            elif (np.abs(row1.ap - row2.ap) < ap_min) and (
                np.abs(row1.ap - row2.ap) > ap_wiggle
            ):

                continue
            else:
                valid[ii, jj] = True
    valid = np.bitwise_or(valid, valid.T)
    return valid


def transform_matrix_from_angles_and_target(AP, ML, Target):
    R = (
        Rotation.from_euler(
            "XYZ", np.array([np.deg2rad(AP), np.deg2rad(ML), 0])
        )
        .as_matrix()
        .squeeze()
    )
    T = np.zeros([4, 4])
    T[:3, :3] = R
    T[:3, 3] = Target
    return T


def get_implant_targets(implant_vol):
    odict = {
        k: implant_vol.GetMetaData(k) for k in implant_vol.GetMetaDataKeys()
    }
    label_dict = sf.find_seg_nrrd_header_segment_info(odict)

    implant_names = []
    implant_targets = []
    implant_pts = []
    implant_vol_arr = sitk.GetArrayFromImage(implant_vol)
    for key in label_dict.keys():
        val = label_dict[key]
        idx_tup = np.nonzero(implant_vol_arr == val)
        if len(idx_tup[0]) == 0:
            continue
        implant_pos = np.vstack(
            [
                implant_vol.TransformIndexToPhysicalPoint(tup[::-1])
                for tup in zip(idx_tup)
            ]
        )
        implant_pts.append(implant_pos)
        implant_targets.append(np.mean(implant_pos, axis=0))
        this_key = key.split("-")[-1].split("_")[-1]
        implant_names.append(int(this_key))
    implant_targets = np.vstack(implant_targets)
    return implant_targets, implant_names


def make_scene_for_insertion(
    headframe_mesh,
    cone,
    transformed_implant,
    transformed_annotation,
    match_insertions,
    df,
    probe_mesh,
):
    headframe_mesh = trimesh.Trimesh()
    headframe_mesh.vertices = headframe_lps
    headframe_mesh.faces = headframe_faces

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
    target_names = tuple(preferred_pts.keys())

    ndf = pd.DataFrame(
        {
            "Target": target_names,
            "ML": -transformed_annotation[:, 0],
            "AP": -transformed_annotation[:, 1],
            "DV": transformed_annotation[:, 2],
        }
    )
    ndf.to_csv(transformed_targets_save_path)

    # If implant has holes that are segmented.
    implant_vol = sitk.ReadImage(implant_holes_path)

    implant_targets, implant_names = get_implant_targets(implant_vol)

    # Visualize Holes, list locations
    transformed_implant = rot.extract_data_for_homogeneous_transform(
        np.dot(
            rot.prepare_data_for_homogeneous_transform(implant_targets), trans
        )
    )

    df = candidate_insertions(
        transformed_annotation,
        transformed_implant,
        target_names,
        implant_names,
    )
    valid = valid_insertion_pairs(df)

    # %%
    insert_list = [68, 59, 67]
    works_for_all = set(np.nonzero(valid[insert_list[0], :])[0])

    for ii in range(0, len(insert_list)):
        works_for_all = (
            works_for_all
            & set(np.nonzero(valid[insert_list[ii], :])[0])
            & set(np.nonzero(df.target == "GenFacCran")[0])
        )

    df.iloc[np.concatenate([insert_list, list(works_for_all)])]

    # %%
    headframe_mesh = trimesh.Trimesh()
    headframe_mesh.vertices = headframe_lps
    headframe_mesh.faces = headframe_faces

    S = make_scene_for_insertion(
        headframe_mesh,
        cone,
        transformed_implant,
        transformed_annotation,
        insert_list,
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

    rotations_to_test = [np.arange(0, 360, 45)] * len(insert_list)

    working_angle = test_for_collisions(
        insert_list, probe_mesh, df, rotations_to_test
    )
    S = make_final_insertion_scene(
        working_angle,
        headframe_mesh,
        probe_mesh,
        cone,
        transformed_implant,
        transformed_annotation,
        insert_list,
        df,
        cm,
    )
    S.show()
