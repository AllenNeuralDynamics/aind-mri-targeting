# ---
# jupyter:
#   jupytext:
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
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import trimesh
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.file_io.slicer_files import (
    get_segmented_labels,
)
from aind_mri_utils.plots import make_3d_ax_look_normal
from aind_mri_utils.sitk_volume import find_points_equal_to
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.optimize import fmin

%matplotlib ipympl

# %%
# Paths
mouse_id = 750105
whoami = "galen"
if whoami == "galen":
    scratchdir = Path("/mnt/aind1-vast/scratch/")
    base_save_dir = Path("/home/galen.lynch/")
elif whoami == "yoni":
    scratchdir = Path(r"Y:")
mri_folder = scratchdir / "ephys/persist/data/MRI"
processed_folder = mri_folder / "processed"
mouse_folder = processed_folder / str(mouse_id)
implant_annotations_file = mouse_folder / f"{mouse_id}_ImplantHoles.seg.nrrd"
hole_folder = mri_folder / "HeadframeModels/HoleOBJs"
hole_files = [
    x for x in os.listdir(hole_folder) if ".obj" in x and "Hole" in x
]

# %%
# Load the implant annotations
implant_annotations = sitk.ReadImage(str(implant_annotations_file))

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


# %%
# import the annotations for each hole
implant_annotations_names = get_segmented_labels(implant_annotations)

annotate_hole_pts = {}

fig, ax = plt.subplots()
ax = plt.axes(projection="3d")

for holename, segval in implant_annotations_names.items():
    positions = find_points_equal_to(implant_annotations, segval)
    annotate_hole_pts[int(holename)] = positions

    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
make_3d_ax_look_normal(ax)

# %%
# Optimization functions


def append_ones_column(data):
    return np.hstack([data, np.ones([data.shape[0], 1])])


def create_rigid_transform(rx, ry, rz, cx, cy, cz):
    """
    Creates a rigid transform from 6 variables.
    For consta
    """

    RZ = np.array(
        [
            [np.cos(np.deg2rad(rz)), -np.sin(np.deg2rad(rz)), 0],
            [np.sin(np.deg2rad(rz)), np.cos(np.deg2rad(rz)), 0],
            [0, 0, 1],
        ]
    )
    RX = np.array(
        [
            [1, 0, 0],
            [0, np.cos(np.deg2rad(rx)), -np.sin(np.deg2rad(rx))],
            [0, np.sin(np.deg2rad(rx)), np.cos(np.deg2rad(rx))],
        ]
    )
    RY = np.array(
        [
            [np.cos(np.deg2rad(ry)), 0, np.sin(np.deg2rad(ry))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(ry)), 0, np.cos(np.deg2rad(ry))],
        ]
    )

    R = np.dot(np.dot(RX, RY), RZ)
    R = np.vstack([R, np.array([cx, cy, cz])])
    return R


def cost_function(T, hole_dict, annotate_hole_pts, hole_plot_dict):
    total_distance = 0
    for ii, hole_id in enumerate(hole_dict.keys()):
        if hole_id not in annotate_hole_pts.keys():
            continue
        this_hole_mesh = hole_dict[hole_id]
        this_hole_pts = annotate_hole_pts[hole_id]

        rx = T[0]
        ry = T[1]
        rz = T[2]
        cx = T[3]
        cy = T[4]
        cz = T[5]
        trans = create_rigid_transform(rx, ry, rz, cx, cy, cz)
        transformed_hole_pts = np.dot(append_ones_column(this_hole_pts), trans)

        _, distances, _ = trimesh.proximity.closest_point(
            this_hole_mesh, transformed_hole_pts
        )
        total_distance += np.sum(distances)
    # print(total_distance)
    return total_distance


def distance_to_all_triangles_optimized(mesh, points, n_jobs=-1):
    """
    Calculate the distance between a point and every triangle in the mesh, optimized for speed.

    Parameters:
    - mesh (trimesh.Trimesh): The mesh object to which the distance is calculated.
    - point (numpy.ndarray): The point (x, y, z) from which the distance is calculated.
    - n_jobs (int): The number of parallel jobs to run (-1 means using all processors).

    Returns:
    - numpy.ndarray: An array of distances from the point to each triangle in the mesh.
    - numpy.ndarray: An array of the closest points on each triangle.
    """

    # Extract the triangles from the mesh
    triangles = mesh.triangles

    # Parallel computation across triangles
    results = Parallel(n_jobs=n_jobs)(
        delayed(distance_to_triangle)(triangle) for triangle in triangles
    )

    # Unpack the results
    distances, nearest_points = zip(*results)

    return np.concatenate(np.array(distances)), np.array(nearest_points)


def distance_to_triangle(triangle, points):
    tri_mesh = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]])
    nearest_points, distance, _ = trimesh.proximity.closest_point(
        tri_mesh, points
    )
    return distance, nearest_points


def distance_to_all_triangles_in_mesh(mesh, points, normalize=1):
    triangles = mesh.triangles
    distance = []
    nearest_points = []
    for ii, triangle in enumerate(triangles):
        this_distance, this_nearest_points = distance_to_triangle(
            triangle, points
        )
        distance.append(this_distance)
        nearest_points.append(this_nearest_points)
    if normalize:
        distance = np.array(distance) / len(points)
    return distance, nearest_points


def distance_to_closest_point_for_each_traingle_in_mesh(
    mesh, points, normalize=1
):
    triangles = mesh.triangles
    distance = []
    nearest_points = []
    for ii, triangle in enumerate(triangles):
        this_distance, this_nearest_points = distance_to_triangle(
            triangle, points
        )
        mnidx = np.argmin(this_distance)
        distance.append(this_distance[mnidx])
        nearest_points.append(this_nearest_points[mnidx, :])
    if normalize:
        distance = np.array(distance) / len(triangles)
    return distance, nearest_points


def cost_function_v2(T, hole_dict, annotate_hole_pts):  # ,hole_plot_dict):
    rx = T[0]
    ry = T[1]
    rz = T[2]
    cx = T[3]
    cy = T[4]
    cz = T[5]
    trans = create_rigid_transform(rx, ry, rz, cx, cy, cz)

    parlist = []
    for ii, hole_id in enumerate(hole_dict.keys()):
        if hole_id not in annotate_hole_pts.keys():
            continue
        if hole_id != -1:
            this_hole_mesh = hole_dict[hole_id]
            this_hole_pts = annotate_hole_pts[hole_id]

            transformed_hole_pts = np.dot(
                append_ones_column(this_hole_pts), trans
            )
            parlist.append(
                delayed(distance_to_all_triangles_in_mesh)(
                    this_hole_mesh, transformed_hole_pts
                )
            )
        elif hole_id == -1:
            lower_mesh = hole_dict[hole_id]
            brain_outline = annotate_hole_pts[hole_id]
            transformed_brain_outline = np.dot(
                append_ones_column(brain_outline), trans
            )
            parlist.append(
                delayed(distance_to_closest_point_for_each_traingle_in_mesh)(
                    lower_mesh, transformed_brain_outline
                )
            )

    results = Parallel(n_jobs=-1)(parlist)
    distances, _ = zip(*results)
    total_distance = np.sum(
        np.concatenate([np.array(x).flatten() for x in distances])
    )
    print(total_distance)
    return total_distance


def save_sitk_transform(filename, T):
    trans = create_rigid_transform(T[0], T[1], T[2], T[3], T[4], T[5])
    A = sitk.AffineTransform(3)
    A.SetMatrix(trans[:3, :3].T.flatten())
    A.SetTranslation(trans[3, :])
    sitk.WriteTransform(A.GetInverse(), filename)


# %%
initialization_hole = 4
annotation_mean = np.mean(annotate_hole_pts[initialization_hole], axis=0)
model_mean = np.mean(hole_dict[initialization_hole].vertices, axis=0)
init_offset = model_mean - annotation_mean
T = [0, 0, 0, init_offset[0], init_offset[1], init_offset[2]]


output = fmin(
    cost_function_v2,
    T,
    args=(hole_dict, annotate_hole_pts),  # ,hole_plot_dict),
    xtol=1e-6,
    maxiter=2000,
)
# callback=callback_func,)
save_sitk_transform(
    str(
        mouse_folder
        / f"{mouse_id}_implant_annotations_to_lps_implant_model_with_brain.h5"
    ),
    output,
)

# %%
