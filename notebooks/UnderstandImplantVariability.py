# +
import os
from pathlib import Path

import matplotlib
import numpy as np
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils import coordinate_systems as cs
from aind_mri_utils import rotations as rot
from aind_mri_utils.chemical_shift import compute_chemical_shift
from aind_mri_utils.file_io import simpleitk as mr_sitk
from matplotlib import pyplot as plt

# %matplotlib inline
from skimage import measure


def sitk_to_trimesh_with_metadata(sitk_image):
    """
    Converts a SimpleITK binary mask image to a trimesh object, taking into
    account the image's spatial metadata (origin, spacing, and direction).

    Args:
        sitk_image (SimpleITK.Image): The input SimpleITK binary mask image.

    Returns:
        trimesh.Trimesh: The resulting mesh object in the image's physical
        space.
    """
    # Step 1: Extract the binary mask as a NumPy array
    mask_array = sitk.GetArrayFromImage(sitk_image)

    # Step 2: Perform marching cubes to extract surface (in voxel space)
    vertices, faces, normals, values = measure.marching_cubes(mask_array, level=0.5)

    # Step 3: Extract metadata (origin, spacing, direction)
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)  # 3x3 matrix

    # Step 4: Transform voxel coordinates to physical space
    # The transformation from voxel to physical space is: physical_point =
    # origin + direction @ (voxel_point * spacing)
    vertices_voxel = vertices * spacing  # Apply spacing
    vertices_physical = np.dot(vertices_voxel[:, [2, 1, 0]], direction.T) + origin  # Apply direction and origin

    # Step 5: Create a trimesh object with the transformed vertices
    mesh = trimesh.Trimesh(vertices=vertices_physical, faces=faces)

    return mesh


# +
processed_folder = Path(r"Y:\ephys\persist\data\MRI\processed")

dataloc = {
    738789: processed_folder / str(738789),
    733585: processed_folder / str(733585),
    743697: processed_folder / str(743697) / "UW",
    750105: processed_folder / str(750105),
    743700: processed_folder / str(743700),
    727456: processed_folder / str(727456) / "UW",
    728537: processed_folder / str(728537) / "UW",
    727354: processed_folder / str(727354),
    721679: processed_folder / str(721679),
    721682: processed_folder / str(721682) / "UW2",
    721678: processed_folder / str(721678),
    # 721681:processed_folder/str(721681)/'UW',
    721680: processed_folder / str(721680) / "UW",
    721685: processed_folder / str(721685) / "HF",
    # 717381:processed_folder/str(717381),
}

implant_alignment_path = Path(r"D:\ImplantAlignment")
# -

im = sitk.ReadImage(str(Path(dataloc[738789]) / f"{738789}_100.nii.gz"))
chem_shift = compute_chemical_shift(im, ppm=3.67)
chem_shift

# +
hole_folder = Path(r"Y:\ephys\persist\data\MRI\HeadframeModels") / "HoleOBJs"


hole_files = [x for x in os.listdir(hole_folder) if ".obj" in x and "Hole" in x]
hole_dict = {}
for ii, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder, flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

model_implant_targets = {}
for ii, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

implant_names = [*model_implant_targets]
model_targets = np.vstack(list(model_implant_targets.values()))


# +
S = trimesh.Scene()

implant_mesh = trimesh.load_mesh(os.path.join(r"Y:\ephys\persist\data\MRI\HeadframeModels", "0283-300-04.obj"))
implant_mesh.vertices = cs.convert_coordinate_system(implant_mesh.vertices, "ASR", "LPS")

headframe_mesh = trimesh.load_mesh(os.path.join(r"Y:\ephys\persist\data\MRI\HeadframeModels", "TenRunHeadframe.obj"))
headframe_mesh.vertices = cs.convert_coordinate_system(headframe_mesh.vertices, "ASR", "LPS")

hole_locs = {}
brain_bottom = {}
for ii, mouse in enumerate(dataloc.keys()):
    # if ii>0:
    #     continue
    # try:
    this_mesh = implant_mesh.copy()

    implant_transform = str(
        implant_alignment_path / f"{mouse}_implant_annotations_to_lps_implant_model_with_brain_better_normalization.h5"
    )
    implant_model_trans = mr_sitk.load_sitk_transform(implant_transform, homogeneous=True, invert=False)[0].T

    # Load the computed transform
    trans = mr_sitk.load_sitk_transform(
        Path(dataloc[mouse]) / f"{mouse}_com_plane.h5",
        homogeneous=True,
        invert=True,
    )[0]

    tmp_ = rot.prepare_data_for_homogeneous_transform(this_mesh.vertices) @ implant_model_trans
    this_mesh.vertices = rot.extract_data_for_homogeneous_transform(tmp_ @ trans.T)
    this_mesh.visual.vertex_colors = np.concatenate([np.random.randint(0, 255, (3)), [125]])

    tmp_ = rot.prepare_data_for_homogeneous_transform(model_targets) @ implant_model_trans
    hole_locs[mouse] = rot.extract_data_for_homogeneous_transform(tmp_ @ trans.T)

    sitk_image = sitk.ReadImage(Path(dataloc[mouse]) / f"{mouse}_auto_skull_strip.nrrd")
    brain = sitk_to_trimesh_with_metadata(sitk_image)
    brain.visual.face_colors = [0, 255, 0, 255]
    brain.vertices[:, 1] -= chem_shift

    brain.vertices = rot.extract_data_for_homogeneous_transform(
        rot.prepare_data_for_homogeneous_transform(brain.vertices) @ trans.T
    )
    brain_bottom[mouse] = []
    for ii in range(hole_locs[mouse].shape[0]):
        intersect, _, _ = brain.ray.intersects_location(
            [hole_locs[mouse][ii, :], hole_locs[mouse][ii, :]],
            [[0, 0, 1], [0, 0, -1]],
        )
        brain_bottom[mouse].append(np.max(np.abs(intersect - hole_locs[mouse][ii, :])))

    S.add_geometry(this_mesh)
    # S.add_geometry(brain)
    # except:
    #     print(mouse)
    #     continue


print(":)")
implant_mesh.visual.face_colors = [255, 0, 0, 255]
# S.add_geometry(implant_mesh)
S.add_geometry(headframe_mesh)
S.show(viewer="gl")

this_hole = 4
ax = plt.figure().add_subplot()

idx = implant_names.index(this_hole)
depth = []
for ii, mouse in enumerate(brain_bottom.keys()):
    depth.append(brain_bottom[mouse][idx])
ax.hist(depth)

# +

this_hole = 3
ax = plt.figure().add_subplot()
cmap = matplotlib.cm.get_cmap("Dark2")

for this_hole in implant_names:
    hole_idx = implant_names.index(this_hole)

    for ii, mouse in enumerate(hole_locs.keys()):
        ax.scatter(
            model_targets[hole_idx, 0] - hole_locs[mouse][hole_idx, 0],
            model_targets[hole_idx, 1] - hole_locs[mouse][hole_idx, 1],
            c=cmap(ii),
        )
ax.scatter(0, 0, marker="*", s=500, c="k")
ax.set_aspect("equal")
ax.set_xlabel("<--M  L-->")
ax.set_ylabel("<--P  A-->")

# +

S = trimesh.Scene()

sitk_image = sitk.ReadImage(r"Y:\ephys\persist\data\MRI\processed\738789\738789_auto_skull_strip.nrrd")
mask_array = sitk.GetArrayFromImage(sitk_image)
vertices, faces, normals, values = measure.marching_cubes(mask_array, level=0.5)
mesh = trimesh.Trimesh(vertices=vertices[:, [2, 1, 0]] * 0.1 + sitk_image.GetOrigin(), faces=faces)

S.add_geometry(mesh)
S.add_geometry(implant_mesh)
S.add_geometry(headframe_mesh)
S.show()

# +

sitk_image = sitk.ReadImage(r"Y:\ephys\persist\data\MRI\processed\738789\738789_auto_skull_strip.nrrd")

mesh = sitk_to_trimesh_with_metadata(sitk_image)
mesh.visual.face_colors = [0, 255, 0, 255]

mesh.vertices = rot.extract_data_for_homogeneous_transform(
    rot.prepare_data_for_homogeneous_transform(mesh.vertices) @ trans.T
)
S = trimesh.Scene()

S.add_geometry(mesh)
S.add_geometry(implant_mesh)
S.add_geometry(headframe_mesh)
S.show()
# -

mesh.vertices
