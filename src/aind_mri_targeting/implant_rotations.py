"""Module for implant rotations."""

import re
from pathlib import Path

import SimpleITK as sitk
import trimesh
from aind_anatomical_utils import coordinate_systems as cs
from aind_mri_utils.implant import fit_implant_to_mri, make_hole_seg_dict
from aind_mri_utils.rotations import rotation_matrix_to_sitk


def find_hole_files(directory_or_files, hole_file_pattern=r"Hole(\d+).obj"):
    """
    Find all hole files in a directory or from a list of files and return a
    dictionary mapping hole numbers to file paths.

    Parameters
    ----------
    directory_or_files : Path, str or iterable
        Either the directory path containing the hole files or an iterable of
        file paths.
    hole_file_pattern : str, optional
        The regular expression pattern to match hole file names. Default is
        r"Hole(\\d+).obj". Must contain a single group to extract the hole
        number.

    Returns
    -------
    dict
        A dictionary where the keys are hole numbers (int) extracted from the
        file names and the values are the corresponding file paths (Path
        objects).

    Raises
    ------
    ValueError
        If `directory_or_files` is a string or Path but is not a valid
        directory.

    Examples
    --------
    >>> find_hole_files("/path/to/directory")
    {1: PosixPath('/path/to/directory/Hole1.obj'),
    2: PosixPath('/path/to/directory/Hole2.obj')}

    >>> find_hole_files(["/path/to/directory/Hole1.obj",
        "/path/to/directory/Hole2.obj"])
    {1: PosixPath('/path/to/directory/Hole1.obj'),
    2: PosixPath('/path/to/directory/Hole2.obj')}
    """
    # Is it a directory or a list of files?
    if isinstance(directory_or_files, str):
        directory_or_files = Path(directory_or_files)
    if isinstance(directory_or_files, Path):
        if not directory_or_files.is_dir():
            raise ValueError(f"{directory_or_files} is not a directory.")
        files = directory_or_files.iterdir()
    else:
        files = directory_or_files

    hole_files = {}
    for file in files:
        if not isinstance(file, Path):
            file = Path(file)
        match = re.match(hole_file_pattern, file.name)
        if match:
            hole_files[int(match.group(1))] = file

    return hole_files


def load_convert_trimesh(file, source_cs="ASR", target_cs="LPS"):
    """
    Load a trimesh object from a file and convert its vertices from one
    coordinate system to another.

    Parameters
    ----------
    file : Path or str
        The file path to the trimesh object.
    source_cs : str, optional
        The coordinate system of the vertices in the file. Default is "ASR".
    target_cs : str, optional
        The coordinate system to convert the vertices to. Default is "LPS".

    Returns
    -------
    trimesh.Trimesh
        The trimesh object with vertices converted to the target coordinate
        system.
    """
    mesh = trimesh.load(file)
    if source_cs != target_cs:
        mesh.vertices = cs.convert_coordinate_system(
            mesh.vertices, source_cs, target_cs
        )
    return mesh


def find_load_meshes(
    hole_directory,
    lower_face_file="LowerFace.obj",
    hole_file_kws={},
    load_convert_kws={},
):
    """
    Find all hole files in a directory, load them as trimesh objects and
    convert their vertices to a target coordinate system.

    Parameters
    ----------
    hole_directory : Path or str
        The directory path containing the hole files.
    lower_face_file : str, optional
        The file name of the lower face mesh. Default is "LowerFace.obj".
    hole_file_kws : dict, optional
        Additional keyword arguments to pass to `find_hole_files`. Default is
        an empty dictionary.
    load_convert_kws : dict, optional
        Additional keyword arguments to pass to `load_convert_trimesh`. Default
        is an empty dictionary.

    Returns
    -------
    dict
        A dictionary where the keys are hole numbers (int) extracted from the
        file names and the values are the corresponding trimesh objects with
        vertices converted to the target coordinate system.

    Examples
    --------
    >>> find_load_meshes("/path/to/directory")
    {1: <trimesh.Trimesh>, 2: <trimesh.Trimesh>}

    >>> find_load_meshes("/path/to/dir", source_cs="LPS", target_cs="ASR")
    {1: <trimesh.Trimesh>, 2: <trimesh.Trimesh>}
    """
    hole_files = find_hole_files(hole_directory, **hole_file_kws)
    hole_meshes = {}
    for hole_num, file in hole_files.items():
        hole_meshes[hole_num] = load_convert_trimesh(file, **load_convert_kws)
    hole_meshes[-1] = load_convert_trimesh(
        Path(hole_directory) / lower_face_file, **load_convert_kws
    )
    return hole_meshes


def fit_implant_to_mri_from_files(
    implant_annotations_file,
    hole_directory,
    save_name=None,
    save_inverse=True,
    force=False,
    mouse_name=None,
    find_load_mesh_kws={},
    fit_kws={},
):
    """
    Fit an implant to an MRI volume using hole files and implant annotations.

    Parameters
    ----------
    implant_annotations_file : Path or str
        The file path to the implant annotations.
    hole_directory : Path or str
        The directory path containing the hole files.
    save_name : Path or str, optional
        The file path to save the transformation matrix. Default is None.
    save_inverse : bool, optional
        Whether to save the inverse of the transformation matrix. Default is
        True.
    force : bool, optional
        Whether to overwrite the save file if it already exists. Default is
        False.
    mouse_name : int or str, optional
        The mouse ID. Default is None.
    find_load_mesh_kws : dict, optional
        Additional keyword arguments to pass to `find_load_meshes`. Default is
        an empty dictionary.
    fit_kws : dict, optional
        Additional keyword arguments to pass to `fit_implant_to_mri`. Default
        is an empty dictionary.

    Returns
    -------
    dict
        A dictionary containing the fitted implant and hole rotation matrices
        for each hole and the lower face.

    Examples
    --------
    >>> fit_implant_to_mri_from_files("/path/to/file", "/path/to/directory")
    {'implant': <trimesh.Trimesh>, 1: <numpy.ndarray>, 2: <numpy.ndarray>}
    """
    if save_name is not None:
        save_path = Path(save_name)
        if save_path.is_dir():
            if mouse_name is not None:
                file_name = f"{mouse_name}_implant_fit.h5"
            else:
                file_name = "implant_fit.h5"
            save_file_path = save_path / file_name
        elif save_path.suffix == "h5":
            save_file_path = save_path
        else:
            raise ValueError("save_name must be a directory or an h5 file.")
        if not force and Path(save_file_path).is_file():
            raise FileExistsError(f"{save_file_path} already exists.")
    else:
        save_file_path = None

    implant_annotations = sitk.ReadImage(str(implant_annotations_file))
    hole_seg_dict = make_hole_seg_dict(implant_annotations)
    hole_mesh_dict = find_load_meshes(hole_directory, **find_load_mesh_kws)

    rotation_matrix, translation = fit_implant_to_mri(
        hole_seg_dict, hole_mesh_dict, **fit_kws
    )

    if save_file_path is not None:
        transform = rotation_matrix_to_sitk(
            rotation=rotation_matrix, translation=translation
        )
        if save_inverse:
            transform = transform.GetInverse()
        sitk.WriteTransform(transform, str(save_file_path))

    return rotation_matrix, translation
