"""
Routines for finding the centers of mass and rotations of headframes
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import nrrd
import SimpleITK as sitk
from aind_mri_utils import headframe_rotation as hr
from aind_mri_utils import rotations as mrrot
from aind_mri_utils.file_io import slicer_files as sf

from . import util as mrt_ut


def try_open_sitk(path: str) -> sitk.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    return sitk.ReadImage(path)


def create_savenames(
    savepath: Path,
    save_format: str,
    orient_names: Tuple[str, str],
    ap_names: Tuple[str, str],
) -> Dict[str, Dict[str, Path]]:
    save_names = dict()
    for orient in orient_names:
        save_names[orient] = dict()
        for ap in ap_names:
            save_names[orient][ap] = savepath / save_format.format(ap, orient)
    return save_names


def headframe_centers_of_mass(
    mri_path: str,
    segmentation_path: str,
    output_path: str = None,
    segment_format: str = None,
    mouse_id: str = None,
    ap_names: Tuple[str, str] = ("anterior", "posterior"),
    orient_names: Tuple[str, str] = ("horizontal", "vertical"),
    force: bool = False,
    ignore_list: List[str] = [],
) -> None:
    """
    Compute the centers of mass for headframe segments in MRI images and save
    them to files.

    Parameters
    ----------
    mri_path : str
        The file path to the MRI image.
    segmentation_path : str
        The file path to the segmentation image.
    output_path : str or None, optional
        The directory where the output files will be saved. If None, the
        current working directory is used.  The default is None.
    segment_format : str or None, optional
        The format string for segment names. The default is None, in which case
        "{}_{}" will be used.  the string will be formatted with
        `segment_format.format(ap, orient)`.
    mouse_id : str or None, optional
        The ID of the mouse. If None, the output file names will not include a
        mouse ID.  The default is None.
    ap_names : tuple of str, optional
        The names of the anterior-posterior axis segments. The default is
        ("anterior", "posterior").
    orient_names : tuple of str, optional
        The names of the orientation segments. The default is ("horizontal",
        "vertical").
    force : bool, optional
        If True, overwrite existing files in the output directory. If False,
        raise an error if a file already exists.  The default is False.
    ignore_list : list of str, optional
        List of segment names to ignore. If empty, no segments are ignored. The
        default is [].

    Returns
    -------
    None
        This function does not return any value.

    Creates
    -------
    Center of mass fcsv files
        The function saves files with the computed centers of mass for each
        segment in the specified output directory.

    Raises
    ------
    NotADirectoryError
        If the output path is not a directory.
    FileExistsError
        If a file already exists and `force` is set to False.
    ValueError
        If no segments are found based on the provided key format.

    Notes
    -----
    - The function computes the centers of mass for specified segments in MRI
    and segmentation images.
    - The output file names can include the mouse ID and follow a specified
    format.
    - Existing files in the output directory will be skipped unless `force` is
    set to True.

    Examples
    --------
    Compute centers of mass and save to the current working directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii")

    Compute centers of mass and save to a specified directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii",
    output_path="/path/to/output")

    Force overwrite existing files in the specified directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii",
    output_path="/path/to/output", force=True)
    """
    savepath = mrt_ut.check_output_path(output_path)

    if mouse_id is None:
        savename_format = "{}_{}_coms.fcsv"
    else:
        savename_format = f"{mouse_id}_{{}}_{{}}_coms.fcsv"

    save_names = create_savenames(
        savepath, savename_format, orient_names, ap_names
    )
    if not force:
        filenames = [
            f for orient in orient_names for f in save_names[orient].values()
        ]
        mrt_ut.err_if_file_exists(filenames)

    img = try_open_sitk(mri_path)
    seg_img = try_open_sitk(segmentation_path)
    _, seg_odict = nrrd.read(segmentation_path)
    seg_vals = hr.segment_dict_from_seg_odict(
        seg_odict, segment_format, ap_names, orient_names, ignore_list
    )

    coms_dict = hr.estimate_coms_from_image_and_segmentation(
        img, seg_img, seg_vals
    )

    for orient in orient_names:
        for ap in ap_names:
            if ap in seg_vals[orient]:
                coms = coms_dict[orient][ap]
                save_filename = savepath / savename_format.format(ap, orient)
                ptdict = {i: coms[i, :] for i in range(coms.shape[0])}
                sf.create_slicer_fcsv(save_filename, ptdict)
    return


def theta_to_sitk_affine(
    theta: List[float], inverse: bool = False
) -> sitk.AffineTransform:
    """
    Convert a set of theta parameters to a SimpleITK affine transformation.

    Parameters
    ----------
    theta : list
        A list of theta parameters representing rotation angles and translation
        values.
    inverse : bool, optional
        Flag indicating whether to compute the inverse transformation. Default
        is False.

    Returns
    -------
    sitk.AffineTransform
        The SimpleITK affine transformation.

    """
    rotmat = mrrot.combine_angles(*theta[:3])
    translation = theta[3:]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rotmat.flatten())
    affine.SetTranslation(translation.tolist())
    if inverse:
        affine = affine.GetInverse()
    return affine


def calculate_headframe_transforms(
    img_path: str,
    seg_path: str,
    lower_plane_path: str,
    output_path: str = None,
    mouse_name: str = None,
    volume_transforms: bool = True,
    segment_format: str = "{}_{}",
    force: bool = False,
    ignore_list: List[str] = [],
) -> None:
    """
    Calculate rotations from segmentation.

    This function calculates rotations from a given segmentation and lower
    plane information.  It saves the calculated rotations as transforms in the
    specified output path.

    Parameters
    ----------
    img_path : str
        Path to the image file.
    seg_path : str
        Path to the segmentation file.
    lower_plane_path : str
        Path to the lower plane file.
    output_path : str
        Path to save the calculated rotations.
    mouse_name : str, optional
        Name of the mouse. Defaults to None.
    volume_transforms : bool, optional
        Whether to write the inverse of the calculated rotations. If you are
        transforming a volume with SITK, you need to use the inverse. Defaults
        to True.
    segment_format : str, optional
        Format for segment names. Defaults to "{}_{}".
    force : bool, optional
        Whether to overwrite existing output files. Defaults to False.
    ignore_list : list of str, optional
        List of segment names to ignore. Defaults to [].

    Returns
    -------
    None
        This function does not return anything.

    """
    savepath = mrt_ut.check_output_path(output_path)
    output_fnames = [
        "hf_hole_angles.h5",
        "com.h5",
        "com_plane.h5",
    ]
    if mouse_name is not None:
        output_fnames = [f"{mouse_name}_{f}" for f in output_fnames]
    save_paths = [savepath / fname for fname in output_fnames]
    if not force:
        mrt_ut.err_if_files_exist(save_paths)
    img = try_open_sitk(img_path)
    seg_img = try_open_sitk(seg_path)
    if not os.path.exists(lower_plane_path):
        raise FileNotFoundError(f"File {lower_plane_path} not found")
    plane_pts = sf.markup_json_to_numpy(lower_plane_path)[0]
    _, seg_odict = nrrd.read(seg_path)
    theta_angle, theta_coms, theta_coms_plane = (
        hr.find_hf_rotation_from_seg_and_lowerplane(
            img,
            seg_img,
            seg_odict,
            plane_pts,
            segment_format,
            ignore_list=ignore_list,
        )
    )
    for fname, theta in zip(
        save_paths, [theta_angle, theta_coms, theta_coms_plane]
    ):
        affine = theta_to_sitk_affine(theta, inverse=volume_transforms)
        sitk.WriteTransform(affine, str(fname))
    return
