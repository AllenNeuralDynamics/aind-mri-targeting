"""
Routines for finding the centers of mass and rotations of headframes
"""

import os
from pathlib import Path

import SimpleITK as sitk
import nrrd
from aind_mri_utils import headframe_rotation as hr
from aind_mri_utils.file_io import slicer_files as sf


def try_open_sitk(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    return sitk.ReadImage(path)


def make_segment_dict(segment_info, segment_format, ap_names, orient_names):
    seg_vals = dict()
    for orient in orient_names:
        seg_vals[orient] = dict()
        for ap in ap_names:
            key_name = segment_format.format(ap, orient)
            if key_name in segment_info:
                seg_vals[orient][ap] = segment_info[key_name]
    return seg_vals


def create_savenames(savepath, save_format, orient_names, ap_names):
    save_names = dict()
    for orient in orient_names:
        save_names[orient] = dict()
        for ap in ap_names:
            save_names[orient][ap] = savepath / save_format.format(ap, orient)
    return save_names


def headframe_centers_of_mass(
    mri_path,
    segmentation_path,
    output_path=None,
    segment_format=None,
    mouse_id=None,
    ap_names=("anterior", "posterior"),
    orient_names=("horizontal", "vertical"),
    force=False,
):
    """
    Compute the centers of mass for headframe segments in MRI images and save them to files.

    Parameters
    ----------
    mri_path : str
        The file path to the MRI image.
    segmentation_path : str
        The file path to the segmentation image.
    output_path : str or None, optional
        The directory where the output files will be saved. If None, the current working directory is used.
        The default is None.
    segment_format : str or None, optional
        The format string for segment names. The default is None, in which case "{}_{}" will be used.
        the string will be formatted with `segment_format.format(ap, orient)`.
    mouse_id : str or None, optional
        The ID of the mouse. If None, the output file names will not include a mouse ID.
        The default is None.
    ap_names : tuple of str, optional
        The names of the anterior-posterior axis segments. The default is ("anterior", "posterior").
    orient_names : tuple of str, optional
        The names of the orientation segments. The default is ("horizontal", "vertical").
    force : bool, optional
        If True, overwrite existing files in the output directory. If False, raise an error if a file already exists.
        The default is False.

    Returns
    -------
    None
        This function does not return any value.

    Creates
    -------
    Center of mass fcsv files
        The function saves files with the computed centers of mass for each segment in the specified output directory.

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
    - The function computes the centers of mass for specified segments in MRI and segmentation images.
    - The output file names can include the mouse ID and follow a specified format.
    - Existing files in the output directory will be skipped unless `force` is set to True.

    Examples
    --------
    Compute centers of mass and save to the current working directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii")

    Compute centers of mass and save to a specified directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii", output_path="/path/to/output")

    Force overwrite existing files in the specified directory:

    >>> headframe_centers_of_mass("mri.nii", "segmentation.nii", output_path="/path/to/output", force=True)
    """

    if output_path is None:
        output_path = os.getcwd()
    if not os.path.isdir(output_path):
        raise NotADirectoryError(
            f"Output path {output_path} is not a directory"
        )
    savepath = Path(output_path)

    if mouse_id is None:
        savename_format = "{}_{}_coms.fcsv"
    else:
        savename_format = f"{mouse_id}_{{}}_{{}}_coms.fcsv"

    if segment_format is None:
        segment_format = "{}_{}"

    save_names = create_savenames(
        savepath, savename_format, orient_names, ap_names
    )
    if not force:
        for orient in orient_names:
            for ap in ap_names:
                file = save_names[orient][ap]
                if os.path.exists(file):
                    raise FileExistsError(f"File {file} already exists")

    img = try_open_sitk(mri_path)
    seg_img = try_open_sitk(segmentation_path)
    _, seg_odict = nrrd.read(segmentation_path)
    segment_info = sf.find_seg_nrrd_header_segment_info(seg_odict)

    seg_vals = make_segment_dict(
        segment_info, segment_format, ap_names, orient_names
    )
    if all([len(d) == 0 for d in seg_vals.values()]):
        raise ValueError(
            "No segments found. Is the key format {key_format} correct?"
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
