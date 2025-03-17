# %%
from pathlib import Path

from aind_mri_targeting import headframe_rotations as hr

# %%
# File paths constant across mice
# mri_path = "/path/to/mri.nii.gz"
# seg_path = "/path/to/segmentation.seg.nrrd"
# lower_plane_file = "/path/to/lower_plane.mrk.json"
# output_path = "/path/to/output"  # If None, will write in cwd

mouse = "771433"
base_path = Path(f"Y:/ephys/persist/data/MRI/processed/{mouse}")
mri_path = base_path / f"{mouse}_100.nii.gz"
seg_path = base_path / f"{mouse}_HeadframeHoles.seg.nrrd"
lower_plane_file = base_path / f"{mouse}_LowerPlane.mrk.json"
output_path = base_path

# Optional
mouse = None  # or string of mouse name
force = False  # True to overwrite existing files
# True to write SITK transform for volumes instead of points
volume_transforms = True
segment_format = None  # default is "{}_{}"
ignore_list = []  # list of segment names to ignore
# %%
hr.calculate_headframe_transforms(
    mri_path,
    seg_path,
    lower_plane_file,
    output_path,
    mouse_name=mouse,
    segment_format=segment_format,
    force=force,
    volume_transforms=volume_transforms,
    ignore_list=ignore_list,
)

# %%
