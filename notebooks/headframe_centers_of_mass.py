# %%
from pathlib import Path

from aind_mri_targeting.headframe_rotations import headframes_centers_of_mass

# %%
# Input files
basepath = Path("/path/to/data")
mri_path = basepath / "mri.nii.gz"
seg_path = basepath / "segmentation.seg.nrrd"

# Output directory
output_dir = "/path/to/output"

# Optional mouse ID
mouse_id = None

# Whether to ovewrite:
force = False

# Default key format is "{}_{}" for orientation and AP direction.
# Must match the segments in the segmentation file. If not, specify the key
# format.
segment_format = None

ignore_list = []  # list of segment names to ignore
# %%
headframes_centers_of_mass(
    mri_path,
    seg_path,
    output_dir,
    mouse_id=mouse_id,
    segment_format=segment_format,
    force=force,
    ignore_list=ignore_list,
)
