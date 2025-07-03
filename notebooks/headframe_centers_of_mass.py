# %%
from pathlib import Path

from aind_mri_targeting.headframe_rotations import headframe_centers_of_mass

# %%
# Input files
basepath = Path(r"Y:\ephys\persist\data\MRI\processed\771433")
mri_path = basepath / "771433_100.nii.gz"
seg_path = basepath / "771433_HeadframeHoles.seg.nrrd"

# Output directory
output_dir = basepath

# Optional mouse ID
mouse_id = 771433

# Whether to overwrite:
force = False

# Default key format is "{}_{}" for orientation and AP direction.
# Must match the segments in the segmentation file. If not, specify the key
# format.
segment_format = None

ignore_list = []  # list of segment names to ignore
# %%
headframe_centers_of_mass(
    mri_path,
    seg_path,
    output_dir,
    mouse_id=mouse_id,
    segment_format=segment_format,
    force=force,
    ignore_list=ignore_list,
)

# %%

# %%
