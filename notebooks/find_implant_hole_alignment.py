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
from pathlib import Path

from aind_mri_targeting.implant_rotations import fit_implant_to_mri_from_files

# %%
# Paths
mouse_id = 771433
whoami = "yoni"
if whoami == "galen":
    scratchdir = Path("/mnt/aind1-vast/scratch/")
    base_save_dir = Path("/home/galen.lynch/")
elif whoami == "yoni":
    scratchdir = Path(r"Y:")
mri_folder = scratchdir / "ephys/persist/data/MRI"
processed_folder = mri_folder / "processed"
mouse_folder = processed_folder / str(mouse_id)

# required files
implant_annotations_file = mouse_folder / f"{mouse_id}_ImplantHoles.seg.nrrd"
hole_folder = mri_folder / "HeadframeModels/HoleOBJs"
save_name = str(mouse_folder)  # f"{mouse_id}_implant_fit.h5")

# optional flags
save_inverse = True  # save the inverse transformation matrix, which works on volumes
force = True

# %%
# Run the fitting
fit_implant_to_mri_from_files(
    implant_annotations_file,
    hole_folder,
    save_name=save_name,
    save_inverse=save_inverse,
    force=force,
    mouse_name=mouse_id,
)

# %%
save_name

# %%
