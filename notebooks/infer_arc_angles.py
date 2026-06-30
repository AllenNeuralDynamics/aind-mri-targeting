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

# %% [markdown]
# # Infer arc angles (AP/ML) from a probe calibration stack-up
#
# Parses a probe calibration stack-up (parallax log directories and/or manual
# reticle `.xlsx` calibrations, combined the same way as the calibration
# notebook) and reports the inferred arc angles for each probe.
#
# The insertion direction is the manipulator z-down axis expressed in bregma
# coordinates. By default the AP angle is reported in the ephys-rig/arc frame
# (the fixed +14° headframe pitch is added), matching what the rig arc reads.

# %%
from pathlib import Path

from aind_mri_targeting.measured_holes import (
    arc_angles_from_calibration,
    check_parallax_calibration,
)

# %%
# Configuration
#
# NOTE: set the reticle actually used. The reticle offset/rotation only matter
# for parallax-only calibrations; manual calibrations carry their own reticle
# metadata (and, when combined, supply it to the parallax calibrations too).
reticle = "H"

base_dir = Path("/mnt/vast/scratch/")
calibration_dir = base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations"

# Parallax calibration directories (a "stack-up"), layered to cover the full
# set of probes; later entries win.
parallax_dirs = [
    calibration_dir / "log_20260610_180434",
    calibration_dir / "log_20260611_150416",
    calibration_dir / "log_20260611_172932",
]
# Manual reticle calibration .xlsx files (highest priority); later entries win.
manual_files = []
# Probes whose manual calibration should be ignored (parallax used instead).
probes_to_ignore_manual = []

# %% [markdown]
# ## Inferred arc angles
# AP in the rig/arc frame (includes the +14° headframe pitch); ML angle is
# frame-independent.

# %%
arc_angles = arc_angles_from_calibration(
    manual_files=manual_files,
    parallax_dirs=parallax_dirs,
    probes_to_ignore_manual=probes_to_ignore_manual,
    reticle=reticle,
    rig_frame=True,
)
arc_angles.round(2)

# %% [markdown]
# ## Sanity check: fitted calibration vs Parallax `transM`
# Confirms our parallax fit reproduces the per-probe transforms Parallax wrote
# to each directory. `rot_max_abs_diff` and `trans_max_abs_diff_mm` should be
# ~0; `pred_*_um` is the manipulator-coordinate agreement over the calibration
# points (sub-µm expected).

# %%
import pandas as pd  # noqa: E402

check_df = pd.concat(
    {Path(d).name: check_parallax_calibration(d, reticle=reticle) for d in parallax_dirs},
    names=["Calibration", "Probe"],
)
check_df.round(4)
