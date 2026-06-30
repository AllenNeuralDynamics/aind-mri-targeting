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
# # Plan off-plane stereotax angles from ephys-rig arc readings
#
# For a planned insertion you know the ephys-rig **arc reading** (AP, ML). To
# reproduce the *same physical insertion* — same probe direction relative to the
# mouse head — on the Kopf 1500 off-plane stereotax, you can re-tilt the head
# with the earbar gimbal and let the Kopf tool take up the rest.
#
# This notebook solves, per insertion, for the earbar pose (pitch about ML, roll
# about AP) that brings the insertion **as close to stereotax-vertical as
# possible** within the earbar's mechanical limits, then reports the Kopf
# off-plane angles (`ry` polar tilt, `rz` azimuth) for the residual.
#
# The arc AP reading is taken in the rig frame (it includes the fixed +14°
# headframe pitch); the solver removes that internally. ML is frame-independent.
#
# Backed by `aind_mri_utils.arc_angles.arc_angles_to_earbar_stereotax`.

# %%
import pandas as pd
from aind_mri_utils.arc_angles import arc_angles_to_earbar_stereotax

# %%
# Configuration
#
# Earbar mechanical limits (degrees). Defaults are the *safe* ranges; the hard
# limits are (-12, 12) pitch and (-30, 30) roll. Pitch is about the ML axis,
# roll about the AP axis.
pitch_bounds = (-10.0, 10.0)
roll_bounds = (-15.0, 15.0)

# Planned insertions as (label, arc AP, arc ML), arc angles in the rig frame.
# Replace with your own. To drive this from a probe calibration instead, run
# `infer_arc_angles` (arc_angles_from_calibration, rig_frame=True) and feed its
# AP_angle / ML_angle columns in here.
insertions = [
    ("probe_A", 14.0, 0.0),
    ("probe_B", 14.0, 10.0),
    ("probe_C", 24.0, 20.0),
    ("probe_D", 34.0, 0.0),
    ("probe_E", 40.0, -25.0),
]


# %%
def solve_insertions(insertions, *, pitch_bounds, roll_bounds):
    """Solve the earbar/Kopf configuration for each (label, arc_AP, arc_ML)."""
    rows = []
    for label, arc_ap, arc_ml in insertions:
        sol = arc_angles_to_earbar_stereotax(
            arc_ap,
            arc_ml,
            pitch_bounds=pitch_bounds,
            roll_bounds=roll_bounds,
        )
        rows.append(
            {
                "insertion": label,
                "arc_AP": arc_ap,
                "arc_ML": arc_ml,
                "earbar_pitch": sol.earbar_pitch,
                "earbar_roll": sol.earbar_roll,
                "kopf_ry": sol.kopf_ry,
                "kopf_rz": sol.kopf_rz,
                "vertical": sol.vertical_achievable,
            }
        )
    return pd.DataFrame(rows).set_index("insertion")


# %% [markdown]
# ## Stereotax configuration per insertion
# `earbar_pitch` / `earbar_roll` are the gimbal settings; `kopf_ry` / `kopf_rz`
# are the off-plane tool angles. `vertical = False` means a bound saturated and
# the Kopf tool must take up the residual `kopf_ry` tilt.

# %%
stereotax = solve_insertions(insertions, pitch_bounds=pitch_bounds, roll_bounds=roll_bounds)
stereotax.round(2)

# %%
# Flag any insertions where the earbar could not fully absorb the tilt.
saturated = stereotax[~stereotax["vertical"]]
if len(saturated):
    print(f"{len(saturated)} insertion(s) leave residual Kopf tilt (earbar bound saturated):")
    print(saturated[["earbar_pitch", "earbar_roll", "kopf_ry"]].round(2).to_string())
else:
    print("All insertions reachable vertically by the earbar alone.")

# %%
# Optionally save the table
save_path = None  # e.g. Path("stereotax_config.csv")
if save_path is not None:
    stereotax.to_csv(save_path)
