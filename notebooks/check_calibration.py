# %%
# Imports
# %%
import importlib
import itertools as itr

import numpy as np
from aind_mri_utils import parallax_calibration as pc
from aind_mri_utils import reticle_calibrations as rc
from matplotlib import pyplot as plt

importlib.reload(rc)
# %%
# Define the calibration file and probe
cal_file = "/mnt/aind1-vast/scratch/ephys/persist/data/probe_calibrations/CSVCalibrations/calibration_info_np2_2024_08_01T11_23_00.xlsx"  # noqa E501
probe = 46116


# %%
# Functions
def calc_pairwise_distances(pts):
    npt = pts.shape[0]
    pairwise_idxs = list(itr.product(range(npt), range(npt)))

    distances = np.empty((npt, npt))

    for i, j in pairwise_idxs:
        distances[i, j] = np.linalg.norm(pts[i, :] - pts[j, :])
    return distances


# %%
# Load the calibration file
(
    adjusted_pairs_by_probe,
    global_offset,
    global_rotation_degrees,
    reticle_name,
) = rc.read_reticle_calibration(cal_file)

reticle_pts, probe_pts = adjusted_pairs_by_probe[probe]

# %%
reticle_distances = calc_pairwise_distances(reticle_pts)
probe_distances = calc_pairwise_distances(probe_pts)
# %%
drift = reticle_distances - probe_distances

# %%
term_kwargs = dict(xtol=1e-12, gtol=1e-12, ftol=1e-15)
R, scaling, translation, res = rc.fit_rotation_params(
    reticle_pts, probe_pts, find_scaling=True, **term_kwargs
)
R_unscaled, translation_unscaled, res_unscaled = rc.fit_rotation_params(
    reticle_pts, probe_pts, find_scaling=False, **term_kwargs
)

rt = pc.RotationTransformation()
translation_parallax, R_parallax, scale_parallax, _ = rt.fit_params(
    probe_pts, reticle_pts
)
# %%
fit_reticle_pts = rc.transform_probe_to_reticle(
    probe_pts, R, translation, scaling
)
fit_reticle_pts_unscaled = rc.transform_probe_to_reticle(
    probe_pts, R_unscaled, translation_unscaled
)
fit_probe_pts = rc.transform_reticle_to_probe(
    reticle_pts, R, translation, scaling
)
fit_probe_pts_unscaled = rc.transform_reticle_to_probe(
    reticle_pts, R_unscaled, translation_unscaled
)
# %%
f, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection="3d"))
# %%
errs = fit_reticle_pts - reticle_pts
errs_unscaled = fit_reticle_pts_unscaled - reticle_pts
errs_probe = fit_probe_pts - probe_pts
errs_probe_unscaled = fit_probe_pts_unscaled - probe_pts
med_err = np.median(np.linalg.norm(errs, axis=1))
med_err_unscaled = np.median(np.linalg.norm(errs_unscaled, axis=1))
med_err_probe = np.median(np.linalg.norm(errs_probe, axis=1))
med_err_probe_unscaled = np.median(np.linalg.norm(errs_probe_unscaled, axis=1))
X, Y, Z = [reticle_pts[:, i] for i in range(3)]
U, V, W = [errs[:, i] for i in range(3)]
# ax.quiver3D(X, Y, Z, U, V, W, angles="xy", scale_units="xy", scale=1)
# %%
probe_distances_scaled = calc_pairwise_distances(probe_pts * scaling)
drift_scaled = reticle_distances - probe_distances_scaled

# %%
