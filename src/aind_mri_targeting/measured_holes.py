"""Compare calibrated-probe hole measurements to implant-fit hole locations.

This module supports a QC workflow: a calibrated probe is used to physically
measure the location of each implant hole opening on a mouse, and those
measurements are compared against the hole locations predicted by the implant
fitting process (see :mod:`aind_mri_targeting.implant_rotations`).

Coordinate chain
----------------
Two independent estimates of each hole are brought into the same
**bregma-relative RAS frame** (axes ordered ML, AP, DV in mm):

* **Measured** — probe/manipulator coordinates (µm) are transformed to bregma
  with a reticle calibration (parallax log directory or manual ``.xlsx``) via
  :func:`aind_mri_utils.reticle_calibrations.transform_probe_to_bregma`.
* **Predicted** — implant CAD hole points (LPS, mm) are mapped to MRI image
  space by the inverse of the saved ``*_implant_fit.h5`` transform, then to
  bregma by the inverse of the saved ``*_com_plane.h5`` transform. (The
  transforms are stored inverted for Slicer, so they must be re-inverted to use
  on points, matching ``plan_target_insertions_implantmodel``.)

No chemical-shift correction is applied: the implant segmentation and the
headframe fiducials are both referenced to vaseline, so they are not shifted
relative to one another.

Hole shaft bottoms
------------------
Per-hole shaft geometry comes from an ``aind-low-point`` ``*.holes.yml`` file.
Each hole has two oval cross-sections; the *bottom opening of the shaft* is the
section with the more negative ``s_mm`` (equivalently the lowest LPS S
coordinate). Those bottom-opening ``center_LPS`` points are what we compare.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from aind_mri_utils import rotations as rot
from aind_mri_utils.arc_angles import vector_to_arc_angles
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.reticle_calibrations import (
    combine_parallax_and_manual_calibrations,
    find_probe_insertion_vector,
    fit_rotation_params_from_parallax,
    read_parallax_calibration_dir,
    reticle_metadata_transform,
    transform_bregma_to_probe,
    transform_probe_to_bregma,
)
from numpy.typing import NDArray

# Parallax writes its own probe transform per probe as transM_SN<sn>_<ts>.csv,
# with a header row R_0_0..R_2_2, T_0..T_2, avg_err. The convention (see the
# parallax repo, probe_calibration.py) is local = R @ global + T, in microns.
_TRANSM_RE = re.compile(r"(?i)transM_SN(\d+)(?:_.*)?\.csv$")

# A manual calibration is a single ``.xlsx`` file; a parallax calibration is a
# directory of ``points_SN*.csv`` files.
_MANUAL_SUFFIXES = {".xlsx", ".xls"}

# Headframe pitch built into the AIND ephys-rig arc: a probe vertical in the
# lambda-bregma (plan) frame reads this AP angle on the rig arc.
HEADFRAME_RX_IN_ARC = 14.0

# Known reticle offsets (bregma-relative mm) and rotations (deg, +clockwise from
# above). Extend as needed; values taken from the calibration notebooks.
RETICLE_OFFSETS: dict[str, NDArray[np.floating[Any]]] = {
    "H": np.array([0.076, 0.062, 0.311]),
}
RETICLE_ROTATIONS: dict[str, float] = {
    "H": 0.0,
}


def load_measured_holes(xlsx_path: str | Path) -> pd.DataFrame:
    """Read a measured-holes spreadsheet.

    Expects columns ``Hole #``, ``Probe #`` (or ``Probe``), ``X``, ``Y``, ``Z``
    (manipulator coordinates in µm) and ``Calibration`` (a calibration
    reference, e.g. a parallax log directory name).

    Parameters
    ----------
    xlsx_path : str or Path
        Path to the measured-holes ``.xlsx`` file.

    Returns
    -------
    pandas.DataFrame
        One row per measured hole, with normalized column names ``Hole``,
        ``Probe``, ``X``, ``Y``, ``Z``, ``Calibration``.
    """
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={"Hole #": "Hole", "Probe #": "Probe"})
    required = {"Hole", "Probe", "X", "Y", "Z", "Calibration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measured-holes file is missing columns: {sorted(missing)}")
    df["Hole"] = df["Hole"].astype(int)
    df["Probe"] = df["Probe"].astype(int)
    return df


def load_hole_shaft_bottoms(holes_yml: str | Path) -> dict[int, NDArray[np.floating[Any]]]:
    """Load the shaft bottom-opening center of each hole from a holes YAML.

    The bottom opening of a shaft is the cross-section with the more negative
    ``s_mm`` (the lowest LPS S coordinate).

    Parameters
    ----------
    holes_yml : str or Path
        Path to an ``aind-low-point`` ``*.holes.yml`` file.

    Returns
    -------
    dict
        Maps hole id (int) to its bottom-opening ``center_LPS`` as a length-3
        numpy array in implant CAD LPS coordinates (mm).
    """
    with open(holes_yml) as f:
        spec = yaml.safe_load(f)
    bottoms: dict[int, NDArray[np.floating[Any]]] = {}
    for hole in spec["holes"]:
        sections = hole["sections"]
        bottom = min(sections, key=lambda s: s["s_mm"])
        center = np.asarray(bottom["center_LPS"], dtype=float)
        # Sanity check: the more-negative-s_mm section should also be the most
        # ventral (lowest LPS S). Warn rather than fail if the geometry is odd.
        other = max(sections, key=lambda s: s["s_mm"])
        if center[2] > np.asarray(other["center_LPS"], dtype=float)[2]:
            import warnings

            warnings.warn(
                f"Hole {hole['id']}: min-s_mm section is not the lowest LPS S; check the holes YAML.",
                stacklevel=2,
            )
        bottoms[int(hole["id"])] = center
    return bottoms


def predicted_hole_bottoms_bregma(
    holes_yml: str | Path,
    implant_fit_h5: str | Path,
    com_plane_h5: str | Path,
) -> dict[int, NDArray[np.floating[Any]]]:
    """Map implant-fit hole shaft bottoms into bregma-relative RAS coordinates.

    Composes (implant CAD LPS) -> (MRI image LPS) -> (bregma RAS), re-inverting
    the saved transforms (which are stored inverted for Slicer).

    Parameters
    ----------
    holes_yml : str or Path
        Path to the ``*.holes.yml`` file with hole definitions.
    implant_fit_h5 : str or Path
        Path to the saved ``*_implant_fit.h5`` transform.
    com_plane_h5 : str or Path
        Path to the saved ``*_com_plane.h5`` transform.

    Returns
    -------
    dict
        Maps hole id to its bottom-opening center in bregma RAS (ML, AP, DV) mm.
    """
    bottoms = load_hole_shaft_bottoms(holes_yml)
    # Re-invert: stored inverse -> forward (model -> image, image -> bregma).
    implant_to_image = load_sitk_transform(str(implant_fit_h5), homogeneous=True, invert=True)[0]
    image_to_bregma = load_sitk_transform(str(com_plane_h5), homogeneous=True, invert=True)[0]

    hole_ids = list(bottoms.keys())
    pts_cad = np.vstack([bottoms[h] for h in hole_ids])
    homog = rot.prepare_data_for_homogeneous_transform(pts_cad)
    bregma_lps = rot.extract_data_for_homogeneous_transform(homog @ implant_to_image.T @ image_to_bregma.T)
    # The com_plane transform operates in ITK/LPS space, so the result is
    # LPS-signed. Convert to bregma RAS (ML, AP, DV) to match the probe
    # measurements, which transform_probe_to_bregma already returns in RAS.
    # (Same [-1, -1, 1] flip used in plan_target_insertions_implantmodel.)
    bregma_ras = bregma_lps * np.array([-1.0, -1.0, 1.0])
    return {h: bregma_ras[i] for i, h in enumerate(hole_ids)}


def _resolve_calibration_ref(ref: str | Path, calibration_dir: str | Path | None) -> str:
    """Resolve a calibration reference, falling back to ``calibration_dir``.

    An absolute path or one that already exists is used as-is; otherwise the
    reference is looked up relative to ``calibration_dir`` (if given).
    """
    path = Path(ref)
    if path.is_absolute() or path.exists() or calibration_dir is None:
        return str(path)
    return str(Path(calibration_dir) / path)


def classify_calibration_refs(
    refs: Sequence[str | Path],
) -> tuple[list[str], list[str]]:
    """Split calibration references into manual files and parallax directories.

    A manual calibration is a single ``.xlsx``/``.xls`` file; a parallax
    calibration is a directory of ``points_SN*.csv`` files.

    Parameters
    ----------
    refs : list of str or Path
        Calibration references (already resolved to existing paths).

    Returns
    -------
    tuple of (list of str, list of str)
        ``(manual_files, parallax_dirs)``, preserving input order.
    """
    manual_files: list[str] = []
    parallax_dirs: list[str] = []
    for ref in refs:
        path = Path(ref)
        if path.is_dir():
            parallax_dirs.append(str(path))
        elif path.suffix.lower() in _MANUAL_SUFFIXES:
            manual_files.append(str(path))
        else:
            raise ValueError(f"Unrecognized calibration reference: {ref}")
    return manual_files, parallax_dirs


def calibration_by_probe(
    manual_files: Sequence[str | Path] | None = None,
    parallax_dirs: Sequence[str | Path] | None = None,
    probes_to_ignore_manual: list[int] | None = None,
    reticle: str = "H",
    reticle_offset: NDArray[np.floating[Any]] | None = None,
    reticle_rotation: float | None = None,
    **fit_kwargs: Any,
) -> dict[int, tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]:
    """Fit probe->bregma calibration from manual and/or parallax calibrations.

    Mirrors the calibration notebook: manual and parallax calibrations may be
    combined, with manual taking priority (unless a probe is listed in
    ``probes_to_ignore_manual``). When any manual calibration is supplied the
    reticle metadata (offset/rotation) is read from it and applied to the
    parallax calibrations; for parallax-only calibration the reticle offset and
    rotation must be supplied (looked up from ``reticle`` by default).

    Parameters
    ----------
    manual_files : list of str or Path, optional
        Manual reticle calibration ``.xlsx`` files; later files take priority.
    parallax_dirs : list of str or Path, optional
        Parallax calibration directories; later dirs take priority.
    probes_to_ignore_manual : list of int, optional
        Probes whose manual calibration should be ignored (parallax used).
    reticle : str, optional
        Reticle name for default offset/rotation (parallax-only). Default "H".
    reticle_offset : numpy.ndarray, optional
        Override the reticle offset (bregma-relative mm), parallax-only.
    reticle_rotation : float, optional
        Override the reticle rotation (degrees), parallax-only.
    **fit_kwargs
        Additional keyword arguments forwarded to the rotation fit.

    Returns
    -------
    dict
        Maps probe id (int) to ``(R, t)`` calibration parameters, where ``R``
        and ``t`` map bregma points to probe points (both in mm). Pass these to
        :func:`aind_mri_utils.reticle_calibrations.transform_probe_to_bregma`.
    """
    manual_files = [str(f) for f in (manual_files or [])]
    parallax_dirs = [str(d) for d in (parallax_dirs or [])]
    if not manual_files and not parallax_dirs:
        raise ValueError("No calibration references provided")

    if manual_files:
        cal_by_probe, _, _ = combine_parallax_and_manual_calibrations(
            manual_files,
            parallax_dirs,
            probes_to_ignore_manual,
            **fit_kwargs,
        )
        return cal_by_probe

    # Parallax only: reticle metadata is not embedded, so it must be supplied.
    if reticle_offset is None:
        reticle_offset = RETICLE_OFFSETS[reticle]
    if reticle_rotation is None:
        reticle_rotation = RETICLE_ROTATIONS[reticle]
    parallax_cal: dict[int, tuple[NDArray, NDArray]] = {}
    for parallax_dir in parallax_dirs:
        this_cal, _ = fit_rotation_params_from_parallax(parallax_dir, reticle_offset, reticle_rotation, **fit_kwargs)
        parallax_cal.update(this_cal)
    return parallax_cal


def read_parallax_transM(
    parallax_dir: str | Path,
) -> dict[int, tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]:
    """Read Parallax's own per-probe transform matrices from a calibration dir.

    Parses the ``transM_SN<sn>_<ts>.csv`` files written by Parallax. Each maps
    global (reticle) coordinates to local (manipulator) coordinates in microns:
    ``local = R @ global + T``.

    Parameters
    ----------
    parallax_dir : str or Path
        Parallax calibration directory.

    Returns
    -------
    dict
        Maps probe id (int) to ``(R, T)``: a 3x3 rotation matrix and a length-3
        translation vector (microns).
    """
    out: dict[int, tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]] = {}
    for file in Path(parallax_dir).iterdir():
        match = _TRANSM_RE.search(file.name)
        if not match:
            continue
        row = pd.read_csv(file).iloc[0]
        R = np.array([row[f"R_{i}_{j}"] for i in range(3) for j in range(3)], dtype=float).reshape(3, 3)
        T = np.array([row[f"T_{i}"] for i in range(3)], dtype=float)
        out[int(match.group(1))] = (R, T)
    return out


def check_parallax_calibration(
    parallax_dir: str | Path,
    reticle: str = "H",
    reticle_offset: NDArray[np.floating[Any]] | None = None,
    reticle_rotation: float | None = None,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Verify our fitted parallax calibration against Parallax's ``transM`` files.

    For each probe, our calibration (``probe_mm = R_ours @ bregma_mm + t_ours``)
    is compared against Parallax's own transform (``local_um = R_px @ global_um
    + T_px``). Accounting for the reticle correction
    (``bregma_mm = (R_reticle @ global_um) / 1000 + reticle_offset``) and unit
    scaling, the two should satisfy ``R_ours = R_px @ R_reticle.T`` and
    ``t_ours = T_px / 1000 - R_ours @ reticle_offset``.

    Two checks are reported: the algebraic difference between our parameters and
    those implied by ``transM``, and an end-to-end agreement obtained by pushing
    the calibration global points through both transforms and comparing the
    predicted manipulator coordinates (convention-independent).

    Parameters
    ----------
    parallax_dir : str or Path
        Parallax calibration directory (containing ``points_SN*.csv`` and
        ``transM_SN*.csv``).
    reticle : str, optional
        Reticle name for default offset/rotation. Default "H".
    reticle_offset : numpy.ndarray, optional
        Override the reticle offset (bregma-relative mm).
    reticle_rotation : float, optional
        Override the reticle rotation (degrees).
    **fit_kwargs
        Additional keyword arguments forwarded to the calibration fit.

    Returns
    -------
    pandas.DataFrame
        Indexed by probe id, with columns: ``n_points``,
        ``rot_max_abs_diff`` (max abs element of ``R_ours - R_px @ R_reticle.T``),
        ``trans_max_abs_diff_mm`` (max abs of ``t_ours`` vs the transM-implied
        value), and ``pred_rms_um`` / ``pred_max_um`` (manipulator-coordinate
        agreement between the two transforms over the global points).
    """
    if reticle_offset is None:
        reticle_offset = RETICLE_OFFSETS[reticle]
    if reticle_rotation is None:
        reticle_rotation = RETICLE_ROTATIONS[reticle]
    R_reticle = reticle_metadata_transform(reticle_rotation)

    ours = calibration_by_probe(
        parallax_dirs=[parallax_dir],
        reticle=reticle,
        reticle_offset=reticle_offset,
        reticle_rotation=reticle_rotation,
        **fit_kwargs,
    )
    transm = read_parallax_transM(parallax_dir)
    points = read_parallax_calibration_dir(str(parallax_dir))

    rows = []
    for probe in sorted(set(ours) & set(transm)):
        R_ours, t_ours = ours[probe]
        R_px, T_px = transm[probe]

        R_ours_pred = R_px @ R_reticle.T
        t_ours_pred = T_px / 1000.0 - R_ours_pred @ reticle_offset
        rot_diff = float(np.abs(R_ours - R_ours_pred).max())
        trans_diff = float(np.abs(t_ours - t_ours_pred).max())

        global_um = points[probe][0]
        # Predicted manipulator coords (µm) via Parallax: local = R_px @ g + T.
        local_px = global_um @ R_px.T + T_px
        # Predicted via our calibration: global_um -> bregma_mm -> probe_mm -> µm.
        bregma_mm = (global_um @ R_reticle.T) / 1000.0 + reticle_offset
        local_ours = transform_bregma_to_probe(bregma_mm, R_ours, t_ours) * 1000.0
        resid = np.linalg.norm(local_px - local_ours, axis=1)

        rows.append(
            {
                "Probe": probe,
                "n_points": int(global_um.shape[0]),
                "rot_max_abs_diff": rot_diff,
                "trans_max_abs_diff_mm": trans_diff,
                "pred_rms_um": float(np.sqrt(np.mean(resid**2))),
                "pred_max_um": float(resid.max()),
            }
        )
    return pd.DataFrame(rows).set_index("Probe")


def measured_holes_bregma(
    measured: pd.DataFrame,
    calibration_dir: str | Path | None = None,
    manual_files: Sequence[str | Path] | None = None,
    parallax_dirs: Sequence[str | Path] | None = None,
    include_calibration_column: bool = True,
    probes_to_ignore_manual: list[int] | None = None,
    reticle: str = "H",
    reticle_offset: NDArray[np.floating[Any]] | None = None,
    reticle_rotation: float | None = None,
    **fit_kwargs: Any,
) -> dict[int, NDArray[np.floating[Any]]]:
    """Transform measured probe hole locations into bregma-relative RAS.

    Calibrations may be *layered* to cover the full set of probes: multiple
    manual ``.xlsx`` files and parallax directories are combined into a single
    probe->bregma calibration. Priority increases along each list, manual beats
    parallax, and the explicit ``manual_files`` / ``parallax_dirs`` lists take
    priority over any calibrations named in the spreadsheet ``Calibration``
    column. Each reference is resolved relative to ``calibration_dir`` when it
    is a bare name rather than an existing/absolute path.

    Parameters
    ----------
    measured : pandas.DataFrame
        Output of :func:`load_measured_holes`.
    calibration_dir : str or Path, optional
        Directory bare calibration references are resolved against.
    manual_files : list of str or Path, optional
        Manual calibration ``.xlsx`` files to layer in (highest priority, in
        order).
    parallax_dirs : list of str or Path, optional
        Parallax calibration directories to layer in.
    include_calibration_column : bool, optional
        If True (default), the calibrations named in the ``Calibration`` column
        are included as a base layer (below the explicit lists). Set False to
        use only ``manual_files`` / ``parallax_dirs``.
    probes_to_ignore_manual : list of int, optional
        Probes whose manual calibration should be ignored.
    reticle : str, optional
        Reticle name for default offset/rotation (parallax-only). Default "H".
    reticle_offset : numpy.ndarray, optional
        Override the reticle offset (bregma-relative mm).
    reticle_rotation : float, optional
        Override the reticle rotation (degrees).
    **fit_kwargs
        Additional keyword arguments forwarded to the calibration fit.

    Returns
    -------
    dict
        Maps hole id to measured location in bregma RAS (ML, AP, DV) mm.
    """
    col_manual: list[str] = []
    col_parallax: list[str] = []
    if include_calibration_column and "Calibration" in measured.columns:
        col_refs = [
            _resolve_calibration_ref(ref, calibration_dir) for ref in dict.fromkeys(measured["Calibration"].astype(str))
        ]
        col_manual, col_parallax = classify_calibration_refs(col_refs)  # type: ignore[arg-type]

    explicit_manual, explicit_parallax = classify_calibration_refs(
        [_resolve_calibration_ref(r, calibration_dir) for r in (manual_files or [])]
        + [_resolve_calibration_ref(r, calibration_dir) for r in (parallax_dirs or [])]
    )

    cal_by_probe = calibration_by_probe(
        manual_files=col_manual + explicit_manual,
        parallax_dirs=col_parallax + explicit_parallax,
        probes_to_ignore_manual=probes_to_ignore_manual,
        reticle=reticle,
        reticle_offset=reticle_offset,
        reticle_rotation=reticle_rotation,
        **fit_kwargs,
    )

    out: dict[int, NDArray[np.floating[Any]]] = {}
    for _, row in measured.iterrows():
        probe = int(row["Probe"])
        if probe not in cal_by_probe:
            raise KeyError(f"Probe {probe} not found in the combined calibration")
        R, t = cal_by_probe[probe]
        probe_pt_mm = np.array([row["X"], row["Y"], row["Z"]], dtype=float) / 1000.0
        out[int(row["Hole"])] = transform_probe_to_bregma(probe_pt_mm, R, t)
    return out


def compare_holes(
    measured_bregma: dict[int, NDArray[np.floating[Any]]],
    predicted_bregma: dict[int, NDArray[np.floating[Any]]],
) -> pd.DataFrame:
    """Build a per-hole comparison table of measured vs predicted locations.

    Parameters
    ----------
    measured_bregma : dict
        Hole id -> measured bregma RAS (ML, AP, DV) mm.
    predicted_bregma : dict
        Hole id -> predicted bregma RAS (ML, AP, DV) mm.

    Returns
    -------
    pandas.DataFrame
        Indexed by hole id, with measured/predicted ML/AP/DV columns, signed
        error components (measured - predicted), and the 3D error magnitude
        ``error_mm``. Only holes present in both inputs are included.
    """
    dims = ["ML", "AP", "DV"]
    rows = []
    for hole in sorted(set(measured_bregma) & set(predicted_bregma)):
        meas = np.asarray(measured_bregma[hole], dtype=float)
        pred = np.asarray(predicted_bregma[hole], dtype=float)
        err = meas - pred
        row: dict[str, Any] = {"Hole": hole}
        for i, d in enumerate(dims):
            row[f"meas_{d}"] = meas[i]
            row[f"pred_{d}"] = pred[i]
            row[f"err_{d}"] = err[i]
        row["error_mm"] = float(np.linalg.norm(err))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Hole")


def plot_hole_errors(
    comparison: pd.DataFrame,
    arrow_scale: float = 5.0,
    title: str | None = None,
):
    """Plot measured-vs-predicted hole errors.

    Produces three panels: an ML-AP map of the predicted hole positions with
    (magnified) error arrows pointing to the measured positions, a grouped bar
    chart of the signed ML/AP/DV error components, and a bar chart of the 3D
    error magnitude per hole.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`compare_holes` (indexed by hole id).
    arrow_scale : float, optional
        Factor by which in-plane (ML/AP) error arrows are magnified on the map
        panel so sub-mm errors are visible. Default 5.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    df = comparison.sort_index()
    holes = df.index.to_numpy()
    labels = [str(h) for h in holes]
    x = np.arange(len(holes))

    fig, (ax_map, ax_comp, ax_mag) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: ML-AP map with magnified error arrows (predicted -> measured).
    sc = ax_map.scatter(df["pred_ML"], df["pred_AP"], c=df["error_mm"], cmap="viridis", s=60, zorder=3)
    ax_map.quiver(
        df["pred_ML"],
        df["pred_AP"],
        df["err_ML"] * arrow_scale,
        df["err_AP"] * arrow_scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="crimson",
        width=0.004,
        zorder=2,
    )
    for h, mlv, apv in zip(labels, df["pred_ML"], df["pred_AP"]):
        ax_map.annotate(h, (mlv, apv), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax_map.set_xlabel("ML (mm)")
    ax_map.set_ylabel("AP (mm)")
    ax_map.set_title(f"Hole map (error arrows ×{arrow_scale:g})")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax_map, label="3D error (mm)")

    # Panel 2: signed error components.
    width = 0.27
    for offset, dim, color in zip((-width, 0, width), ("ML", "AP", "DV"), ("tab:blue", "tab:orange", "tab:green")):
        ax_comp.bar(x + offset, df[f"err_{dim}"], width, label=dim, color=color)
    ax_comp.axhline(0, color="k", lw=0.8)
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(labels)
    ax_comp.set_xlabel("Hole")
    ax_comp.set_ylabel("Signed error, measured − predicted (mm)")
    ax_comp.set_title("Error components")
    ax_comp.legend()
    ax_comp.grid(True, axis="y", alpha=0.3)

    # Panel 3: 3D error magnitude with mean line.
    ax_mag.bar(x, df["error_mm"], color="slategray")
    mean_err = df["error_mm"].mean()
    ax_mag.axhline(mean_err, color="crimson", ls="--", label=f"mean {mean_err:.2f} mm")
    ax_mag.set_xticks(x)
    ax_mag.set_xticklabels(labels)
    ax_mag.set_xlabel("Hole")
    ax_mag.set_ylabel("3D error (mm)")
    ax_mag.set_title("Error magnitude")
    ax_mag.legend()
    ax_mag.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_hole_errors_3d(
    comparison: pd.DataFrame,
    arrow_scale: float = 1.0,
    title: str | None = None,
    ax: Any = None,
):
    """Plot measured-vs-predicted hole errors in interactive 3D.

    Draws predicted and measured hole positions in bregma RAS space, joined by
    error segments colored by 3D error magnitude. Intended for use with an
    interactive backend (e.g. ``%matplotlib widget`` via ipympl) so the view
    can be rotated.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`compare_holes` (indexed by hole id). Must contain the
        ``meas_*``, ``pred_*`` and ``error_mm`` columns.
    arrow_scale : float, optional
        Factor by which the predicted->measured error segment is magnified so
        sub-mm errors are visible. Default 1 (true scale).
    title : str, optional
        Axes title.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Existing 3D axes to draw into. If None, a new figure and 3D axes are
        created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 3D axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    df = comparison.sort_index()
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    pred = df[["pred_ML", "pred_AP", "pred_DV"]].to_numpy()
    meas = df[["meas_ML", "meas_AP", "meas_DV"]].to_numpy()
    # Optionally magnify the displayed measured point about the prediction.
    meas_disp = pred + (meas - pred) * arrow_scale
    err = df["error_mm"].to_numpy()

    norm = colors.Normalize(vmin=float(err.min()), vmax=float(err.max()))
    cmap = plt.get_cmap("viridis")

    for i, hole in enumerate(df.index):
        color = cmap(norm(err[i]))
        ax.plot(
            [pred[i, 0], meas_disp[i, 0]],
            [pred[i, 1], meas_disp[i, 1]],
            [pred[i, 2], meas_disp[i, 2]],
            color=color,
            lw=2,
        )
        ax.text(meas_disp[i, 0], meas_disp[i, 1], meas_disp[i, 2], f" {hole}", fontsize=8)

    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker="o", s=45, c="k", label="predicted (implant fit)")
    meas_sc = ax.scatter(
        meas_disp[:, 0],
        meas_disp[:, 1],
        meas_disp[:, 2],
        marker="^",
        s=55,
        c=err,
        cmap="viridis",
        norm=norm,
        label="measured (probe)",
    )

    ax.set_xlabel("ML (mm)")
    ax.set_ylabel("AP (mm)")
    ax.set_zlabel("DV (mm)")

    # Equal unit spacing on all axes so the 3D geometry is undistorted: center
    # each axis on the data and give them a common span and a cubic box aspect.
    all_pts = np.vstack([pred, meas_disp])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    half_span = float((maxs - mins).max()) / 2 or 1.0
    ax.set_xlim(centers[0] - half_span, centers[0] + half_span)
    ax.set_ylim(centers[1] - half_span, centers[1] + half_span)
    ax.set_zlim(centers[2] - half_span, centers[2] + half_span)
    ax.set_box_aspect((1, 1, 1))

    ax.legend(loc="upper left")
    fig.colorbar(meas_sc, ax=ax, label="3D error (mm)", shrink=0.6, pad=0.1)
    if arrow_scale != 1.0:
        ax.set_title((title + " " if title else "") + f"(error ×{arrow_scale:g})")
    elif title:
        ax.set_title(title)
    return fig


def arc_angles_from_calibration(
    manual_files: Sequence[str | Path] | None = None,
    parallax_dirs: Sequence[str | Path] | None = None,
    probes_to_ignore_manual: list[int] | None = None,
    reticle: str = "H",
    reticle_offset: NDArray[np.floating[Any]] | None = None,
    reticle_rotation: float | None = None,
    rig_frame: bool = True,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Infer per-probe arc angles (AP/ML) from a calibration stack-up.

    The probe insertion direction is the manipulator z-down axis expressed in
    bregma coordinates; its arc angles are computed with
    :func:`aind_mri_utils.arc_angles.vector_to_arc_angles`. Manual and parallax
    calibrations may be combined (see :func:`calibration_by_probe`).

    Parameters
    ----------
    manual_files : list of str or Path, optional
        Manual reticle calibration ``.xlsx`` files; later files take priority.
    parallax_dirs : list of str or Path, optional
        Parallax calibration directories; later dirs take priority.
    probes_to_ignore_manual : list of int, optional
        Probes whose manual calibration should be ignored.
    reticle : str, optional
        Reticle name for default offset/rotation (parallax-only). Default "H".
    reticle_offset : numpy.ndarray, optional
        Override the reticle offset (bregma-relative mm).
    reticle_rotation : float, optional
        Override the reticle rotation (degrees).
    rig_frame : bool, optional
        If True (default), report the AP angle in the ephys-rig/arc frame by
        adding the ``HEADFRAME_RX_IN_ARC`` (14°) headframe pitch. If False,
        report the raw headframe-plan AP angle.
    **fit_kwargs
        Additional keyword arguments forwarded to the calibration fit.

    Returns
    -------
    pandas.DataFrame
        Indexed by probe id, with columns ``AP_angle`` and ``ML_angle`` (deg).
        A ``frame`` column records whether AP is in the rig/arc or plan frame.
    """
    cal_by_probe = calibration_by_probe(
        manual_files=manual_files,
        parallax_dirs=parallax_dirs,
        probes_to_ignore_manual=probes_to_ignore_manual,
        reticle=reticle,
        reticle_offset=reticle_offset,
        reticle_rotation=reticle_rotation,
        **fit_kwargs,
    )
    rows = []
    for probe, (R, _) in sorted(cal_by_probe.items()):
        insertion_vec = find_probe_insertion_vector(R)
        angles = vector_to_arc_angles(insertion_vec)
        if angles is None:
            continue
        ap, ml = angles
        if rig_frame:
            ap = ap + HEADFRAME_RX_IN_ARC
        rows.append(
            {
                "Probe": int(probe),
                "AP_angle": ap,
                "ML_angle": ml,
                "frame": "rig/arc" if rig_frame else "plan",
            }
        )
    return pd.DataFrame(rows).set_index("Probe")
