# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# Imports
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import IntFlag, auto
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)
from warnings import warn

import fcl
import ipywidgets as widgets
import k3d
import matplotlib
import numpy as np
import SimpleITK as sitk
import trimesh
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_mri_utils.arc_angles import arc_angles_to_affine
from aind_mri_utils.chemical_shift import (
    chemical_shift_transform,
    compute_chemical_shift,
)
from aind_mri_utils.file_io.simpleitk import load_sitk_transform
from aind_mri_utils.implant import make_hole_seg_dict
from aind_mri_utils.meshes import (
    load_newscale_trimesh,
    mask_to_trimesh,
)
from aind_mri_utils.plots import hex_string_to_int, rgb_to_hex_string
from aind_mri_utils.reticle_calibrations import (
    combine_parallax_and_manual_calibrations,
    find_probe_angle,
    fit_rotation_params_from_parallax,
)
from aind_mri_utils.rotations import (
    apply_rotate_translate,
    compose_transforms,
    invert_rotate_translate,
)
from ipyevents import Event
from IPython.display import display
from numpy.typing import NDArray
from omegaconf import OmegaConf
from pydantic import BaseModel, DirectoryPath, Field, FilePath, computed_field, field_validator, model_validator

Pair = Tuple[str, str]
# %%
# Uncomment the following line to enable interactive plotting in Jupyter
# notebooks
# %matplotlib ipympl

# %%
# Set the log verbosity to get debug statements
logging.basicConfig(format="%(message)s", level=logging.DEBUG)

# %%
# Location of configuration files, if present
config_files = {
    "app_config": Path("/home/galen.lynch/786864-planning-config.yml"),
    "probe_config": Path("/home/galen.lynch/786864-probe-config.yml"),
}


# %%
# Classes


## Configuration classes
class CalibrationInfo(BaseModel):
    """
    Information about the calibration of each probe
    """

    probe_for_target: Dict[str, str]  # structure → probe ID
    calibration_path: Path
    calibration_files: List[Path]
    parallax_calibration_dirs: List[Path] = Field(default_factory=list)


class AppConfig(BaseModel):
    """
    Global configuration and location of required data assets
    """

    mouse: str
    target_structures: List[str]
    reticle_offset: List[float]
    reticle_rotation: float

    base_path: DirectoryPath

    annotations_path: DirectoryPath
    image_path: FilePath
    headframe_annotations_path: FilePath
    brain_mask_path: FilePath
    structure_mask_path: DirectoryPath

    headframe_transform_file: FilePath

    implant_annotation_path: DirectoryPath
    implant_annotation_file: FilePath
    implant_fit_transform_file: FilePath

    model_path: DirectoryPath
    hole_model_path: DirectoryPath
    probe_model_files: Dict[str, FilePath]

    headframe_file: FilePath
    cone_file: FilePath
    well_file: FilePath
    implant_file: FilePath

    calibration_info: Optional[CalibrationInfo] = None

    plan_save_path: Path

    @field_validator("probe_model_files", mode="before")
    def ensure_string_keys(cls, v):
        return {str(k): v[k] for k in v}

    @computed_field(return_type=Dict[str, FilePath])
    @property
    def structure_files(self) -> Dict[str, FilePath]:
        return {
            struct: self.structure_mask_path / f"{self.mouse}-{struct}-Mask.nrrd" for struct in self.target_structures
        }

    @computed_field(return_type=Dict[int, FilePath])
    @property
    def hole_model_files(self) -> Dict[int, FilePath]:
        hole_model_path = self.hole_model_path
        all_hole_files = hole_model_path.glob("Hole*.obj")
        hole_dict = {}
        for f in all_hole_files:
            hole_num = int(f.stem.split("Hole")[-1])
            hole_dict[hole_num] = f
        lower_face_file = hole_model_path / "LowerFace.obj"
        hole_dict[-1] = lower_face_file
        return hole_dict


class ProbeInfo(BaseModel):
    """
    Information about the probe type and its configuration for a specific structure.
    """

    type: str = Field(default="2.1", description="Type of the probe")
    arc: str = Field(default="a", description="Arc identifier")
    slider_ml: float = Field(default=0.0, description="Slider ML position")
    spin: float = Field(default=0.0, description="Spin angle")


class TargetInfo(BaseModel):
    """
    Information about the target structure, including offsets, depth, and hole number.
    """

    offsets_RA: List[float] = Field(
        [0.0, 0.0],
        min_items=2,
        max_items=2,
        description="RA Offset in the RAS coordinate system of the mouse, not manipulator",
    )
    depth: float = Field(default=0.0, description="Depth of the target structure")
    target_name: Optional[str] = Field(default=None, description="Target structure or landmark name, e.g. PL or hole-1")
    target_coordinates_RAS: Optional[List[float]] = Field(
        default=None, description=("RAS coordinates of the target point, should be None if target_name is used")
    )

    # Ensure that either target_name or target_coordinates_RAS is set
    @model_validator(mode="after")
    def check_target_info(self):
        target_name = self.target_name
        target_coordinates_RAS = self.target_coordinates_RAS
        if not target_name and not target_coordinates_RAS:
            raise ValueError("Either target_name or target_coordinates_RAS must be set.")
        if target_name and target_coordinates_RAS:
            raise ValueError("Only one of target_name or target_coordinates_RAS must be set.")
        return self


class ProbeConfig(BaseModel):
    """
    Configuration for the probes used in the experiment, including mappings to
    target structures, probe information, and target information.
    """

    arcs: Dict[str, float]
    probe_info: Dict[str, ProbeInfo]
    target_info_by_probe: Dict[str, TargetInfo]

    @model_validator(mode="after")
    def check_arc_keys(self) -> "ProbeConfig":
        arc_keys = set(self.arcs.keys())
        for struct, probe in self.probe_info.items():
            if probe.arc not in arc_keys:
                raise ValueError(
                    f"arc '{probe.arc}' in probe_info[{struct}] is not defined in arcs: {sorted(arc_keys)}"
                )
        return self


## Run-time classes
Float3x3 = NDArray[np.float64]  # shape (3, 3)
Float3 = NDArray[np.float64]  # shape (3,)
FloatNx3 = NDArray[np.float64]  # shape (3, N)
RawT_co = TypeVar("RawT_co", covariant=True)


@runtime_checkable
class SupportsRigidTransform(Protocol[RawT_co]):
    @property
    def raw(self) -> RawT_co: ...
    def transformed(self, R: NDArray[np.float64], t: NDArray[np.float64]) -> RawT_co: ...


W = TypeVar("W", bound=SupportsRigidTransform[RawT_co])


@dataclass(frozen=True)
class AffineTransform:
    rotation: Float3x3 = field(default_factory=lambda: np.eye(3), repr=False)
    translation: Float3 = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]), repr=False)
    inverted: bool = False

    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls()

    @classmethod
    def from_sitk_path(cls, path: Path, inverted=False) -> "AffineTransform":
        R, t, _ = load_sitk_transform(str(path))
        return cls(rotation=R, translation=t, inverted=inverted)

    @cached_property
    def rotate_translate(self) -> Tuple[Float3x3, Float3]:
        if self.inverted:
            R, t = invert_rotate_translate(self.rotation, self.translation)
        else:
            R, t = self.rotation, self.translation
        return R, t

    def apply_to(self, pts: FloatNx3) -> FloatNx3:
        """Apply the transform to a set of points."""
        R, t = self.rotate_translate
        return apply_rotate_translate(pts, R, t)

    def invert(self) -> "AffineTransform":
        """Invert the transform."""
        return AffineTransform(
            rotation=np.copy(self.rotation),
            translation=np.copy(self.translation),
            inverted=not self.inverted,
        )


@dataclass(frozen=True)
class TransformChain:
    elements: Tuple[AffineTransform, ...]

    def __post_init__(self):
        # Allow users to pass a list; store as an immutable tuple.
        if not isinstance(self.elements, tuple):
            object.__setattr__(self, "elements", tuple(self.elements))

    @cached_property
    def composed_transform(self) -> Tuple[NDArray, NDArray]:
        """Get the combined rotation and translation from all transforms in the chain."""
        pairs = []
        for e in self.elements:
            R, t = e.rotate_translate
            pairs.append(R)
            pairs.append(t)
        return compose_transforms(*pairs)

    def apply_to(self, pts: NDArray) -> NDArray:
        """Apply the transform chain to a set of points."""
        R, t = self.composed_transform
        return apply_rotate_translate(pts, R, t)

    def invert(self) -> "TransformChain":
        """Invert the transform chain."""
        return TransformChain(tuple(e.invert() for e in reversed(self.elements)))

    @classmethod
    def new(cls, items: Iterable[AffineTransform]) -> "TransformChain":
        # Convenience constructor to make intent explicit.
        return cls(tuple(items))


@dataclass(frozen=True, slots=True)
class MeshTransformable(SupportsRigidTransform[trimesh.Trimesh]):
    _raw: trimesh.Trimesh

    @property
    def raw(self) -> trimesh.Trimesh:
        return self._raw

    def transformed(self, R: Float3x3, t: Float3) -> trimesh.Trimesh:
        v = apply_rotate_translate(self._raw.vertices, R, t)
        return trimesh.Trimesh(vertices=v, faces=self._raw.faces, process=False)


@dataclass(frozen=True, slots=True)
class PointsTransformable(SupportsRigidTransform[FloatNx3]):
    _raw: FloatNx3  # (N,3) float

    def __post_init__(self) -> None:
        a = self._raw
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError("PointsWrap expects shape (N,3)")
        if a.dtype.kind != "f":
            raise TypeError("PointsWrap expects float dtype")

    @property
    def raw(self) -> FloatNx3:
        return self._raw

    def transformed(self, R: Float3x3, t: Float3) -> FloatNx3:
        return apply_rotate_translate(self._raw, R, t)


@overload
def as_transformable(x: trimesh.Trimesh) -> MeshTransformable: ...
@overload
def as_transformable(x: FloatNx3) -> PointsTransformable: ...
def as_transformable(x):
    if isinstance(x, MeshTransformable) or isinstance(x, PointsTransformable):
        return x
    if isinstance(x, trimesh.Trimesh):
        return MeshTransformable(x)
    if isinstance(x, np.ndarray):
        return PointsTransformable(x)
    raise TypeError(f"Unsupported type {type(x)}")


@dataclass(frozen=True)
class Transformed(Generic[W, RawT_co]):
    original: W
    chain: TransformChain

    def __post_init__(self) -> None:
        # Runtime structural check (optional but nice)
        if not isinstance(self.original, SupportsRigidTransform):
            raise TypeError(f"`original` must implement SupportsRigidTransform; got {type(self.original).__name__}")

    @cached_property
    def raw(self) -> RawT_co:
        R, t = self.chain.composed_transform
        return self.original.transformed(R, t)

    # If you often need the untransformed payload too:
    @property
    def original_raw(self) -> RawT_co:
        return self.original.raw


TransformedMesh: TypeAlias = Transformed[MeshTransformable, trimesh.Trimesh]
TransformedPoints: TypeAlias = Transformed[PointsTransformable, FloatNx3]


def _trimesh_from_sitk_mask(mask: sitk.Image) -> trimesh.Trimesh:
    """Convert a SimpleITK mask image to a trimesh."""
    structure_mesh = mask_to_trimesh(mask)
    trimesh.repair.fix_normals(structure_mesh)
    trimesh.repair.fix_inversion(structure_mesh)
    return structure_mesh


def _load_trimesh_lps(path: Path, src_coordinate_system: str = "ASR") -> trimesh.Trimesh:
    """Load a trimesh from a SimpleITK image file."""
    mesh = trimesh.load(str(path))
    vertices_lps = convert_coordinate_system(mesh.vertices, src_coordinate_system, "LPS")
    mesh.vertices = vertices_lps
    return mesh


def _get_colormap_colors(N, colormap_name="viridis"):
    """
    Generate N colors evenly spaced across the given colormap.

    Parameters:
        N (int): Number of colors needed.
        colormap (str): Name of the matplotlib colormap to use.

    Returns:
        list: A list of N RGB tuples.
    """
    cmap = matplotlib.colormaps[colormap_name].resampled(N)
    return [cmap(i) for i in range(N)]


def _chem_shift_t_from_path(brain_image_path):
    brain_image = sitk.ReadImage(str(brain_image_path))
    chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(compute_chemical_shift(brain_image, ppm=3.7))
    return chem_shift_pt_R, chem_shift_pt_t


def _process_implant_segmentation(implant_seg_vol_path):
    implant_seg_vol = sitk.ReadImage(str(implant_seg_vol_path))
    implant_targets_by_hole = make_hole_seg_dict(implant_seg_vol, fun=lambda x: np.mean(x, axis=0))
    implant_names = list(implant_targets_by_hole.keys())
    implant_targets = np.vstack(list(implant_targets_by_hole.values()))
    return implant_names, implant_targets

GeometryOut = Union[
    trimesh.Trimesh,             # surface mesh
    NDArray[np.float64],         # (N,3) points
]

# ---- Registry core --------------------------------------------------------
_LOADER_REGISTRY: Dict[str, Callable[..., GeometryOut]] = {}

def register_loader(name: Optional[str] = None):
    """Decorator to register a loader function by name."""
    def _wrap(fn: Callable[..., GeometryOut]):
        key = name or fn.__name__
        if key in _LOADER_REGISTRY:
            raise KeyError(f"Loader '{key}' already registered")
        _LOADER_REGISTRY[key] = fn
        return fn
    return _wrap

def load_geometry(src: Union[str, Path], loader: str, **kwargs) -> GeometryOut:
    """Dispatch to a named loader. kwargs are passed to the loader."""
    fn = _LOADER_REGISTRY.get(loader)
    if fn is None:
        raise KeyError(
            f"Unknown loader '{loader}'. Known: {', '.join(sorted(_LOADER_REGISTRY)) or '(none)'}"
        )
    return fn(Path(src), **kwargs)

SourceGeo = Union[trimesh.Trimesh, NDArray[np.float64]]
ReduceOut  = NDArray[np.float64]  # usually (3,) single point; could be (N,3)

_REDUCER_REGISTRY: Dict[str, Callable[..., ReduceOut]] = {}

def register_reducer(name: Optional[str] = None):
    def _wrap(fn: Callable[..., ReduceOut]):
        key = name or fn.__name__
        if key in _REDUCER_REGISTRY:
            raise KeyError(f"Reducer '{name}' already registered")
        _REDUCER_REGISTRY[key] = fn
        return fn
    return _wrap

def reduce_target(source: SourceGeo, reducer: str, **kwargs) -> ReduceOut:
    fn = _REDUCER_REGISTRY.get(reducer)
    if fn is None:
        raise KeyError(
            f"Unknown reducer '{reducer}'. Known: {', '.join(sorted(_REDUCER_REGISTRY)) or '(none)'}"
        )
    return fn(source, **kwargs)

class Capability(IntFlag):
    RENDERABLE = auto()
    MOVABLE = auto()
    COLLIDABLE = auto()
    SELECTABLE = auto()
    DEFORMABLE = auto()  # future-proofing (skinning, morphs)
    SAVABLE = auto()  # include in plan exports

class Role(str, Enum):
    GEOMETRY = "geometry"  # meshes/lines/points used as scene geometry
    TARGET   = "target"    # logical target(s), typically non-collidable landmarks
    LANDMARK = "landmark"  # fiducials, reference points, etc.

@dataclass(frozen=True)
class CanonicalizationInfo:
    source_space: Literal["ASR","RAS","LPS","FILE_NATIVE"]
    unit_scale: float                   # e.g. 0.001 if µm → mm
    transform_file_to_canonical: TransformChain
    applied: bool                       # baked into vertices?
    version: str = "canon-v1"           # bump if your rules change
    fingerprint: str = ""               # hash(source_path, mtime, fields above)

Float3 = np.ndarray  # shape (3,)
FloatAABB = np.ndarray  # shape (2,3)

@dataclass(frozen=True)
class BaseSpec:
    # WHAT it is
    key: str                                    # unique id, e.g. "probe:2.1", "structure:PL", "target:hole:1"
    kind: Literal["mesh", "points", "lines"]
    role: Role = Role.GEOMETRY
    default_material: "Material" = field(default_factory=lambda: Material("default"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set) # free-form (scene/UI grouping)
    canonicalization: CanonicalizationInfo = field(
        default_factory=lambda: CanonicalizationInfo(
            source_space="FILE_NATIVE", unit_scale=1.0,
            transform_file_to_canonical="TransformChain.identity()",  # type: ignore
            applied=True)
    )

    # HOW it behaves (capabilities & collision policy)
    caps: Capability = Capability.RENDERABLE
    collidable_group: int = 0                   # label-compiled group bit (0 = none)
    collidable_mask: int = 0                    # set of groups it can collide with (bitmask)

    # Optional quick UI/layout hints (applies to meshes/points; ignored otherwise)
    pivot_LPS: Optional[Float3] = None          # rotation center in canonical local asset space
    bbox_hint: Optional[FloatAABB] = None       # AABB (2×3) or sphere radius (use metadata if preferred)

    # NOTE: BaseSpec does NOT carry concrete geometry; subclasses do.

# ---------------------------------------------------------------------------
# AssetSpec: concrete geometry (catalog items)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetSpec(BaseSpec):
    # SOURCE (how to load the asset)
    source_path: Optional[Path] = None
    loader: Optional[str] = None                # name of a registered loader (e.g. "trimesh", "sitk_mask_to_trimesh")

    # CANONICAL GEOMETRY (post-load, guaranteed in canonical LPS mm when applied=True)
    mesh: Optional["MeshTransformable"] = None
    points: Optional["PointsTransformable"] = None
    # (lines, volume, etc. could be added later)

    def __post_init__(self):
        # A few light invariants to catch common mistakes
        if self.kind == "mesh" and self.mesh is None and self.points is not None:
            raise ValueError(f"{self.key}: kind='mesh' but only points were provided")
        if self.kind == "points" and self.points is None and self.mesh is not None:
            raise ValueError(f"{self.key}: kind='points' but only mesh was provided")
        if self.role != Role.GEOMETRY and self.caps & Capability.COLLIDABLE:
            # Non-geometry roles default to non-collidable unless explicitly chosen
            object.__setattr__(self, "collidable_mask", 0)
            object.__setattr__(self, "collidable_group", 0)

# ---------------------------------------------------------------------------
# TargetSpec: logical targets (derived or explicit points)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec(BaseSpec):
    # For targets we default to points, role TARGET, and non-collidable caps
    kind: Literal["points", "derived_point"] = "points"
    role: Role = Role.TARGET
    caps: Capability = Capability.RENDERABLE

    # SOURCE: either load explicit points, or derive from another asset via a reducer
    # - If 'source_path' + 'loader' given → explicit points (like AssetSpec points)
    # - If 'source_key' + 'reducer' given → derive from another AssetSpec already in catalog
    source_path: Optional[Path] = None
    loader: Optional[str] = None                # e.g. "numpy_points"
    source_key: Optional[str] = None            # e.g. "structure:PL"
    reducer: Optional[str] = None               # registered reducer name
    reducer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # DERIVED/LOADED canonical points
    points: Optional["PointsTransformable"] = None

    # Hints useful for planning/visualization (targets are often landmarks)
    approach_vector: Optional[Float3] = None    # preferred insertion direction (LPS)
    uncertainty_mm: Optional[float] = None      # radius for UI (confidence, snap tolerance, etc.)

    def __post_init__(self):
        # Enforce typical non-collidable defaults for targets
        if self.caps & Capability.COLLIDABLE:
            raise ValueError(f"{self.key}: targets should not be collidable by default")
        # Require either explicit points (source_path+loader) or derived (source_key+reducer)
        explicit = (self.source_path is not None and self.loader is not None)
        derived  = (self.source_key is not None and self.reducer is not None)
        if not explicit and not derived and self.points is None:
            raise ValueError(f"{self.key}: must provide explicit points or a (source_key, reducer)")

# TODO: change this into asset loader
def build_asset_catalog(cfg) -> dict:
    catalog = {}
    # load meshes/points
    for a in cfg["assets"]:
        if a["kind"] == "mesh":
            geo = load_geometry(a["src"], loader=a["loader"])
        elif a["kind"] == "points" and "src" in a:
            geo = load_geometry(a["src"], loader=a.get("loader", "numpy_points"))
        elif a["kind"] == "points" and "source" in a:
            source_geo = catalog[a["source"]]["geo"]  # use previously loaded asset
            p = reduce_target(source_geo, reducer=a.get("reducer", "explicit"))
            geo = p[None, :] if p.ndim == 1 else p
        else:
            raise ValueError(f"Unsupported asset spec: {a}")

        catalog[a["key"]] = {"kind": a["kind"], "geo": geo, **{k:v for k,v in a.items() if k not in ("src","loader")}}
    return catalog

@dataclass(slots=True)
class NodeInstance:
    id: str  # unique per-node, e.g., "probe:PL"
    asset_key: str  # foreign key to AssetSpec.key, e.g., "probe:2.1"
    transform: TransformChain = field(default_factory=lambda: TransformChain.new([AffineTransform.identity()]))
    tags: Set[str] = field(default_factory=set)
    material_override: Optional[Material] = None
    enabled: bool = True

    # Per-instance constraints/locks (e.g., calibration)
    locked_axes: Set[str] = field(default_factory=set)  # {"ap_tilt", "ml_tilt", "spin", "x", "y", "z"}
    extras: Dict[str, Any] = field(default_factory=dict)  # e.g., calibration_rt


@dataclass(frozen=True, slots=True)
class AppAssets:
    specs: Dict[str, AssetSpec]  # asset catalog
    # Keep your current fields too (brain_mesh, probe_models...), or build specs from them
    # Example: specs["probe:2.1"].mesh holds the MeshTransformable
    calibrations: Optional[Dict[str, AffineTransform]] = None
    targets: Dict[str, TransformedPoints] = field(default_factory=dict)  # you added this

    @classmethod
    def from_app_config(cls, app_config: AppConfig) -> "AppAssets":
        brain_image_transform = AffineTransform.from_sitk_path(app_config.headframe_transform_file)
        chem_shift_pt_R, chem_shift_pt_t = _chem_shift_t_from_path(app_config.image_path)
        chem_shift_image_transform = AffineTransform(
            rotation=chem_shift_pt_R, translation=chem_shift_pt_t, inverted=True
        )
        brain_chem_image_transform = TransformChain([chem_shift_image_transform, brain_image_transform])
        # This is the transform to use on points in the brain, with chemical shift
        brain_chem_pt_transform = brain_chem_image_transform.invert()

        specs: Dict[str, AssetSpec] = {}
        brain_mesh = MeshTransformable(_trimesh_from_sitk_mask(sitk.ReadImage(str(app_config.brain_mask_path))))
        specs["brain"] = AssetSpec(
            key="brain", kind="mesh", mesh=brain_mesh,  # base mesh
            default_material=Material("brain", "#EFC3CA", opacity=0.1),
            caps=Capability.RENDERABLE,
        )
        for name, path in app_config.structure_files.items():
            structure_mesh = MeshTransformable(_trimesh_from_sitk_mask(sitk.ReadImage(str(path))))
            structure_target = np.array(structure_mesh.raw.center_mass)



@dataclass(frozen=True, slots=True)
class AppAssets:
    """
    These are the assets for the app, including the brain mask, brain
    structures, implant segmentation, and probe models.

    This should be the loaded data for objects that do not change location during planning, and
    should be used as the basis for all transformations.

    Probe models are loaded here but not transformed into insertion position
    """

    brain_mesh: TransformedMesh
    brain_structures: Dict[str, TransformedMesh]  # structure name → mesh
    implant_segmentation: TransformedPoints
    probe_models: Dict[str, MeshTransformable]
    hole_models: Dict[int, TransformedMesh]
    implant_mesh: TransformedMesh
    well_mesh: MeshTransformable
    cone_mesh: MeshTransformable
    headframe_mesh: MeshTransformable
    targets: Dict[str, TransformedPoints]
    calibrations: Optional[Dict[str, AffineTransform]]  # structure → AffineTransform

    @classmethod
    def from_app_config(cls, app_config: AppConfig) -> "AppAssets":
        # Find the brain transform for points
        brain_image_transform = AffineTransform.from_sitk_path(app_config.headframe_transform_file)
        chem_shift_pt_R, chem_shift_pt_t = _chem_shift_t_from_path(app_config.image_path)
        chem_shift_image_transform = AffineTransform(
            rotation=chem_shift_pt_R, translation=chem_shift_pt_t, inverted=True
        )
        brain_chem_image_transform = TransformChain([chem_shift_image_transform, brain_image_transform])
        # This is the transform to use on points in the brain, with chemical shift
        brain_chem_pt_transform = brain_chem_image_transform.invert()

        # Downsample the brain mask to 1000 points for visualization
        brain_mesh = MeshTransformable(_trimesh_from_sitk_mask(sitk.ReadImage(str(app_config.brain_mask_path))))
        brain_mesh_txd = TransformedMesh(brain_mesh, brain_chem_pt_transform)

        # Targets
        targets = {}

        # Load brain structures
        brain_structures = {}
        for name, path in app_config.structure_files.items():
            structure_mesh = MeshTransformable(_trimesh_from_sitk_mask(sitk.ReadImage(str(path))))
            structure_target = np.array(structure_mesh.raw.center_mass)
            brain_structures[name] = TransformedMesh(structure_mesh, brain_chem_pt_transform)
            targets[f"structure:{name}"] = TransformedPoints(
                as_transformable(structure_target), brain_chem_pt_transform
            )

        # Load implant segmentation
        # No chemical shift: implant visible with same material as headframe
        _, implant_targets = _process_implant_segmentation(app_config.implant_annotation_file)
        brain_pt_transform = brain_image_transform.invert()
        implant_seg_txd = TransformedPoints(as_transformable(implant_targets), TransformChain([brain_pt_transform]))

        # Load hole models
        # Rotate the model to the original image: inverted because it works on points
        implant_to_image_pt_transform = AffineTransform.from_sitk_path(
            app_config.implant_fit_transform_file, inverted=True
        )
        # Then apply the brain transform without chemical shift
        implant_pt_transform = TransformChain([implant_to_image_pt_transform, brain_pt_transform])
        hole_models = {}
        for hole_num, hole_path in app_config.hole_model_files.items():
            hole_mesh = MeshTransformable(_load_trimesh_lps(hole_path))
            implant_target = np.array(hole_mesh.raw.centroid)
            hole_models[hole_num] = TransformedMesh(hole_mesh, implant_pt_transform)
            targets[f"hole:{hole_num}"] = TransformedPoints(as_transformable(implant_target), implant_pt_transform)
        # Apply to entire implant mesh
        implant_mesh = MeshTransformable(_load_trimesh_lps(app_config.implant_file))
        implant_mesh = TransformedMesh(implant_mesh, implant_pt_transform)

        # Load probe models
        probe_models = {}
        for probe_name, probe_path in app_config.probe_model_files.items():
            probe_trimesh = load_newscale_trimesh(probe_path)
            probe_models[probe_name] = MeshTransformable(probe_trimesh)

        # Other models
        well_mesh = MeshTransformable(_load_trimesh_lps(app_config.well_file))
        cone_mesh = MeshTransformable(_load_trimesh_lps(app_config.cone_file))
        headframe_mesh = MeshTransformable(_load_trimesh_lps(app_config.headframe_file))

        # Load calibrations
        if app_config.calibration_info is None:
            calibrations = None
        else:
            if len(app_config.calibration_info.calibration_files) == 0:
                cal_by_probe_combined, _ = fit_rotation_params_from_parallax(
                    app_config.calibration_info.parallax_calibration_dirs,
                    app_config.reticle_offset,
                    app_config.reticle_rotation,
                )
            else:
                cal_by_probe_combined, _, _ = combine_parallax_and_manual_calibrations(
                    manual_calibration_files=app_config.calibration_info.calibration_files,
                    parallax_directories=app_config.calibration_info.parallax_calibration_dirs,
                )
            calibrations = {}
            for structure, probe_id in app_config.calibration_info.probe_for_target.items():
                if probe_id in cal_by_probe_combined:
                    R, t = cal_by_probe_combined[probe_id]
                    calibrations[structure] = AffineTransform(R, t)

        # Return the constructed AppAssets instance
        return cls(
            brain_mesh=brain_mesh_txd,
            brain_structures=brain_structures,
            implant_segmentation=implant_seg_txd,
            probe_models=probe_models,
            hole_models=hole_models,
            implant_mesh=implant_mesh,
            well_mesh=well_mesh,
            cone_mesh=cone_mesh,
            headframe_mesh=headframe_mesh,
            calibrations=calibrations,
        )


@dataclass(frozen=True, slots=True)
class AppData:
    probe_config: ProbeConfig
    assets: AppAssets


@dataclass(slots=True)
class ProbePose:
    # rig convention: positive is mouse pitch down (CW looking into right ML
    # axis), 0 is vertical
    ap: float = 0.0
    # rig convention: positive is mouse roll right (CCW looking into the front
    # AP axis), 0 is midline
    ml: float = 0.0
    # rig convention: positive is mouse yaw right, 0 is sites facing (left?)
    # (CW looking into superior DV axis)
    spin: float = 0.0
    tip: NDArray = field(default_factory=lambda: np.zeros(3))  # LPS

    def transform(self) -> AffineTransform:
        # Compute the transformation matrix from the probe's location
        R = arc_angles_to_affine(self.ap, self.ml, self.spin)
        t = self.tip
        return AffineTransform(R, t)

    def chain(self) -> TransformChain:
        return TransformChain([self.transform()])

    @classmethod
    def from_app_data(
        cls, data: AppData, probe_name: str, calibration: Optional[AffineTransform] = None
    ) -> "ProbePose":
        # AP ML angles and spin
        probe_config = data.probe_config
        if calibration:
            ap, ml = find_probe_angle(calibration.rotation)
            spin = probe_config.probe_info[probe_name].spin
        else:
            arc = probe_config.probe_info[probe_name].arc
            ap = probe_config.arcs[arc]
            ml = probe_config.probe_info[probe_name].slider_ml
            spin = probe_config.probe_info[probe_name].spin
        # Tip location
        target_info = probe_config.target_info_by_probe[probe_name]
        target_name = target_info.target_name
        if target_name:
            assets = data.assets
            structure_target_id = f"structure:{target_name}"
            if target_name.startswith("hole-"):
                # Handle hole targets
                hole_num = int(target_name.split("-")[-1])
                hole_target_id = f"hole:{hole_num}"
                if hole_target_id in assets.targets:
                    target = assets.targets[hole_target_id].raw
                else:
                    warn(f"Missing hole model for: {target_name}")
                    target = np.zeros(3)
            elif structure_target_id in assets.targets:
                # Handle brain structure targets
                target = assets.targets[structure_target_id].raw
            else:
                warn(f"Missing target for: {target_name}")
                target = np.zeros(3)
            trim = np.zeros(3)
            trim[:2] = target_info.offsets_RA
            trim = convert_coordinate_system(trim, "RAS", "LPS")
            adjusted_target = target + trim
        elif target_info.target_coordinates_RAS:
            trim = np.zeros(3)
            trim[:2] = target_info.offsets_RA
            adjust_target_ras = target_info.target_coordinates_RAS + trim
            adjusted_target = convert_coordinate_system(np.array(adjust_target_ras), "RAS", "LPS")
        else:
            warn(f"No valid target information for probe: {probe_name}")
            adjusted_target = np.zeros(3)
        # Calculate insertion vector
        R_probe_mesh = arc_angles_to_affine(ap, ml, spin)
        insertion_vector = R_probe_mesh @ np.array([0.0, 0.0, -target_info.depth])
        tip_location = adjusted_target + insertion_vector
        return cls(ap=ap, ml=ml, spin=spin, tip=tip_location)


@dataclass(slots=True)
class Probe:
    probe_type: str
    pose: ProbePose
    calibrated: bool = False
    calibration_transform: Optional[AffineTransform] = None

    @classmethod
    def from_app_data(cls, data: AppData, probe_name: str, calibration: Optional[AffineTransform] = None) -> "Probe":
        probe_pose = ProbePose.from_app_data(data, probe_name, calibration=calibration)
        calibrated = calibration is not None
        probe_type = data.probe_config.probe_info[probe_name].type
        return cls(
            probe_type=probe_type, pose=probe_pose, calibrated=calibrated, calibration_transform=calibration
        )


@dataclass(slots=True)
class AppGeometry:
    data: AppData
    probes: dict[str, Probe]

    @classmethod
    def from_app_data(cls, data: AppData, ignore_calibrations: bool | List[str] = False) -> "AppGeometry":
        # Test if ignore_calibrations is a list
        calibrated_structures = list(data.assets.calibrations.keys()) if data.assets.calibrations else []
        if isinstance(ignore_calibrations, list):
            # Handle the case where a list of structures to ignore is provided
            calibrations_to_use = list(set(calibrated_structures) - set(ignore_calibrations))
        elif ignore_calibrations:
            calibrations_to_use = []
        else:
            calibrations_to_use = calibrated_structures
        probes = {}
        for probe_name in data.probe_config.target_info_by_probe.keys():
            if probe_name in calibrations_to_use:
                maybe_calibration = data.assets.calibrations[probe_name]
            else:
                maybe_calibration = None
            this_probe = Probe.from_app_data(data, probe_name, calibration=maybe_calibration)
            probes[probe_name] = this_probe
        return cls(data=data, probes=probes)


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    color_hex_str: str = "#C8C8C8"
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True


BlendMode = Literal["replace", "multiply", "screen", "alpha_over"]


@dataclass(frozen=True)
class OverlaySpec:
    color: int  # 0xRRGGBB
    alpha: float = 0.6  # 0..1
    blend: BlendMode = "alpha_over"
    priority: int = 0  # higher wins when conflicts
    source: str = "generic"  # "collision" | "hover" | "selection" | ...
    ttl_ms: Optional[int] = None  # optional auto-expire; None = persistent


@dataclass(slots=True)
class OverlayState:
    # node_id -> list of overlays currently active
    by_node: Dict[str, List[OverlaySpec]] = field(default_factory=dict)

    def set(self, node_id: str, *specs: OverlaySpec) -> None:
        self.by_node[node_id] = list(specs)

    def set_for_source(self, node_ids: list[str], spec: OverlaySpec) -> None:
        for nid in node_ids:
            lst = [s for s in self.by_node.get(nid, []) if s.source != spec.source]
            lst.append(spec)
            self.by_node[nid] = lst

    def add(self, node_id: str, spec: OverlaySpec) -> None:
        self.by_node.setdefault(node_id, []).append(spec)

    def clear_source(self, source: str, node_ids: list[str] = []) -> None:
        if not node_ids:
            node_ids = list(self.by_node.keys())
        for nid in node_ids:
            lst = self.by_node.get(nid, [])
            kept = [s for s in lst if s.source != source]
            if kept:
                self.by_node[nid] = kept
            else:
                self.by_node.pop(nid)

    def clear_node(self, node_id: str) -> None:
        self.by_node.pop(node_id, None)

    def clear_all(self) -> None:
        self.by_node.clear()


@dataclass(frozen=True)
class CollisionOverlayStyle:
    default_color: int = 0xFF0000  # red
    default_alpha: float = 0.65


def _blend_over(base_rgb: int, over_rgb: int, alpha: float) -> int:
    br, bg, bb = (base_rgb >> 16) & 255, (base_rgb >> 8) & 255, base_rgb & 255
    or_, og, ob = (over_rgb >> 16) & 255, (over_rgb >> 8) & 255, over_rgb & 255
    r = int(round((1 - alpha) * br + alpha * or_))
    g = int(round((1 - alpha) * bg + alpha * og))
    b = int(round((1 - alpha) * bb + alpha * ob))
    return (r << 16) | (g << 8) | b


def material_to_view(m: Material) -> ViewMaterial:
    color_int = hex_string_to_int(m.color_hex_str) if isinstance(m.color_hex_str, str) else int(m.color_hex_str)
    return ViewMaterial(color=color_int, opacity=float(m.opacity), wireframe=bool(m.wireframe), visible=bool(m.visible))


@dataclass
class OverlayResolver:
    overlays: OverlayState

    def apply(self, node_id: str, base_vm: ViewMaterial) -> ViewMaterial:
        specs = self.overlays.by_node.get(node_id)
        if not specs:
            return base_vm
        # choose highest priority (or fold in order; customize if needed)
        spec = max(specs, key=lambda s: s.priority)
        if spec.blend == "replace":
            return ViewMaterial(
                color=spec.color, opacity=base_vm.opacity, wireframe=base_vm.wireframe, visible=base_vm.visible
            )
        # default alpha-over on color only
        new_color = _blend_over(base_vm.color, spec.color, spec.alpha)
        return ViewMaterial(
            color=new_color, opacity=base_vm.opacity, wireframe=base_vm.wireframe, visible=base_vm.visible
        )


@dataclass
class CollisionOverlay:
    overlay_color: int = 0xFF0000  # red
    overlay_alpha: float = 0.65  # mix-in strength

    def color_for(self, base_color: int, colliding: bool) -> int:
        return _blend_over(base_color, self.overlay_color, self.overlay_alpha) if colliding else base_color


@dataclass(frozen=True, slots=True)
class GeometryRef:
    """Reference to geometry living in AppAssets (Trimesh) or generated points."""

    key: str  # e.g. 'brain', 'structure:PL', 'probe:NP-2.1:a', 'points:implant-targets'
    kind: str  # 'mesh' | 'points' | 'lines'
    # optionally: bbox, vertex count, etc. for fast diffs


@dataclass(slots=True)
class Node:
    id: str
    name: str
    geom: GeometryRef
    material: Material
    tags: Set[str] = field(default_factory=set)  # e.g. {'static'} or {'dynamic', 'probe'}


@dataclass(slots=True)
class Scene:
    nodes: Dict[str, NodeInstance] = field(default_factory=dict)

    def upsert(self, node: NodeInstance):
        self.nodes[node.id] = node

    def remove(self, node_id: str):
        self.nodes.pop(node_id, None)

    def by_tag(self, tag: str):
        return [n for n in self.nodes.values() if tag in n.tags]


def scene_from_app(
    data: AppData, geom: AppGeometry, colormap: str = "rainbow", remove_last_color: bool = True
) -> Scene:
    assets = data.assets
    scene = Scene()

    # Materials
    mat_brain = Material("brain", "#EFC3CA", opacity=0.1)
    mat_implant = Material("implant", "#FF00FF", opacity=1.0)
    mat_well = Material("well", "#E7DDFF", opacity=1.0)
    mat_head = Material("head", "#E2EAF4", opacity=1.0)
    mat_cone = Material("cone", "#FFC8C8", opacity=1.0)

    # Static nodes
    scene.upsert(Node(id="brain", name="Brain", geom=GeometryRef("brain", "mesh"), material=mat_brain, tags={"static"}))
    scene.upsert(
        Node(
            id="implant",
            name="Implant",
            geom=GeometryRef("implant", "mesh"),
            material=mat_implant,
            tags={"static", "implant"},
        )
    )
    scene.upsert(
        Node(
            id="head",
            name="Headframe",
            geom=GeometryRef("headframe", "mesh"),
            material=mat_head,
            tags={"static", "headframe"},
        )
    )
    scene.upsert(
        Node(
            id="well",
            name="Well",
            geom=GeometryRef("well", "mesh"),
            material=mat_well,
            tags={"static", "well"},
        )
    )
    scene.upsert(
        Node(
            id="cone",
            name="Cone",
            geom=GeometryRef("cone", "mesh"),
            material=mat_cone,
            tags={"static", "cone"},
        )
    )
    structures = list(assets.brain_structures.keys())
    probes = list(data.probe_config.target_info_by_probe.keys())
    names = list(set(structures) | set(probes))
    colors = _get_colormap_colors(len(names) + 1, colormap_name=colormap)
    colors = colors[:-1] if remove_last_color else colors
    # Skip the first colormap_skip colors
    color_lookup = {name: rgb_to_hex_string(*(255 * np.array(c)[:3]).astype(int)) for name, c in zip(names, colors)}
    for structure_name in assets.brain_structures.keys():
        color = color_lookup.get(structure_name, "#C8C8C8")
        scene.upsert(
            Node(
                id=f"structure:{structure_name}",
                name=structure_name,
                geom=GeometryRef(f"structure:{structure_name}", "mesh"),
                material=Material(structure_name, color_hex_str=color, opacity=0.1),
                tags={"static", "structure"},
            )
        )
    # Probes (dynamic)
    for pname in geom.probes.keys():
        # geom key identifies canonical geometry; transform is the pose
        ptype = geom.probes[pname].probe_type
        color = color_lookup.get(pname, "#C8C8C8")
        scene.upsert(
            Node(
                id=f"probe:{pname}",
                name=pname,
                geom=GeometryRef(f"probe:{ptype}", "mesh"),
                material=Material(pname, color_hex_str=color, opacity=1.0),
                tags={"dynamic", "probe"},
            )
        )

    return scene


# Commands describe *intent* (what changed), not how to draw
@dataclass(frozen=True)
class SetProbeParams:
    name: str
    ap: Optional[float] = None
    ml: Optional[float] = None
    spin: Optional[float] = None
    tip: Optional[NDArray[np.float64]] = None  # LPS coordinates, if provided


@dataclass(frozen=True)
class SetArcAngle:
    arc_id: str
    angle_deg: float
    propagate: bool = True  # update all non-calibrated probes on this arc


@dataclass(frozen=True)
class AssignProbeArc:
    probe_name: str
    new_arc_id: str
    # Optional: keep AP consistent by setting the arc's angle to current AP for this probe
    snap_arc_to_current_ap: bool = False
    propagate: bool = True  # after assignment, (optionally) update that probe's AP


@dataclass(frozen=True)
class SetProbeCalibration:
    probe_name: str
    calibration_transform: Optional[AffineTransform] = None


# Union of the supported commands (add more later if needed)
AnyCommand = Union[SetProbeParams, SetArcAngle, AssignProbeArc, SetProbeCalibration]


# Pure function: apply a command to geometry and return a *new* geometry + which probes changed
# --- Utilities used by reducer --------------------------------------------
def _is_calibrated(geom: AppGeometry, probe_name: str) -> bool:
    calibration = getattr(geom.probes[probe_name], "calibration_transform", None)
    return calibration is not None


def _iter_probes_on_arc(geom: AppGeometry, arc_id: str) -> Iterable[str]:
    for name, pinfo in geom.data.probe_config.probe_info.items():
        if pinfo.arc == arc_id:
            yield name


def _set_probe_pose(geom: AppGeometry, name: str, *, ap=None, ml=None, spin=None, tip=None) -> None:
    pr = geom.probes[name]
    pose = pr.pose
    geom.probes[name].pose = replace(
        pose,
        ap=pose.ap if ap is None else ap,
        ml=pose.ml if ml is None else ml,
        spin=pose.spin if spin is None else spin,
        tip=pose.tip if tip is None else tip,
    )


# --- Reducer extension -----------------------------------------------------
def apply_command(geom: AppGeometry, cmd: AnyCommand) -> tuple[AppGeometry, list[str]]:
    # Existing SetProbeParams branch (unchanged)
    if isinstance(cmd, SetProbeParams):
        name = cmd.name
        _set_probe_pose(geom, name, ap=cmd.ap, ml=cmd.ml, spin=cmd.spin, tip=cmd.tip)
        return geom, [name]

    # New: set an arc's absolute AP angle (coupled update)
    if isinstance(cmd, SetArcAngle):
        pcfg = geom.data.probe_config
        pcfg.arcs[cmd.arc_id] = float(cmd.angle_deg)

        changed: list[str] = []
        if cmd.propagate:
            for pname in _iter_probes_on_arc(geom, cmd.arc_id):
                if _is_calibrated(geom, pname):  # do not move calibrated probes
                    continue
                _set_probe_pose(geom, pname, ap=float(cmd.angle_deg))
                changed.append(pname)
        return geom, changed

    # New: assign a probe to a different arc
    if isinstance(cmd, AssignProbeArc):
        pcfg = geom.data.probe_config
        pname = cmd.probe_name
        old_arc = pcfg.probe_info[pname].arc
        pcfg.probe_info[pname].arc = cmd.new_arc_id

        # Optionally snap the arc to the probe's current AP so reassignment is continuous
        if cmd.snap_arc_to_current_ap:
            curr_ap = float(geom.probes[pname].pose.ap)
            pcfg.arcs[cmd.new_arc_id] = curr_ap

        # Propagate AP to this probe (but never override calibrated)
        changed = []
        if cmd.propagate and not _is_calibrated(geom, pname):
            new_ap = float(pcfg.arcs[cmd.new_arc_id])
            _set_probe_pose(geom, pname, ap=new_ap)
            changed.append(pname)

        return geom, changed

    # New: toggle calibration lock for a probe
    if isinstance(cmd, SetProbeCalibration):
        pname = cmd.probe_name
        probe = geom.probes[pname]
        probe.calibration_transform = cmd.calibration_transform
        # When turning ON calibration, freeze current AP/ML; when OFF, AP will again follow its arc on next arc update.
        return geom, [pname]

    # Fallback (no-op)
    return geom, []


@dataclass(frozen=True)
class ViewMaterial:
    color: int
    opacity: float
    wireframe: bool
    visible: bool


class RenderBackend(Protocol):
    def create_mesh(
        self, node_id: str, *, name: str, vertices: np.ndarray, indices: np.ndarray, material: ViewMaterial
    ) -> None: ...
    def update_mesh(
        self,
        node_id: str,
        *,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        material: ViewMaterial | None = None,
    ) -> None: ...
    def create_points(
        self, node_id: str, *, name: str, positions: np.ndarray, material: ViewMaterial, point_size: float = 1.0
    ) -> None: ...
    def update_points(
        self, node_id: str, *, positions: np.ndarray | None = None, material: ViewMaterial | None = None
    ) -> None: ...
    def remove(self, node_ids: Iterable[str]) -> None: ...


## LRU of pose
# ---- pose signature ----
def _pose_signature(R: np.ndarray, t: np.ndarray, *, tol: float = 1e-6) -> bytes:
    qR = np.round(R / tol).astype(np.int64).ravel()
    qt = np.round(t / tol).astype(np.int64).ravel()
    return b"v1|" + qR.tobytes() + b"|" + qt.tobytes()


Key = Tuple[str, bytes]  # (mesh_id, pose_sig)


# ---- cache entry ----
@dataclass(slots=True)
class _CacheEntry:
    vertices: np.ndarray  # (N,3) float64
    faces: np.ndarray  # (M,3) int32/64 (passed through; backend re-casts as needed)


# ---- LRU cache of transformed vertices ----
class _TransformCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = int(maxsize)
        self._od: "OrderedDict[Key, _CacheEntry]" = OrderedDict()

    def get_or_compute(self, mesh_id: str, base: trimesh.Trimesh, R: np.ndarray, t: np.ndarray) -> _CacheEntry:
        key = (mesh_id, _pose_signature(R, t))
        hit = self._od.get(key)
        if hit is not None:
            self._od.move_to_end(key)
            return hit
        # transform
        v = (base.vertices @ R.T) + t
        entry = _CacheEntry(vertices=v.astype(np.float64, copy=False), faces=base.faces)
        self._od[key] = entry
        if len(self._od) > self.maxsize:
            self._od.popitem(last=False)
        return entry

    def clear(self) -> None:
        self._od.clear()

    def invalidate_mesh(self, mesh_id: str) -> None:
        """Drop all cache entries derived from a particular mesh key (e.g., topology changed)."""
        to_del = [k for k in self._od.keys() if k[0] == mesh_id]
        for k in to_del:
            self._od.pop(k, None)


@dataclass
class RendererAdapter:
    backend: RenderBackend
    scene: Scene
    assets: AppAssets
    cache: _TransformCache = _TransformCache(maxsize=256)
    overlays: OverlayResolver | None = None

    # ----- public API -----
    def build(self, geom: AppGeometry, coll: CollisionState | None = None) -> None:
        hot = coll.hot if coll else frozenset()
        for node in self.scene.nodes.values():
            self._upsert_node(node, geom, node.id in hot)

    def sync_nodes(self, geom: AppGeometry, nodes: Iterable[Node], coll: CollisionState | None = None) -> None:
        hot = coll.hot if coll else frozenset()
        for node in nodes:
            self._upsert_node(node, geom, node.id in hot)

    def remove(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    def invalidate_mesh_key(self, mesh_key: str) -> None:
        """Call this if a base mesh topology changes (forces re-transform)."""
        self.cache.invalidate_mesh(mesh_key)

    # ----- internals -----
    def _upsert_node(self, node: Node, geom: AppGeometry, colliding: bool) -> None:
        base_vm = material_to_view(node.material)
        vm = self.overlays.apply(node.id, base_vm) if self.overlays else base_vm

        if node.geom.kind == "mesh":
            base = self._resolve_mesh(node.geom.key, self.assets)

            # Default pose is identity; probes use live pose
            R, t = self._pose_for_node(node, geom)

            # Use LRU only for PROBE meshes (dynamic). Key the cache by probe TYPE string in geom key.
            if node.id.startswith("probe:") and node.geom.key.startswith("probe:"):
                mesh_id = node.geom.key  # e.g., "probe:2.1"
                entry = self.cache.get_or_compute(mesh_id, base, R, t)
                v, f = entry.vertices, entry.faces
            else:
                v = (base.vertices @ R.T) + t
                f = base.faces

            if node.id in getattr(self.backend, "_handles", {}):
                self.backend.update_mesh(node.id, vertices=v, indices=None, material=vm)
            else:
                self.backend.create_mesh(node.id, name=node.name, vertices=v, indices=f, material=vm)

        elif node.geom.kind == "points":
            pts = self._resolve_points(node.geom.key, self.assets)
            if node.id in getattr(self.backend, "_handles", {}):
                self.backend.update_points(node.id, positions=pts, material=vm)
            else:
                self.backend.create_points(node.id, name=node.name, positions=pts, material=vm, point_size=0.5)
        else:
            raise ValueError(f"Unsupported node kind: {node.geom.kind}")

    def _pose_for_node(self, node: Node, geom: AppGeometry) -> Tuple[np.ndarray, np.ndarray]:
        if node.id.startswith("probe:"):
            pname = node.id.split(":", 1)[1]
            return geom.probes[pname].pose.chain().composed_transform
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _resolve_mesh(self, key: str, a: AppAssets) -> trimesh.Trimesh:
        if key == "brain":
            return a.brain_mesh.raw
        if key == "implant":
            return a.implant_mesh.raw
        if key == "headframe":
            return a.headframe_mesh.raw
        if key == "well":
            return a.well_mesh.raw
        if key == "cone":
            return a.cone_mesh.raw
        if key.startswith("structure:"):
            return a.brain_structures[key.split(":", 1)[1]].raw
        if key.startswith("hole:"):
            return a.hole_models[int(key.split(":", 1)[1])].raw
        if key.startswith("probe:"):
            return a.probe_models[key.split(":", 1)[1]].raw  # TYPE, not name
        raise KeyError(f"Unknown mesh key: {key}")

    def _resolve_points(self, key: str, a: AppAssets) -> np.ndarray:
        if key == "implant-targets":
            return a.implant_segmentation.raw
        raise KeyError(f"Unknown points key: {key}")


@dataclass
class K3DBackend(RenderBackend):
    plot: k3d.Plot
    _handles: Dict[str, Any] = field(default_factory=dict)
    _kinds: Dict[str, str] = field(default_factory=dict)  # 'mesh'|'points'

    def create_mesh(self, node_id, *, name, vertices, indices, material):
        h = k3d.mesh(
            vertices.astype(float),
            indices.astype(np.uint32),
            name=name,
            color=int(material.color),
            opacity=float(material.opacity),
            wireframe=bool(material.wireframe),
        )
        if hasattr(h, "visible"):
            h.visible = bool(material.visible)
        self.plot += h
        self._handles[node_id] = h
        self._kinds[node_id] = "mesh"

    def update_mesh(self, node_id, *, vertices=None, indices=None, material=None):
        h = self._handles.get(node_id)
        if h is None or self._kinds.get(node_id) != "mesh":
            return
        if vertices is not None:
            h.vertices = vertices.astype(float)
        if indices is not None:
            h.indices = indices.astype(np.uint32)
        if material is not None:
            h.color = int(material.color)
            h.opacity = float(material.opacity)
            if hasattr(h, "wireframe"):
                h.wireframe = bool(material.wireframe)
            if hasattr(h, "visible"):
                h.visible = bool(material.visible)

    def create_points(self, node_id, *, name, positions, material, point_size=1.0):
        h = k3d.points(
            positions=positions.astype(float), name=name, color=int(material.color), point_size=float(point_size)
        )
        if hasattr(h, "visible"):
            h.visible = bool(material.visible)
        self.plot += h
        self._handles[node_id] = h
        self._kinds[node_id] = "points"

    def update_points(self, node_id, *, positions=None, material=None):
        h = self._handles.get(node_id)
        if h is None or self._kinds.get(node_id) != "points":
            return
        if positions is not None:
            h.positions = positions.astype(float)
        if material is not None:
            h.color = int(material.color)
            if hasattr(h, "visible"):
                h.visible = bool(material.visible)

    def remove(self, node_ids: Iterable[str]) -> None:
        for nid in node_ids:
            h = self._handles.pop(nid, None)
            self._kinds.pop(nid, None)
            if h is not None:
                try:
                    self.plot -= h
                except Exception:
                    if hasattr(h, "visible"):
                        h.visible = False


Subscriber = Callable[[AppGeometry, List[str]], None]


class GeometryStore:
    def __init__(self, initial: AppGeometry):
        self._state = initial
        self._subs: List[Subscriber] = []

    @property
    def state(self) -> AppGeometry:
        return self._state

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subs.append(fn)

        def _unsub():
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return _unsub

    def _notify(self, changed: List[str]) -> None:
        for fn in list(self._subs):
            fn(self._state, changed)

    def dispatch(self, cmd: AnyCommand) -> None:
        new_state, changed = apply_command(self._state, cmd)  # your existing pure function
        self._state = new_state
        self._notify(changed)


class DebouncedCoalescer:
    """
    Wrap a (state, changed_ids) sink with a debounce/coalesce buffer.
    Thread-safe; suitable for slider drags calling dispatch() rapidly.
    """

    def __init__(self, sink: Subscriber, interval_ms: int = 16):
        self._sink = sink
        self._interval = interval_ms / 1000.0
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending_ids: Set[str] = set()
        self._latest_state: Optional[AppGeometry] = None

    def __call__(self, state: AppGeometry, changed_ids: List[str]) -> None:
        with self._lock:
            self._latest_state = state
            self._pending_ids.update(changed_ids)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._interval, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            ids = list(self._pending_ids)
            self._pending_ids.clear()
            state = self._latest_state
            self._timer = None
        if state is not None and ids:
            self._sink(state, ids)


## Collision detection
# ---- result types ----
@dataclass(frozen=True)
class Contact:
    position: np.ndarray  # (3,), float64
    normal: np.ndarray  # (3,), float64 (from o1 into o2)
    penetration_depth: float


@dataclass(frozen=True)
class CollisionPair:
    id1: str
    id2: str
    contacts: Tuple[Contact, ...]  # empty if enable_contact=False


# ---- specs the backend accepts (domain-free) ----
@dataclass(frozen=True)
class ObjSpec:
    node_id: str
    geom: fcl.CollisionGeometry  # already built (BVH, box, etc.)
    transform: fcl.Transform  # pose in world coords (LPS)


@dataclass(slots=True)
class CollisionState:
    # all pairs in collision (sorted tuple so (a,b)==(b,a))
    pairs: FrozenSet[Pair] = field(default_factory=frozenset)
    # convenience: any node that participates in *any* collision
    hot: FrozenSet[str] = field(default_factory=frozenset)

    def replace(self, pairs: set[Pair]) -> "CollisionState":
        spairs = frozenset(tuple(sorted(p)) for p in pairs)
        hot = frozenset({nid for p in spairs for nid in p})
        return CollisionState(pairs=spairs, hot=hot)


class CollisionBackend(Protocol):
    def rebuild(self, specs: Iterable[ObjSpec]) -> None: ...
    def sync(self, specs: Iterable[ObjSpec]) -> None: ...
    def remove(self, node_ids: Iterable[str]) -> None: ...
    def collide_internal(self, *, enable_contacts: bool, max_contacts: int) -> List[CollisionPair]: ...
    def collide_one_to_many(
        self, spec: ObjSpec, *, enable_contacts: bool, max_contacts: int
    ) -> List[CollisionPair]: ...


@dataclass
class FCLBackend(CollisionBackend):
    _mgr: fcl.DynamicAABBTreeCollisionManager = field(default_factory=fcl.DynamicAABBTreeCollisionManager)
    _node_to_obj: Dict[str, fcl.CollisionObject] = field(default_factory=dict)
    _geomid_to_node: Dict[int, str] = field(default_factory=dict)  # id(CollisionGeometry) -> node_id
    _node_to_geomid: Dict[str, int] = field(default_factory=dict)  # node_id -> id(CollisionGeometry)

    def rebuild(self, specs: Iterable[ObjSpec]) -> None:
        self._mgr.clear()
        self._node_to_obj.clear()
        self._geomid_to_node.clear()
        self._node_to_geomid.clear()

        objs: List[fcl.CollisionObject] = []
        for s in specs:
            geom_id = id(s.geom)
            cob = fcl.CollisionObject(s.geom, s.transform)
            objs.append(cob)
            self._node_to_obj[s.node_id] = cob
            self._geomid_to_node[geom_id] = s.node_id
            self._node_to_geomid[s.node_id] = geom_id

        if objs:
            self._mgr.registerObjects(objs)
        self._mgr.setup()

    def sync(self, specs: Iterable[ObjSpec]) -> None:
        for s in specs:
            cob = self._node_to_obj.get(s.node_id)
            if cob is None:
                # new
                geom_id = id(s.geom)
                cob = fcl.CollisionObject(s.geom, s.transform)
                self._node_to_obj[s.node_id] = cob

                self._geomid_to_node[geom_id] = s.node_id
                self._node_to_geomid[s.node_id] = geom_id
                self._mgr.registerObject(cob)
            else:
                # pose update only (geometry assumed same)
                cob.setTransform(s.transform)
                self._mgr.update(cob)
        self._mgr.update()

    def remove(self, node_ids: Iterable[str]) -> None:
        for nid in node_ids:
            cob = self._node_to_obj.pop(nid, None)
            if cob is not None:
                try:
                    self._mgr.unregisterObject(cob)
                finally:
                    geom_id = self._node_to_geomid.pop(nid, None)
                    if geom_id is not None:
                        self._geomid_to_node.pop(geom_id, None)
        self._mgr.update()

    # ---- queries (docs-style, no custom callbacks) ----
    def collide_internal(self, *, enable_contacts: bool, max_contacts: int) -> List[CollisionPair]:
        req = fcl.CollisionRequest(enable_contact=bool(enable_contacts), num_max_contacts=int(max_contacts))
        cdata = fcl.CollisionData(request=req)
        self._mgr.collide(cdata, fcl.defaultCollisionCallback)
        return self._pairs_from_contacts(cdata.result.contacts)

    def collide_one_to_many(self, spec: ObjSpec, *, enable_contacts: bool, max_contacts: int) -> List[CollisionPair]:
        req = fcl.CollisionRequest(enable_contact=bool(enable_contacts), num_max_contacts=int(max_contacts))
        cdata = fcl.CollisionData(request=req)
        ext = fcl.CollisionObject(spec.geom, spec.transform)
        self._mgr.collide(ext, cdata, fcl.defaultCollisionCallback)
        # add a temporary mapping for the external object, using its geometry id
        ext_name_map = {id(ext.collision_geometry): spec.node_id}
        return self._pairs_from_contacts(cdata.result.contacts, extra_map=ext_name_map)

    # ---- helpers ----
    def _pairs_from_contacts(
        self,
        contacts: Iterable[fcl.Contact],
        *,
        extra_map: Optional[Dict[int, str]] = None,
    ) -> List[CollisionPair]:
        gid_to_name: Dict[int, str] = dict(self._geomid_to_node)
        if extra_map:
            gid_to_name.update(extra_map)

        groups: Dict[Tuple[str, str], List[Contact]] = {}
        for c in contacts:
            n1 = gid_to_name.get(id(c.o1))
            n2 = gid_to_name.get(id(c.o2))
            if n1 is None or n2 is None:
                continue
            k = (n1, n2) if n1 <= n2 else (n2, n1)
            cc = Contact(
                position=np.asarray(c.pos, dtype=np.float64),
                normal=np.asarray(c.normal, dtype=np.float64),
                penetration_depth=float(c.penetration_depth),
            )
            groups.setdefault(k, []).append(cc)

        return [CollisionPair(id1=a, id2=b, contacts=tuple(cs)) for (a, b), cs in groups.items()]


# ---- simple guards at the fcl boundary ----
class FCLInputError(ValueError):
    pass


def _ensure_fcl_arrays(mesh: trimesh.Trimesh, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise FCLInputError(f"{name}: vertices must be (N,3)")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or mesh.faces.size == 0:
        raise FCLInputError(f"{name}: faces must be (M,3) and non-empty")
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    f = np.ascontiguousarray(mesh.faces, dtype=np.int32)  # FCL wants int32
    if not np.isfinite(v).all():
        raise FCLInputError(f"{name}: NaN/Inf in vertices")
    vmax = v.shape[0] - 1
    if f.min() < 0 or f.max() > vmax:
        raise FCLInputError(f"{name}: face index out of range [0..{vmax}]")
    return v, f


def _bvh_from_mesh(mesh: trimesh.Trimesh, *, name: str) -> fcl.BVHModel:
    v, f = _ensure_fcl_arrays(mesh, name=name)
    m = fcl.BVHModel()
    m.beginModel(v.shape[0], f.shape[0])
    m.addSubModel(v, f)
    m.endModel()
    return m


def _rt_to_transform(R: np.ndarray, t: np.ndarray, *, name: str) -> fcl.Transform:
    R = np.ascontiguousarray(R, dtype=np.float64).reshape(3, 3)
    t = np.ascontiguousarray(t, dtype=np.float64).reshape(
        3,
    )
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise FCLInputError(f"{name}: non-finite R/t")
    return fcl.Transform(R, t)


# ---- inclusion predicate (domain-level) ----
def default_include(node: Node) -> bool:
    # Exclude brain + structures; include only meshes
    if node.geom.kind != "mesh":
        return False
    if node.geom.key == "brain" or node.geom.key.startswith("structure:"):
        return False
    return True


@dataclass
class CollisionAdapter:
    backend: CollisionBackend
    scene: Scene
    assets: AppAssets
    include: Callable[[Node], bool] = default_include

    # ---- lifecycle wiring ----
    def rebuild(self, geom: AppGeometry) -> None:
        specs = [s for n in self.scene.nodes.values() if self.include(n) for s in [self._spec_for_node(n, geom)] if s]
        self.backend.rebuild(specs)

    def on_store_change(self, geom: AppGeometry, changed_probe_names: List[str]) -> None:
        # Only probes move; map probe names -> scene nodes
        nodes: List[Node] = []
        for pname in changed_probe_names:
            nid = f"probe:{pname}"
            node = self.scene.nodes.get(nid)
            if node and self.include(node):
                nodes.append(node)
        if not nodes:
            return
        specs = [self._spec_for_node(n, geom) for n in nodes]
        self.backend.sync([s for s in specs if s is not None])

    def remove_nodes(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    # ---- queries (pass-through to backend) ----
    def collide_internal(self, *, enable_contacts: bool = True, max_contacts: int = 8) -> List[CollisionPair]:
        return self.backend.collide_internal(enable_contacts=enable_contacts, max_contacts=max_contacts)

    def collide_one_to_many(
        self, mesh: trimesh.Trimesh, R: np.ndarray, t: np.ndarray, *, name: str
    ) -> List[CollisionPair]:
        spec = ObjSpec(
            node_id=name, geom=_bvh_from_mesh(mesh, name=name), transform=_rt_to_transform(R, t, name=f"pose:{name}")
        )
        return self.backend.collide_one_to_many(spec, enable_contacts=True, max_contacts=8)

    # ---- domain → backend spec ----
    def _spec_for_node(self, node: Node, geom: AppGeometry) -> Optional[ObjSpec]:
        if node.geom.kind != "mesh":
            return None
        base = self._resolve_mesh(node.geom.key, self.assets)
        bvh = _bvh_from_mesh(base, name=node.geom.key)  # unique geometry per node
        R, t = self._pose_for_node(node, geom)
        tf = _rt_to_transform(R, t, name=f"pose:{node.id}")
        return ObjSpec(node_id=node.id, geom=bvh, transform=tf)

    def _pose_for_node(self, node: Node, geom: AppGeometry) -> Tuple[np.ndarray, np.ndarray]:
        if node.id.startswith("probe:"):
            pname = node.id.split(":", 1)[1]
            return geom.probes[pname].pose.chain().composed_transform
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _resolve_mesh(self, key: str, a: AppAssets) -> trimesh.Trimesh:
        if key == "implant":
            return a.implant_mesh.raw
        if key == "headframe":
            return a.headframe_mesh.raw
        if key == "well":
            return a.well_mesh.raw
        if key == "cone":
            return a.cone_mesh.raw
        if key.startswith("hole:"):
            return a.hole_models[int(key.split(":", 1)[1])].raw
        if key.startswith("probe:"):
            return a.probe_models[key.split(":", 1)[1]].raw  # TYPE, not name
        # brain/structures are intentionally excluded by include()
        raise KeyError(f"Unknown mesh key for collision: {key}")


def objects_in_collision(collision_pairs: List[CollisionPair]) -> List[Tuple[str, str]]:
    objects_in_collision = set()
    for coll_pair in collision_pairs:
        objects_in_collision.add((coll_pair.id1, coll_pair.id2))
    return list(objects_in_collision)


@dataclass
class StoreSubscriber:
    store: GeometryStore
    on_event: Callable[[AppGeometry, List[str]], None]

    def __post_init__(self):
        self._unsubscribe = self.store.subscribe(self.on_event)

    def dispose(self):
        try:
            self._unsubscribe()
        except Exception:
            pass


@dataclass
class RenderHandler:
    scene: Scene
    adapter: RendererAdapter
    # optional shared view-state (e.g., overlays from collisions)
    get_collision_state: Callable[[], CollisionState] | None = None

    def __call__(self, geom: AppGeometry, changed_ids: List[str]) -> None:
        # map probe ids → scene nodes; extend as needed
        nodes = [self.scene.nodes.get(f"probe:{pid}") for pid in changed_ids]
        nodes = [n for n in nodes if n is not None]
        # let the adapter apply overlays if provided
        self.adapter.sync_nodes(geom, nodes, coll=self.get_collision_state() if self.get_collision_state else None)


def _diff_hot(curr: CollisionState, prev: CollisionState | None = None) -> set[str]:
    # nodes that flipped collision state
    if prev is None:
        return set(curr.hot)
    return set((prev.hot - curr.hot) | (curr.hot - prev.hot))


@dataclass
class CollisionHandler:
    scene: Scene
    adapter: CollisionAdapter
    state: CollisionState = field(default_factory=CollisionState)
    on_state_changed: Optional[Callable[[CollisionState, Set[str], AppGeometry], None]] = None
    _prev_state: CollisionState | None = None

    def __call__(self, geom: AppGeometry, changed_ids: List[str]) -> None:
        # keep backend up-to-date for moved probes
        moved = [pid for pid in changed_ids if f"probe:{pid}" in self.scene.nodes]
        if moved:
            self.adapter.on_store_change(geom, moved)

        # recompute collisions
        pairs = self.adapter.collide_internal(enable_contacts=False)
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)

        # notify only if something flipped
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, geom)


def on_collisions_changed_lambda(renderer_adapter: RendererAdapter, scene: Scene, overlays_state: OverlayState):
    def _on_collisions_changed(state: CollisionState, flips: Set[str], geom: AppGeometry) -> None:
        # update overlays by source "collision"
        overlays_state.clear_source("collision")
        if state.hot:
            spec = OverlaySpec(color=0xFF0000, alpha=0.65, source="collision", priority=30)
            overlays_state.set_for_source(list(state.hot), spec)

        # repaint only nodes whose hot/cold status flipped
        nodes = [scene.nodes[nid] for nid in flips if nid in scene.nodes]
        if nodes:
            renderer_adapter.sync_nodes(geom, nodes)  # adapter reads overlays internally

    return _on_collisions_changed


@dataclass
class ProbeWidgetController:
    data: AppData
    store: GeometryStore
    assets: AppAssets
    plot: k3d.Plot
    render_adapter: RendererAdapter
    collision_handler: CollisionHandler
    overlays_resolver: OverlayResolver

    # Coupling behavior (AP always coupled via arc; ML can be optionally coupled)
    couple_ml: bool = False

    # UI containers
    controls: widgets.VBox = field(init=False)
    view: widgets.HBox = field(init=False)

    # Core widgets
    probe_dd: widgets.Dropdown = field(init=False)
    arc_label: widgets.HTML = field(init=False)

    # Position (RAS)
    pos_ap_ras: widgets.FloatSlider = field(init=False)
    pos_ml_ras: widgets.FloatSlider = field(init=False)
    pos_dv_ras: widgets.FloatSlider = field(init=False)

    # Rotations (about tip)
    ap_tilt_deg: widgets.FloatSlider = field(init=False)  # AP coupled by arc unless calibrated
    ml_tilt_deg: widgets.FloatSlider = field(init=False)  # per-probe (or optionally coupled)
    spin_deg: widgets.IntSlider = field(init=False)  # per-probe

    arc_assign_dd: widgets.Dropdown = field(init=False)
    # Target snap (optional convenience)
    target_dd: widgets.Dropdown = field(init=False)
    goto_btn: widgets.Button = field(init=False)

    status_out: widgets.Output = field(init=False)

    # Keyboard helpers (optional)
    kb_panel: widgets.HTML = field(init=False)
    kb_event: Event | None = field(init=False, default=None)
    step_pos_mm: float = 0.05
    step_tilt_deg: float = 0.5
    step_spin_deg: float = 1.0

    def __post_init__(self):
        # enable collision overlays in renderer
        self.render_adapter.overlays = self.overlays_resolver

        self._build_widgets()
        self._wire_events()
        self._populate_initial()

        kb_help = widgets.HTML(
            "<div style='font-size:12px;line-height:1.3'>"
            "<b>Keyboard</b> (click panel): Move <code>W/S</code>=AP ±, <code>A/D</code>=ML ∓, <code>R/F</code>=DV ±; "
            "Tilt <code>I/K</code>=AP ±, <code>J/L</code>=ML ±; Spin <code>U/O</code> ±. "
            "Shift×10, Ctrl×0.2."
            "</div>"
        )
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.probe_dd, self.arc_label, self.arc_assign_dd, self.target_dd, self.goto_btn]),
                kb_help,
                self.kb_panel,
                widgets.HTML("<b>Position (RAS)</b>"),
                widgets.HBox([self.pos_ap_ras, self.pos_ml_ras, self.pos_dv_ras]),
                widgets.HTML("<b>Orientation about tip (°)</b>"),
                widgets.HBox([self.ap_tilt_deg, self.ml_tilt_deg, self.spin_deg]),
                self.status_out,
            ]
        )
        self.view = widgets.HBox([self.controls, self.plot])

    # ---------- helpers ----------
    def _current_probe(self):
        pname = self.probe_dd.value
        return pname, self.store.state.probes[pname] if pname else (None, None)

    def _probe_arc_id(self, probe_name: str) -> str:
        return self.data.probe_config.probe_info[probe_name].arc

    def _arc_angle(self, arc_id: str) -> float:
        return float(self.data.probe_config.arcs[arc_id])

    def _target_names(self) -> list[str]:
        return sorted(self.assets.targets.keys())

    def _target_point_LPS(self, key: str) -> np.ndarray:
        pts = self.assets.targets[key].raw  # (N,3)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            raise ValueError(f"targets['{key}'] must be (N,3)")
        return np.asarray(pts.mean(axis=0), dtype=float)

    # ---------- domain pushes ----------
    def _dispatch_tip_only(self, probe_name: str, tip_lps: np.ndarray):
        probe = self.store.state.probes[probe_name]
        self.store.dispatch(
            SetProbeParams(
                name=probe_name,
                ap=probe.pose.ap,  # unchanged
                ml=probe.pose.ml,  # unchanged
                spin=probe.pose.spin,  # unchanged
                tip=tip_lps,
            )
        )

    def _dispatch_pose(self, probe_name: str, ap: float, ml: float, spin: float, tip_lps: np.ndarray | None = None):
        probe = self.store.state.probes[probe_name]
        self.store.dispatch(
            SetProbeParams(
                name=probe_name,
                ap=ap,
                ml=ml,
                spin=spin,
                tip=probe.pose.tip if tip_lps is None else tip_lps,
            )
        )

    # ---------- apply from UI (live) ----------
    def _apply_xyz_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        tip_ras = np.array([self.pos_ml_ras.value, self.pos_ap_ras.value, self.pos_dv_ras.value], dtype=float)
        tip_lps = convert_coordinate_system(tip_ras, "RAS", "LPS")  # convert RAS to LPS
        self._dispatch_tip_only(pname, tip_lps)

    def _apply_spin_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        # spin is always allowed, even if calibrated
        self._dispatch_pose(pname, ap=probe.pose.ap, ml=probe.pose.ml, spin=float(self.spin_deg.value))

    def _apply_ml_live(self):
        pname, probe = self._current_probe()
        if not pname:
            return
        if getattr(probe, "calibrated", False):
            # locked
            self.ml_tilt_deg.value = float(probe.pose.ml)
            return
        new_ml = float(self.ml_tilt_deg.value)
        if self.couple_ml:
            # propagate to all non-calibrated probes sharing this arc
            arc_id = self._probe_arc_id(pname)
            for other_name, other in self.store.state.probes.items():
                if getattr(other, "calibrated", False):
                    continue
                if self._probe_arc_id(other_name) == arc_id:
                    self._dispatch_pose(other_name, ap=other.pose.ap, ml=new_ml, spin=other.pose.spin)
        else:
            self._dispatch_pose(pname, ap=probe.pose.ap, ml=new_ml, spin=probe.pose.spin)

    def _apply_ap_via_arc_live(self):
        """AP tilt is coupled by arc: edit arc angle and propagate to all non-calibrated probes on that arc."""
        pname, probe = self._current_probe()
        if not pname:
            return
        arc_id = self._probe_arc_id(pname)
        self.store.dispatch(SetArcAngle(arc_id=arc_id, angle_deg=float(self.ap_tilt_deg.value), propagate=True))

    # ---------- load UI from domain ----------
    def _load_probe_into_widgets(self, probe_name: str):
        p = self.store.state.probes[probe_name]
        arc_id = self._probe_arc_id(probe_name)

        # keep options in sync (in case arcs were added/removed elsewhere)
        self.arc_assign_dd.options = sorted(self.data.probe_config.arcs.keys())
        self.arc_assign_dd.value = arc_id  # select current arc

        self.arc_label.value = f"<b>Arc:</b> {arc_id} &nbsp; (<i>AP coupled</i>)"

        # XYZ
        tip_ras = convert_coordinate_system(p.pose.tip, "RAS", "LPS")
        self.pos_ml_ras.value = float(tip_ras[0])
        self.pos_ap_ras.value = float(tip_ras[1])
        self.pos_dv_ras.value = float(tip_ras[2])

        # AP from arc; ML, Spin from probe
        self.ap_tilt_deg.value = self._arc_angle(arc_id)
        self.ml_tilt_deg.value = float(p.pose.ml)
        self.spin_deg.value = int(round(float(p.pose.spin)))

        # lock AP/ML when calibrated
        is_cal = getattr(p, "calibrated", False)
        self.ap_tilt_deg.disabled = is_cal  # AP control will still show arc value but be disabled if calibrated
        self.ml_tilt_deg.disabled = is_cal

        with self.status_out:
            self.status_out.clear_output(wait=True)
            print(f"[UI] Loaded '{probe_name}'  (arc={arc_id}, {'calibrated' if is_cal else 'free'})")

    # ---------- UI build & events ----------
    def _build_widgets(self):
        self.probe_dd = widgets.Dropdown(
            options=sorted(self.store.state.probes.keys()),
            description="Probe:",
            layout={"width": "220px"},
        )
        self.arc_label = widgets.HTML(layout={"width": "180px"})

        self.arc_assign_dd = widgets.Dropdown(
            options=sorted(self.data.probe_config.arcs.keys()),
            description="Arc:",
            layout={"width": "140px"},
        )

        # Position sliders (RAS)
        self.pos_ap_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="AP (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.pos_ml_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="ML (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.pos_dv_ras = widgets.FloatSlider(
            value=0.0,
            min=-10,
            max=10,
            step=0.05,
            description="DV (mm)",
            continuous_update=False,
            layout={"width": "220px"},
        )

        # Orientation about tip
        self.ap_tilt_deg = widgets.FloatSlider(
            value=0.0,
            min=-60,
            max=60,
            step=0.5,
            description="AP tilt (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.ml_tilt_deg = widgets.FloatSlider(
            value=0.0,
            min=-60,
            max=60,
            step=0.5,
            description="ML tilt (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )
        self.spin_deg = widgets.IntSlider(
            value=0,
            min=-180,
            max=180,
            step=1,
            description="Spin (°)",
            continuous_update=False,
            layout={"width": "220px"},
        )

        # Targets
        self.target_dd = widgets.Dropdown(
            options=self._target_names(), description="Target:", layout={"width": "220px"}
        )
        self.goto_btn = widgets.Button(description="Go to target", button_style="info")

        self.status_out = widgets.Output(
            layout={"border": "1px solid lightgray", "max_height": "120px", "overflow": "auto"}
        )

        # Keyboard panel
        self.kb_panel = widgets.HTML(
            value="<div style='border:1px dashed #999;padding:6px;border-radius:6px;background:#fafafa;'>"
            "<b>Click here</b> to enable keyboard control</div>",
            layout=widgets.Layout(width="320px"),
        )
        if Event is not None:
            self.kb_event = Event(source=self.kb_panel, watched_events=["keydown"], prevent_default_action=True)
        else:
            self.kb_event = None

    def _wire_events(self):
        # Live position
        for w in (self.pos_ap_ras, self.pos_ml_ras, self.pos_dv_ras):
            w.observe(lambda ch: self._apply_xyz_live() if ch["name"] == "value" else None, names="value")

        # Live rotations
        self.spin_deg.observe(lambda ch: self._apply_spin_live() if ch["name"] == "value" else None, names="value")
        self.ml_tilt_deg.observe(lambda ch: self._apply_ml_live() if ch["name"] == "value" else None, names="value")
        self.ap_tilt_deg.observe(
            lambda ch: self._apply_ap_via_arc_live() if ch["name"] == "value" else None, names="value"
        )

        # Probe switch
        self.probe_dd.observe(
            lambda ch: self._load_probe_into_widgets(ch["new"]) if ch["name"] == "value" else None, names="value"
        )

        # Go to target (XYZ only)
        def _goto(_):
            if not self.target_dd.value:
                return
            tip_lps = self._target_point_LPS(self.target_dd.value)
            tip_ras = convert_coordinate_system(tip_lps, "LPS", "RAS")  # convert LPS to RAS
            self.pos_ml_ras.value = float(tip_ras[0])
            self.pos_ap_ras.value = float(tip_ras[1])
            self.pos_dv_ras.value = float(tip_ras[2])
            self._apply_xyz_live()

        self.goto_btn.on_click(_goto)

        def _on_arc_assign(change):
            if change["name"] != "value" or change["new"] is None:
                return
            pname, probe = self._current_probe()
            if not pname:
                return
            new_arc_id = change["new"]
            old_arc_id = self._probe_arc_id(pname)
            if new_arc_id == old_arc_id:
                return

            # If calibrated, we still change the assignment but don't move AP/ML
            is_cal = getattr(probe, "calibrated", False)

            # Snap the new arc to the probe's current AP for continuity, then (optionally)
            # propagate to this probe (no propagation if calibrated).
            self.store.dispatch(
                AssignProbeArc(
                    probe_name=pname, new_arc_id=new_arc_id, snap_arc_to_current_ap=True, propagate=not is_cal
                )
            )

            # Refresh UI labels/sliders to reflect the new arc
            self.arc_label.value = f"<b>Arc:</b> {new_arc_id} &nbsp; (<i>AP coupled</i>)"
            # AP slider reflects arc angle (it may be disabled if calibrated)
            self.ap_tilt_deg.value = self._arc_angle(new_arc_id)

        self.arc_assign_dd.observe(_on_arc_assign, names="value")

        # Keyboard (optional)
        if self.kb_event is not None:

            def on_key(event):
                key = (event or {}).get("key", "")
                shift = bool((event or {}).get("shiftKey", False))
                ctrl = bool((event or {}).get("ctrlKey", False))
                mul = 10.0 if shift else 0.2 if ctrl else 1.0
                dpos = self.step_pos_mm * mul
                dtilt = self.step_tilt_deg * mul
                dspin = self.step_spin_deg * mul
                handled = True

                # XYZ (RAS)
                if key in ("w", "W", "ArrowUp"):
                    self.pos_ap_ras.value += dpos
                elif key in ("s", "S", "ArrowDown"):
                    self.pos_ap_ras.value -= dpos
                elif key in ("a", "A", "ArrowLeft"):
                    self.pos_ml_ras.value -= dpos
                elif key in ("d", "D", "ArrowRight"):
                    self.pos_ml_ras.value += dpos
                elif key in ("r", "R"):
                    self.pos_dv_ras.value += dpos
                elif key in ("f", "F"):
                    self.pos_dv_ras.value -= dpos

                # Rotations (about tip)
                elif key in ("i", "I"):
                    self.ap_tilt_deg.value += dtilt
                elif key in ("k", "K"):
                    self.ap_tilt_deg.value -= dtilt
                elif key in ("j", "J"):
                    self.ml_tilt_deg.value -= dtilt
                elif key in ("l", "L"):
                    self.ml_tilt_deg.value += dtilt
                elif key in ("u", "U"):
                    self.spin_deg.value -= dspin
                elif key in ("o", "O"):
                    self.spin_deg.value += dspin
                else:
                    handled = False

                if handled:
                    with self.status_out:
                        print(f"[KB] {key} (x{mul:g})")

            self.kb_event.on_dom_event(on_key)

    def _populate_initial(self):
        # Select first probe and load its state
        if self.probe_dd.options:
            self.probe_dd.value = list(self.probe_dd.options)[0]
            self._load_probe_into_widgets(self.probe_dd.value)

    # ---------- public ----------
    def display(self):
        display(self.view)


# Instantiate
# %%
# Load YAML with OmegaConf and resolve all ${}
app_cfg = OmegaConf.load(config_files["app_config"])
app_cfg_resolved = OmegaConf.to_container(app_cfg, resolve=True)
# Create validated Pydantic model
app_config = AppConfig(**app_cfg_resolved)

probe_config = config_files.get("probe_config", None)
if probe_config:
    probe_cfg = OmegaConf.load(probe_config)
    probe_cfg_resolved = OmegaConf.to_container(probe_cfg, resolve=True)
    probe_config = ProbeConfig(**probe_cfg_resolved)
else:
    structures = app_config.target_structures
    probe_config = ProbeConfig(
        arcs={"a": 0.0},
        probe_info={struct: ProbeInfo(arc="a") for struct in structures},
        target_info_by_probe={struct: TargetInfo() for struct in structures},
    )
# %%
assets = AppAssets.from_app_config(app_config)
# %%
data = AppData(assets=assets, probe_config=probe_config)
geom = AppGeometry.from_app_data(data, ignore_calibrations=True)
scene = scene_from_app(data, geom)
store = GeometryStore(geom)
overlays_state = OverlayState()
overlays_resolver = OverlayResolver(overlays_state)
# backends
plot = k3d.plot(grid_visible=True, background_color=0xFFFFFF)
fcl_backend = FCLBackend()
k3d_backend = K3DBackend(plot=plot)
render_adapter = RendererAdapter(backend=k3d_backend, scene=scene, assets=assets, overlays=overlays_resolver)
collision_adapter = CollisionAdapter(backend=fcl_backend, scene=scene, assets=assets)

# Initial build
render_adapter.build(geom)
collision_adapter.rebuild(geom)

# Collision view-state
collision_handler = CollisionHandler(
    scene=scene,
    adapter=collision_adapter,
    on_state_changed=on_collisions_changed_lambda(render_adapter, scene, overlays_state),
)
render_handler = RenderHandler(scene=scene, adapter=render_adapter, get_collision_state=lambda: collision_handler.state)
# Optional: wrap handlers with different debouncers
render_cb = DebouncedCoalescer(render_handler, interval_ms=16)
collision_cb = DebouncedCoalescer(collision_handler, interval_ms=33)

# Subscribers (identical shape)
render_sub = StoreSubscriber(store, render_cb)
collision_sub = StoreSubscriber(store, collision_cb)

# --- Instantiate and display --------------------------------------------
controller = ProbeWidgetController(
    data=data,
    store=store,
    assets=assets,
    plot=plot,
    render_adapter=render_adapter,
    collision_handler=collision_handler,
    overlays_resolver=overlays_resolver,
    couple_ml=False,  # set True if you also want ML tilt coupled across probes on the same arc
)

controller.display()
# %%
