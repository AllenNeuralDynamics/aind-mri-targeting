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
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from functools import cached_property
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
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
from aind_mri_utils.meshes import (
    mask_to_trimesh,
)
from aind_mri_utils.plots import hex_string_to_int
from aind_mri_utils.reticle_calibrations import (
    find_probe_angle,
    fit_rotation_params_from_manual_calibration,
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
from pydantic import BaseModel, DirectoryPath, Field, FilePath, field_validator, model_validator
from scipy.spatial.transform import Rotation

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


# %%
# Classes


## Run-time classes
Float3x3 = NDArray[np.float64]  # shape (3, 3)
Float3 = NDArray[np.float64]  # shape (3,)
FloatNx3 = NDArray[np.float64]  # shape (3, N)
FloatAABB = NDArray[np.float64]  # shape (2, 3)
RawT_co = TypeVar("RawT_co", covariant=True)

# -----------------------------------------------------------------------------
# Enums & Flags (config-facing; parsed from strings in YAML)
# -----------------------------------------------------------------------------


class Capability(IntFlag):
    RENDERABLE = auto()
    MOVABLE = auto()
    COLLIDABLE = auto()
    SELECTABLE = auto()
    DEFORMABLE = auto()
    SAVABLE = auto()


class Role(str, Enum):
    GEOMETRY = "geometry"
    TARGET = "target"
    LANDMARK = "landmark"
    ANATOMY = "anatomy"


class Kind(str, Enum):
    MESH = "mesh"
    POINTS = "points"
    LINES = "lines"


# -----------------------------------------------------------------------------
# Basic building blocks
# -----------------------------------------------------------------------------


class ImagingModel(BaseModel):
    magnet_frequency_MHz: float
    chem_shift_ppm_default: float = 3.7
    chem_shift_apply_by_role: List[Role] = Field(default_factory=lambda: [Role.ANATOMY])
    # optionally, where to read the reference image from if needed by your library
    image_path: Optional[FilePath] = None


ChemMode = Literal["on", "off", "auto"]


class MaterialModel(BaseModel):
    name: str = "default"
    color: str = Field("#C8C8C8", description="Hex #RRGGBB")
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True

    @field_validator("opacity")
    @classmethod
    def _opacity_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("opacity must be in [0,1]")
        return v


class CanonicalizationDefModel(BaseModel):
    source_space: Literal["ASR", "RAS", "LPS", "FILE_NATIVE"]
    scale_to_mm: float = 1.0
    transform: Optional[TransformRefModel] = None  # name in your transforms registry
    version: str = "canon-v1"


class CanonicalizationOverrideModel(BaseModel):
    # all optional: only supplied fields override the referenced def
    source_space: Optional[Literal["ASR", "RAS", "LPS", "FILE_NATIVE"]] = None
    scale_to_mm: Optional[float] = None
    transform: Optional[TransformRefModel] = None
    version: Optional[str] = None


class ResourceModel(BaseModel):
    """
    A load-once file. The loader may return a structured container:
    - dict[str, np.ndarray] of named points
    - dict[str|int, trimesh.Trimesh] for labelmaps
    - GLTF scene graph keyed by node paths, etc.
    """

    key: str
    kind: Kind  # POINTS, MESH, LABELS, GLTF, etc. (choose the set you support)
    src: str
    loader: str  # e.g., "named_points_npz", "labelmap_to_meshes", "gltf"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)
    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None


class SelectorBase(BaseModel):
    kind: Literal["name", "index", "path", "label"]

    def select(self, payload: Any) -> Any:
        raise NotImplementedError


class NameSelector(SelectorBase):
    kind: Literal["name"]
    name: str

    def select(self, payload: Any) -> Any:
        return payload[self.name]


class IndexSelector(SelectorBase):
    kind: Literal["index"]
    index: int

    def select(self, payload: Any) -> Any:
        return payload[self.index]


class PathSelector(SelectorBase):
    kind: Literal["path"]  # e.g., HDF5 dataset path or GLTF node path
    path: str

    def select(self, payload: Any) -> Any:
        return payload[self.path]


class LabelSelector(SelectorBase):
    kind: Literal["label"]  # e.g., integer label id or string label name
    label: Union[int, str]

    def select(self, payload: Any) -> Any:
        return payload[self.label]


Selector = Annotated[Union[NameSelector, IndexSelector, PathSelector, LabelSelector], Field(discriminator="kind")]


def select_from_resource(payload: Any, selector: Selector) -> Any:
    return selector.select(payload)


class CollisionPolicyModel(BaseModel):
    """Label-based policy; compile to bitmasks in loader."""

    group: Optional[str] = Field(default=None, description="e.g., STATIC, FIXTURE, PROBE")
    mask: List[str] = Field(default_factory=list, description="Labels it can collide with")


class _TxOpBase(BaseModel):
    invert: bool = False

    def to_affine(self):
        raise NotImplementedError("to_affine must be implemented by subclasses")


class TranslateTxOpModel(_TxOpBase):
    kind: Literal["translate_mm"] = "translate_mm"
    delta: List[float] = Field(..., min_length=3, max_length=3)

    def to_affine(self):
        return AffineTransform(np.eye(3), np.array(self.delta))


class RotateEulerTxOpModel(_TxOpBase):
    kind: Literal["rotate_euler_deg"] = "rotate_euler_deg"
    order: Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx"] = "ZYX"
    angles_deg: List[float] = Field(..., min_length=3, max_length=3)

    def to_affine(self):
        # Convert degrees to radians
        # Create a rotation object using scipy
        rotation = Rotation.from_euler(self.order, self.angles_deg, degrees=True)
        # Get the rotation matrix
        R = rotation.as_matrix()
        return AffineTransform(R, np.zeros(3))


class LoadSITKTxOpModel(_TxOpBase):
    kind: Literal["sitk_file"] = "sitk_file"
    path: FilePath
    inverted: bool = False

    def to_affine(self):
        R, t, _ = load_sitk_transform(self.path)
        return AffineTransform(R, t, self.inverted)


TransformOp = Annotated[
    Union[TranslateTxOpModel, RotateEulerTxOpModel, LoadSITKTxOpModel],
    Field(discriminator="kind"),
]


class TransformRecipeModel(BaseModel):
    """Sequence of ops; accepts a single op or a list and normalizes to list."""

    sequence: List[TransformOp] = Field(default_factory=list)

    # Allow top-level single-op form:
    #   transforms:
    #     fit: { kind: sitk_file, path: ... }
    @model_validator(mode="before")
    @classmethod
    def _coerce_root_single_op(cls, data: Any):
        if isinstance(data, dict) and "sequence" not in data and "kind" in data:
            return {"sequence": [data]}
        return data

    # Allow 'sequence' itself to be a single op (dict or parsed model)
    @field_validator("sequence", mode="before")
    @classmethod
    def _coerce_sequence(cls, v: Any):
        if v is None:
            return []
        # if already a list, keep it
        if isinstance(v, list):
            return v
        # if a single op dict (has 'kind'), wrap it
        if isinstance(v, dict) and "kind" in v:
            return [v]
        # if a single parsed op model, wrap it
        if isinstance(v, _OpBase):
            return [v]
        raise TypeError("sequence must be a list[TransformOp] or a single TransformOp")


# Optional: key-or-inline reference, with the same single-op convenience
class TransformRefModel(BaseModel):
    key: Optional[str] = None
    inline: Optional[TransformRecipeModel] = None

    # Allow: inline: { kind: translate_mm, delta: [1,2,3] }
    @field_validator("inline", mode="before")
    @classmethod
    def _coerce_inline(cls, v: Any):
        if v is None:
            return None
        if isinstance(v, dict) and "sequence" not in v and "kind" in v:
            return {"sequence": [v]}
        return v

    @model_validator(mode="after")
    def _xor(self):
        if bool(self.key) == bool(self.inline):
            raise ValueError("TransformRefModel: provide exactly one of {key | inline}")
        return self


def compile_recipe_to_chain(recipe: TransformRecipeModel) -> TransformChain:
    """
    Compile a recipe to a TransformChain. We keep individual ops as separate
    AffineTransforms (nice for debugging) but you could also pre-compose.
    """
    affines = [op.to_affine() for op in recipe.sequence]
    return TransformChain.new(affines if affines else [AffineTransform.identity()])



# -----------------------------------------------------------------------------
# Catalog specs (WHAT an asset/target is; not where placed)
# -----------------------------------------------------------------------------


class BaseSpecModel(BaseModel):
    key: str
    kind: Kind
    role: Role = Role.GEOMETRY
    default_material: MaterialModel = Field(default_factory=MaterialModel)  # type: ignore[arg-type]
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    canonicalization_ref: Optional[str] = None
    canonicalization: Optional[CanonicalizationDefModel] = None  # inline (legacy/one-off)
    canonicalization_override: Optional[CanonicalizationOverrideModel] = None

    # capabilities are parsed from strings like ["RENDERABLE", "COLLIDABLE"]
    caps: List[Capability] = Field(default_factory=lambda: [Capability.RENDERABLE])
    collision: CollisionPolicyModel = Field(default_factory=CollisionPolicyModel)

    # UI/layout hints
    pivot_LPS: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    bbox_hint: Optional[List[List[float]]] = Field(default=None)

    chem_shift: ChemMode = "auto"
    chem_shift_ppm: Optional[float] = None

    @model_validator(mode="after")
    def _check_canon_choice(self):
        # allow: (ref) or (inline); not both
        if self.canonicalization_ref and self.canonicalization:
            raise ValueError("Provide either canonicalization_ref or canonicalization, not both.")
        return self

    @field_validator("bbox_hint")
    @classmethod
    def _bbox_shape(cls, v):
        if v is None:
            return v
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(row, list) and len(row) == 3 for row in v)):
            raise ValueError("bbox_hint must be [[minx,miny,minz],[maxx,maxy,maxz]]")
        return v


class AssetSpecModel(BaseSpecModel):
    """Geometry/points/lines that can be loaded by a named loader."""

    src: Optional[Path] = None
    loader: Optional[str] = None
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    from_resource: Optional[str] = None
    selector: Optional[Selector] = None

    @model_validator(mode="after")
    def _check_src_loader(self):
        if (self.src is None) ^ (self.loader is None):
            raise ValueError("Asset must provide both 'src' and 'loader', or neither (if injected elsewhere).")
        if (self.src and self.loader) and (self.from_resource or self.selector):
            raise ValueError("Choose either (src+loader) or (from_resource+selector), not both.")
        if (self.from_resource is None) ^ (self.selector is None):
            # only one given
            raise ValueError("When using from_resource, you must also provide a selector.")
        return self


class TargetSpecModel(BaseSpecModel):
    """Targets are points; explicit (src+loader) or derived (source_key+reducer)."""

    kind: Kind = Kind.POINTS
    role: Role = Role.TARGET

    # Explicit points (file)
    src: Optional[Path] = None
    loader: Optional[str] = None  # e.g., "numpy_points"
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Or derived from an existing asset in catalog
    source_key: Optional[str] = None
    reducer: Optional[str] = None
    reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Or resource
    from_resource: Optional[str] = None
    selector: Optional[Selector] = None
    post_reducer: Optional[str] = None  # optional final reduction (e.g., COM of a selected mesh)
    post_reducer_kwargs: dict[str, Any] = Field(default_factory=dict)

    approach_vector: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)
    uncertainty_mm: Optional[float] = None

    @model_validator(mode="after")
    def _exactly_one_source(self):
        explicit = self.src is not None and self.loader is not None
        derived = self.source_key is not None and self.reducer is not None
        from_res = (self.from_resource is not None) and (self.selector is not None)
        paths = sum([explicit, derived, from_res])
        if paths != 1:
            raise ValueError(
                f"{self.key}: provide exactly one of (src+loader) | (source_key+reducer) | (from_resource+selector)"
            )
        return self

    @model_validator(mode="after")
    def _noncollidable_default(self):
        if Capability.COLLIDABLE in self.caps:
            raise ValueError(f"{self.key}: targets should not be collidable by default.")
        return self


# -----------------------------------------------------------------------------
# Scene (WHERE: instances and bindings)
# -----------------------------------------------------------------------------


class SceneNodeModel(BaseModel):
    id: str
    asset: str = Field(description="Key of an AssetSpec in catalog")
    tags: List[str] = Field(default_factory=list)

    # Reference a named transform (from ConfigModel.transforms) or leave None for identity
    transform: Optional[TransformRefModel] = None

    # Optional domain binding for pose (use for probes): ties node to domain.probes[name]
    pose_source_probe: Optional[str] = Field(
        default=None,
        description="If set, renderer should take pose from domain.probes[pose_source_probe].",
    )


class SceneModel(BaseModel):
    nodes: List[SceneNodeModel] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Domain (mechanics: arcs, probes, calibrations, target declarations)
# -----------------------------------------------------------------------------


class ProbeDeclModel(BaseModel):
    kind: str
    arc: str
    slider_ml: float = 0.0
    spin: float = 0.0

    target: str = Field(description="Key of a target (TargetSpecModel.key)")
    past_target_mm: float = 0.0
    offsets_RA: List[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)

    calibrated: bool = False  # initial lock state; actual calibration affine comes from 'calibrations' map


class CalibrationRefModel(BaseModel):
    """Reference a specific calibration entry inside a calibration file."""

    cal_id: str  # key into CalibrationsModel.files
    probe_code: str  # 5-digit code in the file (keep as str; accept ints)

    # allow shorthand "cal_id:probe_code"
    @classmethod
    def from_string(cls, s: str) -> "CalibrationRefModel":
        if ":" not in s:
            raise ValueError("Expected '<cal_id>:<probe_code>'")
        cal_id, probe_code = s.split(":", 1)
        return cls(cal_id=cal_id.strip(), probe_code=str(probe_code).strip())


class CalibrationReticleModel(BaseModel):
    """Model for calibration reticle used in calibrations"""

    offset_RAS: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    rotation_z: float = 0.0


class CalibrationSourceModel(BaseModel):
    """
    One calibration 'bank' source:
      - EITHER a single file (e.g., .xlsx). In this case NO reticle is allowed.
      - OR a directory for parallax. In this case a reticle IS REQUIRED.
    """

    file: Optional[FilePath] = Field(default=None, description="Path to a single calibration file (e.g., .xlsx)")
    directory: Optional[DirectoryPath] = Field(default=None, description="Path to a parallax calibration directory")
    reticle: Optional[str] = Field(default=None, description="Name of reticle (required when 'directory' is set)")

    @model_validator(mode="after")
    def _xor_and_require_reticle(self):
        has_file = self.file is not None
        has_dir = self.directory is not None
        if has_file == has_dir:
            # both set or both None → invalid
            raise ValueError("Specify exactly one of 'file' or 'directory' in a calibration source")

        if has_file and self.reticle is not None:
            # forbid reticle with file
            raise ValueError("'reticle' must not be provided when 'file' is used")

        if has_dir and not self.reticle:
            # require reticle with directory
            raise ValueError("'reticle' is required when 'directory' is used")

        return self


class CalibrationsModel(BaseModel):
    files: dict[str, CalibrationSourceModel] = Field(default_factory=dict)
    # domain_probe_name → either {"cal_id": "...", "probe_code": "..."} OR "cal_id:probe_code"
    probe_to_ref: dict[str, Union[CalibrationRefModel, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_refs(self):
        # convert any string refs to CalibrationRefModel
        normalized: dict[str, CalibrationRefModel] = {}
        for probe_name, ref in self.probe_to_ref.items():
            if isinstance(ref, str):
                normalized[probe_name] = CalibrationRefModel.from_string(ref)
            else:
                normalized[probe_name] = ref
        object.__setattr__(self, "probe_to_ref", normalized)
        return self


class PlanningModel(BaseModel):
    arcs: dict[str, float] = Field(default_factory=dict, description="arc_id → AP angle (deg)")
    probes: dict[str, ProbeDeclModel] = Field(default_factory=dict, description="probe_name → probe declaration")
    reticles: dict[str, CalibrationReticleModel] = Field(default_factory=dict)
    calibrations: CalibrationsModel = Field(default_factory=CalibrationsModel)
    # Targets can live in the catalog (TargetSpecModel), or you can also allow simple inline targets here if desired.


# -----------------------------------------------------------------------------
# Transforms, Paths, Options
# -----------------------------------------------------------------------------


class PathsModel(BaseModel):
    """Freeform helper; keep loose so Hydra/OmegaConf interpolation is easy."""

    __root__: dict[str, Any] = Field(default_factory=dict)


class OptionsModel(BaseModel):
    color_map: str = "rainbow"
    remove_last_color: bool = True


# -----------------------------------------------------------------------------
# Root config (everything together) + cross-reference validation
# -----------------------------------------------------------------------------


class ConfigModel(BaseModel):
    version: int = 1

    paths: PathsModel = Field(default_factory=PathsModel)
    imaging: Optional[ImagingModel] = None
    # Catalog
    resources: List[ResourceModel] = Field(default_factory=list)
    assets: List[AssetSpecModel] = Field(default_factory=list)
    targets: List[TargetSpecModel] = Field(default_factory=list)

    # Scene
    scene: SceneModel = Field(default_factory=SceneModel)

    # Domain
    plan: PlanningModel = Field(default_factory=PlanningModel)

    # Named transforms & misc
    transforms: dict[str, TransformRecipeModel] = Field(default_factory=dict)
    canonicalizations: dict[str, CanonicalizationDefModel] = Field(default_factory=dict)
    options: OptionsModel = Field(default_factory=OptionsModel)

    # ---------- Cross-file integrity checks ----------
    @model_validator(mode="after")
    def _xref(self):
        asset_keys = {a.key for a in self.assets}
        target_keys = {t.key for t in self.targets}
        arc_ids = set(self.plan.arcs.keys())
        probe_names = set(self.plan.probes.keys())
        transform_keys = set(self.transforms.keys())

        errors: List[str] = []
        # Scene nodes must point to existing assets
        for n in self.scene.nodes:
            if n.asset not in asset_keys:
                errors.append(f"scene.nodes[{n.id}].asset '{n.asset}' not found in assets catalog")
            if n.transform is not None and n.transform not in transform_keys:
                errors.append(f"scene.nodes[{n.id}].transform_key='{n.transform}' not found in transforms")
            if n.pose_source_probe is not None and n.pose_source_probe not in probe_names:
                errors.append(f"scene.nodes[{n.id}].pose_source_probe='{n.pose_source_probe}' not in domain.probes")

        # Targets (derived) must reference existing assets
        for t in self.targets:
            if t.source_key is not None and t.source_key not in asset_keys:
                errors.append(f"target '{t.key}' source_key '{t.source_key}' not found in assets")

        # Probes must reference valid arcs and targets
        for pname, p in self.plan.probes.items():
            if p.arc not in arc_ids:
                errors.append(f"domain.probes['{pname}'] arc '{p.arc}' not found in domain.arcs")
            if p.target not in target_keys:
                errors.append(f"domain.probes['{pname}'] target '{p.target}' not found in targets")

        # Reticles must exist and be referenced by calibration files
        reticle_names = set(self.plan.reticles.keys())
        cal_files = self.plan.calibrations.files
        probe_names = set(self.plan.probes.keys())

        # Every calibration file must reference an existing reticle
        for cal_id, cal in cal_files.items():
            if cal.reticle not in reticle_names:
                errors.append(
                    f"calibrations.files['{cal_id}'] references reticle '{cal.reticle}' "
                    f"which is not defined in domain.reticles"
                )

        # probe_to_ref must reference existing probes and calibration file IDs
        for probe_name, ref in self.plan.calibrations.probe_to_ref.items():
            if probe_name not in probe_names:
                errors.append(f"calibrations.probe_to_ref contains '{probe_name}' not in domain.probes")
            if ref.cal_id not in cal_files:  # type: ignore[reportAttributeAccessIssue]
                errors.append(
                    f"calibrations.probe_to_ref['{probe_name}'].cal_id '{ref.cal_id}' not found in calibrations.files"  # type: ignore[reportAttributeAccessIssue]
                )
        # Canonicalization
        # --- helper to resolve & check a spec-like object (Resource/Asset/Target) ---

        def _check_transform_ref(ref: Optional["TransformRefModel"], where: str) -> None:
            """If a TransformRef uses a key, it must exist in config.transforms."""
            if ref and ref.key and ref.key not in self.transforms:
                errors.append(f"{where}: transform key '{ref.key}' not found in transforms")

        def _check_canon_def(cdef: Optional["CanonicalizationDefModel"], where: str) -> None:
            if not cdef:
                return
            _check_transform_ref(cdef.transform, f"{where}.canonicalization.transform")

        def _check_spec_like(spec, where: str) -> None:
            # 1) canonicalization_ref must exist (if provided)
            cref = getattr(spec, "canonicalization_ref", None)
            if cref:
                base = self.canonicalizations.get(cref)
                if base is None:
                    errors.append(f"{where} '{getattr(spec, 'key', '?')}': canonicalization_ref '{cref}' not found")
                else:
                    # referenced canonicalization may itself carry a transform ref
                    _check_transform_ref(base.transform, f"canonicalizations['{cref}'].transform")

            # 2) inline canonicalization and override may each carry a transform ref
            _check_canon_def(getattr(spec, "canonicalization", None), f"{where} '{getattr(spec, 'key', '?')}'")
            _check_canon_def(
                getattr(spec, "canonicalization_override", None), f"{where} '{getattr(spec, 'key', '?')}'.override"
            )

        # --- check canonicalization definitions themselves -------------------
        for cname, cdef in self.canonicalizations.items():
            _check_transform_ref(cdef.transform, f"canonicalizations['{cname}'].transform")

        # --- check resources / assets / targets ------------------------------
        for r in self.resources:
            _check_spec_like(r, "resource")
        for a in self.assets:
            _check_spec_like(a, "asset")
        for t in self.targets:
            _check_spec_like(t, "target")

        for n in self.scene.nodes:
            _check_transform_ref(getattr(n, "transform", None), f"scene.nodes['{getattr(n, 'id', '?')}'].transform")

        def _check_spec_canon(spec, where: str) -> None:
            # 1) canonicalization_ref exists (if provided)
            if spec.canonicalization_ref:
                if spec.canonicalization_ref not in self.canonicalizations:
                    errors.append(
                        f"{where} '{getattr(spec, 'key', '?')}': "
                        f"canonicalization_ref '{spec.canonicalization_ref}' not found"
                    )

            # 2) collect all transform_keys that *might* be used for this spec
            tkeys: List[Tuple[str, Optional[str]]] = []

            # from referenced canonicalization
            if spec.canonicalization_ref:
                base = self.canonicalizations.get(spec.canonicalization_ref)
                if base and base.transform:
                    tkeys.append((f"canonicalizations[{spec.canonicalization_ref}]", base.transform))

            # from inline canonicalization
            if spec.canonicalization and spec.canonicalization.transform_key:
                tkeys.append(("inline canonicalization", spec.canonicalization.transform_key))

            # from override
            if spec.canonicalization_override and spec.canonicalization_override.transform_key:
                tkeys.append(("canonicalization_override", spec.canonicalization_override.transform_key))

            # 3) each transform_key must exist in transforms
            for origin, tk in tkeys:
                if tk and tk not in self.transforms:
                    errors.append(
                        f"{where} '{getattr(spec, 'key', '?')}': {origin} transform_key '{tk}' not found in transforms"
                    )

        # --- check all spec containers ---
        for r in self.resources:
            _check_spec_canon(r, "resource")

        for a in self.assets:
            _check_spec_canon(a, "asset")

        for t in self.targets:
            _check_spec_canon(t, "target")

        # --- optional: sanity checks on canonicalization defs themselves ---
        for cname, cdef in self.canonicalizations.items():
            # if applied=False and source_space == FILE_NATIVE, a transform_key is required
            if (cdef.source_space == "FILE_NATIVE") and not cdef.transform:
                errors.append(
                    f"canonicalizations[{cname}]: source_space=FILE_NATIVE without transform_key; "
                    "provide a transform_key or ensure the loader normalizes to LPS."
                )
            # transform_key (if present) must exist
            if cdef.transform and cdef.transform not in self.transforms:
                errors.append(f"canonicalizations[{cname}]: transform_key '{cdef.transform}' not found in transforms")

        if errors:
            raise ValueError("Config cross-reference errors:\n  - " + "\n  - ".join(errors))
        return self


@runtime_checkable
class SupportsRigidTransform(Protocol[RawT_co]):
    @property
    def raw(self) -> RawT_co: ...
    def transformed(self, R: NDArray[np.float64], t: NDArray[np.float64]) -> RawT_co: ...


W = TypeVar("W", bound=SupportsRigidTransform[Any])


@dataclass(frozen=True)
class AffineTransform:
    rotation: Float3x3 = field(default_factory=lambda: np.eye(3), repr=False)
    translation: Float3 = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]), repr=False)
    inverted: bool = False

    @classmethod
    def identity(cls) -> AffineTransform:
        return cls()

    @classmethod
    def from_sitk_path(cls, path: Path, inverted=False) -> AffineTransform:
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


GeometryOut = Union[
    trimesh.Trimesh,  # surface mesh
    NDArray[np.float64],  # (N,3) points
]

# ---- Registry core --------------------------------------------------------
_LOADER_REGISTRY: dict[str, Callable[..., GeometryOut]] = {}


def register_loader_fn(fn: Callable[..., GeometryOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _LOADER_REGISTRY:
        raise KeyError(f"Loader '{key}' already registered")
    _LOADER_REGISTRY[key] = fn
    return fn


def register_loader(arg: str | Callable[..., GeometryOut] | None = None):
    """Decorator to register a loader function by name."""

    def _wrap(fn: Callable[..., GeometryOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_loader_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def load_geometry(src: Union[str, Path], loader: str, **kwargs) -> GeometryOut:
    """Dispatch to a named loader. kwargs are passed to the loader."""
    fn = _LOADER_REGISTRY.get(loader)
    if fn is None:
        raise KeyError(f"Unknown loader '{loader}'. Known: {', '.join(sorted(_LOADER_REGISTRY)) or '(none)'}")
    return fn(Path(src), **kwargs)


@register_loader
def trimesh_from_sitk_mask(mask: sitk.Image) -> trimesh.Trimesh:
    """Convert a SimpleITK mask image to a trimesh."""
    structure_mesh = mask_to_trimesh(mask)
    trimesh.repair.fix_normals(structure_mesh)
    trimesh.repair.fix_inversion(structure_mesh)
    return structure_mesh


@register_loader
def load_trimesh_lps(path: Path, src_coordinate_system: str = "ASR") -> trimesh.Trimesh:
    """Load a trimesh from a SimpleITK image file."""
    mesh = trimesh.load(str(path))
    vertices_lps = convert_coordinate_system(mesh.vertices, src_coordinate_system, "LPS")
    mesh.vertices = vertices_lps
    return mesh


SourceGeo = Union[trimesh.Trimesh, NDArray[np.float64]]
ReduceOut = NDArray[np.float64]  # usually (3,) single point; could be (N,3)

_REDUCER_REGISTRY: dict[str, Callable[..., ReduceOut]] = {}


def register_reducer_fn(fn: Callable[..., ReduceOut], name: Optional[str] = None):
    key = name or fn.__name__
    if key in _REDUCER_REGISTRY:
        raise KeyError(f"Reducer '{key}' already registered")
    _REDUCER_REGISTRY[key] = fn
    return fn


def register_reducer(arg: str | Callable[..., ReduceOut] | None = None):
    def _wrap(fn: Callable[..., ReduceOut]):
        name = arg.__name__ if callable(arg) else arg
        return register_reducer_fn(fn, name)

    if callable(arg):
        return _wrap(arg)
    else:
        return _wrap


def reduce_target(source: SourceGeo, reducer: str, **kwargs) -> ReduceOut:
    fn = _REDUCER_REGISTRY.get(reducer)
    if fn is None:
        raise KeyError(f"Unknown reducer '{reducer}'. Known: {', '.join(sorted(_REDUCER_REGISTRY)) or '(none)'}")
    return fn(source, **kwargs)


@dataclass(frozen=True)
class CanonicalizationRuntime:
    source_space: Literal["ASR", "RAS", "FILE_NATIVE"]
    scale_to_mm: float  # e.g. 0.001 if µm → mm
    transform_file_to_canonical: AffineTransform


@dataclass(frozen=True)
class BaseSpec:
    # WHAT it is
    key: str  # unique id, e.g. "probe:2.1", "structure:PL", "target:hole:1"
    kind: Literal["mesh", "points", "lines"]
    role: Role = Role.GEOMETRY
    default_material: Material = field(default_factory=lambda: Material("default"))
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)  # free-form (scene/UI grouping)

    # HOW it behaves (capabilities & collision policy)
    caps: Capability = Capability.RENDERABLE
    collidable_group: int = 0  # label-compiled group bit (0 = none)
    collidable_mask: int = 0  # set of groups it can collide with (bitmask)

    # Optional quick UI/layout hints (applies to meshes/points; ignored otherwise)
    pivot_LPS: Optional[Float3] = None  # rotation center in canonical local asset space
    bbox_hint: Optional[FloatAABB] = None  # AABB (2×3) or sphere radius (use metadata if preferred)

    # NOTE: BaseSpec does NOT carry concrete geometry; subclasses do.


# ---------------------------------------------------------------------------
# AssetSpec: concrete geometry (catalog items)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetSpec(BaseSpec):
    # SOURCE (how to load the asset)
    source_path: Optional[Path] = None
    loader: Optional[str] = None  # name of a registered loader (e.g. "trimesh", "trimesh_from_sitk_mask")

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
    loader: Optional[str] = None  # e.g. "numpy_points"
    source_key: Optional[str] = None  # e.g. "structure:PL"
    reducer: Optional[str] = None  # registered reducer name
    reducer_kwargs: dict[str, Any] = field(default_factory=dict)

    # DERIVED/LOADED canonical points
    points: Optional["PointsTransformable"] = None

    # Hints useful for planning/visualization (targets are often landmarks)
    approach_vector: Optional[Float3] = None  # preferred insertion direction (LPS)
    uncertainty_mm: Optional[float] = None  # radius for UI (confidence, snap tolerance, etc.)

    def __post_init__(self):
        # Enforce typical non-collidable defaults for targets
        if self.caps & Capability.COLLIDABLE:
            raise ValueError(f"{self.key}: targets should not be collidable by default")
        # Require either explicit points (source_path+loader) or derived (source_key+reducer)
        explicit = self.source_path is not None and self.loader is not None
        derived = self.source_key is not None and self.reducer is not None
        if not explicit and not derived and self.points is None:
            raise ValueError(f"{self.key}: must provide explicit points or a (source_key, reducer)")


@dataclass(frozen=True)
class ChemShiftContext:
    enabled: bool
    magnet_MHz: float
    default_ppm: float = 3.7
    apply_by_role: set[Role] = field(default_factory=set)
    # transforms to apply to geometry in image/LPS space
    points_transform: AffineTransform = AffineTransform.identity()


def build_chemshift_context(cfg: ConfigModel) -> ChemShiftContext:
    im = cfg.imaging
    if im is None:
        return ChemShiftContext(False, 0.0, 0.0)
    # Build correction using your existing aind_mri_utils helpers.
    # If your `compute_chemical_shift` accepts only ppm, scale ppm if you want
    # frequency-awareness; otherwise pass ppm through (common in practice).
    if im.image_path is not None:
        brain_image = sitk.ReadImage(str(im.image_path))
        chem_shift_pt_R, chem_shift_pt_t = chemical_shift_transform(
            compute_chemical_shift(brain_image, ppm=im.chem_shift_ppm_default)
        )
        chem_shift_pt_transform = AffineTransform(chem_shift_pt_R, chem_shift_pt_t)
    else:
        chem_shift_pt_transform = AffineTransform.identity()
    return ChemShiftContext(
        enabled=True,
        magnet_MHz=im.magnet_frequency_MHz,
        default_ppm=im.chem_shift_ppm_default,
        apply_by_role=set(im.chem_shift_apply_by_role),
        points_transform=chem_shift_pt_transform,
    )


def _should_apply_chem(asset_model: BaseSpecModel, chem: ChemShiftContext) -> bool:
    if not chem.enabled:
        return False
    mode = asset_model.chem_shift  # "on"|"off"|"auto"
    if mode == "on":
        return True
    if mode == "off":
        return False
    # "auto": follow role defaults
    return asset_model.role in chem.apply_by_role


@dataclass(frozen=True, slots=True)
class AssetCatalog:
    assets: dict[str, AssetSpec]  # asset catalog
    targets: dict[str, TargetSpec] = field(default_factory=dict)


# Plan for probe location
@dataclass(slots=True)
class ProbePlan:
    probe_type: str
    arc_id: Optional[str]  # which arc this probe belongs to (None = not bound to any arc)
    # angle sources / bindings
    bind_ap_to_arc: bool = True  # if True and not calibrated → AP comes from arc
    # per-probe local angles (always present so you can edit them; used when
    # not bound / not calibrated)
    ap_local: float = 0.0  # deg
    ml_local: float = 0.0  # deg
    spin: float = 0.0  # deg
    # targeting
    past_target_mm: float = 0.0
    offsets_RA: Tuple[float, float] = (0.0, 0.0)
    target_key: Optional[str] = None  # preferred: reference into asset catalog
    target_point_RAS: Optional[Tuple[float, float, float]] = None  # ad-hoc fallback
    # calibration policy
    calibrated: bool = False  # if True and calibration exists → AP/ML come from calibration


@dataclass(frozen=True, slots=True)
class JointRange:
    lo: float
    hi: float

    def clamp(self, v: float) -> float:
        return float(min(max(v, self.lo), self.hi))


@dataclass(frozen=True, slots=True)
class PoseLimits:
    # angular limits (deg)
    ap_deg: JointRange = field(default_factory=lambda: JointRange(-60.0, 60.0))
    ml_deg: JointRange = field(default_factory=lambda: JointRange(-60.0, 60.0))
    spin_deg: JointRange = field(default_factory=lambda: JointRange(-180.0, 180.0))
    # translational work envelope (mm); set to None if unbounded
    x_mm: Optional[JointRange] = None
    y_mm: Optional[JointRange] = None
    z_mm: Optional[JointRange] = None

    def clamp_angles(self, ap: float, ml: float, spin: float) -> Tuple[float, float, float]:
        return (
            self.ap_deg.clamp(ap),
            self.ml_deg.clamp(ml),
            self.spin_deg.clamp(spin),
        )

    def clamp_xyz(self, tip_lps: np.ndarray) -> np.ndarray:
        t = np.asarray(tip_lps, dtype=np.float64).copy()
        if self.x_mm:
            t[0] = self.x_mm.clamp(t[0])
        if self.y_mm:
            t[1] = self.y_mm.clamp(t[1])
        if self.z_mm:
            t[2] = self.z_mm.clamp(t[2])
        return t


# --- Kinematics model (no knowledge of probes) ------------------------------


@dataclass(slots=True)
class Kinematics:
    """
    Rig-wide kinematics parameters.
    - arc_angles: shared AP tilt per arc id (deg)
    - limits: mechanical/operational joint limits
    - coupled_axes: which DOFs are shared by all probes on the same arc
      (by convention these names match your ProbePose fields: 'ap_deg', 'ml_deg', 'spin_deg', 'x_mm', 'y_mm', 'z_mm')
    """

    arc_angles: dict[str, float] = field(default_factory=dict)  # e.g., {"a": 12.0, "b": -8.0}
    limits: PoseLimits = field(default_factory=PoseLimits)
    coupled_axes: Set[str] = field(default_factory=lambda: {"ap_deg"})  # today: AP tilt is arc-coupled

    # convenience helpers
    def get_arc(self, arc_id: str) -> float:
        return float(self.arc_angles[arc_id])

    def set_arc(self, arc_id: str, ap_deg: float) -> float:
        """Clamp and store AP for an arc; return the value actually stored."""
        clamped = self.limits.ap_deg.clamp(ap_deg)
        self.arc_angles[arc_id] = clamped
        return clamped

    def clamp_angles(self, ap: float, ml: float, spin: float) -> Tuple[float, float, float]:
        return self.limits.clamp_angles(ap, ml, spin)

    def clamp_xyz(self, tip_lps: np.ndarray) -> np.ndarray:
        return self.limits.clamp_xyz(tip_lps)

    def is_axis_coupled(self, axis_name: str) -> bool:
        """UI can call this to gray controls; mechanics layer just declares policy."""
        return axis_name in self.coupled_axes


@dataclass(slots=True)
class PlanningState:
    kinematics: Kinematics
    probes: dict[str, ProbePlan]
    calibrations: dict[str, AffineTransform] = field(default_factory=dict)  # probe_name → calibration transform
    target_index: dict[str, Float3] = field(default_factory=dict)


def _resolve_target_LPS_from_plan(
    plan: ProbePlan,
    target_index: dict[str, np.ndarray],
    assets_fallback: Optional[dict[str, TransformedPoints]] = None,
) -> np.ndarray:
    """Return a single (3,) LPS point for the plan's target."""
    # Inline ad-hoc target
    if plan.target_point_RAS is not None:
        ras = np.asarray(plan.target_point_RAS, dtype=float)
        return convert_coordinate_system(ras, "RAS", "LPS")

    # Catalog target by key
    if plan.target_key:
        pts = target_index.get(plan.target_key)
        if pts is None and assets_fallback is not None:
            tp = assets_fallback.get(plan.target_key)
            if tp is not None:
                pts = tp.raw  # already in LPS if your assets pipeline canonicalized it
        if pts is None:
            warn(f"Missing target for key: {plan.target_key!r}; using origin.")
            return np.zeros(3, dtype=float)
        return pts if pts.ndim == 1 else pts.mean(axis=0)

    warn("ProbePlan has neither target_key nor target_point_RAS; using origin.")
    return np.zeros(3, dtype=float)


def _resolved_angles(name: str, ps: PlanningState) -> tuple[float, float, float]:
    plan = ps.probes[name]
    cal = ps.calibrations.get(name)

    if plan.calibrated and cal is not None:
        ap, ml = find_probe_angle(cal.rotation)  # locked to calibration
    else:
        # AP: from arc if bound, else local; ML: always per-probe local unless you add another binding flag
        ap = ps.kinematics.get_arc(plan.arc_id) if (plan.arc_id and plan.bind_ap_to_arc) else plan.ap_local
        ml = plan.ml_local
    # clamp to rig limits
    ap, ml, spin = ps.kinematics.clamp_angles(ap, ml, plan.spin)
    return ap, ml, spin


# Run time
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
    def from_planning_state(
        cls,
        ps: PlanningState,
        probe_name: str,
        *,
        assets_targets_fallback: Optional[dict[str, TransformedPoints]] = None,
    ) -> ProbePose:
        """
        Resolve a live pose from PlanningState (no mutations).
        - AP comes from calibration if plan.calibrated and matrix is present,
        else from arc if bound, else local.
        - ML comes from calibration if present/allowed, else local.
        - Spin is always the per-probe plan spin.
        - Target is taken from planning.target_index (or assets fallback) +
        offsets_RA.
        """
        plan = ps.probes[probe_name]
        cal = ps.calibrations.get(probe_name)

        # --- angles (AP/ML) ---
        ap_deg, ml_deg, spin_deg = _resolved_angles(probe_name, ps)

        # --- target + offsets (RAS→LPS) ---
        tgt_LPS = _resolve_target_LPS_from_plan(plan, ps.target_index, assets_fallback=assets_targets_fallback)
        off_RAS = np.array([plan.offsets_RA[0], plan.offsets_RA[1], 0.0], dtype=np.float64)
        off_LPS = convert_coordinate_system(off_RAS, "RAS", "LPS")
        adjusted_target = tgt_LPS + off_LPS

        # --- tip from depth and orientation ---
        R_probe = arc_angles_to_affine(ap_deg, ml_deg, spin_deg)
        insertion_vec = R_probe @ np.array([0.0, 0.0, -float(plan.past_target_mm)], dtype=np.float64)
        tip = adjusted_target + insertion_vec
        tip = ps.kinematics.clamp_xyz(tip)

        return cls(ap=ap_deg, ml=ml_deg, spin=spin_deg, tip=tip)


# run time
@dataclass(slots=True)
class Probe:
    probe_type: str
    pose: ProbePose


# get_pivot_for_asset: asset_key -> local-space pivot (LPS mm) or None
GetPivotFn = Callable[[str], Optional[np.ndarray]]


@dataclass
class PoseResolver:
    planning: PlanningState
    get_pivot_for_asset: GetPivotFn = lambda _key: None  # default: rotate around asset origin

    # ---- dynamic pose for a probe (no scene knowledge) ----
    def _probe_chain(self, probe_name: str) -> TransformChain:
        pose = ProbePose.from_planning_state(self.planning, probe_name)
        return pose.chain()

    # ---- dynamic transform for a scene node (may be identity) ----
    def dynamic_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        probe_name: Optional[str] = node.extras.get("pose_source_probe")
        if not probe_name:
            return TransformChain.new([AffineTransform.identity()])

        dyn = self._probe_chain(probe_name)

        # If the asset needs rotation about a local pivot (e.g., tip),
        # wrap the dynamic pose with +pivot / -pivot translations
        pivot = self.get_pivot_for_asset(node.asset_key)
        if pivot is not None:
            T_p = AffineTransform(rotation=np.eye(3), translation=np.asarray(pivot, float))
            T_m = AffineTransform(rotation=np.eye(3), translation=-np.asarray(pivot, float))
            return TransformChain.new([T_p, *dyn.elements, T_m])

        return dyn

    # ---- final world transform = base ∘ dynamic ----
    def world_chain_for_node(self, node: "NodeInstance") -> TransformChain:
        base = node.transform
        dyn = self.dynamic_chain_for_node(node)
        return TransformChain.new([*base.elements, *dyn.elements])

    def world_rt_for_node(self, node: "NodeInstance") -> tuple[np.ndarray, np.ndarray]:
        return self.world_chain_for_node(node).composed_transform


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    color_hex_str: str = "#C8C8C8"
    opacity: float = 1.0
    wireframe: bool = False
    visible: bool = True


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
    extras: dict[str, Any] = field(default_factory=dict)  # e.g., calibration_rt


@dataclass(slots=True)
class Scene:
    nodes: dict[str, NodeInstance] = field(default_factory=dict)

    def upsert(self, node: NodeInstance):
        self.nodes[node.id] = node

    def remove(self, node_id: str):
        self.nodes.pop(node_id, None)

    def by_tag(self, tag: str):
        return [n for n in self.nodes.values() if tag in n.tags]


def _load_calibration_bank(
    cal_file: CalibrationSourceModel, reticles: dict[str, CalibrationReticleModel]
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a calibration file that contains multiple probe entries.
    Return a dict mapping probe_code (string) -> (R,t).
    """
    # --- EXAMPLE STUBS (replace with real parsing) -------------------------
    if cal_file.directory:
        if cal_file.reticle is None:
            raise ValueError("Reticle model is required for directory calibration")
        reticle = reticles.get(cal_file.reticle)
        offset = np.array(reticle.offset_RAS, dtype=float)
        rotation = reticle.rotation_z
        cal_by_probe = fit_rotation_params_from_parallax(cal_file.directory, offset, rotation)[0]
    else:
        cal_by_probe = fit_rotation_params_from_manual_calibration(cal_file.file)[0]
    return {str(k): v for k, v in cal_by_probe.items()}


def _get_calibration_rt(
    calibrations: CalibrationsModel,
    reticles: dict[str, CalibrationReticleModel] = {},
) -> dict[str, "AffineTransform"]:
    """
    For each domain probe name, resolve (cal_id → path) then (probe_code → R,t).
    Cache each file load so it’s read once.
    """
    cal_files = calibrations.files
    probe_to_ref = calibrations.probe_to_ref
    cache: dict[str, dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    out: dict[str, AffineTransform] = {}

    for probe_name, ref in probe_to_ref.items():
        # load or reuse the bank
        if ref.cal_id not in cache:
            cal_file = cal_files[ref.cal_id]
            bank = _load_calibration_bank(cal_file, reticles)
            cache[ref.cal_id] = bank
        else:
            bank = cache[ref.cal_id]

        code = str(ref.probe_code)
        if code not in bank:
            # Clear error message showing available keys
            avail = ", ".join(sorted(bank.keys())[:8])
            raise KeyError(
                f"Calibration probe_code '{code}' not found in cal_id '{ref.cal_id}'. "
                f"Examples available: {avail}{' …' if len(bank) > 8 else ''}"
            )

        R, t = bank[code]
        out[probe_name] = AffineTransform(rotation=np.asarray(R, float), translation=np.asarray(t, float))

    return out


def _compile_collision_labels(labels_in_use: Iterable[str]) -> dict[str, int]:
    """
    Assign each collision label a bit. Bit 0 is reserved for 'NONE' (unused).
    """
    labels = [l for l in dict.fromkeys(labels_in_use) if l]  # unique, drop falsy
    mapping: dict[str, int] = {}
    for i, lab in enumerate(labels, start=1):  # start bits at 1
        mapping[lab] = 1 << i
    return mapping


def _apply_canonicalization_mesh(
    mesh: trimesh.Trimesh,
    source_space: str,
    unit_scale: float,
) -> trimesh.Trimesh:
    """
    Make a shallow copy of mesh in LPS mm.
    - Apply unit scaling.
    - Convert coordinate system to LPS.
    """
    m = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    if unit_scale != 1.0:
        m.vertices *= float(unit_scale)
    if source_space != "LPS":
        # convert_coordinate_system expects (N,3) array
        m.vertices = convert_coordinate_system(m.vertices, source_space, "LPS")
    return m


def _apply_canonicalization_points(
    pts: np.ndarray,
    source_space: str,
    unit_scale: float,
) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    if unit_scale != 1.0:
        p = p * float(unit_scale)
    if source_space != "LPS":
        p = convert_coordinate_system(p, source_space, "LPS")
    return p


def _transform_chain_from_ref(transforms_model, key: str | None) -> TransformChain:
    """
    Resolve a ConfigModel.transforms entry into a TransformChain.
    Falls back to identity when key is None.
    """
    if not key:
        return TransformChain.new([AffineTransform.identity()])

    ref = transforms_model.get(key)
    if ref is None:
        raise KeyError(f"Unknown transform_key '{key}'")

    if ref.kind == "identity":
        return TransformChain.new([AffineTransform.identity()])

    if ref.kind == "sitk_file":
        return TransformChain.new([AffineTransform.from_sitk_path(Path(ref.path), inverted=ref.invert)])

    raise ValueError(f"Unsupported transform kind: {ref.kind!r}")


# -------------------------------------------------------------------------
# Output bundle
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class CollisionLabelIndex:
    label_to_bit: dict[str, int]
    bit_to_label: dict[int, str]


@dataclass(frozen=True)
class RuntimeBundle:
    # Catalog after loading/canonicalizing
    asset_catalog: AssetCatalog  # runtime AssetSpec (with Mesh/PointsTransformable)
    targets_pts: dict[str, np.ndarray]  # key -> (N,3) points in LPS mm
    # Scene ready to render
    scene: Scene
    # Collision label bits (for adapters)
    collision_labels: CollisionLabelIndex
    plan_state: PlanningState


# -------------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------------


def _capabilities_from_list(lst) -> Capability:
    val = 0
    for c in lst or []:
        # if Pydantic parsed as Capability already, bit-or directly
        if isinstance(c, int):
            val |= c
        else:
            # string fallback
            val |= getattr(Capability, str(c))
    return Capability(val)


def _material_from_model(m: MaterialModel) -> Material:
    return Material(
        name=m.name,
        color_hex_str=m.color,
        opacity=m.opacity,
        wireframe=m.wireframe,
        visible=m.visible,
    )


def _collision_bits(policy: CollisionPolicyModel, label_to_bit: dict[str, int]) -> tuple[int, int]:
    group_bits = label_to_bit.get(policy.group or "", 0)
    mask_bits = 0
    for lab in policy.mask:
        mask_bits |= label_to_bit.get(lab, 0)
    return group_bits, mask_bits


def _resolve_canonicalization(
    spec: BaseSpecModel,
    cfg: ConfigModel,
) -> Optional[CanonicalizationRuntime]:
    # pick base: from ref, or inline, or safe default
    if spec.canonicalization_ref:
        try:
            base = cfg.canonicalizations[spec.canonicalization_ref]
        except KeyError:
            raise KeyError(f"Unknown canonicalization_ref '{spec.canonicalization_ref}' for '{spec.key}'")
    elif spec.canonicalization:
        base = spec.canonicalization
    else:
        base = CanonicalizationDefModel(source_space="LPS", scale_to_mm=1.0, applied=True)

    # overlay overrides (only provided fields)
    if spec.canonicalization_override:
        ov = spec.canonicalization_override
        base = base.model_copy(update={k: v for k, v in ov.model_dump().items() if v is not None})

    # materialize runtime CanonicalizationInfo
    # (compile transform_key → TransformChain here if you want it eager)
    tc = TransformChain.new([AffineTransform.identity()])
    if base.transform:
        tc = _transform_chain_from_ref(cfg.transforms, base.transform)

    return CanonicalizationRuntime(
        source_space=base.source_space,
        scale_to_mm=base.scale_to_mm,
        transform_file_to_canonical=tc,
        applied=base.applied,
        version=base.version,
        fingerprint=base.fingerprint,
    )


def _canon_from_model(c: CanonicalizationDefModel) -> CanonicalizationRuntime:
    # Keep transform_file_to_canonical as identity at runtime unless you actually bake/load it.
    return CanonicalizationRuntime(
        source_space=c.source_space,
        scale_to_mm=c.scale_to_mm,
        transform_file_to_canonical=TransformChain.new([AffineTransform.identity()]),
        applied=c.applied,
        version=c.version,
    )


def _base_spec_kwargs_from_model(
    m: BaseSpecModel,
    label_to_bit: dict[str, int],
) -> dict[str, Any]:
    group_bits, mask_bits = _collision_bits(m.collision, label_to_bit)
    return dict(
        key=m.key,
        kind=m.kind.value,  # "mesh" | "points" | "lines"
        role=m.role,  # keep enum if your runtime type expects it; else use m.role.value
        default_material=_material_from_model(m.default_material),
        metadata=dict(m.metadata),
        tags=set(m.tags),
        caps=_capabilities_from_list(m.caps),
        collidable_group=group_bits,
        collidable_mask=mask_bits,
        pivot_LPS=np.array(m.pivot_LPS, float) if m.pivot_LPS else None,
        bbox_hint=np.array(m.bbox_hint, float) if m.bbox_hint else None,
    )


# --- asset builder -----------------------------------------------------------


def build_asset_spec(a: AssetSpecModel, label_to_bit: dict[str, int], chem: ChemShiftContext) -> AssetSpec:
    base_kwargs = _base_spec_kwargs_from_model(a, label_to_bit)

    mesh_tf: MeshTransformable | None = None
    pts_tf: PointsTransformable | None = None

    if a.src and a.loader:
        loader_kwargs = a.loader_kwargs or {}
        geo = load_geometry(Path(a.src), loader=a.loader, **loader_kwargs)
        if a.kind == Kind.MESH:
            if not isinstance(geo, trimesh.Trimesh):
                raise TypeError(f"Asset '{a.key}' loader returned points but kind=MESH")
            canon_mesh = _apply_canonicalization_mesh(
                geo, a.canonicalization.source_space, a.canonicalization.scale_to_mm
            )
            if _should_apply_chem(a, chem):
                shifted_vertices = chem.points_transform.apply_to(canon_mesh.vertices)
                # TODO make sure inversion is correct
                canon_mesh = trimesh.Trimesh(vertices=shifted_vertices, faces=canon_mesh.faces, process=False)
            mesh_tf = MeshTransformable(canon_mesh)
        elif a.kind == Kind.POINTS:
            if isinstance(geo, trimesh.Trimesh):
                raise TypeError(f"Asset '{a.key}' loader returned mesh but kind=POINTS")
            pts = _apply_canonicalization_points(geo, a.canonicalization.source_space, a.canonicalization.scale_to_mm)
            if _should_apply_chem(a, chem):
                pts = chem.points_transform.apply_to(pts)
            pts_tf = PointsTransformable(pts)
        elif a.kind == Kind.LINES:
            raise NotImplementedError("kind='lines' not implemented in loader")

    return AssetSpec(
        **base_kwargs,
        source_path=Path(a.src) if a.src else None,
        loader=a.loader,
        mesh=mesh_tf,
        points=pts_tf,
    )


# --- target builder (returns spec + resolved points array) -------------------
CompiledTransforms = dict[str, AffineTransform]


def compile_all_transforms(transforms: dict[str, TransformRecipeModel]) -> CompiledTransforms:
    compiled: CompiledTransforms = {}
    for key, recipe in transforms.items():
        chain = compile_recipe_to_chain(recipe)
        R, t = chain.composed_transform
        compiled[key] = AffineTransform(R, t)
    return compiled


def resolve_transform_key_cached(key: Optional[str], cache: CompiledTransforms) -> AffineTransform:
    if not key:
        return AffineTransform.identity()
    try:
        return cache[key]
    except KeyError:
        raise KeyError(f"Unknown transform_key '{key}' (not found in compiled transforms)")


def resolve_transform_ref_cached(ref: Optional[TransformRefModel], cache: CompiledTransforms) -> AffineTransform:
    if ref is None:
        return AffineTransform.identity()
    if ref.key:
        return resolve_transform_key_cached(ref.key, cache)
    # Inline recipe: compile on the fly (not in cache by design)
    return compile_recipe_to_chain(ref.inline)  # type: ignore[arg-type]


def build_target_spec(
    t: TargetSpecModel,
    runtime_assets: dict[str, AssetSpec],
    label_to_bit: dict[str, int],
    chem: ChemShiftContext,
    reducer_registry: dict[str, Callable[..., np.ndarray]],
) -> tuple[TargetSpec, np.ndarray]:
    base_kwargs = _base_spec_kwargs_from_model(t, label_to_bit)

    # Targets must be non-collidable by default; enforce here (even if config forgot).
    base_kwargs["caps"] = Capability.RENDERABLE
    base_kwargs["collidable_group"] = 0
    base_kwargs["collidable_mask"] = 0

    # Resolve points (explicit file or derived by reducer)
    if t.src and t.loader:
        loader_kwargs = t.loader_kwargs or {}
        pts = load_geometry(Path(t.src), loader=t.loader, **loader_kwargs)
        if isinstance(pts, trimesh.Trimesh):
            raise TypeError(f"Target '{t.key}' loader returned mesh; expected points")
        pts = _apply_canonicalization_points(pts, t.canonicalization.source_space, t.canonicalization.scale_to_mm)
        if _should_apply_chem(t, chem):
            pts = chem.points_transform.apply_to(pts)
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
    else:
        # Derived: fetch source asset geometry first
        src_key = t.source_key or ""
        src_asset = runtime_assets.get(src_key)
        if src_asset is None:
            raise KeyError(f"Target '{t.key}' source_key '{src_key}' not found in loaded assets")
        src_geo = (
            src_asset.mesh.raw
            if src_asset.mesh is not None
            else (src_asset.points.raw if src_asset.points is not None else None)
        )
        if src_geo is None:
            raise ValueError(f"Target '{t.key}': source asset '{src_key}' has no geometry loaded")

        reducer_name = t.reducer or ""
        reducer_fn = reducer_registry.get(reducer_name)
        if reducer_fn is None:
            raise KeyError(f"Unknown target reducer '{reducer_name}' for target '{t.key}'")

        pt = reducer_fn(src_geo, **(t.reducer_kwargs or {}))  # should return (3,) or (1,3)
        pts = np.asarray(pt, dtype=np.float64).reshape(1, 3)

    spec = TargetSpec(
        **base_kwargs,
        kind="points",  # targets are points in runtime
        role=Role.TARGET,
        source_path=Path(t.src) if t.src else None,
        loader=t.loader,
        source_key=t.source_key,
        reducer=t.reducer,
        reducer_kwargs=dict(t.reducer_kwargs),
        points=PointsTransformable(pts),
        approach_vector=np.array(t.approach_vector, float) if t.approach_vector else None,
        uncertainty_mm=t.uncertainty_mm,
    )
    return spec, pts


def build_runtime_from_config(cfg: ConfigModel) -> RuntimeBundle:
    # 1) collision labels → bit mapping
    labels: list[str] = []
    for a in cfg.assets:
        if a.collision.group:
            labels.append(a.collision.group)
        labels.extend(a.collision.mask)
    for t in cfg.targets:
        if t.collision.group:
            labels.append(t.collision.group)
        labels.extend(t.collision.mask)
    label_to_bit = _compile_collision_labels(labels)
    bit_to_label = {v: k for k, v in label_to_bit.items()}
    label_index = CollisionLabelIndex(label_to_bit=label_to_bit, bit_to_label=bit_to_label)
    # 2) assets
    chem = build_chemshift_context(cfg)
    runtime_assets: dict[str, AssetSpec] = {}
    for a in cfg.assets:
        runtime_assets[a.key] = build_asset_spec(a, label_to_bit, chem)

    # 3) targets (specs + points index)
    runtime_targets: dict[str, TargetSpec] = {}
    targets_pts: dict[str, Float3] = {}
    for t in cfg.targets:
        tspec, pts = build_target_spec(t, runtime_assets, label_to_bit, chem, _REDUCER_REGISTRY)
        runtime_targets[tspec.key] = tspec
        targets_pts[tspec.key] = pts

    asset_catalog = AssetCatalog(assets=runtime_assets, targets=runtime_targets)

    # 4) scene
    scene = Scene()
    for n in cfg.scene.nodes:
        asset_key = n.asset
        if asset_key not in runtime_assets:
            raise KeyError(f"Scene node '{n.id}' references unknown asset '{asset_key}'")

        node_tf = _transform_chain_from_ref(cfg.transforms, n.transform)

        extras: dict[str, Any] = {}
        locked_axes: set[str] = set()
        if n.pose_source_probe:
            extras["pose_source_probe"] = n.pose_source_probe
            decl = cfg.plan.probes.get(n.pose_source_probe)
            if decl and decl.calibrated:
                locked_axes.update({"ap_tilt", "ml_tilt"})

        scene.upsert(
            NodeInstance(
                id=n.id,
                asset_key=asset_key,
                transform=node_tf,
                tags=set(n.tags),
                material_override=None,
                enabled=True,
                locked_axes=locked_axes,
                extras=extras,
            )
        )

    # 5) kinematics, calibrations, plans (build PlanningState)
    kinematics = Kinematics(arc_angles=dict(cfg.plan.arcs))
    calibrations = _get_calibration_rt(cfg.plan.calibrations, cfg.plan.reticles)
    probes: dict[str, ProbePlan] = {}
    for probe_name, probe_decl in cfg.plan.probes.items():
        probe_calibrated = probe_name in calibrations
        if probe_calibrated:
            ap, ml = find_probe_angle(calibrations[probe_name])
        else:
            ap = kinematics.get_arc(probe_decl.arc)
            ml = probe_decl.slider_ml
        probes[probe_name] = ProbePlan(
            probe_type=probe_decl.kind,
            arc_id=probe_decl.arc,
            bind_ap_to_arc=probe_calibrated,
            ap_local=ap,
            ml_local=ml,
            spin=probe_decl.spin,
            past_target_mm=probe_decl.past_target_mm,
            offsets_RA=probe_decl.offsets_RA,
            target_key=probe_decl.target,
            target_point_RAS=None,
            calibrated=probe_calibrated,
        )
    plan_state = PlanningState(
        kinematics=kinematics,
        probes=probes,
        calibrations=calibrations,
        target_index=targets_pts,
    )

    return RuntimeBundle(
        asset_catalog=asset_catalog,
        targets_pts=targets_pts,
        scene=scene,
        collision_labels=label_index,
        plan_state=plan_state,
    )


@dataclass(frozen=True)
class SetProbeLocalAngles:
    """Edit per-probe local angles (used when not bound to arc or when you unbind)."""

    name: str
    ap_local: Optional[float] = None  # deg
    ml_local: Optional[float] = None  # deg
    spin: Optional[float] = None  # deg


@dataclass(frozen=True)
class SetProbeOffsetsRA:
    """Set absolute R/A offsets (in mm)."""

    name: str
    R_mm: Optional[float] = None
    A_mm: Optional[float] = None


@dataclass(frozen=True)
class NudgeProbeOffsetsRA:
    """Nudge offsets (delta in mm)."""

    name: str
    dR_mm: float = 0.0
    dA_mm: float = 0.0


@dataclass(frozen=True)
class SetProbePastTarget:
    """Set relative depth (mm). Positive increases insertion past the target."""

    name: str
    past_target_mm: float


@dataclass(frozen=True)
class NudgeProbePastTarget:
    """Nudge depth (delta mm)."""

    name: str
    d_mm: float


@dataclass(frozen=True)
class SetProbeTarget:
    """
    Choose a target. Exactly one of target_key or target_point_RAS must be provided.
    Passing None clears that field; use to switch between the two forms.
    """

    name: str
    target_key: Optional[str] = None
    target_point_RAS: Optional[Tuple[float, float, float]] = None


# ---------- Arc & policy edits ----------


@dataclass(frozen=True)
class SetArcAngle:
    """Set AP angle for an arc (deg)."""

    arc_id: str
    ap_deg: float


@dataclass(frozen=True)
class AssignProbeArc:
    """Assign/unassign probe to an arc and optionally (un)bind AP to arc."""

    name: str
    arc_id: Optional[str]  # None = unassign from arc
    bind_ap_to_arc: Optional[bool] = None


@dataclass(frozen=True)
class BindProbeAPToArc:
    """Bind/unbind AP to the probe’s current arc."""

    name: str
    bind: bool
    freeze_effective_on_unbind: bool = True  # capture current AP into ap_local on unbind


@dataclass(frozen=True)
class SetProbeCalibrated:
    """Mark plan as 'use calibration if available'."""

    name: str
    calibrated: bool


PlanningCommand = Union[
    SetProbeLocalAngles,
    SetProbeOffsetsRA,
    NudgeProbeOffsetsRA,
    SetProbePastTarget,
    NudgeProbePastTarget,
    SetProbeTarget,
    SetArcAngle,
    AssignProbeArc,
    BindProbeAPToArc,
    SetProbeCalibrated,
]


def apply_planning_command(ps: PlanningState, cmd: PlanningCommand) -> List[str]:
    """
    Mutates PlanningState in place.
    Returns a list of probe names that should be re-resolved/re-rendered.
    """
    changed: Set[str] = set()

    if isinstance(cmd, SetArcAngle):
        # clamp to limits (and apply any separation policy if you added it)
        ap = ps.kinematics.set_arc(cmd.arc_id, cmd.ap_deg)
        # any non-calibrated probe bound to this arc is affected
        for name, plan in ps.probes.items():
            if plan.arc_id == cmd.arc_id and plan.bind_ap_to_arc:
                # calibrated probes ignore arc changes
                if not (plan.calibrated and name in ps.calibrations):
                    changed.add(name)
        return sorted(changed)

    if isinstance(cmd, AssignProbeArc):
        plan = ps.probes[cmd.name]
        plan.arc_id = cmd.arc_id
        if cmd.bind_ap_to_arc is not None:
            plan.bind_ap_to_arc = bool(cmd.bind_ap_to_arc)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, BindProbeAPToArc):
        plan = ps.probes[cmd.name]
        if cmd.bind:
            plan.bind_ap_to_arc = True
        else:
            # freeze current effective AP into ap_local so unbinding doesn't jump
            if cmd.freeze_effective_on_unbind:
                eff_ap, _, _ = _resolved_angles(cmd.name, ps)  # helper from earlier reply
                plan.ap_local = eff_ap
            plan.bind_ap_to_arc = False
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeCalibrated):
        plan = ps.probes[cmd.name]
        plan.calibrated = bool(cmd.calibrated)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeLocalAngles):
        plan = ps.probes[cmd.name]
        if cmd.ap_local is not None:
            plan.ap_local = ps.kinematics.limits.ap_deg.clamp(cmd.ap_local)
        if cmd.ml_local is not None:
            plan.ml_local = ps.kinematics.limits.ml_deg.clamp(cmd.ml_local)
        if cmd.spin is not None:
            plan.spin = ps.kinematics.limits.spin_deg.clamp(cmd.spin)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeOffsetsRA):
        plan = ps.probes[cmd.name]
        R, A = plan.offsets_RA
        if cmd.R_mm is not None:
            R = float(cmd.R_mm)
        if cmd.A_mm is not None:
            A = float(cmd.A_mm)
        plan.offsets_RA = (R, A)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, NudgeProbeOffsetsRA):
        plan = ps.probes[cmd.name]
        R, A = plan.offsets_RA
        plan.offsets_RA = (R + float(cmd.dR_mm), A + float(cmd.dA_mm))
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbePastTarget):
        plan = ps.probes[cmd.name]
        plan.past_target_mm = float(cmd.past_target_mm)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, NudgeProbePastTarget):
        plan = ps.probes[cmd.name]
        plan.past_target_mm = float(plan.past_target_mm + cmd.d_mm)
        changed.add(cmd.name)
        return sorted(changed)

    if isinstance(cmd, SetProbeTarget):
        plan = ps.probes[cmd.name]
        # ensure exactly one is set
        if (cmd.target_key is None) == (cmd.target_point_RAS is None):
            raise ValueError("SetProbeTarget: specify exactly one of target_key or target_point_RAS")
        plan.target_key = cmd.target_key
        plan.target_point_RAS = cmd.target_point_RAS
        changed.add(cmd.name)
        return sorted(changed)

    # Unknown command: no-op
    return sorted(changed)


# --- Reducer extension -----------------------------------------------------


@dataclass(frozen=True)
class ViewMaterial:
    color: int
    opacity: float
    wireframe: bool
    visible: bool


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
    by_node: dict[str, List[OverlaySpec]] = field(default_factory=dict)

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
    assets: AssetCatalog
    cache: _TransformCache = _TransformCache(maxsize=256)
    overlays: OverlayResolver | None = None

    # ----- public API -----
    def build(self, plan: PlanningState, coll: CollisionState | None = None) -> None:
        hot = coll.hot if coll else frozenset()
        for node in self.scene.nodes.values():
            self._upsert_node(node, plan, node.id in hot)

    def sync_nodes(self, plan: PlanningState, nodes: Iterable[Node], coll: CollisionState | None = None) -> None:
        hot = coll.hot if coll else frozenset()
        for node in nodes:
            self._upsert_node(node, plan, node.id in hot)

    def remove(self, node_ids: Iterable[str]) -> None:
        self.backend.remove(node_ids)

    def invalidate_mesh_key(self, mesh_key: str) -> None:
        """Call this if a base mesh topology changes (forces re-transform)."""
        self.cache.invalidate_mesh(mesh_key)

    # ----- internals -----
    def _upsert_node(self, node: Node, plan: PlanningState, colliding: bool) -> None:
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

    def _pose_for_node(self, node: Node, plan: PlanningState) -> Tuple[np.ndarray, np.ndarray]:
        if node.id.startswith("probe:"):
            pname = node.id.split(":", 1)[1]
            return plan.probes[pname].pose.chain().composed_transform
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _resolve_mesh(self, key: str, a: AssetCatalog) -> trimesh.Trimesh:
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

    def _resolve_points(self, key: str, a: AssetCatalog) -> np.ndarray:
        raise KeyError(f"Unknown points key: {key}")


@dataclass
class K3DBackend(RenderBackend):
    plot: k3d.Plot
    _handles: dict[str, Any] = field(default_factory=dict)
    _kinds: dict[str, str] = field(default_factory=dict)  # 'mesh'|'points'

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


Subscriber = Callable[[PlanningState, List[str]], None]


class PlanStore:
    def __init__(self, initial: PlanningState):
        self._state = initial
        self._subs: List[Subscriber] = []

    @property
    def state(self) -> PlanningState:
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

    def dispatch(self, cmd: PlanningCommand) -> None:
        changed = apply_planning_command(self._state, cmd)  # your existing pure function
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
        self._latest_state: Optional[PlanningState] = None

    def __call__(self, state: PlanningState, changed_ids: List[str]) -> None:
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
    _node_to_obj: dict[str, fcl.CollisionObject] = field(default_factory=dict)
    _geomid_to_node: dict[int, str] = field(default_factory=dict)  # id(CollisionGeometry) -> node_id
    _node_to_geomid: dict[str, int] = field(default_factory=dict)  # node_id -> id(CollisionGeometry)

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
        extra_map: Optional[dict[int, str]] = None,
    ) -> List[CollisionPair]:
        gid_to_name: dict[int, str] = dict(self._geomid_to_node)
        if extra_map:
            gid_to_name.update(extra_map)

        groups: dict[Tuple[str, str], List[Contact]] = {}
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
    assets: AssetCatalog
    include: Callable[[Node], bool] = default_include

    # ---- lifecycle wiring ----
    def rebuild(self, plan: PlanningState) -> None:
        specs = [s for n in self.scene.nodes.values() if self.include(n) for s in [self._spec_for_node(n, plan)] if s]
        self.backend.rebuild(specs)

    def on_store_change(self, plan: PlanningState, changed_probe_names: List[str]) -> None:
        # Only probes move; map probe names -> scene nodes
        nodes: List[Node] = []
        for pname in changed_probe_names:
            nid = f"probe:{pname}"
            node = self.scene.nodes.get(nid)
            if node and self.include(node):
                nodes.append(node)
        if not nodes:
            return
        specs = [self._spec_for_node(n, plan) for n in nodes]
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
    def _spec_for_node(self, node: Node, plan: PlanningState) -> Optional[ObjSpec]:
        if node.geom.kind != "mesh":
            return None
        base = self._resolve_mesh(node.geom.key, self.assets)
        bvh = _bvh_from_mesh(base, name=node.geom.key)  # unique geometry per node
        R, t = self._pose_for_node(node, plan)
        tf = _rt_to_transform(R, t, name=f"pose:{node.id}")
        return ObjSpec(node_id=node.id, geom=bvh, transform=tf)

    def _pose_for_node(self, node: Node, plan: PlanningState) -> Tuple[np.ndarray, np.ndarray]:
        if node.id.startswith("probe:"):
            pname = node.id.split(":", 1)[1]
            return plan.probes[pname].pose.chain().composed_transform
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _resolve_mesh(self, key: str, a: AssetCatalog) -> trimesh.Trimesh:
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
    store: PlanStore
    on_event: Callable[[PlanningState, List[str]], None]

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

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
        # map probe ids → scene nodes; extend as needed
        nodes = [self.scene.nodes.get(f"probe:{pid}") for pid in changed_ids]
        nodes = [n for n in nodes if n is not None]
        # let the adapter apply overlays if provided
        self.adapter.sync_nodes(plan, nodes, coll=self.get_collision_state() if self.get_collision_state else None)


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
    on_state_changed: Optional[Callable[[CollisionState, Set[str], PlanningState], None]] = None
    _prev_state: CollisionState | None = None

    def __call__(self, plan: PlanningState, changed_ids: List[str]) -> None:
        # keep backend up-to-date for moved probes
        moved = [pid for pid in changed_ids if f"probe:{pid}" in self.scene.nodes]
        if moved:
            self.adapter.on_store_change(plan, moved)

        # recompute collisions
        pairs = self.adapter.collide_internal(enable_contacts=False)
        new_pairs = {(p.id1, p.id2) for p in pairs}
        new_state = self.state.replace(new_pairs)

        # notify only if something flipped
        flips = _diff_hot(new_state, self._prev_state)
        self._prev_state = new_state
        self.state = new_state
        if flips and self.on_state_changed:
            self.on_state_changed(new_state, flips, plan)


def on_collisions_changed_lambda(renderer_adapter: RendererAdapter, scene: Scene, overlays_state: OverlayState):
    def _on_collisions_changed(state: CollisionState, flips: Set[str], plan: PlanningState) -> None:
        # update overlays by source "collision"
        overlays_state.clear_source("collision")
        if state.hot:
            spec = OverlaySpec(color=0xFF0000, alpha=0.65, source="collision", priority=30)
            overlays_state.set_for_source(list(state.hot), spec)

        # repaint only nodes whose hot/cold status flipped
        nodes = [scene.nodes[nid] for nid in flips if nid in scene.nodes]
        if nodes:
            renderer_adapter.sync_nodes(plan, nodes)  # adapter reads overlays internally

    return _on_collisions_changed


@dataclass
class ProbeWidgetController:
    data: AppData
    store: PlanStore
    assets: AssetCatalog
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
        return self.data.plan.probe_info[probe_name].arc

    def _arc_angle(self, arc_id: str) -> float:
        return float(self.data.plan.arcs[arc_id])

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
            SetProbePlanPose(
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
            SetProbePlanPose(
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
        self.store.dispatch(SetArcAngle(arc_id=arc_id, angle_deg=float(self.ap_tilt_deg.value)))

    # ---------- load UI from domain ----------
    def _load_probe_into_widgets(self, probe_name: str):
        p = self.store.state.probes[probe_name]
        arc_id = self._probe_arc_id(probe_name)

        # keep options in sync (in case arcs were added/removed elsewhere)
        self.arc_assign_dd.options = sorted(self.data.plan.arcs.keys())
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
            options=sorted(self.data.plan.arcs.keys()),
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
# %%
assets = AssetCatalog.from_app_config(app_config)
# %%
data = AppData(assets=assets, plan=probe_config)
plan = PlanningState.from_app_data(data, ignore_calibrations=True)
scene = scene_from_app(data, plan)
store = PlanStore(plan)
overlays_state = OverlayState()
overlays_resolver = OverlayResolver(overlays_state)
# backends
plot = k3d.plot(grid_visible=True, background_color=0xFFFFFF)
fcl_backend = FCLBackend()
k3d_backend = K3DBackend(plot=plot)
render_adapter = RendererAdapter(backend=k3d_backend, scene=scene, assets=assets, overlays=overlays_resolver)
collision_adapter = CollisionAdapter(backend=fcl_backend, scene=scene, assets=assets)

# Initial build
render_adapter.build(plan)
collision_adapter.rebuild(plan)

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
