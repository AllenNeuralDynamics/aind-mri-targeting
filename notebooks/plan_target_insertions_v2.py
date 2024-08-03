from aind_mri_targeting.planning import (
    candidate_insertions,
    valid_insertion_pairs,
    transform_matrix_from_angles_and_target,
    get_implant_targets,
    make_final_insertion_scene,
    make_scene_for_insertion,
    test_for_collisions,
    plan_insertion,
)

from pathlib import Path
import trimesh

mouse = "743700"
whoami = "galen"
if whoami == "galen":
    base_dir = Path("/mnt/aind1-vast/scratch/")
    base_save_dir = Path("/home/galen.lynch/")
elif whoami == "yoni":
    base_dir = Path("Y:/")
    base_save_dir = Path(
        "C:/Users/yoni.browning/OneDrive - Allen Institute/Desktop/"
    )
else:
    raise ValueError("Who are you again?")

headframe_model_dir = base_dir / "ephys/persist/data/MRI/HeadframeModels/"
probe_model_file = (
    headframe_model_dir / "dovetailtweezer_oneShank_centered_corrected.obj"
)  # "modified_probe_holder.obj"
annotations_path = base_dir / "ephys/persist/data/MRI/processed/{}/UW2".format(
    mouse
)

headframe_path = headframe_model_dir / "TenRunHeadframe.obj"
holes_path = headframe_model_dir / "OneOff_HolesOnly.obj"


implant_holes_path = str(
    annotations_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
)

image_path = str(
    annotations_path / "{}_100.nii.gz".format(mouse)
)  # '_100.nii.gz'))
labels_path = str(
    annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse)
)  # 'Segmentation.seg.nrrd')#
brain_mask_path = str(
    annotations_path / ("{}_auto_skull_strip.nrrd".format(mouse))
)
manual_annotation_path = str(
    annotations_path / (f"{mouse}_ManualAnnotations.fcsv")
)
cone_path = (
    base_dir
    / "ephys/persist/Software/PinpointBuilds/WavefrontFiles/Cone_0160-200-53.obj"
)

uw_yoni_annotation_path = (
    annotations_path / f"targets-{mouse}-transformed.fcsv"
)

newscale_file_name = headframe_path / "Centered_Newscale_2pt0.obj"
#


calibration_filename = "calibration_info_np2_2024_04_22T11_15_00.xlsx"
calibration_dir = (
    base_dir / "ephys/persist/data/probe_calibrations/CSVCalibrations/"
)
calibration_file = calibration_dir / calibration_filename
measured_hole_centers = annotations_path / "measured_hole_centers.xlsx"


# manual_hole_centers_file = annotations_path / 'hole_centers.mrk.json'

transformed_targets_save_path = annotations_path / (
    mouse + "TransformedTargets.csv"
)
test_probe_translation_save_path = str(
    base_save_dir / "test_probe_translation.h5"
)
transform_filename = str(annotations_path / (mouse + "_com_plane.h5"))
