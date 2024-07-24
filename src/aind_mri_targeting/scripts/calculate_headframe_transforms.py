"""
Command line script to calculate find candidate headframe transforms
"""

import argparse

from .. import headframe_rotations as hr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate headframe transforms from MRI and segmentation"
    )
    parser.add_argument("mri", help="path to the MRI file")
    parser.add_argument("segmentation", help="path to the segmentation file")
    parser.add_argument("lower_plane", help="path to the lower plane file")
    parser.add_argument("output", nargs="?", help="path to the output file")
    parser.add_argument("-m", "--mouse", default=None, help="mouse ID")
    parser.add_argument(
        "-s", "--segment_format", default=None, help="segment name format"
    )
    parser.add_argument(
        "-v",
        "--volume-transform",
        action="store_true",
        default=True,
        help="write transform for volume",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="force overwrite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hr.calculate_headframe_transforms(
        args.mri,
        args.segmentation,
        args.lower_plane,
        args.output,
        mouse_name=args.mouse,
        segment_format=args.segment_format,
        force=args.force,
        volume_transforms=args.volume_transform,
    )
    return 0


if __name__ == "__main__":
    main()