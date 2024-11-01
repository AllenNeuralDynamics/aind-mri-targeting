"""
Command line script to calculate the center of mass of headframe segments
"""

import argparse

from .. import headframe_rotations as hr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the center of mass of the headframes"
    )
    parser.add_argument("mri", help="path to the MRI file")
    parser.add_argument("segmentation", help="path to the segmentation file")
    parser.add_argument("output", nargs="?", help="path to the output file")
    parser.add_argument("-m", "--mouse", default=None, help="mouse ID")
    parser.add_argument(
        "-s", "--segment_format", default=None, help="segment name format"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="force overwrite",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        nargs="+",
        default=[],
        help="list of segment names to ignore",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hr.headframe_centers_of_mass(
        args.mri,
        args.segmentation,
        args.output,
        mouse_id=args.mouse,
        segment_format=args.segment_format,
        force=args.force,
        ignore_list=args.ignore,
    )
    return 0


if __name__ == "__main__":
    main()
