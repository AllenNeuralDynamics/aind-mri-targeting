"""
Command line script to calculate the transformation matrix to fit the implant
"""

import argparse

from aind_mri_targeting.implant_rotations import fit_implant_to_mri_from_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the volume transformation matrix of an implant"
    )
    parser.add_argument("segmentation", help="path to the segmentation file")
    parser.add_argument("hole_dir", help="directory containing the hole files")
    parser.add_argument("output", help="path to the output file")
    parser.add_argument(
        "-F",
        "--forward",
        action="store_true",
        default=False,
        help="save the forward transformation matrix instead of the inverse",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="force overwrite",
    )
    parser.add_argument("-m", "--mouse", default=None, help="mouse ID")
    return parser.parse_args()


def main():
    args = parse_args()
    fit_implant_to_mri_from_files(
        args.segmentation,
        args.hole_dir,
        save_name=args.output,
        save_inverse=not args.forward,
        force=args.force,
        mouse_name=args.mouse,
    )
    return 0


if __name__ == "__main__":
    main()
