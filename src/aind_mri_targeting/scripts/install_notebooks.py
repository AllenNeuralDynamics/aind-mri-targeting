"""
Command line script to install notebooks
"""

import argparse

from .. import util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install notebooks into a specified directory"
    )
    parser.add_argument("output", nargs="?", help="path to the output file")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="overwrite files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    util.install_notebooks(args.output, force=args.force)
    return 0


if __name__ == "__main__":
    main()
