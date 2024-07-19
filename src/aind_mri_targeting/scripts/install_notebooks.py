"""
Command line script to install notebooks
"""

import argparse
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the center of mass of the headframes"
    )
    parser.add_argument("output", nargs="?", help="path to the output file")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        output = os.getcwd()
    else:
        output = args.output
    output_dir = Path(output) / "notebooks"
    notebook_dir = (
        Path(__file__).resolve().parent.parent.parent.parent / "notebooks"
    )
    shutil.copytree(str(notebook_dir), str(output_dir))
    return 0


if __name__ == "__main__":
    main()
