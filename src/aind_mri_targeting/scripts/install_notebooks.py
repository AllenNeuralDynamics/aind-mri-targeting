"""
Command line script to install notebooks
"""

import argparse
import os
import shutil
from pathlib import Path
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the center of mass of the headframes"
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
    if args.output is None:
        output = os.getcwd()
    else:
        output = args.output
    output_dir = Path(output) / "notebooks"
    notebook_dir = (
        Path(__file__).resolve().parent.parent.parent.parent / "notebooks"
    )
    files_to_write = glob.glob(str(notebook_dir / "*.py"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files_in_output = set(os.listdir(output_dir))
    for file in files_to_write:
        filename = Path(file).name
        if filename in files_in_output and not args.force:
            print(f"Skipping {filename}")
            continue
        print(f"Copying {filename}")
        shutil.copy2(str(file), str(output_dir))
    return 0


if __name__ == "__main__":
    main()
