"""
utilities
"""

import os
import shutil
from pathlib import Path
import glob


def install_notebooks(output=None, force=False):
    """
    Install notebooks by copying them to the specified output directory.

    Parameters
    ----------
    output : str or None, optional
        The directory where the notebooks should be installed. If None, the
        current working directory is used.  The default is None.
    force : bool, optional
        If True, overwrite existing files in the output directory. If False,
        skip copying if the file already exists.  The default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    - The function looks for notebooks in the 'notebooks' directory, which is
      three levels up from the location of this script.
    - If the output directory does not exist, it will be created.
    - Only files with a `.py` extension in the source 'notebooks' directory
      will be copied.
    - Existing files in the output directory will be skipped unless `force` is
      set to True.

    Examples
    --------
    Install notebooks to the current working directory, creating a 'notebooks'
    directory if it doesn't exist:

    >>> install_notebooks()

    Install notebooks to a specified directory:

    >>> install_notebooks(output='/path/to/destination')

    Force overwrite existing notebooks in the specified directory:

    >>> install_notebooks(output='/path/to/destination', force=True)
    """
    if output is None:
        output = os.getcwd()
    output_dir = Path(output) / "notebooks"
    notebook_dir = Path(__file__).resolve().parent.parent.parent / "notebooks"
    files_to_write = glob.glob(str(notebook_dir / "*.py"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files_in_output = set(os.listdir(output_dir))
    for file in files_to_write:
        filename = Path(file).name
        if filename in files_in_output and not force:
            print(f"Skipping {filename}")
            continue
        print(f"Copying {filename}")
        shutil.copy2(str(file), str(output_dir))
    return
