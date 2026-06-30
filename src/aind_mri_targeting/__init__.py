"""Tools to plan and execute MRI-guided targeting experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-mri-targeting")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
