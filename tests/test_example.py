"""Example test module."""

import aind_mri_targeting


def test_version():
    """Test that version is defined."""
    assert aind_mri_targeting.__version__ is not None
    assert isinstance(aind_mri_targeting.__version__, str)
