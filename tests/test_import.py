"""Import smoke tests for the ehrfm package."""

import ehrfm


def test_import_exposes_version() -> None:
    """The package exposes a non-empty version string."""
    assert isinstance(ehrfm.__version__, str)
    assert ehrfm.__version__ != ""
