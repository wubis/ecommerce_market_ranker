"""Smoke tests for the initial package skeleton."""

from market_rank import __version__


def test_package_imports_with_expected_version() -> None:
    """The package should import from the configured source layout."""
    assert __version__ == "0.1.0"
