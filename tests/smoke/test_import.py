"""Fresh-process smoke tests for package and lifecycle imports."""

import subprocess
import sys

from market_rank import __version__


def test_package_imports_with_expected_version() -> None:
    """The package should import from the configured source layout."""
    assert __version__ == "0.1.0"


def test_qualification_imports_in_a_fresh_process_without_cycles() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import market_rank.qualification"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_import_does_not_eagerly_load_lightgbm() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import market_rank.cli; assert 'lightgbm' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_runtime_launcher_reaches_cli_in_a_fresh_process() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "market_rank.launcher", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage: market-rank" in completed.stdout
