"""Native-runtime-safe launcher for the MarketRank CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_LIBOMP_CANDIDATES = (
    Path("/opt/homebrew/opt/libomp/lib"),
    Path("/usr/local/opt/libomp/lib"),
)
_REEXEC_MARKER = "MARKET_RANK_LIBOMP_REEXEC"


def _homebrew_libomp_dir() -> Path:
    for candidate in _LIBOMP_CANDIDATES:
        if (candidate / "libomp.dylib").is_file():
            return candidate
    raise RuntimeError(
        "Homebrew libomp is required on macOS; install it with `brew install libomp`"
    )


def _reexec_with_shared_libomp() -> None:
    if sys.platform != "darwin":
        return
    libomp_dir = _homebrew_libomp_dir()
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    entries = tuple(entry for entry in current.split(os.pathsep) if entry)
    if str(libomp_dir) in entries:
        return
    if os.environ.get(_REEXEC_MARKER) == "1":
        raise RuntimeError("macOS removed DYLD_LIBRARY_PATH during the libomp relaunch")
    environment = os.environ.copy()
    environment["DYLD_LIBRARY_PATH"] = os.pathsep.join((str(libomp_dir), *entries))
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        sys.executable,
        (sys.executable, "-m", "market_rank.cli", *sys.argv[1:]),
        environment,
    )


def main() -> int:
    """Relaunch once with one macOS OpenMP runtime, then enter the real CLI."""
    try:
        _reexec_with_shared_libomp()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from market_rank.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
