from __future__ import annotations

from pathlib import Path


REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "requirements.txt"


def raise_missing_dependency_error(
    exc: ModuleNotFoundError,
    *,
    script_name: str,
) -> None:
    missing_module = exc.name or "unknown"
    message = (
        f"Missing Python dependency '{missing_module}' while starting {script_name}.\n\n"
        f"Install the project requirements from the repository root:\n"
        f"  python3 -m pip install -r {REQUIREMENTS_PATH.name}\n\n"
        "If installation fails in Python 3.13 on your machine, create a Python 3.12 virtual environment "
        "and install the same requirements there, because that is the runtime used to validate these scripts."
    )
    raise ModuleNotFoundError(message) from exc
