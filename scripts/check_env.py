from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


SUPPORTED_PYTHONS = {(3, 11), (3, 12), (3, 13), (3, 14)}

CORE_IMPORTS = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "tqdm",
    "trajectories_of_change",
)

OPTIONAL_IMPORTS = {
    "plotting": ("kaleido", "plotly"),
    "provenance": ("yaml",),
    "notebooks": ("ipykernel", "notebook"),
}


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def main() -> int:
    executable = Path(sys.executable).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info[:2] not in SUPPORTED_PYTHONS:
        errors.append(
            "Python version mismatch: "
            "expected one of 3.11, 3.12, 3.13, or 3.14, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )

    missing_core = _missing_modules(CORE_IMPORTS)
    if missing_core:
        errors.append("Missing core imports: " + ", ".join(missing_core))

    for extra, modules in OPTIONAL_IMPORTS.items():
        missing = _missing_modules(modules)
        if missing:
            warnings.append(f"Optional extra '{extra}' not fully available: {', '.join(missing)}")

    if errors:
        print("Environment check failed:")
        for error in errors:
            print(f"- {error}")
        for warning in warnings:
            print(f"- warning: {warning}")
        print(f"- sys.executable: {executable}")
        print()
        print("Recommended development setup:")
        print("- uv sync --all-extras --group dev")
        print("- uv run python scripts/check_env.py")
        return 1

    print("Environment check passed.")
    print(f"- python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"- sys.executable: {executable}")
    if warnings:
        print("Optional extras:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
