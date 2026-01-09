"""Utility functions for resolving project paths."""

from pathlib import Path


def get_project_root() -> Path:
    """
    Find project root by locating pyproject.toml.

    This function searches for the project root by looking for pyproject.toml
    in the current working directory or by walking up from this file's location.
    This works from scripts, notebooks, and installed packages.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If pyproject.toml cannot be found.
    """
    # Try current working directory first (useful for notebooks and scripts)
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd

    # Otherwise, walk up from this file's location
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Could not find project root. Make sure pyproject.toml exists "
        "in the project root directory."
    )


# Cache the project root to avoid repeated lookups
PROJECT_ROOT = get_project_root()
