"""Read prepared MuJoCo assets without importing the MJX training stack.

These are the three asset-only APIs used from mjx_env by pinned DimOS. Asset
installation remains scripts/bootstrap_dimos.sh's job, never a runtime download.
"""

from __future__ import annotations

import importlib.util
from glob import iglob
import os
from pathlib import Path


def _missing(path: object) -> FileNotFoundError:
    return FileNotFoundError(
        f"Prepared MuJoCo asset is missing: {path}. "
        "Run scripts/bootstrap_dimos.sh with DIMOS_FETCH_SIM_ASSETS=1 before starting the dashboard."
    )


def menagerie_path() -> Path:
    # Looking up a top-level package spec does not execute its __init__.py.
    spec = importlib.util.find_spec("mujoco_playground")
    if spec is None or spec.origin is None:
        raise _missing("mujoco_playground package / Menagerie assets")
    return Path(spec.origin).parent / "external_deps" / "mujoco_menagerie"


def __getattr__(name: str) -> Path:
    if name == "MENAGERIE_PATH":
        return menagerie_path()
    raise AttributeError(name)


def ensure_menagerie_exists() -> None:
    root = menagerie_path()
    for robot in ("unitree_go1", "unitree_g1"):
        assets = root / robot / "assets"
        if not assets.is_dir() or not any(p.is_file() for p in assets.iterdir()):
            raise _missing(assets)


def get_data(name: str | Path) -> Path:
    """Resolve only preinstalled data; do not invoke upstream LFS/clone helpers."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("MuJoCo asset names must be relative to the prepared data directory")
    configured = os.environ.get("DIMOS_UPSTREAM_DIR")
    if configured:
        upstream = Path(configured)
    else:
        spec = importlib.util.find_spec("dimos")
        if spec is None or spec.origin is None:
            raise _missing("DimOS installation")
        upstream = Path(spec.origin).parent.parent
    path = upstream / "data" / relative
    if not path.exists():
        raise _missing(path)
    return path


def update_assets(
    assets: dict[str, bytes], path: str | Path, glob: str = "*", recursive: bool = False
) -> None:
    """Match mjx_env's basename keys, globbing, recursion and overwrite order."""
    directory = Path(path)
    if not directory.is_dir():
        raise _missing(directory)
    # etils' local backend uses glob.glob, which excludes dotfiles unless the
    # pattern explicitly requests them. Path.glob would include macOS ._ assets.
    for match in iglob(str(directory / glob)):
        file = Path(match)
        if file.is_file():
            assets[file.name] = file.read_bytes()
        elif file.is_dir() and recursive:
            update_assets(assets, file, glob, recursive)
