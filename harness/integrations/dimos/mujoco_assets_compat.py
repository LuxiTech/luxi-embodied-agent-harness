"""Bind pinned DimOS' two MuJoCo modules to local, asset-only dependencies.

Execute the original module code with a scoped import function. No upstream file
or global package is replaced, and model/controller functions remain unchanged.
Install before importing either target, in both the CLI and simulator process.
"""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import sys
from types import SimpleNamespace

from harness.robots.g1.mujoco import local_assets


_TARGETS = frozenset({
    "dimos.robot.unitree.mujoco_connection",
    "dimos.simulation.mujoco.model",
})


def _asset_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == "mujoco_playground._src" and tuple(fromlist) == ("mjx_env",):
        return SimpleNamespace(mjx_env=local_assets)
    if level == 0 and name == "dimos.utils.data" and tuple(fromlist) == ("get_data",):
        return SimpleNamespace(get_data=local_assets.get_data)
    return builtins.__import__(name, globals, locals, fromlist, level)


class _AssetLoader:
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": _asset_import}
        self.loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _AssetFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _AssetLoader(spec.loader)
        return spec


def install_mujoco_asset_compat() -> None:
    if any(isinstance(finder, _AssetFinder) for finder in sys.meta_path):
        return
    loaded = _TARGETS.intersection(sys.modules)
    if loaded:
        raise RuntimeError(f"MuJoCo asset compatibility must be installed before importing {sorted(loaded)}")
    sys.meta_path.insert(0, _AssetFinder())
