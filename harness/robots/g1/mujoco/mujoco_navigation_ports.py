"""Thin MuJoCo adapters for the shared terminal navigation Skills."""

from __future__ import annotations

import math
from pathlib import Path
from threading import RLock
import time
from typing import Any
import unicodedata

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.perception.spatial_memory_spec import SpatialMemorySpec
from dimos.types.robot_location import RobotLocation

from harness.robots.g1.isaac.isaac_protocol import SCHEMA_VERSION, atomic_write_json, read_json
from harness.robots.g1.isaac.location_tagging import configured_tagged_locations_path


class MujocoLocationTagConfig(ModuleConfig):
    storage_path: Path = Field(default_factory=configured_tagged_locations_path)
    max_locations: int = Field(default=256, ge=1, le=4096)


class MujocoExplorationCostmapPort(Module):
    """Expose the existing unknown-preserving map under the shared Port name.

    This adapter does not copy, inflate, clear or otherwise reinterpret cells.
    The shared frontier selector therefore sees exactly the same live map as
    the existing MuJoCo planner, and this module owns no command output.
    """

    global_costmap: In[OccupancyGrid]
    exploration_costmap: Out[OccupancyGrid]

    @staticmethod
    def forward(costmap: OccupancyGrid) -> OccupancyGrid:
        """Keep object identity so the adapter cannot mutate map semantics."""

        return costmap

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(
            Disposable(
                self.global_costmap.subscribe(
                    lambda costmap: self.exploration_costmap.publish(
                        self.forward(costmap)
                    )
                )
            )
        )


class MujocoLocationTagSkillContainer(Module):
    """Retain the legacy MuJoCo ``tag_location`` API without its async nav."""

    config: MujocoLocationTagConfig
    odom: In[PoseStamped]
    _spatial_memory: SpatialMemorySpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._lock = RLock()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))

    def _on_odom(self, odom: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = odom

    @staticmethod
    def _normalized_name(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        name = unicodedata.normalize("NFKC", value).strip()
        if not name or len(name) > 80:
            return None
        if any(unicodedata.category(character).startswith("C") for character in name):
            return None
        return name

    def _persist_exact_pose(self, name: str, odom: PoseStamped) -> bool:
        path = Path(self.config.storage_path).expanduser()
        payload = read_json(path, max_bytes=256 * 1024) if path.exists() else None
        if payload is None:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "backend": "mujoco",
                "locations": {},
            }
        locations = payload.get("locations")
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("backend") != "mujoco"
            or not isinstance(locations, dict)
            or (name.casefold() not in locations and len(locations) >= self.config.max_locations)
        ):
            return False
        values = (
            odom.position.x,
            odom.position.y,
            odom.position.z,
            odom.orientation.x,
            odom.orientation.y,
            odom.orientation.z,
            odom.orientation.w,
            odom.ts,
        )
        if not all(math.isfinite(float(value)) for value in values):
            return False
        locations[name.casefold()] = {
            "name": name,
            "frame_id": "world",
            "position": [
                float(odom.position.x),
                float(odom.position.y),
                float(odom.position.z),
            ],
            "quaternion_xyzw": [
                float(odom.orientation.x),
                float(odom.orientation.y),
                float(odom.orientation.z),
                float(odom.orientation.w),
            ],
            "pose_timestamp": float(odom.ts),
            "tagged_at": time.time(),
        }
        try:
            atomic_write_json(path, payload)
        except OSError:
            return False
        return True

    @skill
    def tag_location(self, location_name: str) -> str:
        """Associate the current measured MuJoCo pose with one short name."""

        name = self._normalized_name(location_name)
        if name is None:
            return "Error: tag_location_failed: location name is invalid"
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return "No odometry data received yet, cannot tag location."
        position = odom.position
        rotation = odom.orientation
        location = RobotLocation(
            name=name,
            position=(position.x, position.y, position.z),
            # Preserve the pinned DimOS tag representation exactly.
            rotation=(rotation.x, rotation.y, rotation.z),
        )
        if not self._spatial_memory.tag_location(location):
            return f"Error: Failed to store '{name}' in the spatial memory"
        if not self._persist_exact_pose(name, odom):
            return f"Error: Failed to persist exact pose for '{name}'"
        return f"Tagged '{name}': ({position.x},{position.y})."
