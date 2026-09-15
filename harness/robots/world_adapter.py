"""Composition boundary between shared AgentOS tasks and simulator backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from harness.robots.entity_port import EntityManipulationPort

WORLD_STATE_SCHEMA_VERSION = 1
SHARED_MOTION_SKILLS = frozenset({"move", "relative_move", "reset_pose"})


def read_world_json(path: Path, *, max_bytes: int = 256 * 1024) -> dict[str, Any] | None:
    """Read a bounded adapter-owned JSON document for the shared task layer."""

    try:
        if path.stat().st_size > max_bytes:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class WorldCapabilities:
    aligned_rgbd: bool
    online_mapping: bool
    frontier_exploration: bool
    persistent_semantic_memory: bool
    manipulation: bool


@dataclass(frozen=True)
class RobotWorldAdapter:
    """Factories and module types supplied at the AgentOS composition root."""

    adapter_id: str
    connection_module: type[Any]
    motion_module: type[Any]
    camera_info: Any
    entity_port_factory: Callable[[], EntityManipulationPort]
    capabilities: WorldCapabilities

    def create_entity_port(self) -> EntityManipulationPort:
        return self.entity_port_factory()

    def validate_contract(self) -> None:
        missing = sorted(
            name
            for name in SHARED_MOTION_SKILLS
            if not callable(getattr(self.motion_module, name, None))
        )
        if missing:
            raise RuntimeError(
                f"{self.adapter_id} adapter is missing motion skills: {missing}"
            )


def get_world_adapter(adapter_id: str) -> RobotWorldAdapter:
    """Resolve one simulator adapter without leaking it into shared skills."""

    normalized = str(adapter_id).strip().casefold()
    if normalized == "mujoco":
        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        from harness.robots.g1.mujoco.entity_manipulation import EntityControlChannel
        from harness.robots.g1.g1_simulation_skills import G1MujocoAdapter

        adapter = RobotWorldAdapter(
            adapter_id="mujoco",
            connection_module=MujocoConnection,
            motion_module=G1MujocoAdapter,
            camera_info=MujocoConnection.camera_info_static,
            entity_port_factory=EntityControlChannel,
            capabilities=WorldCapabilities(
                aligned_rgbd=True,
                online_mapping=True,
                frontier_exploration=True,
                persistent_semantic_memory=True,
                manipulation=True,
            ),
        )
        adapter.validate_contract()
        return adapter
    if normalized == "isaac-g1":
        from harness.robots.g1.g1_simulation_skills import G1IsaacAdapter
        from harness.robots.g1.isaac.isaac_entity_port import IsaacEntityControlChannel
        from harness.robots.g1.isaac.isaac_g1_connection import G1IsaacConnection

        adapter = RobotWorldAdapter(
            adapter_id="isaac-g1",
            connection_module=G1IsaacConnection,
            motion_module=G1IsaacAdapter,
            camera_info=G1IsaacConnection.camera_info_static,
            entity_port_factory=IsaacEntityControlChannel,
            capabilities=WorldCapabilities(
                aligned_rgbd=True,
                online_mapping=True,
                frontier_exploration=True,
                persistent_semantic_memory=True,
                manipulation=False,
            ),
        )
        adapter.validate_contract()
        return adapter
    raise ValueError(f"unsupported RobotWorld adapter: {adapter_id!r}")
