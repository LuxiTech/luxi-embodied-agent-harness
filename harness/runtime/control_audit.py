"""Static migration audit for robot command surfaces and Agent tool ownership."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from harness.robots.robot_profiles import G1_ISAAC_SIMULATION
from harness.robots.robot_profiles import G1_SIMULATION

from harness.runtime.capability_policy import (
    accepted_cutover_tools,
)


NON_PHYSICAL_ISAAC_AGENT_TOOLS = frozenset(
    {
        "observe_environment",
        "get_dimos_status",
        "tag_location",
        "analyze_scene",
        "find_visual_target",
        "verify_visual_condition",
    }
)

COMMAND_METHODS = frozenset(
    {
        "write_velocity",
        "publish_follow_command",
        "set_goal",
        "cancel_goal",
        "command_velocity",
        "drive_velocity",
        "_publish_cmd_vel",
    }
)
COMMAND_CHANNEL_NAMES = frozenset({"cmd_vel", "nav_cmd_vel", "stop_movement"})

# These are implementation boundaries, not permission grants. Unknown files fail
# the audit; pending categories remain visible until their backend phase closes.
SURFACE_OWNERS = {
    "dimos_runtime.py": "legacy_mujoco_safety_relay",
    "go2_mujoco.py": "go2_runtime_cutover_pending",
    "go2_ros_robot_node.py": "go2_runtime_cutover_pending",
    "go2_ros_sim_adapter.py": "go2_driver_boundary",
    "isaac_g1_connection.py": "isaac_driver_boundary",
    "isaac_g1_runtime.py": "isaac_driver_boundary",
    "isaac_manual_control.py": "reviewed_operator_motion_gateway_client",
    "isaac_protocol.py": "isaac_driver_boundary",
    "isaac_recovery.py": "reviewed_safety_recovery",
    "location_tagging.py": "reviewed_native_terminal_skill",
    "navigation_benchmark.py": "legacy_mujoco_benchmark",
    "navigation_relay.py": "reviewed_native_navigation_relay",
    "object_navigation_compat.py": "legacy_mujoco_terminal_skill",
    "object_skills.py": "legacy_mujoco_terminal_skill",
    "person_follow.py": "reviewed_native_terminal_skill",
    "motion_skills.py": "shared_robot_motion_gateway",
    "task_skills.py": "shared_robot_task_gateway",
    "rgbd_skills.py": "reviewed_native_terminal_skill",
}


@dataclass(frozen=True)
class CommandSurface:
    path: str
    line: int
    sink: str
    owner: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "sink": self.sink,
            "owner": self.owner,
        }


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _call_sink(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    parts = _attribute_parts(call.func)
    if not parts:
        return None
    method = parts[-1]
    if method in COMMAND_METHODS:
        return ".".join(parts[-2:]) if len(parts) >= 2 else method
    if method == "publish" and len(parts) >= 2 and parts[-2] in COMMAND_CHANNEL_NAMES:
        return ".".join(parts[-2:])
    return None


def scan_command_surfaces(source_root: Path) -> tuple[CommandSurface, ...]:
    surfaces: list[CommandSurface] = []
    for path in sorted(source_root.rglob("*.py")):
        relative_path = path.relative_to(source_root)
        # Preserve the audit's compatibility/driver scope. The unified runtime
        # and isolated BEHAVIOR development backend have separate contracts.
        if relative_path.parts[0] == "runtime" or relative_path.parts[:2] == (
            "evaluation", "behavior"
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        owner = SURFACE_OWNERS.get(path.name, "unreviewed")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            sink = _call_sink(node)
            if sink is not None:
                surfaces.append(
                    CommandSurface(
                        path=f"harness/{relative_path.as_posix()}",
                        line=node.lineno,
                        sink=sink,
                        owner=owner,
                    )
                )
    return tuple(surfaces)


def physical_isaac_agent_tools() -> frozenset[str]:
    return frozenset(G1_ISAAC_SIMULATION.agent_tools) - NON_PHYSICAL_ISAAC_AGENT_TOOLS


def _mujoco_migration_decisions(project_root: Path) -> dict[str, str]:
    path = project_root / "config" / "mujoco-g1-capability-migration.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    capabilities = payload.get("capabilities")
    if payload.get("schema_version") != 1 or not isinstance(capabilities, dict):
        return {}
    return {
        str(name): str(entry.get("decision"))
        for name, entry in capabilities.items()
        if isinstance(entry, dict) and isinstance(entry.get("decision"), str)
    }


def audit_physical_control_paths(project_root: Path) -> dict[str, Any]:
    required = physical_isaac_agent_tools()
    accepted = accepted_cutover_tools(project_root, "isaac-g1")
    gaps = sorted(required - accepted)
    mujoco_accepted = accepted_cutover_tools(project_root, "mujoco")
    mujoco_agent_tools = frozenset(G1_SIMULATION.agent_tools)
    non_physical = NON_PHYSICAL_ISAAC_AGENT_TOOLS | {"tag_location"}
    mujoco_unmigrated = sorted(mujoco_agent_tools - non_physical - mujoco_accepted)
    decisions = _mujoco_migration_decisions(project_root)
    missing_decisions = sorted(set(mujoco_unmigrated) - decisions.keys())
    surfaces = scan_command_surfaces(project_root / "harness")
    unknown = [surface.as_dict() for surface in surfaces if surface.owner == "unreviewed"]
    pending = [
        surface.as_dict()
        for surface in surfaces
        if surface.owner.endswith("_pending")
    ]
    return {
        "schema_version": 1,
        "kind": "physical-control-path-audit",
        "isaac_agent_physical_tools": sorted(required),
        "isaac_accepted_tools": sorted(accepted),
        "isaac_cutover_gaps": gaps,
        "mujoco_accepted_tools": sorted(mujoco_accepted),
        "mujoco_unmigrated_agent_tools": mujoco_unmigrated,
        "mujoco_migration_decisions": {
            name: decisions[name] for name in mujoco_unmigrated if name in decisions
        },
        "mujoco_missing_migration_decisions": missing_decisions,
        "command_surfaces": [surface.as_dict() for surface in surfaces],
        "unknown_command_surfaces": unknown,
        "pending_command_surfaces": pending,
        "passed": not gaps and not missing_decisions and not unknown,
    }


def owner_counts(surfaces: Iterable[CommandSurface]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for surface in surfaces:
        counts[surface.owner] = counts.get(surface.owner, 0) + 1
    return dict(sorted(counts.items()))
