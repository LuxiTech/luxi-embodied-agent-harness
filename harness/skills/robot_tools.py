"""Robot skill composition, independent of model and session management."""

from __future__ import annotations
import os
from pathlib import Path
import threading
from typing import Any, Callable

from harness.runtime.tool_catalog import (
    _tools_for_backend,
)
from harness.skills.tool_results import (
    _blind_safe_payload,
)

from harness.skills.motion_skills import G1MotionSkills
from harness.skills.task_skills import G1TaskSkills
from harness.integrations.mcp.tool_client import RobotMcpClient

class G1RobotTools(G1MotionSkills, G1TaskSkills, RobotMcpClient):
    """G1 skill dependencies and dispatch, with no planning or conversation state."""

    def __init__(self, events, monitor, project_root, *, backend="mujoco",
                 command_runner=None, long_task_runner=None):
        if backend not in {"mujoco", "isaac-g1"}:
            raise ValueError(f"Unsupported G1 backend: {backend!r}")
        self.events = events
        self.monitor = monitor
        self.project_root = Path(project_root)
        self.backend = backend
        self._tools = _tools_for_backend(backend)
        self.blind_mode = os.environ.get("LUXI_BLIND_MODE", "").lower() in {"1", "true", "yes", "on"}
        self._custom_command_runner = command_runner is not None
        self._command_runner = command_runner or self._run_cancelable_subprocess
        self._long_task_runner = long_task_runner
        self._cancel = threading.Event()
        self._observed_this_turn = False
        self._visual_branch_failed = False
        self._trace_local = threading.local()
        self._physical_cancel_probe = None

    def set_physical_cancel_probe(
        self, probe: Callable[[], bool] | None
    ) -> None:
        """Expose Pipeline cancellation/deadline to blocking legacy commands."""

        self._physical_cancel_probe = probe

    def _model_payload(self, value: Any) -> Any:
        return _blind_safe_payload(value) if self.blind_mode else value

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {tool["function"]["name"] for tool in self._tools}
        if name not in allowed:
            return {"ok": False, "error": f"当前 {self.backend} 后端未授权工具: {name}"}
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "observe_environment": self._observe_environment,
            "get_dimos_status": self._get_dimos_status,
            "move_robot": self._move_robot,
            "move_distance": self._move_distance,
            "stop_robot": self._stop_robot,
            "turn_around": self._turn_around,
            "walk_room_loop": self._walk_room_loop,
            "navigate_with_text": self._navigate_with_text,
            "navigate_to_pose": self._navigate_to_pose,
            "navigate_to_tag": self._navigate_to_tag,
            "stop_navigation": self._stop_navigation,
            "explore_frontiers": self._explore_frontiers,
            "tag_location": self._tag_location,
            "analyze_scene": self._analyze_scene,
            "find_visual_target": self._find_visual_target,
            "verify_visual_condition": self._verify_visual_condition,
            "object_search": self._object_search,
            "follow_person": self._follow_person,
            "approach_visual_target": self._approach_visual_target,
            "approach_person": self._approach_person,
            "fetch_object": self._fetch_object,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"未授权工具: {name}"}
        return handler(arguments)

    def _observation(self) -> dict[str, Any]:
        return self.monitor.agent_snapshot()

    def _observe_environment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"ok": False, "error": "observe_environment 不接受参数"}
        self._observed_this_turn = True
        return {"ok": True, "observation": self._observation()}

    def _get_dimos_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"ok": False, "error": "get_dimos_status 不接受参数"}
        return self._mcp_call("server_status")
