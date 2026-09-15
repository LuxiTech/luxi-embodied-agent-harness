"""AgentOS compatibility Skill for the provider-neutral stop owner."""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
import uuid


from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    IsaacCommandWriter,
    IsaacRuntimePaths,
    atomic_write_json,
    read_json,
)


AGENT_STOP_CONTROL_PATH_ENV = "LUXI_AGENT_STOP_CONTROL_PATH"


def configured_agent_stop_root() -> Path:
    configured = os.environ.get(AGENT_STOP_CONTROL_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return IsaacRuntimePaths.configured().root / "agent-stop"


class IsaacStopControlChannel:
    """Atomic request/ack channel between the DimOS Skill and runtime owner."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or configured_agent_stop_root()).expanduser().resolve()
        self.request_path = self.root / "request.json"
        self.ack_path = self.root / "ack.json"

    def request_stop(
        self,
        *,
        action: str = "stop_robot",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        if action not in {"stop_robot", "stop_navigation"}:
            raise ValueError(f"unsupported stop action: {action}")
        token = uuid.uuid4().hex
        correlation = {
            "turn_id": f"native-turn-{token}",
            "task_id": f"native-task-{token}",
            "step_id": f"native-step-{token}",
            "tool_call_id": f"native-stop-{token}",
        }
        atomic_write_json(
            self.request_path,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "action": action,
                "token": token,
                "requested_at": time.time(),
                **correlation,
            },
        )
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            acknowledgement = read_json(self.ack_path)
            if acknowledgement and acknowledgement.get("token") == token:
                result = acknowledgement.get("result")
                return dict(result) if isinstance(result, Mapping) else {
                    "ok": False,
                    "completed": False,
                    "task_status": "runtime_error",
                    "error": "Agent stop acknowledgement omitted result",
                }
            time.sleep(0.02)
        return self._emergency_fallback("provider stop service timed out")

    @staticmethod
    def _emergency_fallback(reason: str) -> dict[str, Any]:
        try:
            sequence = IsaacCommandWriter(IsaacRuntimePaths.configured()).stop()
        except Exception as exc:  # noqa: BLE001 - report local stop failure
            return {
                "ok": False,
                "completed": False,
                "task_status": "runtime_error",
                "error": f"{reason}; emergency zero failed: {exc}",
                "safety_evidence": {
                    "stop_command_completed": False,
                    "stationary_confirmed": False,
                    "provider_service_fallback": True,
                },
            }
        return {
            "ok": True,
            "completed": False,
            "task_status": "stop_fallback_incomplete",
            "error": reason,
            "safety_evidence": {
                "stop_command_completed": True,
                "stationary_confirmed": False,
                "provider_service_fallback": True,
                "command_sequence": sequence,
            },
        }




class IsaacStopRequestService:
    """Robot-runtime-side consumer; UI and model are not execution owners."""

    def __init__(
        self,
        executor: Any,
        channel: IsaacStopControlChannel,
        *,
        max_request_age_s: float = 10.0,
    ) -> None:
        self.executor = executor
        self.channel = channel
        self.max_request_age_s = max(1.0, float(max_request_age_s))
        self._last_token: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.channel.root.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="luxi-native-stop-service",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            request = read_json(self.channel.request_path)
            if request is None:
                continue
            token = request.get("token")
            if not isinstance(token, str) or not token or token == self._last_token:
                continue
            self._last_token = token
            result = self._execute(request)
            atomic_write_json(
                self.channel.ack_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": BACKEND_NAME,
                    "action": str(request.get("action", "stop_robot")),
                    "token": token,
                    "acknowledged_at": time.time(),
                    "result": result,
                },
            )

    def _execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        requested_at = request.get("requested_at")
        request_age = (
            time.time() - float(requested_at)
            if isinstance(requested_at, (int, float))
            and not isinstance(requested_at, bool)
            else math.inf
        )
        identifiers = {
            name: request.get(name)
            for name in ("turn_id", "task_id", "step_id", "tool_call_id")
        }
        if (
            request.get("schema_version") != SCHEMA_VERSION
            or request.get("backend") != BACKEND_NAME
            or request.get("action") not in {"stop_robot", "stop_navigation"}
            or not math.isfinite(request_age)
            or request_age < -0.25
            or request_age > self.max_request_age_s
            or any(
                not isinstance(value, str) or not value
                for value in identifiers.values()
            )
        ):
            return {
                "ok": False,
                "completed": False,
                "task_status": "tool_denied",
                "error": "AgentOS stop request is stale or malformed",
            }
        return self.executor.request_stop(
            f"native:{request['action']}",
            action=str(request["action"]),
            turn_id=identifiers["turn_id"],
            task_id=identifiers["task_id"],
            step_id=identifiers["step_id"],
            tool_call_id=identifiers["tool_call_id"],
        )
