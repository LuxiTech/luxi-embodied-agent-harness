"""Shared robot execution boundary, independent of model providers."""
from __future__ import annotations
from datetime import datetime
import math
import threading
import time
from typing import Any, Mapping
from .contracts import CancellationToken, ToolRequest, ToolResult
from .ports import CommandEvidence, RuntimeCommand

def _iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None

class MonitorStopGateway:
    """Local zero-command and fresh-odometry evidence adapter.

    This gateway deliberately has no model, UI or SessionStore dependency.
    The bridge's existing backend-specific stop implementation remains the
    temporary command-port Adapter during migration.
    """

    def __init__(
        self,
        bridge: Any,
        *,
        max_pose_age_s: float = 1.5,
        emergency_stop_callback: Any | None = None,
    ) -> None:
        self.bridge = bridge
        self.max_pose_age_s = max_pose_age_s
        self.emergency_stop_callback = emergency_stop_callback
        self._samples: list[dict[str, Any]] = []
        self._last_pose_stamp: float | None = None
        self._post_stop_pose = None
        self._stop_wall_time = 0.0
        self._command_lock = threading.RLock()

    def execute_tool(
        self, name: str, arguments: Mapping[str, Any], cancel: CancellationToken
    ) -> Mapping[str, Any]:
        """The sole migrated physical Adapter-to-command-port boundary."""

        cancel.raise_if_cancelled()
        with self._command_lock:
            return self.bridge._dispatch_tool(name, dict(arguments))

    def publish_zero(self, robot_id: str) -> float:
        del robot_id
        if self.emergency_stop_callback is not None:
            self.emergency_stop_callback()
        result = self.bridge._stop_robot({})
        if result.get("ok") is not True:
            raise RuntimeError(str(result.get("error") or "backend zero command failed"))
        self._stop_wall_time = time.time()
        self._post_stop_pose = None
        self._samples.clear()
        self._last_pose_stamp = None
        return time.monotonic()

    def stationary_samples(
        self, robot_id: str, after_monotonic: float
    ) -> list[Mapping[str, Any]]:
        del robot_id
        snapshot = self.bridge._observation()
        pose = snapshot.get("pose")
        motion = snapshot.get("motion")
        if not isinstance(pose, Mapping) or not isinstance(motion, Mapping):
            return list(self._samples)
        pose_stamp = pose.get("timestamp")
        if not isinstance(pose_stamp, (int, float)) or isinstance(pose_stamp, bool):
            return list(self._samples)
        pose_stamp = float(pose_stamp)
        if not math.isfinite(pose_stamp) or time.time() - pose_stamp > self.max_pose_age_s:
            return list(self._samples)
        if self._last_pose_stamp is not None and pose_stamp <= self._last_pose_stamp:
            return list(self._samples)
        if pose_stamp <= self._stop_wall_time:
            return list(self._samples)
        speed = motion.get("planar_speed")
        yaw_rate = motion.get("yaw_rate")
        coordinates = tuple(pose.get(key) for key in ("x", "y", "yaw"))
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in coordinates):
            current = (pose_stamp, *coordinates)
            previous = self._post_stop_pose
            self._post_stop_pose = current
            if previous is None:
                self._last_pose_stamp = pose_stamp
                return list(self._samples)
            elapsed = pose_stamp - previous[0]
            speed = math.hypot(current[1]-previous[1], current[2]-previous[2]) / elapsed
            yaw_delta = current[3]-previous[3]
            yaw_rate = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)) / elapsed
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            return list(self._samples)
        if not isinstance(yaw_rate, (int, float)) or isinstance(yaw_rate, bool):
            return list(self._samples)
        self._last_pose_stamp = pose_stamp
        self._samples.append(
            {
                "timestamp_monotonic": max(time.monotonic(), after_monotonic + 1e-6),
                "pose_timestamp": pose_stamp,
                "planar_speed_mps": abs(float(speed)),
                "yaw_rate_rps": float(yaw_rate),
            }
        )
        return list(self._samples)

class GatewayPhysicalToolAdapter:
    """Prevent a migrated physical Tool Adapter from bypassing its gateway."""

    def __init__(self, gateway: Any, name: str) -> None:
        self.gateway = gateway
        self.name = name

    def execute(
        self, request: ToolRequest, cancel: CancellationToken
    ) -> Mapping[str, Any]:
        return self.gateway.execute_tool(self.name, request.arguments, cancel)

class SafetyStopIntentAdapter:
    """Let the Safety Kernel own the only physical effect of ``stop_robot``."""

    def execute(
        self, request: ToolRequest, cancel: CancellationToken
    ) -> Mapping[str, Any]:
        del request, cancel
        return {
            "ok": True,
            "completed": False,
            "task_status": "stop_requested",
        }

class CombinedCancellationToken(CancellationToken):
    """Combine Turn/deadline cancellation with RuntimeHost health cancellation."""

    def __init__(
        self,
        upstream: CancellationToken,
        runtime_token: CancellationToken,
    ) -> None:
        super().__init__()
        self.upstream = upstream
        self.runtime_token = runtime_token

    @property
    def cancelled(self) -> bool:
        return bool(
            super().cancelled
            or self.upstream.cancelled
            or self.runtime_token.cancelled
        )

class ExistingSafeCommandMotionPort:
    """Delegate reviewed physical tools to the existing command gateway.

    This class contains no transport, publisher, shell or zero-command
    implementation. Emergency stop remains owned directly by SafetyKernel.
    """

    def __init__(self, gateway: Any, capability_ids: frozenset[str], native_port=None) -> None:
        self.gateway = gateway
        self.capability_ids = capability_ids
        self.native_port = native_port
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.RLock()

    def bind_boot_epoch(self, boot_epoch):
        if self.native_port is not None:
            self.native_port.bind_boot_epoch(boot_epoch)

    def command_operator_velocity(self, command):
        if self.native_port is None:
            raise PermissionError("No operator velocity port is configured")
        return self.native_port.command_operator_velocity(command)

    def execute(
        self,
        command: RuntimeCommand,
        cancel: CancellationToken,
    ) -> Mapping[str, Any]:
        with self._lock:
            if command.cancel_token_id in self._tokens:
                raise PermissionError("RuntimeCommand cancel token replay rejected")
            self._tokens[command.cancel_token_id] = cancel
        bridge = getattr(self.gateway, "bridge", None)
        set_probe = getattr(bridge, "set_physical_cancel_probe", None)
        if callable(set_probe):
            set_probe(lambda: cancel.cancelled)
        try:
            evidence = self.command_motion(command)
            raw = evidence.evidence.get("raw")
            if not isinstance(raw, Mapping):
                raise RuntimeError("existing command gateway returned no structured result")
            return dict(raw)
        finally:
            if callable(set_probe):
                set_probe(None)
            with self._lock:
                self._tokens.pop(command.cancel_token_id, None)

    def command_motion(self, command: RuntimeCommand) -> CommandEvidence:
        capability_id = command.payload.get("capability_id")
        arguments = command.payload.get("arguments")
        if capability_id not in self.capability_ids or not isinstance(arguments, Mapping):
            return CommandEvidence(
                False,
                "unsupported",
                time.monotonic(),
                {
                    "error": (
                        "MotionPort accepts only reviewed capabilities: "
                        f"{sorted(self.capability_ids)}"
                    )
                },
            )
        with self._lock:
            cancel = self._tokens.get(command.cancel_token_id)
        if cancel is None:
            raise PermissionError("RuntimeCommand has no active cancellation owner")
        raw = self.gateway.execute_tool(str(capability_id), arguments, cancel)
        return CommandEvidence(
            True,
            str(raw.get("task_status", "tool_succeeded")),
            time.monotonic(),
            {"raw": dict(raw)},
        )

    def command_zero(self, command: RuntimeCommand) -> CommandEvidence:
        del command
        return CommandEvidence(
            False,
            "emergency_stop_not_owned",
            time.monotonic(),
            {
                "reason": (
                    "SafetyKernel retains the independent existing zero-command path"
                )
            },
        )

class RuntimeCommandFence:
    """Fence selected tools with RuntimeHost without becoming a second Loop."""

    def __init__(
        self,
        runtime_host: Any,
        motion_port: ExistingSafeCommandMotionPort,
        *,
        fenced_tools: frozenset[str],
        navigation_port=None,
    ) -> None:
        self.runtime_host = runtime_host
        self.motion_port = motion_port
        self.fenced_tools = fenced_tools
        self.navigation_port = navigation_port
        self._active: dict[str, tuple[RuntimeCommand, CancellationToken]] = {}
        self._lock = threading.RLock()

    def applies(self, request: ToolRequest) -> bool:
        return request.capability_id in self.fenced_tools

    def _command(self, request: ToolRequest) -> RuntimeCommand:
        if request.boot_epoch is None:
            raise PermissionError("RuntimeHost-fenced tool requires boot_epoch")
        if request.deadline_monotonic is None:
            raise PermissionError("RuntimeHost-fenced tool requires deadline")
        return RuntimeCommand(
            robot_id=self.runtime_host.robot_id,
            boot_epoch=request.boot_epoch,
            task_id=request.task_id,
            tool_call_id=request.tool_call_id,
            deadline_monotonic=request.deadline_monotonic,
            cancel_token_id=f"runtime-cancel:{request.tool_call_id}",
            payload={
                "capability_id": request.capability_id,
                "arguments": dict(request.arguments),
            },
        )

    def admit(
        self,
        request: ToolRequest,
        cancel: CancellationToken,
    ) -> CancellationToken:
        command = self._command(request)
        runtime_token = self.runtime_host.accept(command)
        combined = CombinedCancellationToken(cancel, runtime_token)
        with self._lock:
            if request.tool_call_id in self._active:
                self.runtime_host.cancel("runtime_fence_duplicate")
                raise PermissionError("RuntimeCommand replay rejected")
            self._active[request.tool_call_id] = (command, combined)
        return combined

    def execute(
        self,
        request: ToolRequest,
        cancel: CancellationToken,
    ) -> Mapping[str, Any]:
        with self._lock:
            active = self._active.get(request.tool_call_id)
        if active is None:
            raise PermissionError("RuntimeCommand was not admitted")
        command, admitted_cancel = active
        if cancel is not admitted_cancel:
            raise PermissionError("RuntimeCommand cancellation owner changed")
        port = (self.navigation_port if self.navigation_port is not None
                and request.capability_id in self.navigation_port.supported_capabilities else self.motion_port)
        return port.execute(command, cancel)

    def finish(self, request: ToolRequest, result: ToolResult) -> None:
        with self._lock:
            active = self._active.pop(request.tool_call_id, None)
        if active is None:
            raise PermissionError("RuntimeCommand completion has no active owner")
        self.runtime_host.finish(
            request.task_id,
            stationary_confirmed=(
                result.evidence.get("stationary_confirmed") is True
            ),
        )

class RuntimeFencedPhysicalToolAdapter:
    def __init__(self, fence: RuntimeCommandFence) -> None:
        self.fence = fence

    def execute(
        self,
        request: ToolRequest,
        cancel: CancellationToken,
    ) -> Mapping[str, Any]:
        return self.fence.execute(request, cancel)
