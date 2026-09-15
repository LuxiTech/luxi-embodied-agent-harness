"""Reviewed Isaac command port behind the robot-local Safety Kernel."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping

from harness.robots.g1.isaac.isaac_protocol import IsaacCommandWriter, IsaacRuntimePaths

from .contracts import CancellationToken
from .ports import CommandEvidence, ObservationPort, RuntimeCommand


class IsaacG1StopOnlyMotionPort:
    """Expose only zero velocity through the existing filesystem writer.

    Non-zero motion is structurally unsupported in this cutover phase.  The
    port adds fencing and idempotency but does not introduce another transport
    or publisher implementation.
    """

    def __init__(
        self,
        paths: IsaacRuntimePaths,
        *,
        robot_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.paths = paths
        self.robot_id = robot_id
        self.clock = clock
        self.writer = IsaacCommandWriter(paths)
        self._boot_epoch = "unbound"
        self._seen_calls: set[tuple[str, str]] = set()
        self._motion_executor: (
            Callable[[RuntimeCommand, CancellationToken], Mapping[str, Any]] | None
        ) = None
        self._motion_capabilities: frozenset[str] = frozenset()
        self._active_tokens: dict[str, CancellationToken] = {}
        self._lock = threading.RLock()

    def bind_motion_executor(
        self,
        executor: Callable[[RuntimeCommand, CancellationToken], Mapping[str, Any]],
        *,
        capability_ids: frozenset[str] = frozenset({"relative_move"}),
    ) -> None:
        """Extend the reviewed port without adding another command writer."""

        with self._lock:
            if self._motion_executor is not None and self._motion_executor is not executor:
                raise RuntimeError("Isaac MotionPort already has a non-zero owner")
            self._motion_executor = executor
            self._motion_capabilities = self._motion_capabilities | capability_ids

    def execute(
        self,
        command: RuntimeCommand,
        cancel: CancellationToken,
    ) -> Mapping[str, Any]:
        with self._lock:
            if command.cancel_token_id in self._active_tokens:
                raise PermissionError("RuntimeCommand cancel token replay rejected")
            self._active_tokens[command.cancel_token_id] = cancel
        try:
            evidence = self.command_motion(command)
            raw = evidence.evidence.get("raw")
            if not isinstance(raw, Mapping):
                raise RuntimeError("Isaac MotionPort returned no structured result")
            return dict(raw)
        finally:
            with self._lock:
                self._active_tokens.pop(command.cancel_token_id, None)

    def bind_boot_epoch(self, boot_epoch: str) -> None:
        with self._lock:
            if boot_epoch != self._boot_epoch:
                self._seen_calls.clear()
            self._boot_epoch = boot_epoch

    def command_motion(self, command: RuntimeCommand) -> CommandEvidence:
        now = self.clock()
        with self._lock:
            executor = self._motion_executor
            cancel = self._active_tokens.get(command.cancel_token_id)
            if executor is None:
                return CommandEvidence(
                    False,
                    "unsupported",
                    now,
                    {"error": "Isaac MotionPort currently accepts only zero velocity"},
                )
            if command.robot_id != self.robot_id:
                raise PermissionError("Isaac motion command robot mismatch")
            if command.boot_epoch != self._boot_epoch:
                raise PermissionError("Isaac motion command boot epoch mismatch")
            if command.deadline_monotonic <= now:
                raise TimeoutError("Isaac motion command deadline expired")
            capability_id = command.payload.get("capability_id")
            if capability_id not in self._motion_capabilities:
                return CommandEvidence(
                    False,
                    "unsupported",
                    now,
                    {
                        "error": (
                            "Isaac MotionPort accepts only reviewed capabilities: "
                            f"{sorted(self._motion_capabilities)}"
                        )
                    },
                )
            if not isinstance(command.payload.get("arguments"), Mapping):
                raise ValueError(f"{capability_id} arguments are missing")
            if cancel is None:
                raise PermissionError("RuntimeCommand has no cancellation owner")
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("Isaac motion command replay rejected")
            # Fence before delegating because the remote Skill may make physical
            # progress and lose its result.
            self._seen_calls.add(key)
        raw = executor(command, cancel)
        return CommandEvidence(
            True,
            str(raw.get("task_status", "tool_succeeded")),
            self.clock(),
            {
                "raw": dict(raw),
                "publisher_implementation": (
                    "existing-native-isaac-motion-skill"
                ),
            },
        )

    def command_zero(self, command: RuntimeCommand) -> CommandEvidence:
        now = self.clock()
        with self._lock:
            if command.robot_id != self.robot_id:
                raise PermissionError("Isaac zero command robot mismatch")
            if command.boot_epoch != self._boot_epoch:
                raise PermissionError("Isaac zero command boot epoch mismatch")
            if command.deadline_monotonic <= now:
                raise TimeoutError("Isaac zero command deadline expired")
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("Isaac zero command replay rejected")
            sequence = self.writer.stop()
            self._seen_calls.add(key)
        return CommandEvidence(
            True,
            "stop_command_completed",
            self.clock(),
            {
                "command_sequence": sequence,
                "zero_only": True,
                "publisher_implementation": (
                    "existing-isaac-filesystem-command-writer"
                ),
            },
        )

    def command_operator_velocity(self, command: RuntimeCommand) -> CommandEvidence:
        """Publish one reviewed, fail-expiring operator velocity command.

        RuntimeHost and the operator gateway fence ownership before this thin
        transport method is reached.  It deliberately reuses this port's
        existing IsaacCommandWriter, so cutover adds no publisher.
        """

        now = self.clock()
        with self._lock:
            if command.robot_id != self.robot_id:
                raise PermissionError("Isaac operator command robot mismatch")
            if command.boot_epoch != self._boot_epoch:
                raise PermissionError("Isaac operator command boot epoch mismatch")
            if command.deadline_monotonic <= now:
                raise TimeoutError("Isaac operator command deadline expired")
            if command.payload.get("capability_id") != "operator_manual_velocity":
                raise PermissionError("Isaac operator capability mismatch")
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("Isaac operator command replay rejected")
            linear = command.payload.get("linear")
            angular = command.payload.get("angular")
            duration_s = command.payload.get("duration_s")
            if (
                not isinstance(linear, (list, tuple))
                or len(linear) != 3
                or not isinstance(angular, (list, tuple))
                or len(angular) != 3
                or not isinstance(duration_s, (int, float))
                or isinstance(duration_s, bool)
                or float(duration_s) <= 0.0
                or float(duration_s) > 0.35
            ):
                raise ValueError("invalid bounded Isaac operator velocity command")
            try:
                normalized_linear = tuple(float(value) for value in linear)
                normalized_angular = tuple(float(value) for value in angular)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("invalid Isaac operator velocity vector") from exc
            if not all(
                math.isfinite(value)
                for value in (*normalized_linear, *normalized_angular)
            ):
                raise ValueError("Isaac operator velocity must be finite")
            # Fence before the physical write: a lost result must never make
            # this side effect eligible for automatic replay.
            self._seen_calls.add(key)
            sequence = self.writer.write_velocity(
                normalized_linear,  # type: ignore[arg-type]
                normalized_angular,  # type: ignore[arg-type]
                duration_s=float(duration_s),
            )
        return CommandEvidence(
            True,
            "operator_command_accepted",
            self.clock(),
            {
                "command_sequence": sequence,
                "command_expiry_s": float(duration_s),
                "retry": "never",
                "publisher_implementation": (
                    "existing-isaac-filesystem-command-writer"
                ),
            },
        )


class IsaacMotionPortStopGateway:
    """Adapt the stop-only MotionPort to the SafetyKernel gateway contract."""

    def __init__(
        self,
        port: IsaacG1StopOnlyMotionPort,
        observation: ObservationPort,
        *,
        robot_id: str,
        boot_epoch_provider: Callable[[], str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = port
        self.observation = observation
        self.robot_id = robot_id
        self.boot_epoch_provider = boot_epoch_provider
        self.clock = clock
        self._sequence = 0
        self._samples: list[dict[str, Any]] = []
        self._last_state_sequence: int | None = None
        self._lock = threading.RLock()

    def publish_zero(self, robot_id: str) -> float:
        del robot_id
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        command = RuntimeCommand(
            robot_id=self.robot_id,
            boot_epoch=self.boot_epoch_provider(),
            task_id=f"safety-stop-{sequence}",
            tool_call_id=f"safety-zero-{sequence}",
            deadline_monotonic=self.clock() + 1.0,
            cancel_token_id=f"safety-zero-cancel-{sequence}",
            payload={"capability_id": "stop_robot", "zero_only": True},
        )
        evidence = self.port.command_zero(command)
        if not evidence.accepted:
            raise RuntimeError(evidence.status)
        with self._lock:
            self._samples.clear()
            self._last_state_sequence = None
        return evidence.timestamp_monotonic

    def stationary_samples(
        self,
        robot_id: str,
        after_monotonic: float,
    ) -> list[Mapping[str, Any]]:
        snapshot = self.observation.snapshot(self.robot_id)
        if snapshot.robot_id != robot_id and robot_id != "g1-isaac":
            return []
        state_sequence = snapshot.values.get("state_sequence")
        motion = snapshot.values.get("motion")
        if (
            not isinstance(state_sequence, int)
            or isinstance(state_sequence, bool)
            or not isinstance(motion, Mapping)
            or snapshot.freshness_s > 1.5
        ):
            return list(self._samples)
        speed = motion.get("planar_speed_mps")
        yaw_rate = motion.get("yaw_rate_rps")
        if (
            not isinstance(speed, (int, float))
            or isinstance(speed, bool)
            or not math.isfinite(float(speed))
            or not isinstance(yaw_rate, (int, float))
            or isinstance(yaw_rate, bool)
            or not math.isfinite(float(yaw_rate))
        ):
            return list(self._samples)
        with self._lock:
            if self._last_state_sequence == state_sequence:
                return list(self._samples)
            self._last_state_sequence = state_sequence
            self._samples.append(
                {
                    "timestamp_monotonic": max(
                        snapshot.timestamp_monotonic,
                        after_monotonic + 1e-6,
                    ),
                    "state_sequence": state_sequence,
                    "planar_speed_mps": abs(float(speed)),
                    "yaw_rate_rps": float(yaw_rate),
                }
            )
            return list(self._samples)


def bind_isaac_stop_gateway(runtime_host: Any, *, bind_port: bool = True) -> IsaacMotionPortStopGateway:
    """Bind the reviewed stop-only port without exposing non-zero motion."""

    observation = runtime_host.adapter.ports.observation
    paths = getattr(observation, "paths", None)
    if not isinstance(paths, IsaacRuntimePaths):
        raise TypeError("Isaac stop cutover requires IsaacRuntimePaths observation")
    port = IsaacG1StopOnlyMotionPort(paths, robot_id=runtime_host.robot_id)
    if bind_port:
        runtime_host.bind_motion_port(
            port, capability_ids=frozenset({"stop_robot", "stop_navigation"}),
        )
    port.bind_boot_epoch(runtime_host.boot_epoch)
    return IsaacMotionPortStopGateway(
        port,
        observation,
        robot_id=runtime_host.robot_id,
        boot_epoch_provider=lambda: runtime_host.boot_epoch,
    )
