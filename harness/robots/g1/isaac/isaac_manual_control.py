"""Fail-expiring operator keyboard control for the Isaac G1 adapter."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    read_json,
)


class IsaacPersonControlWriter:
    """Write fail-expiring operator commands to the task-apartment actor."""

    COMMAND_DURATION_S = 0.35

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        self._sequence = 0

    def _write(self, mode: str, *, forward: float = 0.0, turn: float = 0.0) -> int:
        with self._lock:
            existing = read_json(self.path, max_bytes=8_192)
            existing_sequence = 0
            if isinstance(existing, Mapping):
                value = existing.get("sequence")
                if isinstance(value, int) and not isinstance(value, bool):
                    existing_sequence = max(0, value)
            self._sequence = max(self._sequence, existing_sequence) + 1
            now = self.clock()
            atomic_write_json(
                self.path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": BACKEND_NAME,
                    "sequence": self._sequence,
                    "written_at": now,
                    "expires_at": now + self.COMMAND_DURATION_S,
                    "mode": mode,
                    "forward": float(forward),
                    "turn": float(turn),
                },
            )
            return self._sequence

    def manual(self, *, forward: float, turn: float) -> int:
        return self._write("manual", forward=forward, turn=turn)

    def pause(self) -> int:
        return self._write("paused")

    def resume_auto(self) -> int:
        return self._write("auto")


class IsaacManualControl:
    FORWARD_SPEED_MPS = 0.40
    YAW_RATE_RAD_S = 0.80
    TURNING_CREEP_MPS = 0.25
    COMMAND_DURATION_S = 0.35
    MIN_TRANSLATION_CENTER_DISTANCE_M = 0.55

    def __init__(
        self,
        events: Any,
        operator_gateway: Any,
        *,
        simulation_status: Callable[[], dict[str, Any]],
        agent_status: Callable[[], dict[str, Any]],
        recovery_status: Callable[[], dict[str, Any]],
        proximity_payload: Callable[[], dict[str, Any] | None],
        person_writer: IsaacPersonControlWriter | None = None,
        scene_status: Callable[[], str | None] | None = None,
        person_status: Callable[[], dict[str, Any] | None] | None = None,
        stop_callback: Callable[[str], Any] | None = None,
        control_owner_status: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.operator_gateway = operator_gateway
        self.simulation_status = simulation_status
        self.agent_status = agent_status
        self.recovery_status = recovery_status
        self.proximity_payload = proximity_payload
        self.person_writer = person_writer
        self.scene_status = scene_status or (lambda: None)
        self.person_status = person_status or (lambda: None)
        self.stop_callback = stop_callback
        self.control_owner_status = control_owner_status
        self.clock = clock
        self._lock = threading.Lock()
        self._enabled = False
        self._last_command_at: float | None = None
        self._last_forward = 0.0
        self._last_turn = 0.0
        self._target = "robot"
        self._person_mode = "auto"

    def _stop_robot(self, source: str) -> None:
        self.operator_gateway.stop(source)

    def _control_available(self) -> tuple[bool, str]:
        if not self.simulation_status().get("bridge_ready"):
            return False, "Isaac 尚未就绪"
        if self._target == "person":
            if self.person_writer is None or self.scene_status() != "task_apartment":
                return False, "当前场景没有可控制的人物"
            return True, ""
        if self.agent_status().get("busy"):
            return False, "Agent 正在执行任务，不能抢占为键盘控制"
        if self.control_owner_status is not None:
            owner = self.control_owner_status()
            state = getattr(owner, "state", None)
            state_name = str(getattr(state, "value", state or "")).upper()
            gateway_owns_runtime = bool(
                state_name == "BUSY"
                and self.operator_gateway.owns_runtime_control()
            )
            if state_name and state_name != "READY" and not gateway_owns_runtime:
                return False, f"RuntimeHost 当前为 {state_name}，键盘不能抢占控制权"
        recovery = self.recovery_status()
        if recovery.get("active") or recovery.get("safety_hold"):
            return False, "安全恢复正在锁定机器人"
        return True, ""

    def set_enabled(self, enabled: bool) -> tuple[bool, str]:
        enabled = bool(enabled)
        with self._lock:
            if enabled:
                available, message = self._control_available()
                if not available:
                    return False, message
                if self._target == "person":
                    if self.person_writer is None or self.scene_status() != "task_apartment":
                        return False, "当前场景没有可控制的人物"
                    self.person_writer.manual(forward=0.0, turn=0.0)
                    self._person_mode = "manual"
                else:
                    admitted, message = self.operator_gateway.enable()
                    if not admitted:
                        return False, message
                self._enabled = True
                self._last_command_at = None
                self._last_forward = 0.0
                self._last_turn = 0.0
                self.events.append(
                    "ui",
                    "manual_control",
                    "Isaac keyboard control enabled",
                    "W/S controls translation; A/D uses the G1 policy's stable stepping turn. "
                    "Commands expire after 0.35 s.",
                    level="warning",
                )
                return True, f"{('人物' if self._target == 'person' else '机器人')}键盘控制已开启"

            if self._target == "person" and self.person_writer is not None:
                self.person_writer.pause()
                self._person_mode = "paused"
            else:
                self.operator_gateway.disable("manual_control_disable")
            was_enabled = self._enabled
            self._enabled = False
            self._last_command_at = None
            self._last_forward = 0.0
            self._last_turn = 0.0
            if was_enabled:
                self.events.append(
                    "ui",
                    "manual_control",
                    "Isaac keyboard control disabled",
                    "A zero-velocity command was written.",
                )
            return True, "键盘控制已关闭并停车"

    def set_target(self, target: Any) -> tuple[bool, str]:
        if target not in {"robot", "person"}:
            return False, "控制对象必须是 robot 或 person"
        with self._lock:
            if target == "person" and (
                self.person_writer is None or self.scene_status() != "task_apartment"
            ):
                return False, "人物控制仅在 Isaac 实景公寓场景可用"
            if target == "robot" and self.agent_status().get("busy"):
                return False, "Agent 正在控制机器人，不能切换到机器人键盘控制"
            if self._target == "robot" and self._enabled:
                self.operator_gateway.disable("manual_control_target_change")
            if self._target == "person" and self.person_writer is not None:
                self.person_writer.pause()
                self._person_mode = "paused"
            self._target = target
            self._enabled = False
            self._last_command_at = None
            self._last_forward = 0.0
            self._last_turn = 0.0
            return True, f"控制对象已切换为{('人物' if target == 'person' else '机器人')}"

    def person_action(self, action: Any) -> tuple[bool, str]:
        if action not in {"pause", "resume"}:
            return False, "人物动作必须是 pause 或 resume"
        with self._lock:
            if self.person_writer is None or self.scene_status() != "task_apartment":
                return False, "当前场景没有可控制的人物"
            if action == "pause":
                self.person_writer.pause()
                self._person_mode = "paused"
                self._last_command_at = None
                return True, "人物已暂停"
            self.person_writer.resume_auto()
            self._person_mode = "auto"
            self._enabled = False
            self._last_command_at = None
            self._last_forward = 0.0
            self._last_turn = 0.0
            return True, "人物已从当前位置恢复预设路线"

    @staticmethod
    def _axis(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(normalized) or normalized < -1.0 or normalized > 1.0:
            return None
        return normalized

    def _translation_clear(self, forward: float) -> tuple[bool, str]:
        payload = self.proximity_payload()
        if not isinstance(payload, dict) or not payload.get("available"):
            return False, "没有新鲜雷达，平移命令被拒绝"
        sectors = payload.get("sectors_m")
        if not isinstance(sectors, dict):
            return False, "雷达方向数据缺失，平移命令被拒绝"
        names = (
            ("front", "front_left", "front_right")
            if forward > 0
            else ("rear", "rear_left", "rear_right")
        )
        try:
            distances = [float(sectors[name]) for name in names if name in sectors]
        except (TypeError, ValueError, OverflowError):
            return False, "雷达方向数据无效，平移命令被拒绝"
        if not all(math.isfinite(value) for value in distances):
            return False, "雷达方向数据无效，平移命令被拒绝"
        if not distances:
            candidate_points = payload.get("candidate_points")
            if (
                isinstance(candidate_points, bool)
                or not isinstance(candidate_points, int)
                or candidate_points < 0
            ):
                return False, "雷达方向数据缺失，平移命令被拒绝"
            # The fresh full scan explicitly found no body-height return in
            # the requested sectors. Sparse sector maps encode clear sectors
            # by omission; this is not a reduced clearance threshold.
            return True, ""
        nearest = min(distances)
        if nearest <= self.MIN_TRANSLATION_CENTER_DISTANCE_M:
            return False, f"行进方向障碍仅 {nearest:.2f} m，已停车"
        return True, ""

    def command(self, *, forward: Any, turn: Any) -> tuple[bool, str]:
        normalized_forward = self._axis(forward)
        normalized_turn = self._axis(turn)
        if normalized_forward is None or normalized_turn is None:
            return False, "键盘控制轴必须在 -1 到 1 之间"

        with self._lock:
            if not self._enabled:
                return False, "请先开启键盘控制模式"
            available, message = self._control_available()
            if not available:
                self._stop_robot("manual_control_unavailable")
                self._last_command_at = None
                self._last_forward = 0.0
                self._last_turn = 0.0
                return False, message
            if self._target == "person":
                assert self.person_writer is not None
                sequence = self.person_writer.manual(
                    forward=normalized_forward,
                    turn=normalized_turn,
                )
                self._person_mode = "manual"
                self._last_command_at = self.clock()
                self._last_forward = normalized_forward
                self._last_turn = normalized_turn
                return True, f"person manual command sequence={sequence}"

            commanded_forward = normalized_forward * self.FORWARD_SPEED_MPS
            if abs(normalized_forward) <= 1e-9 and abs(normalized_turn) > 1e-9:
                commanded_forward = self.TURNING_CREEP_MPS
            if abs(commanded_forward) > 1e-9:
                clear, message = self._translation_clear(commanded_forward)
                if not clear:
                    self._stop_robot("manual_control_risk_blocked")
                    self._last_command_at = None
                    self._last_forward = 0.0
                    self._last_turn = 0.0
                    self.events.append(
                        "safety",
                        "manual_control",
                        "Isaac keyboard translation blocked",
                        message,
                        level="warning",
                    )
                    return False, message

            try:
                sequence = self.operator_gateway.command_velocity(
                    (
                        commanded_forward,
                        0.0,
                        0.0,
                    ),
                    (
                        0.0,
                        0.0,
                        normalized_turn * self.YAW_RATE_RAD_S,
                    ),
                    duration_s=self.COMMAND_DURATION_S,
                )
            except PermissionError:
                self._stop_robot("manual_control_safety_hold")
                self._last_command_at = None
                self._last_forward = 0.0
                self._last_turn = 0.0
                return False, "安全回撤已锁定控制权，键盘命令被拒绝"
            except (RuntimeError, TimeoutError) as error:
                self._last_command_at = None
                self._last_forward = 0.0
                self._last_turn = 0.0
                return False, str(error)
            self._last_command_at = self.clock()
            self._last_forward = normalized_forward
            self._last_turn = normalized_turn
            return True, f"manual command sequence={sequence}"

    def stop(self) -> None:
        self.set_enabled(False)

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime_person = self.person_status()
            runtime_mode = (
                runtime_person.get("mode")
                if isinstance(runtime_person, dict)
                and runtime_person.get("mode") in {"auto", "manual", "paused"}
                else self._person_mode
            )
            age = (
                None
                if self._last_command_at is None
                else max(0.0, self.clock() - self._last_command_at)
            )
            active = bool(
                self._enabled
                and age is not None
                and age <= self.COMMAND_DURATION_S
                and (
                    abs(self._last_forward) > 1e-9
                    or abs(self._last_turn) > 1e-9
                )
            )
            return {
                "supported": True,
                "enabled": self._enabled,
                "active": active,
                "target": self._target,
                "person_available": bool(
                    self.person_writer is not None
                    and self.scene_status() == "task_apartment"
                ),
                "person_mode": runtime_mode,
                "person_collision_blocked": bool(
                    isinstance(runtime_person, dict)
                    and runtime_person.get("collision_blocked") is True
                ),
                "last_command_age_seconds": (
                    None if age is None else round(age, 3)
                ),
                "forward": self._last_forward if active else 0.0,
                "turn": self._last_turn if active else 0.0,
                "forward_speed_mps": self.FORWARD_SPEED_MPS,
                "yaw_rate_rad_s": self.YAW_RATE_RAD_S,
                "turning_creep_mps": self.TURNING_CREEP_MPS,
                "command_expiry_seconds": self.COMMAND_DURATION_S,
            }
