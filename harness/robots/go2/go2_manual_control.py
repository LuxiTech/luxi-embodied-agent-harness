"""Fail-expiring operator control for the Go2 simulation person."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping

from harness.robots.go2.go2_protocol import Go2RuntimePaths, atomic_write_json, read_json


class Go2PersonControlWriter:
    COMMAND_DURATION_S = 0.35

    def __init__(self, paths: Go2RuntimePaths, *, clock: Callable[[], float] = time.time) -> None:
        self.path = paths.person_control
        self.clock = clock
        self._lock = threading.Lock()
        self._sequence = 0

    def _write(self, mode: str, *, forward: float = 0.0, turn: float = 0.0) -> int:
        with self._lock:
            current = read_json(self.path)
            current_sequence = current.get("sequence", 0) if isinstance(current, Mapping) else 0
            if not isinstance(current_sequence, int) or isinstance(current_sequence, bool):
                current_sequence = 0
            self._sequence = max(self._sequence, current_sequence) + 1
            now = self.clock()
            atomic_write_json(
                self.path,
                {
                    "schema_version": 1,
                    "backend": "mujoco-go2",
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


class Go2PersonManualControl:
    """Dashboard-facing person-only WASD control."""

    COMMAND_DURATION_S = Go2PersonControlWriter.COMMAND_DURATION_S
    FORWARD_SPEED_MPS = 0.40
    YAW_RATE_RAD_S = 0.90

    def __init__(
        self,
        events: Any,
        writer: Go2PersonControlWriter,
        *,
        simulation_status: Callable[[], dict[str, Any]],
        person_status: Callable[[], dict[str, Any] | None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.writer = writer
        self.simulation_status = simulation_status
        self.person_status = person_status
        self.clock = clock
        self._lock = threading.Lock()
        self._enabled = False
        self._last_command_at: float | None = None
        self._last_forward = 0.0
        self._last_turn = 0.0
        self._person_mode = "auto"

    @staticmethod
    def _axis(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) and -1.0 <= result <= 1.0 else None

    def set_enabled(self, enabled: bool) -> tuple[bool, str]:
        with self._lock:
            if enabled:
                if not self.simulation_status().get("bridge_ready"):
                    return False, "Go2 MuJoCo 尚未就绪"
                self.writer.manual(forward=0.0, turn=0.0)
                self._enabled = True
                self._person_mode = "manual"
                self._last_command_at = None
                self._last_forward = self._last_turn = 0.0
                self.events.append(
                    "ui", "manual_control", "Go2 person keyboard control enabled",
                    "W/S moves the person; A/D turns it. Commands expire after 0.35 s.",
                    level="warning",
                )
                return True, "人物键盘控制已开启"
            self.writer.pause()
            was_enabled = self._enabled
            self._enabled = False
            self._person_mode = "paused"
            self._last_command_at = None
            self._last_forward = self._last_turn = 0.0
            if was_enabled:
                self.events.append(
                    "ui", "manual_control", "Go2 person keyboard control disabled",
                    "The person was stopped with a fail-expiring zero command.",
                )
            return True, "键盘控制已关闭，人物已暂停"

    def set_target(self, target: Any) -> tuple[bool, str]:
        if target != "person":
            return False, "Go2 操作台当前只开放人物键盘控制"
        return True, "控制对象已切换为人物"

    def person_action(self, action: Any) -> tuple[bool, str]:
        if action not in {"pause", "resume"}:
            return False, "人物动作必须是 pause 或 resume"
        with self._lock:
            if action == "pause":
                self.writer.pause()
                self._person_mode = "paused"
                self._enabled = False
                self._last_command_at = None
                return True, "人物已暂停"
            self.writer.resume_auto()
            self._person_mode = "auto"
            self._enabled = False
            self._last_command_at = None
            self._last_forward = self._last_turn = 0.0
            return True, "人物已从当前位置恢复自动路线"

    def command(self, *, forward: Any, turn: Any) -> tuple[bool, str]:
        normalized_forward = self._axis(forward)
        normalized_turn = self._axis(turn)
        if normalized_forward is None or normalized_turn is None:
            return False, "键盘控制轴必须在 -1 到 1 之间"
        with self._lock:
            if not self._enabled:
                return False, "请先开启人物键盘控制"
            if not self.simulation_status().get("bridge_ready"):
                self.writer.pause()
                self._enabled = False
                return False, "Go2 MuJoCo 已离线，人物已停车"
            sequence = self.writer.manual(forward=normalized_forward, turn=normalized_turn)
            self._last_command_at = self.clock()
            self._last_forward = normalized_forward
            self._last_turn = normalized_turn
            self._person_mode = "manual"
            return True, f"person manual command sequence={sequence}"

    def stop(self) -> None:
        self.set_enabled(False)

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime = self.person_status()
            runtime_mode = (
                runtime.get("mode")
                if isinstance(runtime, dict) and runtime.get("mode") in {"auto", "manual", "paused"}
                else self._person_mode
            )
            age = None if self._last_command_at is None else max(0.0, self.clock() - self._last_command_at)
            active = bool(
                self._enabled and age is not None and age <= self.COMMAND_DURATION_S
                and (abs(self._last_forward) > 1e-9 or abs(self._last_turn) > 1e-9)
            )
            return {
                "supported": True,
                "enabled": self._enabled,
                "active": active,
                "target": "person",
                "robot_available": False,
                "person_available": True,
                "person_mode": runtime_mode,
                "person_collision_blocked": bool(isinstance(runtime, dict) and runtime.get("boundary_blocked")),
                "last_command_age_seconds": None if age is None else round(age, 3),
                "forward": self._last_forward if active else 0.0,
                "turn": self._last_turn if active else 0.0,
                "forward_speed_mps": self.FORWARD_SPEED_MPS,
                "yaw_rate_rad_s": self.YAW_RATE_RAD_S,
                "turning_creep_mps": 0.0,
                "command_expiry_seconds": self.COMMAND_DURATION_S,
            }
