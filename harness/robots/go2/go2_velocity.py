"""Shared Go2 ``cmd_vel`` contract for MuJoCo and Unitree SportMode.

The planner-facing boundary is deliberately ROS-shaped without requiring ROS
inside the standalone simulator: ``linear.x``, ``linear.y`` and ``angular.z``.
The real adapter can publish the same values as ``geometry_msgs/Twist`` while
the simulator consumes them through the torque gait controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any


SPORT_API_STOP_MOVE = 1003
SPORT_API_MOVE = 1008


@dataclass(frozen=True)
class VelocityLimits:
    max_linear_x: float = 0.5
    max_linear_y: float = 0.5
    max_angular_z: float = 1.0
    deadband: float = 1.0e-4
    timeout_s: float = 0.2


@dataclass(frozen=True)
class Go2VelocityCommand:
    sequence: int
    linear_x: float
    linear_y: float
    angular_z: float
    source: str
    issued_at: float

    @property
    def is_zero(self) -> bool:
        return self.linear_x == 0.0 and self.linear_y == 0.0 and self.angular_z == 0.0

    def twist(self) -> dict[str, Any]:
        return {
            "linear": {"x": self.linear_x, "y": self.linear_y, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": self.angular_z},
        }

    def sport_mode_request(self) -> dict[str, Any]:
        if self.is_zero:
            return {"api_id": SPORT_API_STOP_MOVE, "parameter": ""}
        parameter = json.dumps(
            {"x": self.linear_x, "y": self.linear_y, "z": self.angular_z},
            separators=(",", ":"),
            allow_nan=False,
        )
        return {"api_id": SPORT_API_MOVE, "parameter": parameter}


class Go2VelocityGateway:
    """Bound, audit and expire the common planner-to-base command."""

    def __init__(self, *, limits: VelocityLimits = VelocityLimits()) -> None:
        if limits.timeout_s <= 0.0:
            raise ValueError("cmd_vel timeout must be positive")
        self.limits = limits
        self._sequence = 0
        self._last_command = Go2VelocityCommand(
            sequence=0,
            linear_x=0.0,
            linear_y=0.0,
            angular_z=0.0,
            source="startup_stop",
            issued_at=time.monotonic(),
        )
        self._last_nonzero: Go2VelocityCommand | None = None
        self._source_counts: dict[str, int] = {}
        self._timed_out = False

    @staticmethod
    def _finite(value: float) -> float:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0

    def _bounded(self, value: float, limit: float) -> float:
        numeric = max(-abs(limit), min(abs(limit), self._finite(value)))
        return 0.0 if abs(numeric) <= abs(self.limits.deadband) else numeric

    def submit(
        self,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        *,
        source: str,
        now: float | None = None,
    ) -> Go2VelocityCommand:
        if not source.strip():
            raise ValueError("cmd_vel source must not be empty")
        self._sequence += 1
        self._last_command = Go2VelocityCommand(
            sequence=self._sequence,
            linear_x=self._bounded(linear_x, self.limits.max_linear_x),
            linear_y=self._bounded(linear_y, self.limits.max_linear_y),
            angular_z=self._bounded(angular_z, self.limits.max_angular_z),
            source=source.strip(),
            issued_at=time.monotonic() if now is None else float(now),
        )
        self._source_counts[self._last_command.source] = (
            self._source_counts.get(self._last_command.source, 0) + 1
        )
        if not self._last_command.is_zero:
            self._last_nonzero = self._last_command
        self._timed_out = False
        return self._last_command

    def stop(self, *, source: str, now: float | None = None) -> Go2VelocityCommand:
        return self.submit(0.0, 0.0, 0.0, source=source, now=now)

    def current(self, *, now: float | None = None) -> Go2VelocityCommand:
        current_time = time.monotonic() if now is None else float(now)
        age = max(0.0, current_time - self._last_command.issued_at)
        if not self._last_command.is_zero and age > self.limits.timeout_s:
            self._timed_out = True
            return Go2VelocityCommand(
                sequence=self._last_command.sequence,
                linear_x=0.0,
                linear_y=0.0,
                angular_z=0.0,
                source="cmd_vel_watchdog_stop",
                issued_at=current_time,
            )
        return self._last_command

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        current_time = time.monotonic() if now is None else float(now)
        active = self.current(now=current_time)
        return {
            "contract": "geometry_msgs/Twist",
            "topic": "/cmd_vel",
            "sequence": self._last_command.sequence,
            "source": self._last_command.source,
            "age_s": round(max(0.0, current_time - self._last_command.issued_at), 4),
            "watchdog_timeout_s": self.limits.timeout_s,
            "watchdog_stopped": self._timed_out,
            "requested_twist": self._last_command.twist(),
            "active_twist": active.twist(),
            "sport_mode_preview": active.sport_mode_request(),
            "last_nonzero": (
                {
                    "sequence": self._last_nonzero.sequence,
                    "source": self._last_nonzero.source,
                    "twist": self._last_nonzero.twist(),
                    "sport_mode_preview": self._last_nonzero.sport_mode_request(),
                }
                if self._last_nonzero is not None
                else None
            ),
            "source_counts": dict(self._source_counts),
            "limits": {
                "max_linear_x": self.limits.max_linear_x,
                "max_linear_y": self.limits.max_linear_y,
                "max_angular_z": self.limits.max_angular_z,
            },
            "simulation_consumer": "go2_torque_trot",
            "real_consumer": "go2_cmd_vel_bridge->unitree_sport_mode",
        }
