"""Exclusive, fail-expiring recovery command channel for Isaac G1."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable


class IsaacSafetyCommandChannel:
    """Own non-zero filesystem commands while critical recovery is latched."""

    def __init__(
        self,
        writer: Any,
        *,
        max_planar_speed_mps: float = 0.40,
        command_duration_s: float = 0.35,
        stop_callback: Callable[[str], Any] | None = None,
    ) -> None:
        self.writer = writer
        self.max_planar_speed_mps = min(
            0.40,
            max(0.25, float(max_planar_speed_mps)),
        )
        self.command_duration_s = min(
            0.35,
            max(0.10, float(command_duration_s)),
        )
        self._stop_callback = stop_callback
        self._lock = threading.RLock()
        self._token: str | None = None

    def force_stop(self) -> None:
        self.begin_hold()
        callback = self._stop_callback
        if callback is not None:
            callback("critical_recovery_hold")

    def bind_stop_callback(self, callback: Callable[[str], Any]) -> None:
        with self._lock:
            if self._stop_callback is not None and self._stop_callback is not callback:
                raise RuntimeError("recovery stop callback already has an owner")
            self._stop_callback = callback

    def begin_hold(self) -> object:
        with self._lock:
            if self._token is not None:
                return self._token
            token = self.writer.begin_safety_hold()
            self._token = token
            return token

    def publish(self, token: object, x: float, y: float) -> bool:
        try:
            command_x = float(x)
            command_y = float(y)
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in (command_x, command_y)):
            return False
        with self._lock:
            active_token = self._token
        if not isinstance(token, str) or token != active_token:
            return False
        magnitude = math.hypot(command_x, command_y)
        if magnitude > 1e-9:
            # The Isaac G1 policy has a measured low-speed dead zone: a
            # continuously refreshed 0.25 m/s target can settle without
            # translating.  Use its proven 0.40 m/s stepping target while the
            # controller continues to bound distance, corridor and command
            # lifetime.  Preserve direction and additionally respect the
            # adapter's narrower lateral envelope.
            scale = self.max_planar_speed_mps / magnitude
            command_x *= scale
            command_y *= scale
            if abs(command_y) > 0.25:
                lateral_scale = 0.25 / abs(command_y)
                command_x *= lateral_scale
                command_y *= lateral_scale
        try:
            self.writer.write_velocity(
                (command_x, command_y, 0.0),
                (0.0, 0.0, 0.0),
                duration_s=self.command_duration_s,
                safety_token=token,
            )
        except (OSError, PermissionError, ValueError):
            return False
        return True

    def end_hold(self, token: object) -> bool:
        if not isinstance(token, str):
            return False
        with self._lock:
            if token != self._token:
                return False
            released = self.writer.end_safety_hold(token)
            if released:
                self._token = None
            return released

    def reset(self, *, clear_stale: bool = False) -> None:
        with self._lock:
            token = self._token
        if token is not None:
            self.end_hold(token)
        elif clear_stale:
            self.writer.clear_safety_hold()
