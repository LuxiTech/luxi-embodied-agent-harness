"""Local DimOS runtime adapters used by the Luxi operator UI.

The pinned DimOS G1 blueprint publishes planned velocity on
``/nav_cmd_vel`` while the MuJoCo connection consumes ``/cmd_vel``.  This
module bridges that integration gap without modifying the upstream checkout.
It also keeps a compact, persistent copy of ``/global_costmap`` for the local
dashboard.

DimOS imports are deliberately lazy so the pure safety and persistence logic
can be tested without importing the full robotics runtime.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from harness.robots.g1.isaac.isaac_gait_control import IsaacPlannerVelocityAdapter


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _bounded_float(value: Any, limit: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("velocity contains a non-finite value")
    return max(-limit, min(limit, number))


@dataclass(frozen=True)
class VelocityCommand:
    x: float
    y: float
    yaw: float

    @property
    def moving(self) -> bool:
        return max(abs(self.x), abs(self.y), abs(self.yaw)) > 1e-4


class NavigationVelocityBridge:
    """Safely relay planner velocity to the simulated G1 command topic.

    Idle zero messages are ignored so an inactive planner cannot continuously
    overwrite short manual commands.  Once a non-zero planner command has been
    observed, a subsequent zero is relayed and a dead-man watchdog guarantees
    a stop if the planner stream disappears.
    """

    def __init__(
        self,
        events: Any,
        *,
        source_transport: Any | None = None,
        target_transport: Any | None = None,
        twist_factory: Callable[[float, float, float], Any] | None = None,
        stop_transport: Any | None = None,
        stop_factory: Callable[[], Any] | None = None,
        max_planar_speed: float = 0.22,
        max_yaw_rate: float = 0.55,
        warning_planar_speed: float = 0.10,
        warning_yaw_rate: float = 0.25,
        recovery_planar_speed: float = 0.06,
        watchdog_seconds: float = 0.65,
        stop_hold_seconds: float = 0.5,
        risk_provider: Callable[[], str] | None = None,
        isaac_gait: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.source_transport = source_transport
        self.target_transport = target_transport
        self.twist_factory = twist_factory
        self.stop_transport = stop_transport
        self.stop_factory = stop_factory
        self.max_planar_speed = max(0.01, float(max_planar_speed))
        self.max_yaw_rate = max(0.01, float(max_yaw_rate))
        self.warning_planar_speed = min(
            self.max_planar_speed, max(0.01, float(warning_planar_speed))
        )
        self.warning_yaw_rate = min(
            self.max_yaw_rate, max(0.01, float(warning_yaw_rate))
        )
        self.recovery_planar_speed = min(
            self.warning_planar_speed,
            max(0.01, float(recovery_planar_speed)),
        )
        self.watchdog_seconds = max(0.1, float(watchdog_seconds))
        self.stop_hold_seconds = max(0.5, float(stop_hold_seconds))
        self.risk_provider = risk_provider
        self.isaac_gait = bool(isaac_gait)
        self.clock = clock
        self._isaac_planner_adapter = IsaacPlannerVelocityAdapter()

        self._unsubscribe: Callable[[], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._output_lock = threading.RLock()
        self._running = False
        self._active = False
        self._last_input_at: float | None = None
        self._last_command = VelocityCommand(0.0, 0.0, 0.0)
        self._blocked_until = 0.0
        self._forwarded = 0
        self._clamped = 0
        self._dropped = 0
        self._watchdog_stops = 0
        self._safety_blocks = 0
        self._deadline_blocks = 0
        self._recovery_token: object | None = None
        self._recovery_commands = 0
        self._recovery_conflict_blocks = 0
        self._last_safety_event: tuple[str, float] | None = None

    @staticmethod
    def _runtime_dependencies() -> tuple[
        Any, Any, Callable[[float, float, float], Any], Any, Callable[[], Any]
    ]:
        from dimos.core.transport import LCMTransport
        from dimos.msgs.geometry_msgs.Twist import Twist
        from dimos.msgs.geometry_msgs.Vector3 import Vector3
        from dimos.msgs.std_msgs.Bool import Bool

        source = LCMTransport("/nav_cmd_vel", Twist)
        target = LCMTransport("/cmd_vel", Twist)
        stop = LCMTransport("/stop_movement", Bool)

        def make_twist(x: float, y: float, yaw: float) -> Any:
            return Twist(
                linear=Vector3(x, y, 0.0),
                angular=Vector3(0.0, 0.0, yaw),
            )

        return source, target, make_twist, stop, lambda: Bool(data=True)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
        if (
            self.source_transport is None
            or self.target_transport is None
            or self.twist_factory is None
        ):
            source, target, factory, stop, stop_factory = self._runtime_dependencies()
            self.source_transport = source
            self.target_transport = target
            self.twist_factory = factory
            self.stop_transport = stop
            self.stop_factory = stop_factory

        self._stop.clear()
        self._unsubscribe = self.source_transport.subscribe(self._on_message)
        with self._lock:
            self._running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="navigation-velocity-watchdog",
            daemon=True,
        )
        self._thread.start()
        self.events.append(
            "navigation",
            "lifecycle",
            "Navigation velocity bridge ready",
            (
                f"/nav_cmd_vel → /cmd_vel · planar≤{self.max_planar_speed:.2f} m/s · "
                f"yaw≤{self.max_yaw_rate:.2f} rad/s · watchdog={self.watchdog_seconds:.2f}s"
            ),
        )

    def _extract(self, message: Any) -> VelocityCommand:
        raw_x = float(message.linear.x)
        raw_y = float(message.linear.y)
        raw_yaw = float(message.angular.z)
        if not all(math.isfinite(value) for value in (raw_x, raw_y, raw_yaw)):
            raise ValueError("velocity contains a non-finite value")
        if self.isaac_gait:
            raw_x, raw_y, raw_yaw = self._isaac_planner_adapter.adapt(
                raw_x,
                raw_y,
                raw_yaw,
            )

        magnitude = math.hypot(raw_x, raw_y)
        if magnitude > self.max_planar_speed:
            scale = self.max_planar_speed / magnitude
            x = raw_x * scale
            y = raw_y * scale
        else:
            x, y = raw_x, raw_y
        yaw = _bounded_float(raw_yaw, self.max_yaw_rate)
        return VelocityCommand(x, y, yaw)

    @staticmethod
    def _limit_command(
        command: VelocityCommand,
        planar_limit: float,
        yaw_limit: float,
    ) -> VelocityCommand:
        magnitude = math.hypot(command.x, command.y)
        if magnitude > planar_limit:
            scale = planar_limit / magnitude
            x = command.x * scale
            y = command.y * scale
        else:
            x, y = command.x, command.y
        return VelocityCommand(x, y, _bounded_float(command.yaw, yaw_limit))

    def set_risk_provider(self, provider: Callable[[], str] | None) -> None:
        self.risk_provider = provider

    def _risk(self) -> str:
        if self.risk_provider is None:
            return "unavailable"
        try:
            risk = str(self.risk_provider()).lower()
        except Exception:
            return "unknown"
        return (
            risk if risk in {"clear", "warning", "critical", "unknown"} else "unknown"
        )

    def _on_message(self, message: Any) -> None:
        try:
            command = self._extract(message)
            raw = VelocityCommand(
                float(message.linear.x),
                float(message.linear.y),
                float(message.angular.z),
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            with self._lock:
                self._dropped += 1
            self.events.append(
                "safety",
                "navigation",
                "Dropped invalid planner velocity",
                str(error),
                level="danger",
            )
            return

        now = self.clock()
        with self._lock:
            recovery_blocked = bool(command.moving and self._recovery_token is not None)
            deadline_blocked = bool(
                command.moving
                and self._recovery_token is None
                and now < self._blocked_until
            )
            if recovery_blocked:
                self._recovery_conflict_blocks += 1
            if deadline_blocked:
                self._active = False
                self._last_command = VelocityCommand(0.0, 0.0, 0.0)
                self._deadline_blocks += 1
        if recovery_blocked:
            return
        if deadline_blocked:
            self._broadcast_stop(repeats=1)
            return

        risk = self._risk()
        if command.moving and risk in {"unknown", "critical"}:
            with self._lock:
                was_active = self._active
                self._active = False
                self._last_command = VelocityCommand(0.0, 0.0, 0.0)
                self._blocked_until = max(
                    self._blocked_until,
                    now + self.stop_hold_seconds,
                )
                self._safety_blocks += 1
                should_announce = bool(
                    self._last_safety_event is None
                    or self._last_safety_event[0] != risk
                    or now - self._last_safety_event[1] >= 2.0
                )
                if should_announce:
                    self._last_safety_event = (risk, now)
            if was_active:
                self._broadcast_stop()
            if should_announce:
                self.events.append(
                    "safety",
                    "navigation",
                    "Planner velocity blocked by proximity state",
                    f"risk={risk}; non-zero /nav_cmd_vel was not forwarded.",
                    level="danger" if risk == "critical" else "warning",
                )
            return
        if command.moving and risk == "warning":
            command = self._limit_command(
                command,
                self.warning_planar_speed,
                self.warning_yaw_rate,
            )

        activated = False
        deactivated = False
        with self._lock:
            if not command.moving and not self._active:
                return
            if command.moving and not self._active:
                self._active = True
                activated = True
            elif not command.moving and self._active:
                self._active = False
                deactivated = True
                self._blocked_until = max(
                    self._blocked_until,
                    now + self.stop_hold_seconds,
                )
            self._last_input_at = now
            self._last_command = command
            self._forwarded += 1
            if command != raw:
                self._clamped += 1

        self._broadcast(command)
        if activated:
            self.events.append(
                "navigation",
                "command",
                "Planner control activated",
                "Forwarding bounded /nav_cmd_vel commands to the G1 simulation.",
                data={"x": command.x, "y": command.y, "yaw": command.yaw},
            )
        elif deactivated:
            self.events.append(
                "navigation",
                "command",
                "Planner control stopped",
                "The planner published zero velocity.",
            )

    def _broadcast(self, command: VelocityCommand) -> None:
        if self.target_transport is None or self.twist_factory is None:
            return
        with self._output_lock:
            self.target_transport.broadcast(
                None,
                self.twist_factory(command.x, command.y, command.yaw),
            )

    def _broadcast_stop(self, repeats: int = 3) -> None:
        stop = VelocityCommand(0.0, 0.0, 0.0)
        with self._output_lock:
            for index in range(max(1, repeats)):
                self._broadcast(stop)
                if index + 1 < repeats:
                    self._stop.wait(0.02)

    def begin_safety_recovery(self) -> object:
        """Block planner output and return a capability for local recovery motion."""

        token = object()
        now = self.clock()
        with self._output_lock:
            with self._lock:
                self._active = False
                self._last_command = VelocityCommand(0.0, 0.0, 0.0)
                self._last_input_at = now
                self._blocked_until = max(
                    self._blocked_until,
                    now + self.stop_hold_seconds,
                )
                self._recovery_token = token
            self._broadcast_stop()
        return token

    def publish_safety_recovery(
        self,
        token: object,
        x: float,
        y: float,
    ) -> bool:
        """Publish one tightly bounded, yaw-free recovery command."""

        try:
            command = self._limit_command(
                VelocityCommand(float(x), float(y), 0.0),
                self.recovery_planar_speed,
                0.01,
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in (command.x, command.y)):
            return False

        with self._output_lock:
            with self._lock:
                if token is not self._recovery_token:
                    return False
                if (
                    not self._running
                    or self.target_transport is None
                    or self.twist_factory is None
                ):
                    return False
                self._last_input_at = self.clock()
                self._last_command = command
                self._recovery_commands += 1
            self._broadcast(command)
        return True

    def end_safety_recovery(self, token: object) -> bool:
        """Release the recovery hold after an explicit new planning decision."""

        now = self.clock()
        with self._output_lock:
            with self._lock:
                if token is not self._recovery_token:
                    return False
                self._recovery_token = None
                self._last_command = VelocityCommand(0.0, 0.0, 0.0)
                self._last_input_at = now
                self._blocked_until = max(
                    self._blocked_until,
                    now + self.stop_hold_seconds,
                )
            self._broadcast_stop()
        return True

    def check_watchdog(self, now: float | None = None) -> bool:
        """Stop stale planner control; exposed for deterministic tests."""

        current = self.clock() if now is None else now
        with self._lock:
            recovery_stale = bool(
                self._recovery_token is not None
                and self._last_input_at is not None
                and current - self._last_input_at > self.watchdog_seconds
            )
            planner_stale = bool(
                self._active
                and self._last_input_at is not None
                and current - self._last_input_at > self.watchdog_seconds
            )
            stale = recovery_stale or planner_stale
            if not stale:
                return False
            self._active = False
            self._last_command = VelocityCommand(0.0, 0.0, 0.0)
            if recovery_stale:
                # A stalled recovery controller must not release the old
                # planner.  Keep the capability latched and periodically zero
                # output until the controller resumes or an operator stops it.
                self._last_input_at = current
            self._blocked_until = max(
                self._blocked_until,
                current + self.stop_hold_seconds,
            )
            self._watchdog_stops += 1
        self._broadcast_stop()
        self.events.append(
            "safety",
            "navigation",
            "Navigation watchdog stopped G1",
            f"No planner velocity arrived for {self.watchdog_seconds:.2f}s.",
            level="warning",
        )
        return True

    def _watchdog_loop(self) -> None:
        interval = min(0.1, self.watchdog_seconds / 3.0)
        while not self._stop.wait(interval):
            self.check_watchdog()

    def force_stop(self, *, announce: bool = False, preserve_recovery: bool = False) -> None:
        with self._output_lock:
            with self._lock:
                was_active = self._active or self._recovery_token is not None
                self._active = False
                if not preserve_recovery:
                    self._recovery_token = None
                self._last_command = VelocityCommand(0.0, 0.0, 0.0)
                self._blocked_until = max(
                    self._blocked_until,
                    self.clock() + self.stop_hold_seconds,
                )
            if self.target_transport is not None:
                self._broadcast_stop()
            # A zero command is not a planner cancellation. Notify the planner
            # in its own worker before the late-command barrier expires, so a
            # detached terminal Skill cannot resume motion.
            if self.stop_transport is not None and self.stop_factory is not None:
                self.stop_transport.broadcast(None, self.stop_factory())
        if announce and (was_active or self._running):
            self.events.append(
                "safety",
                "navigation",
                "Navigation bridge forced to zero",
                "Published three zero-velocity commands.",
                level="warning",
            )

    def reset(self) -> None:
        """Return the bridge to a fresh experiment state without disconnecting LCM."""

        self.force_stop()
        with self._lock:
            self._active = False
            self._last_input_at = None
            self._last_command = VelocityCommand(0.0, 0.0, 0.0)
            self._blocked_until = 0.0
            self._forwarded = 0
            self._clamped = 0
            self._dropped = 0
            self._watchdog_stops = 0
            self._safety_blocks = 0
            self._deadline_blocks = 0
            self._recovery_token = None
            self._recovery_commands = 0
            self._recovery_conflict_blocks = 0
            self._last_safety_event = None

    def stop(self) -> None:
        with self._lock:
            running = self._running
            self._running = False
        if not running:
            return
        self.force_stop()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._unsubscribe:
            try:
                self._unsubscribe()
            except Exception:
                pass
        for transport in (
            self.source_transport,
            self.target_transport,
            self.stop_transport,
        ):
            try:
                transport.stop()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = (
                None
                if self._last_input_at is None
                else max(0.0, self.clock() - self._last_input_at)
            )
            return {
                "running": self._running,
                "active": self._active,
                "recovery_active": self._recovery_token is not None,
                "source_topic": "/nav_cmd_vel",
                "target_topic": "/cmd_vel",
                "max_planar_speed": self.max_planar_speed,
                "max_yaw_rate": self.max_yaw_rate,
                "warning_planar_speed": self.warning_planar_speed,
                "warning_yaw_rate": self.warning_yaw_rate,
                "recovery_planar_speed": self.recovery_planar_speed,
                "watchdog_seconds": self.watchdog_seconds,
                "stop_hold_seconds": self.stop_hold_seconds,
                "risk": self._risk(),
                "last_input_age_seconds": round(age, 3) if age is not None else None,
                "last_command": {
                    "x": self._last_command.x,
                    "y": self._last_command.y,
                    "yaw": self._last_command.yaw,
                },
                "forwarded": self._forwarded,
                "clamped": self._clamped,
                "dropped": self._dropped,
                "watchdog_stops": self._watchdog_stops,
                "safety_blocks": self._safety_blocks,
                "deadline_blocks": self._deadline_blocks,
                "recovery_commands": self._recovery_commands,
                "recovery_conflict_blocks": self._recovery_conflict_blocks,
                "stop_hold_remaining_seconds": round(
                    max(0.0, self._blocked_until - self.clock()),
                    3,
                ),
            }


class CostmapMonitor:
    """Subscribe to, expose and persist the latest DimOS occupancy costmap."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        events: Any,
        snapshot_path: Path,
        *,
        transport: Any | None = None,
        persist_interval_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.events = events
        self.snapshot_path = snapshot_path
        self.transport = transport
        self.persist_interval_seconds = max(0.0, float(persist_interval_seconds))
        self.clock = clock

        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._unsubscribe: Callable[[], None] | None = None
        self._payload: dict[str, Any] | None = None
        self._received_at: float | None = None
        self._last_persisted_at = 0.0
        self._revision = 0
        self._running = False
        self._load_saved()

    @staticmethod
    def _runtime_transport() -> Any:
        from dimos.core.transport import LCMTransport
        from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

        return LCMTransport("/global_costmap", OccupancyGrid)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
        if self.transport is None:
            self.transport = self._runtime_transport()
        self._unsubscribe = self.transport.subscribe(self._on_message)
        with self._lock:
            self._running = True
        self.events.append(
            "mapping",
            "lifecycle",
            "Costmap monitor ready",
            "/global_costmap is connected to the Luxi UI.",
        )

    @staticmethod
    def _grid_bytes(grid: Any) -> bytes:
        tobytes = getattr(grid, "tobytes", None)
        if callable(tobytes):
            try:
                return tobytes(order="C")
            except TypeError:
                return tobytes()
        try:
            view = memoryview(grid)
            if not view.contiguous:
                raise TypeError("grid buffer is not contiguous")
            return view.cast("B").tobytes()
        except (TypeError, ValueError):
            pass

        values: list[int] = []
        for row in grid:
            if isinstance(row, (list, tuple)):
                values.extend(int(value) for value in row)
            else:
                values.append(int(row))
        return bytes(value & 0xFF for value in values)

    @staticmethod
    def _origin(message: Any) -> tuple[float, float, float]:
        origin = message.origin
        position = origin.position
        orientation = origin.orientation
        w = float(getattr(orientation, "w", 1.0))
        x = float(getattr(orientation, "x", 0.0))
        y = float(getattr(orientation, "y", 0.0))
        z = float(getattr(orientation, "z", 0.0))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(position.x), float(position.y), yaw

    def _on_message(self, message: Any) -> None:
        try:
            width = int(message.width)
            height = int(message.height)
            resolution = float(message.resolution)
            if (
                width <= 0
                or height <= 0
                or not math.isfinite(resolution)
                or resolution <= 0
            ):
                raise ValueError("invalid costmap dimensions or resolution")
            raw = self._grid_bytes(message.grid)
            if len(raw) != width * height:
                raise ValueError(
                    f"costmap payload has {len(raw)} cells, expected {width * height}"
                )
            origin_x, origin_y, origin_yaw = self._origin(message)
            timestamp = float(getattr(message, "ts", time.time()))
            if not math.isfinite(timestamp):
                timestamp = time.time()
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self.events.append(
                "mapping",
                "error",
                "Dropped invalid costmap",
                str(error),
                level="danger",
            )
            return

        unknown = raw.count(255)
        free = raw.count(0)
        occupied = sum(1 for value in raw if 50 <= value <= 100)
        saturated_height_cost = raw.count(100)
        high_cost = sum(1 for value in raw if 50 <= value < 100)
        cost = sum(1 for value in raw if 1 <= value < 50)
        now = self.clock()
        with self._lock:
            self._revision += 1
            payload = {
                "available": True,
                "version": self.FORMAT_VERSION,
                "revision": self._revision,
                "source": "live",
                "topic": "/global_costmap",
                "frame_id": str(getattr(message, "frame_id", "world")),
                "timestamp": timestamp,
                "captured_at": _utc_now(),
                "width": width,
                "height": height,
                "resolution": resolution,
                "origin": {"x": origin_x, "y": origin_y, "yaw": origin_yaw},
                "encoding": "base64-int8-row-major",
                "data": base64.b64encode(raw).decode("ascii"),
                "cells": {
                    "total": width * height,
                    "unknown": unknown,
                    "free": free,
                    "cost": cost,
                    "occupied": occupied,
                    "saturated_height_cost": saturated_height_cost,
                    "high_cost": high_cost,
                    "known": width * height - unknown,
                },
            }
            first_live = self._payload is None or self._payload.get("source") != "live"
            self._payload = payload
            self._received_at = now

        if first_live:
            self.events.append(
                "mapping",
                "observation",
                "Live costmap received",
                f"{width}×{height} cells · {resolution:.3f} m/cell",
                data={"cells": payload["cells"]},
            )
        if now - self._last_persisted_at >= self.persist_interval_seconds:
            self.save_now(announce=False)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            if self._payload is None:
                return {"available": False}
            payload = dict(self._payload)
            received_at = self._received_at
        payload["age_seconds"] = (
            None
            if received_at is None
            else round(max(0.0, self.clock() - received_at), 3)
        )
        return payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = self._payload
            running = self._running
            received_at = self._received_at
        if payload is None:
            return {
                "available": False,
                "running": running,
                "topic": "/global_costmap",
                "snapshot_path": str(self.snapshot_path),
            }
        status = {
            key: payload[key]
            for key in (
                "available",
                "revision",
                "source",
                "topic",
                "frame_id",
                "timestamp",
                "captured_at",
                "width",
                "height",
                "resolution",
                "origin",
                "cells",
            )
        }
        age = None if received_at is None else max(0.0, self.clock() - received_at)
        status.update(
            {
                "running": running,
                "age_seconds": round(age, 3) if age is not None else None,
                "snapshot_path": str(self.snapshot_path),
            }
        )
        return status

    def _load_saved(self) -> None:
        try:
            with gzip.open(self.snapshot_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if (
                not isinstance(payload, dict)
                or payload.get("version") != self.FORMAT_VERSION
            ):
                raise ValueError("unsupported costmap snapshot format")
            width = int(payload["width"])
            height = int(payload["height"])
            raw = base64.b64decode(payload["data"], validate=True)
            if len(raw) != width * height:
                raise ValueError("saved costmap cell count does not match dimensions")
        except FileNotFoundError:
            return
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            self.events.append(
                "mapping",
                "error",
                "Could not load saved costmap",
                str(error),
                level="warning",
            )
            return

        payload["available"] = True
        payload["source"] = "saved"
        payload["revision"] = 0
        with self._lock:
            self._payload = payload
            self._received_at = None
        self.events.append(
            "mapping",
            "lifecycle",
            "Loaded saved costmap snapshot",
            f"{width}×{height} cells from {self.snapshot_path}",
        )

    def _pgm_bytes(self, payload: dict[str, Any]) -> bytes:
        width = int(payload["width"])
        height = int(payload["height"])
        raw = base64.b64decode(payload["data"])
        rows = [raw[index * width : (index + 1) * width] for index in range(height)]
        pixels = bytearray()
        for row in reversed(rows):
            for value in row:
                if value == 255:
                    pixels.append(205)
                else:
                    pixels.append(max(0, min(254, round(254 * (1.0 - value / 100.0)))))
        header = (
            f"P5\n# LuxiAgent DimOS global costmap\n{width} {height}\n255\n".encode()
        )
        return header + bytes(pixels)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    def save_now(self, *, announce: bool = True) -> tuple[bool, str]:
        payload = self.payload()
        if not payload.get("available"):
            return False, "还没有可保存的 costmap"
        with self._persist_lock:
            try:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                compressed = gzip.compress(encoded, compresslevel=6, mtime=0)
                self._atomic_write(self.snapshot_path, compressed)

                pgm_path = self.snapshot_path.with_name("latest-costmap.pgm")
                yaml_path = self.snapshot_path.with_name("latest-costmap.yaml")
                self._atomic_write(pgm_path, self._pgm_bytes(payload))
                origin = payload["origin"]
                yaml = (
                    "image: latest-costmap.pgm\n"
                    f"resolution: {float(payload['resolution']):.9g}\n"
                    f"origin: [{float(origin['x']):.9g}, {float(origin['y']):.9g}, "
                    f"{float(origin['yaw']):.9g}]\n"
                    "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
                ).encode("utf-8")
                self._atomic_write(yaml_path, yaml)
                self._last_persisted_at = self.clock()
            except (OSError, ValueError, TypeError, KeyError) as error:
                if announce:
                    self.events.append(
                        "mapping",
                        "error",
                        "Costmap snapshot failed",
                        str(error),
                        level="danger",
                    )
                return False, str(error)

        message = f"已保存到 {self.snapshot_path.parent}"
        if announce:
            self.events.append(
                "mapping",
                "snapshot",
                "Costmap snapshot saved",
                message,
            )
        return True, message

    def reset(self) -> list[str]:
        """Forget the live grid and delete every exported snapshot for this run."""

        snapshot_paths = [
            self.snapshot_path,
            self.snapshot_path.with_name("latest-costmap.pgm"),
            self.snapshot_path.with_name("latest-costmap.yaml"),
        ]
        removed: list[str] = []
        errors: list[str] = []
        with self._persist_lock:
            with self._lock:
                self._payload = None
                self._received_at = None
                self._last_persisted_at = 0.0
                self._revision = 0
            for path in snapshot_paths:
                try:
                    path.unlink()
                    removed.append(str(path))
                except FileNotFoundError:
                    continue
                except OSError as error:
                    errors.append(f"{path}: {error}")

        if errors:
            raise RuntimeError("无法清理地图快照：" + "; ".join(errors))
        return removed

    def stop(self) -> None:
        with self._lock:
            running = self._running
            self._running = False
        if not running:
            return
        self.save_now(announce=False)
        if self._unsubscribe:
            try:
                self._unsubscribe()
            except Exception:
                pass
        try:
            self.transport.stop()
        except Exception:
            pass
