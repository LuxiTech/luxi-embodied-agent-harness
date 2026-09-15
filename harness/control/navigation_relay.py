"""Robot-neutral, bounded relay from planner velocity to base velocity."""

from __future__ import annotations

import math
from threading import Event, RLock, Thread
import time
from typing import Any

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3

from harness.robots.g1.isaac.isaac_gait_control import IsaacPlannerVelocityAdapter


class BoundedNavigationRelayConfig(ModuleConfig):
    max_planar_speed: float = Field(default=0.22, gt=0.0, le=0.5)
    max_yaw_rate: float = Field(default=0.55, gt=0.0, le=1.0)
    watchdog_seconds: float = Field(default=0.65, gt=0.1, le=3.0)
    stop_hold_seconds: float = Field(default=0.5, ge=0.5, le=2.0)
    isaac_gait: bool = False


def clamp_twist(message: Twist, max_planar_speed: float, max_yaw_rate: float) -> Twist:
    x, y = float(message.linear.x), float(message.linear.y)
    norm = math.hypot(x, y)
    if norm > max_planar_speed:
        scale = max_planar_speed / norm
        x, y = x * scale, y * scale
    yaw = max(-max_yaw_rate, min(max_yaw_rate, float(message.angular.z)))
    return Twist(linear=Vector3(x, y, 0.0), angular=Vector3(0.0, 0.0, yaw))


class BoundedNavigationRelay(Module):
    """Connect planners to any base exposing the conventional ``cmd_vel`` stream."""

    config: BoundedNavigationRelayConfig
    nav_cmd_vel: In[Twist]
    cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stop_event = Event()
        self._lock = RLock()
        self._last_input_at: float | None = None
        self._active = False
        self._blocked_until = 0.0
        self._deadline_blocks = 0
        self._clock = time.monotonic
        self._isaac_planner_adapter = IsaacPlannerVelocityAdapter()
        self._thread: Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.nav_cmd_vel.subscribe(self._on_command)))
        self._thread = Thread(target=self._watchdog, name="nav-relay-watchdog", daemon=True)
        self._thread.start()

    def _on_command(self, message: Twist) -> None:
        now = self._clock()
        if getattr(self.config, "isaac_gait", False):
            try:
                adapter = getattr(self, "_isaac_planner_adapter", None)
                if adapter is None:
                    adapter = IsaacPlannerVelocityAdapter()
                    self._isaac_planner_adapter = adapter
                x, y, yaw = adapter.adapt(
                    message.linear.x,
                    message.linear.y,
                    message.angular.z,
                )
                message = Twist(
                    linear=Vector3(x, y, 0.0),
                    angular=Vector3(0.0, 0.0, yaw),
                )
            except (TypeError, ValueError, OverflowError):
                message = Twist.zero()
        bounded = clamp_twist(
            message,
            self.config.max_planar_speed,
            self.config.max_yaw_rate,
        )
        nonzero = max(
            abs(bounded.linear.x), abs(bounded.linear.y), abs(bounded.angular.z)
        ) > 1e-4
        with self._lock:
            self._last_input_at = now
            was_active = self._active
            deadline_blocked = bool(nonzero and now < self._blocked_until)
            if deadline_blocked:
                self._active = False
                self._deadline_blocks += 1
            elif nonzero:
                self._active = True
            else:
                self._active = False
                if was_active:
                    self._blocked_until = max(
                        self._blocked_until,
                        now + self.config.stop_hold_seconds,
                    )
            should_publish = nonzero or was_active
        if deadline_blocked:
            self.cmd_vel.publish(Twist.zero())
        elif should_publish:
            self.cmd_vel.publish(bounded)

    def _watchdog(self) -> None:
        while not self._stop_event.wait(0.05):
            now = self._clock()
            with self._lock:
                stale = (
                    self._active
                    and self._last_input_at is not None
                    and now - self._last_input_at
                    > self.config.watchdog_seconds
                )
                if stale:
                    self._active = False
                    self._blocked_until = max(
                        self._blocked_until,
                        now + self.config.stop_hold_seconds,
                    )
            if stale:
                self.cmd_vel.publish(Twist.zero())

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self.cmd_vel.publish(Twist.zero())
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()
