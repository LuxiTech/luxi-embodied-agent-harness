"""Deterministic local safety authority and single-writer command gateway."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    SafetyDecision,
    SafetyEvidence,
    SafetyRequest,
    SideEffect,
)


class SafetyState(str, Enum):
    IDLE = "IDLE"
    ADMITTED = "ADMITTED"
    ACTIVE = "ACTIVE"
    STOPPING = "STOPPING"
    STATIONARY = "STATIONARY"
    FAULT = "FAULT"


class SafeCommandGateway(Protocol):
    def publish_zero(self, robot_id: str) -> float: ...

    def stationary_samples(
        self, robot_id: str, after_monotonic: float
    ) -> list[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class SafetyPolicy:
    revision: str = "safety-v1"
    max_observation_age_s: float = 0.5
    stationary_speed_mps: float = 0.025
    stationary_yaw_rate_rps: float = 0.05
    min_stationary_samples: int = 2
    min_stationary_duration_s: float = 0.5
    stop_confirmation_timeout_s: float = 2.0


class LuxiSafetyKernel:
    """Per-process implementation; deploy one beside each physical robot."""

    def __init__(
        self,
        gateway: SafeCommandGateway,
        *,
        policy: SafetyPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gateway = gateway
        self.policy = policy or SafetyPolicy()
        self.clock = clock
        self._states: dict[str, SafetyState] = {}
        self._stop_locks: dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    def state(self, robot_id: str) -> SafetyState:
        with self._lock:
            return self._states.get(robot_id, SafetyState.IDLE)

    def admit(self, request: SafetyRequest) -> SafetyDecision:
        if request.side_effect is not SideEffect.PHYSICAL:
            return SafetyDecision(True, "non_physical", self.policy.revision)
        now = self.clock()
        reason = "admitted"
        admitted = True
        emergency_stop = request.capability_id in {"stop_robot", "stop_navigation"}
        if emergency_stop:
            admitted, reason = True, "emergency_stop_always_admitted"
        elif request.deadline_monotonic is not None and request.deadline_monotonic <= now:
            admitted, reason = False, "deadline_expired"
        else:
            timestamp = request.observation.get("timestamp_monotonic")
            if not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
                admitted, reason = False, "observation_timestamp_missing"
            elif now - float(timestamp) > self.policy.max_observation_age_s:
                admitted, reason = False, "observation_stale"
            elif request.observation.get("risk") in {"critical", "fault"}:
                admitted, reason = False, "risk_blocked"
        with self._lock:
            current = self._states.get(request.robot_id, SafetyState.IDLE)
            if not emergency_stop and current in {
                SafetyState.ACTIVE,
                SafetyState.STOPPING,
                SafetyState.FAULT,
            }:
                admitted, reason = False, f"state_{current.value.lower()}"
            if admitted:
                self._states[request.robot_id] = SafetyState.ADMITTED
        return SafetyDecision(
            admitted,
            reason,
            self.policy.revision,
            constraints={"stationary_speed_mps": self.policy.stationary_speed_mps},
        )

    def mark_active(self, robot_id: str) -> None:
        with self._lock:
            if self._states.get(robot_id) != SafetyState.ADMITTED:
                raise RuntimeError("physical command was not admitted")
            self._states[robot_id] = SafetyState.ACTIVE

    def mark_stationary(self, robot_id: str) -> None:
        with self._lock:
            self._states[robot_id] = SafetyState.STATIONARY

    def release(self, robot_id: str) -> None:
        with self._lock:
            if self._states.get(robot_id) not in {
                SafetyState.ACTIVE,
                SafetyState.STOPPING,
            }:
                self._states[robot_id] = SafetyState.IDLE

    def stop(self, robot_id: str, reason: str) -> SafetyEvidence:
        """Stop locally; it never depends on model, network or event storage."""

        with self._lock:
            stop_lock = self._stop_locks.setdefault(robot_id, threading.Lock())
        with stop_lock:
            return self._stop_locked(robot_id, reason)

    def _stop_locked(self, robot_id: str, reason: str) -> SafetyEvidence:
        """Run one complete stop barrier at a time for each robot."""

        with self._lock:
            self._states[robot_id] = SafetyState.STOPPING
        try:
            completed_at = self.gateway.publish_zero(robot_id)
        except Exception as exc:
            with self._lock:
                self._states[robot_id] = SafetyState.FAULT
            return SafetyEvidence(
                robot_id,
                False,
                False,
                None,
                None,
                {"reason": reason, "stop_error": str(exc)[:500]},
            )
        deadline = self.clock() + self.policy.stop_confirmation_timeout_s
        confirmed_at: float | None = None
        consecutive = 0
        while self.clock() < deadline:
            samples = self.gateway.stationary_samples(robot_id, completed_at)
            consecutive = 0
            consecutive_times: list[float] = []
            for sample in samples:
                timestamp = sample.get("timestamp_monotonic")
                speed = sample.get("planar_speed_mps")
                yaw_rate = sample.get("yaw_rate_rps")
                if (
                    not isinstance(timestamp, (int, float))
                    or float(timestamp) <= completed_at
                    or float(timestamp) > self.clock() + 0.05
                ):
                    continue
                yaw_stationary = (
                    yaw_rate is None
                    or (
                        isinstance(yaw_rate, (int, float))
                        and abs(float(yaw_rate))
                        <= self.policy.stationary_yaw_rate_rps
                    )
                )
                if (
                    isinstance(speed, (int, float))
                    and abs(float(speed)) <= self.policy.stationary_speed_mps
                    and yaw_stationary
                ):
                    consecutive += 1
                    consecutive_times.append(float(timestamp))
                    confirmed_at = float(timestamp)
                else:
                    consecutive = 0
                    consecutive_times.clear()
                    confirmed_at = None
            if consecutive >= self.policy.min_stationary_samples:
                stable_duration = (
                    consecutive_times[-1] - consecutive_times[0]
                    if len(consecutive_times) >= 2
                    else 0.0
                )
                if stable_duration < self.policy.min_stationary_duration_s:
                    time.sleep(0.02)
                    continue
                with self._lock:
                    self._states[robot_id] = SafetyState.STATIONARY
                return SafetyEvidence(
                    robot_id,
                    True,
                    True,
                    completed_at,
                    confirmed_at,
                    {
                        "reason": reason,
                        "stationary_samples": consecutive,
                        "stationary_duration_s": stable_duration,
                    },
                )
            time.sleep(0.02)
        with self._lock:
            self._states[robot_id] = SafetyState.FAULT
        return SafetyEvidence(
            robot_id,
            True,
            False,
            completed_at,
            None,
            {"reason": reason, "stationary_samples": consecutive},
        )
