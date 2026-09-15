"""Command conditioning for the reference Isaac G1 locomotion policy."""

from __future__ import annotations

import math
from typing import Sequence


def planner_velocity_for_isaac(
    x: float,
    y: float,
    yaw: float,
    *,
    forward_speed: float = 0.40,
    turn_rate: float = 0.80,
    steering_rate: float = 0.08,
) -> tuple[float, float, float]:
    """Convert nonholonomic planner output into reliable Isaac G1 commands.

    The reference locomotion policy has a practical forward dead band, does
    not provide usable lateral walking in this scene, and stalls when a
    translation command is combined with the planner's comparatively large
    steering rate.  Retain the planner signs while selecting measured
    executable magnitudes and limiting steering during translation.
    Invalid or lateral-only input fails closed.
    """

    values = (float(x), float(y), float(yaw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("planner velocity values must be finite")
    x, y, yaw = values
    if abs(x) > 1e-4:
        steering = max(-steering_rate, min(steering_rate, yaw))
        return math.copysign(forward_speed, x), 0.0, steering
    if abs(yaw) > 1e-4:
        return 0.0, 0.0, math.copysign(turn_rate, yaw)
    # Lateral-only commands are unsupported by this G1 policy.
    return 0.0, 0.0, 0.0


class IsaacPlannerVelocityAdapter:
    """Restore the original bounded stepping turn used by Isaac navigation."""

    TURN_CREEP_SPEED = 0.12
    TURN_CREEP_PHASE_TICKS = 20

    def __init__(self) -> None:
        self._turn_ticks = 0

    def adapt(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        command = planner_velocity_for_isaac(x, y, yaw)
        if abs(float(x)) > 1e-4:
            self.reset()
            return command
        if abs(float(yaw)) > 1e-4:
            phase = (self._turn_ticks // self.TURN_CREEP_PHASE_TICKS) % 2
            self._turn_ticks += 1
            creep = self.TURN_CREEP_SPEED if phase == 0 else -self.TURN_CREEP_SPEED
            return creep, 0.0, math.copysign(0.80, float(yaw))
        self.reset()
        return 0.0, 0.0, 0.0

    def reset(self) -> None:
        self._turn_ticks = 0


class IsaacGaitCommandAdapter:
    """Apply the reference keyboard controller's per-policy-tick slew limits."""

    MAX_DELTA_PER_POLICY_TICK = (0.035, 0.030, 0.050)
    DEFAULT_HEIGHT = 0.8

    def __init__(self) -> None:
        self._current = [0.0, 0.0, 0.0, self.DEFAULT_HEIGHT]

    @staticmethod
    def _validated(command: Sequence[float]) -> tuple[float, float, float, float]:
        if len(command) != 4:
            raise ValueError("G1 gait command must contain vx, vy, yaw_rate, height")
        values = tuple(float(value) for value in command)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("G1 gait command values must be finite")
        return values

    def step(
        self,
        target: Sequence[float],
    ) -> tuple[float, float, float, float]:
        values = self._validated(target)
        # A zero command is a safety boundary, not another gait setpoint.  Do
        # not spend policy ticks slewing through residual translation/yaw:
        # doing so lets the simulated base coast beyond the 500 ms physical
        # stop contract used by terminal navigation skills.  Non-zero targets
        # still use the reference controller's input ramp below.
        if max(abs(value) for value in values[:3]) <= 1e-12:
            self._current[:] = (0.0, 0.0, 0.0, values[3])
            return tuple(self._current)  # type: ignore[return-value]
        for axis, max_delta in enumerate(self.MAX_DELTA_PER_POLICY_TICK):
            delta = values[axis] - self._current[axis]
            delta = min(max_delta, max(-max_delta, delta))
            updated = self._current[axis] + delta
            self._current[axis] = 0.0 if abs(updated) < 1e-12 else updated
        self._current[3] = values[3]
        return tuple(self._current)  # type: ignore[return-value]

    def reset(self) -> tuple[float, float, float, float]:
        self._current[:] = (0.0, 0.0, 0.0, self.DEFAULT_HEIGHT)
        return tuple(self._current)  # type: ignore[return-value]
