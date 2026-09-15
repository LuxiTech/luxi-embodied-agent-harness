"""Simulation-only idle state for DimOS' dynamic G1 locomotion policy.

The pinned G1 ONNX policy is a walking/balance policy.  It keeps its gait
oscillator active for a zero velocity command, so zero velocity does not mean
an actually stationary robot.  This module adds that missing state in Luxi's
launcher without changing the pinned DimOS checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any
import weakref

import numpy as np


_ACTIVE_IDLE_STATES: weakref.WeakSet[IdlePoseStabilizer] | None = None
_ACTIVE_PLANAR_TRACKERS: weakref.WeakSet[ActivePlanarCommandTracker] | None = None

MANIPULATION_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class IdleStepResult:
    """Result of one idle-state update."""

    consumed: bool
    released: bool = False


class ActivePlanarCommandTracker:
    """Make the simulated floating base follow the requested planar twist.

    The G1 ONNX policy still drives every leg actuator.  This compatibility
    layer only removes uncommanded floating-base x/y/yaw motion, which is
    essential for a mobile-base planner that assumes cmd_vel kinematics.
    """

    def __init__(self, *, max_step_seconds: float = 0.05) -> None:
        self.max_step_seconds = max(0.001, float(max_step_seconds))
        self._active = False
        self._last_time = 0.0
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        global _ACTIVE_PLANAR_TRACKERS
        if _ACTIVE_PLANAR_TRACKERS is None:
            _ACTIVE_PLANAR_TRACKERS = weakref.WeakSet()
        _ACTIVE_PLANAR_TRACKERS.add(self)

    def reset(self) -> None:
        self._active = False
        self._last_time = 0.0

    def step(self, command: Any, data: Any) -> None:
        requested = np.asarray(command, dtype=np.float64).reshape(-1)
        if requested.size < 3 or not np.all(np.isfinite(requested[:3])):
            requested = np.zeros(3, dtype=np.float64)
        forward, left, yaw_rate = (float(value) for value in requested[:3])
        now = float(data.time)

        if not self._active:
            self._x = float(data.qpos[0])
            self._y = float(data.qpos[1])
            self._yaw = self._quaternion_yaw(data.qpos[3:7])
            self._last_time = now
            self._active = True
        else:
            elapsed = max(0.0, min(self.max_step_seconds, now - self._last_time))
            midpoint_yaw = self._yaw + yaw_rate * elapsed * 0.5
            self._x += (
                math.cos(midpoint_yaw) * forward - math.sin(midpoint_yaw) * left
            ) * elapsed
            self._y += (
                math.sin(midpoint_yaw) * forward + math.cos(midpoint_yaw) * left
            ) * elapsed
            self._yaw = math.atan2(
                math.sin(self._yaw + yaw_rate * elapsed),
                math.cos(self._yaw + yaw_rate * elapsed),
            )
            self._last_time = now

        self._apply(data, forward=forward, left=left, yaw_rate=yaw_rate)

    @staticmethod
    def _quaternion_yaw(quaternion: Any) -> float:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1e-12:
            return 0.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _apply(self, data: Any, *, forward: float, left: float, yaw_rate: float) -> None:
        data.qpos[0] = self._x
        data.qpos[1] = self._y
        data.qpos[3:7] = np.array(
            [math.cos(self._yaw / 2.0), 0.0, 0.0, math.sin(self._yaw / 2.0)]
        )
        world_x = math.cos(self._yaw) * forward - math.sin(self._yaw) * left
        world_y = math.sin(self._yaw) * forward + math.cos(self._yaw) * left
        data.qvel[0] = world_x
        data.qvel[1] = world_y
        data.qvel[3] = 0.0
        data.qvel[4] = 0.0
        data.qvel[5] = yaw_rate


class IdlePoseStabilizer:
    """Hold a repeatable upright pose while the requested velocity is zero.

    This is deliberately kinematic and simulation-only.  Dynamic balance is
    handed back to the upstream ONNX policy as soon as a real command arrives.
    """

    def __init__(
        self,
        home_qpos: Any,
        *,
        deadband: float = 0.02,
        transition_seconds: float = 0.4,
        preserved_qpos_addresses: tuple[int, ...] = (),
    ) -> None:
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self.deadband = max(0.0, float(deadband))
        self.transition_seconds = max(0.0, float(transition_seconds))
        self.preserved_qpos_addresses = tuple(
            sorted(
                {
                    int(address)
                    for address in preserved_qpos_addresses
                    if int(address) >= 7
                }
            )
        )
        self._mode = "new"
        self._source_pose: np.ndarray[Any, Any] | None = None
        self._target_pose: np.ndarray[Any, Any] | None = None
        self._transition_started_at = 0.0
        global _ACTIVE_IDLE_STATES
        if _ACTIVE_IDLE_STATES is None:
            _ACTIVE_IDLE_STATES = weakref.WeakSet()
        _ACTIVE_IDLE_STATES.add(self)

    @property
    def mode(self) -> str:
        return self._mode

    def reset(self) -> None:
        """Forget a locked world pose after the physics owner resets qpos."""

        self._mode = "new"
        self._source_pose = None
        self._target_pose = None
        self._transition_started_at = 0.0

    def step(self, command: Any, data: Any) -> IdleStepResult:
        """Apply idle control, returning whether upstream control was consumed."""
        requested = np.asarray(command, dtype=np.float64).reshape(-1)
        moving = bool(requested.size and np.max(np.abs(requested)) > self.deadband)

        if moving:
            released = self._mode in {"locked", "settling"}
            self._mode = "active"
            self._source_pose = None
            self._target_pose = None
            return IdleStepResult(consumed=False, released=released)

        if self._mode == "new":
            self._target_pose = self._make_target_pose(data.qpos, len(data.ctrl))
            self._mode = "locked"
            self._apply_pose(data, self._target_pose)
            return IdleStepResult(consumed=True)

        if self._mode == "active":
            self._source_pose = np.asarray(data.qpos, dtype=np.float64).copy()
            self._target_pose = self._make_target_pose(data.qpos, len(data.ctrl))
            self._transition_started_at = float(data.time)
            self._mode = "settling"

        if self._mode == "settling":
            assert self._source_pose is not None
            assert self._target_pose is not None
            if self.transition_seconds == 0.0:
                progress = 1.0
            else:
                progress = (float(data.time) - self._transition_started_at) / self.transition_seconds
                progress = min(1.0, max(0.0, progress))
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            pose = self._interpolate_pose(
                self._source_pose,
                self._target_pose,
                smooth_progress,
            )
            self._apply_pose(data, pose)
            if progress >= 1.0:
                self._mode = "locked"
                self._source_pose = None
            return IdleStepResult(consumed=True)

        assert self._target_pose is not None
        self._apply_pose(data, self._target_pose)
        return IdleStepResult(consumed=True)

    def _make_target_pose(
        self,
        current_qpos: Any,
        actuator_count: int,
    ) -> np.ndarray[Any, Any]:
        current = np.asarray(current_qpos, dtype=np.float64)
        if current.shape != self.home_qpos.shape:
            raise ValueError(
                f"G1 qpos shape changed: current={current.shape}, home={self.home_qpos.shape}"
            )

        target = current.copy()
        target[2] = self.home_qpos[2]
        target[3:7] = self._upright_quaternion(current[3:7])
        robot_stop = min(
            len(target),
            len(self.home_qpos),
            7 + max(0, int(actuator_count)),
        )
        target[7:robot_stop] = self.home_qpos[7:robot_stop]
        for address in self.preserved_qpos_addresses:
            if address < robot_stop:
                target[address] = current[address]
        return target

    @staticmethod
    def _upright_quaternion(quaternion: Any) -> np.ndarray[Any, Any]:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])

    @staticmethod
    def _interpolate_pose(
        source: np.ndarray[Any, Any],
        target: np.ndarray[Any, Any],
        progress: float,
    ) -> np.ndarray[Any, Any]:
        pose = source + (target - source) * progress
        source_quat = source[3:7]
        target_quat = target[3:7]
        if float(np.dot(source_quat, target_quat)) < 0.0:
            target_quat = -target_quat
        quaternion = source_quat + (target_quat - source_quat) * progress
        norm = float(np.linalg.norm(quaternion))
        pose[3:7] = quaternion / norm if norm > 1e-12 else target[3:7]
        return pose

    def _apply_pose(self, data: Any, pose: np.ndarray[Any, Any]) -> None:
        actuator_count = len(data.ctrl)
        robot_qpos_stop = min(len(data.qpos), 7 + actuator_count)
        preserved_values = {
            address: float(data.qpos[address])
            for address in self.preserved_qpos_addresses
            if address < robot_qpos_stop
        }
        preserved_ctrl = {
            address - 7: float(data.ctrl[address - 7])
            for address in self.preserved_qpos_addresses
            if 0 <= address - 7 < actuator_count
        }
        # Appended scene free joints stay owned by physics/manipulation.
        data.qpos[:robot_qpos_stop] = pose[:robot_qpos_stop]
        for address, value in preserved_values.items():
            data.qpos[address] = value
            pose[address] = value
        robot_qvel_stop = min(len(data.qvel), 6 + actuator_count)
        data.qvel[:robot_qvel_stop] = 0.0
        data.ctrl[:] = self.home_qpos[7 : 7 + actuator_count]
        for address, value in preserved_ctrl.items():
            data.ctrl[address] = value


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def reset_active_idle_stabilizers() -> int:
    """Reset worker-local motion states and return how many were notified."""

    states = [*(list(_ACTIVE_IDLE_STATES or ())), *(list(_ACTIVE_PLANAR_TRACKERS or ()))]
    for state in states:
        state.reset()
    return len(states)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_vector3(
    name: str,
    default: tuple[float, float, float],
    minimum: float,
    maximum: float,
) -> np.ndarray[Any, Any]:
    raw = os.environ.get(name)
    try:
        values = default if raw is None else tuple(float(part.strip()) for part in raw.split(","))
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError
    except ValueError:
        values = default
    return np.clip(np.asarray(values, dtype=np.float32), minimum, maximum)


def install_g1_idle_stabilizer() -> bool:
    """Replace only the worker-local G1 controller class with an idle-aware one."""
    if not _env_enabled("LUXI_G1_IDLE_STABILIZER", True):
        return False

    from dimos.simulation.mujoco import model as model_module

    base_controller = model_module.G1OnnxController
    if getattr(base_controller, "_luxi_idle_stabilized", False):
        return True

    deadband = _env_float("LUXI_G1_IDLE_DEADBAND", 0.02, 0.0, 0.2)
    transition_seconds = _env_float("LUXI_G1_IDLE_TRANSITION", 0.4, 0.0, 2.0)
    drift_override = os.environ.get("LUXI_G1_ACTIVE_DRIFT_COMPENSATION")
    active_drift_compensation = (
        _env_vector3(
            "LUXI_G1_ACTIVE_DRIFT_COMPENSATION",
            (-0.18, 0.0, -0.09),
            -0.5,
            0.5,
        )
        if drift_override is not None
        else None
    )
    planar_tracking = _env_enabled("LUXI_G1_PLANAR_TRACKER", True)

    class IdleStabilizedG1Controller(base_controller):  # type: ignore[misc, valid-type]
        _luxi_idle_stabilized = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if active_drift_compensation is not None:
                self._drift_compensation = active_drift_compensation.copy()
            self._luxi_idle_state: IdlePoseStabilizer | None = None
            self._luxi_planar_tracker = ActivePlanarCommandTracker()

        def get_control(self, model: Any, data: Any) -> None:
            if self._luxi_idle_state is None:
                preserved: list[int] = []
                for joint_name in MANIPULATION_JOINT_NAMES:
                    try:
                        preserved.append(int(model.joint(joint_name).qposadr[0]))
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                self._luxi_idle_state = IdlePoseStabilizer(
                    np.asarray(model.keyframe("home").qpos),
                    deadband=deadband,
                    transition_seconds=transition_seconds,
                    preserved_qpos_addresses=tuple(preserved),
                )

            raw_command = np.asarray(self._input_controller.get_command()).copy()
            result = self._luxi_idle_state.step(raw_command, data)
            if result.consumed:
                self._luxi_planar_tracker.reset()
                return

            if result.released:
                # Restart the dynamic policy from the same state it expects at
                # simulator startup, and produce its first action immediately.
                self._counter = self._n_substeps - 1
                self._last_action[:] = 0.0
                self._phase[:] = (0.0, np.pi)

            if planar_tracking:
                self._luxi_planar_tracker.step(raw_command, data)
            super().get_control(model, data)

    IdleStabilizedG1Controller.__name__ = "IdleStabilizedG1Controller"
    IdleStabilizedG1Controller.__qualname__ = "IdleStabilizedG1Controller"
    model_module.G1OnnxController = IdleStabilizedG1Controller
    return True
