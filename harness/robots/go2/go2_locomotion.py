"""Torque-only locomotion controller for the MuJoCo Unitree Go2.

The controller is deliberately small and inspectable.  It generates a
diagonal-trot joint trajectory and realizes it through bounded PD torques on
the twelve Go2 actuators.  It never writes the floating base pose or velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrotConfig:
    frequency_hz: float = 3.0
    position_kp: float = 45.0
    velocity_kd: float = 1.2
    command_slew_per_second: float = 2.5
    stride_per_mps: float = 2.0
    # Go2 ab/adduction response is much stronger than sagittal hip response.
    # The negative sign aligns positive Twist.linear.y with body-left.
    lateral_stride_per_mps: float = -0.35
    stride_per_yaw_rps: float = 0.70
    max_stride_radians: float = 0.38
    swing_knee_lift_radians: float = 0.35
    motion_deadband: float = 0.015


class Go2TorqueTrotController:
    """Generate a stable Go2 trot using actuator torques only."""

    HOME = np.asarray([0.0, 0.9, -1.8] * 4, dtype=np.float64)
    # FL/RR and FR/RL form the two diagonal pairs.
    PHASE_OFFSETS = (0.0, math.pi, math.pi, 0.0)
    LEFT_LEGS = frozenset((0, 2))

    JOINT_NAMES = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )
    ACTUATOR_NAMES = tuple(
        f"{leg}_{joint}"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )

    def __init__(
        self,
        model: Any,
        *,
        name_prefix: str = "",
        config: TrotConfig = TrotConfig(),
    ) -> None:
        import mujoco

        self.model = model
        self.config = config
        self.name_prefix = str(name_prefix)
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{self.name_prefix}{name}",
                )
                for name in self.JOINT_NAMES
            ],
            dtype=np.int32,
        )
        actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{self.name_prefix}{name}",
                )
                for name in self.ACTUATOR_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
            raise RuntimeError(
                f"Go2 actuator contract is incomplete for prefix {self.name_prefix!r}"
            )
        self._qpos_indices = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
        self._qvel_indices = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
        self._actuator_ids = actuator_ids
        self._filtered_linear = 0.0
        self._filtered_lateral = 0.0
        self._filtered_yaw = 0.0
        self.last_torques = np.zeros(12, dtype=np.float64)

    @staticmethod
    def _slew(current: float, target: float, maximum_delta: float) -> float:
        return current + float(np.clip(target - current, -maximum_delta, maximum_delta))

    def reset(self) -> None:
        self._filtered_linear = 0.0
        self._filtered_lateral = 0.0
        self._filtered_yaw = 0.0
        self.last_torques[:] = 0.0

    def apply(
        self,
        data: Any,
        *,
        linear_x_mps: float,
        linear_y_mps: float,
        yaw_rps: float,
        timestep: float,
    ) -> None:
        """Write only actuator torque commands into ``data.ctrl``."""

        limit = self.config.command_slew_per_second * timestep
        self._filtered_linear = self._slew(
            self._filtered_linear, float(linear_x_mps), limit
        )
        filtered_lateral = self._slew(
            self._filtered_lateral,
            float(linear_y_mps),
            limit,
        )
        self._filtered_lateral = filtered_lateral
        self._filtered_yaw = self._slew(self._filtered_yaw, float(yaw_rps), limit)

        target = self.HOME.copy()
        moving = (
            abs(self._filtered_linear) >= self.config.motion_deadband
            or abs(filtered_lateral) >= self.config.motion_deadband
            or abs(self._filtered_yaw) >= self.config.motion_deadband
        )
        if moving:
            phase = 2.0 * math.pi * self.config.frequency_hz * float(data.time)
            for leg, offset in enumerate(self.PHASE_OFFSETS):
                side_sign = -1.0 if leg in self.LEFT_LEGS else 1.0
                stride = (
                    self.config.stride_per_mps * self._filtered_linear
                    + side_sign
                    * self.config.stride_per_yaw_rps
                    * self._filtered_yaw
                )
                stride = float(
                    np.clip(
                        stride,
                        -self.config.max_stride_radians,
                        self.config.max_stride_radians,
                    )
                )
                sine = math.sin(phase + offset)
                target[3 * leg] = (
                    self.HOME[3 * leg]
                    + self.config.lateral_stride_per_mps * filtered_lateral * sine
                )
                target[3 * leg + 1] = self.HOME[3 * leg + 1] + stride * sine
                # The diagonal swing pair retracts its knees.  Reverse travel
                # changes foot fore/aft motion, not the contact schedule.
                target[3 * leg + 2] = (
                    self.HOME[3 * leg + 2]
                    + self.config.swing_knee_lift_radians * max(0.0, sine)
                )

        q = np.asarray(data.qpos[self._qpos_indices])
        qd = np.asarray(data.qvel[self._qvel_indices])
        torque = (
            self.config.position_kp * (target - q)
            - self.config.velocity_kd * qd
        )
        ranges = np.asarray(self.model.actuator_ctrlrange[self._actuator_ids])
        self.last_torques = np.clip(torque, ranges[:, 0], ranges[:, 1])
        data.ctrl[self._actuator_ids] = self.last_torques

    def status(self) -> dict[str, Any]:
        return {
            "controller": "diagonal_trot_cpg_joint_torque",
            "learned_policy": False,
            "floating_base_qpos_writes": False,
            "actuated_joints": 12,
            "name_prefix": self.name_prefix,
            "frequency_hz": self.config.frequency_hz,
            "command": {
                "linear_mps": round(self._filtered_linear, 4),
                "lateral_mps": round(self._filtered_lateral, 4),
                "yaw_rps": round(self._filtered_yaw, 4),
            },
            "peak_abs_torque_nm": round(float(np.max(np.abs(self.last_torques))), 3),
        }
