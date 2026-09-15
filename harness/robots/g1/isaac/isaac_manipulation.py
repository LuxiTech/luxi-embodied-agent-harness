"""Safe waist, arm, and Inspire Hand ownership for the Isaac G1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np

from harness.robots.g1.isaac.isaac_protocol import (
    BACKEND_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    read_json,
)


LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ARM_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
HAND_JOINT_NAMES = (
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)
ARM_JOINTS_BY_HAND = {
    side: tuple(
        f"{side}_{joint}_joint"
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )
    for side in ("left", "right")
}
MANIPULATION_JOINT_NAMES = (
    *WAIST_JOINT_NAMES,
    *ARM_JOINT_NAMES,
    *HAND_JOINT_NAMES,
)


@dataclass(frozen=True)
class JointCommandResult:
    accepted: bool
    sequence: int
    reason: str


def validate_joint_schema(
    joint_names: tuple[str, ...],
    joint_limits: np.ndarray[Any, Any],
) -> None:
    """Fail startup when the pinned articulation no longer has one exact owner."""

    if len(joint_names) != len(set(joint_names)):
        raise RuntimeError("G1 articulation contains duplicate DOF names")
    expected = set(LEG_JOINT_NAMES) | set(MANIPULATION_JOINT_NAMES)
    actual = set(joint_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"G1 manipulation joint schema mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if joint_limits.shape != (len(joint_names), 2):
        raise RuntimeError("G1 manipulation joint limits have an invalid shape")
    if not np.all(np.isfinite(joint_limits)):
        raise RuntimeError("G1 manipulation joint limits must be finite")
    if np.any(joint_limits[:, 0] > joint_limits[:, 1]):
        raise RuntimeError("G1 manipulation joint limits are reversed")


def damped_cartesian_joint_delta(
    jacobian: np.ndarray[Any, Any],
    position_error: np.ndarray[Any, Any],
    *,
    damping: float = 0.08,
    max_delta_rad: float = 0.08,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return one bounded DLS position-only IK increment."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(position_error, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 3 or matrix.shape[1] < 1:
        raise ValueError("Cartesian Jacobian must have shape (3, n)")
    if error.shape != (3,) or not np.all(np.isfinite(matrix)) or not np.all(
        np.isfinite(error)
    ):
        raise ValueError("Cartesian IK inputs must be finite")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("Cartesian IK damping must be positive")
    if not math.isfinite(max_delta_rad) or max_delta_rad <= 0.0:
        raise ValueError("Cartesian IK delta bound must be positive")
    regularized = matrix @ matrix.T + (float(damping) ** 2) * np.eye(3)
    delta = matrix.T @ np.linalg.solve(regularized, error)
    norm = float(np.linalg.norm(delta))
    if norm > max_delta_rad:
        delta *= float(max_delta_rad) / norm
    return np.ascontiguousarray(delta, dtype=np.float64)


def joint_command_payload(
    targets: Mapping[str, float],
    *,
    sequence: int,
    duration_s: float,
    now: float | None = None,
) -> dict[str, Any]:
    if sequence < 1:
        raise ValueError("manipulation command sequence must be positive")
    if not 0.1 <= float(duration_s) <= 10.0:
        raise ValueError("manipulation command duration must be 0.1 to 10 seconds")
    normalized: dict[str, float] = {}
    for name, value in targets.items():
        if name not in MANIPULATION_JOINT_NAMES:
            raise ValueError(f"joint is not owned by Isaac manipulation: {name}")
        target = float(value)
        if not math.isfinite(target):
            raise ValueError("manipulation joint targets must be finite")
        normalized[name] = target
    if not normalized:
        raise ValueError("manipulation command must contain at least one target")
    issued_at = time.time() if now is None else float(now)
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND_NAME,
        "action": "joint_targets",
        "sequence": sequence,
        "issued_at": issued_at,
        "expires_at": issued_at + float(duration_s),
        "targets_rad": normalized,
    }


class IsaacManipulationCommandWriter:
    """Atomic host writer used by the later entity adapter and acceptance tests."""

    def __init__(self, request_path: Path, ack_path: Path) -> None:
        self.request_path = request_path
        self.ack_path = ack_path
        self._sequence = 0
        self._lock = threading.Lock()

    def write_targets(
        self,
        targets: Mapping[str, float],
        *,
        duration_s: float = 3.0,
    ) -> int:
        with self._lock:
            existing = read_json(self.request_path)
            existing_sequence = (
                int(existing.get("sequence", 0))
                if isinstance(existing, Mapping)
                and isinstance(existing.get("sequence"), int)
                and not isinstance(existing.get("sequence"), bool)
                else 0
            )
            self._sequence = max(self._sequence, existing_sequence) + 1
            payload = joint_command_payload(
                targets,
                sequence=self._sequence,
                duration_s=duration_s,
            )
            atomic_write_json(self.request_path, payload)
            return self._sequence

    def wait_for_ack(self, sequence: int, timeout_s: float = 3.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            payload = read_json(self.ack_path)
            if payload and payload.get("sequence") == sequence:
                return payload
            time.sleep(0.02)
        payload = read_json(self.ack_path)
        return payload if payload and payload.get("sequence") == sequence else None


class IsaacManipulationJointController:
    """Runtime owner that overlays non-leg targets after ONNX policy inference."""

    def __init__(
        self,
        *,
        joint_names: tuple[str, ...],
        joint_limits: np.ndarray[Any, Any],
        neutral_positions: np.ndarray[Any, Any],
        request_path: Path,
        ack_path: Path,
        max_speed_rad_s: float = 0.6,
    ) -> None:
        limits = np.asarray(joint_limits, dtype=np.float64)
        neutral = np.asarray(neutral_positions, dtype=np.float64)
        validate_joint_schema(joint_names, limits)
        if neutral.shape != (len(joint_names),) or not np.all(np.isfinite(neutral)):
            raise RuntimeError("G1 manipulation neutral pose has an invalid shape")
        if not math.isfinite(max_speed_rad_s) or not 0.05 <= max_speed_rad_s <= 2.0:
            raise ValueError("G1 manipulation speed limit is invalid")
        self.joint_names = joint_names
        self.joint_to_index = {name: index for index, name in enumerate(joint_names)}
        self.indices = np.asarray(
            [self.joint_to_index[name] for name in MANIPULATION_JOINT_NAMES],
            dtype=np.int32,
        )
        self.limits = limits
        self.neutral = neutral.copy()
        self.current_targets = neutral.copy()
        self.goal_targets = neutral.copy()
        self.request_path = request_path
        self.ack_path = ack_path
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.last_sequence = 0
        self.active_until = 0.0
        self.last_result = JointCommandResult(False, 0, "no_command")

    def reset(self) -> None:
        self.current_targets[:] = self.neutral
        self.goal_targets[:] = self.neutral
        self.active_until = 0.0
        self.last_result = JointCommandResult(True, self.last_sequence, "reset")

    def set_runtime_targets(
        self,
        targets: Mapping[str, float],
        *,
        hold_s: float = 0.25,
        now: float | None = None,
    ) -> None:
        """Apply physics-runtime IK targets through the same safe owner."""

        timestamp = time.time() if now is None else float(now)
        candidate = self.goal_targets.copy()
        for name, raw in targets.items():
            if name not in MANIPULATION_JOINT_NAMES:
                raise ValueError(f"joint is not owned by Isaac manipulation: {name}")
            value = float(raw)
            index = self.joint_to_index[name]
            lower, upper = self.limits[index]
            if not math.isfinite(value):
                raise ValueError("runtime joint target must be finite")
            candidate[index] = np.clip(value, lower, upper)
        self.goal_targets[:] = candidate
        self.active_until = max(self.active_until, timestamp + max(0.05, hold_s))

    def _ack(self, result: JointCommandResult, *, now: float) -> None:
        atomic_write_json(
            self.ack_path,
            {
                "schema_version": SCHEMA_VERSION,
                "backend": BACKEND_NAME,
                "action": "joint_targets",
                "sequence": result.sequence,
                "accepted": result.accepted,
                "reason": result.reason,
                "applied_at": now,
            },
        )

    def poll(self, *, now: float | None = None) -> JointCommandResult | None:
        timestamp = time.time() if now is None else float(now)
        payload = read_json(self.request_path)
        sequence = payload.get("sequence") if payload else None
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= self.last_sequence
        ):
            return None
        self.last_sequence = sequence
        result = JointCommandResult(False, sequence, "invalid_command")
        try:
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("backend") != BACKEND_NAME
                or payload.get("action") != "joint_targets"
            ):
                raise ValueError("schema_or_backend_mismatch")
            issued_at = float(payload["issued_at"])
            expires_at = float(payload["expires_at"])
            if (
                not math.isfinite(issued_at)
                or not math.isfinite(expires_at)
                or timestamp < issued_at - 0.25
                or timestamp > expires_at
                or not 0.1 <= expires_at - issued_at <= 10.001
            ):
                raise ValueError("stale_or_invalid_deadline")
            targets = payload.get("targets_rad")
            if not isinstance(targets, Mapping) or not targets:
                raise ValueError("targets_missing")
            candidate = self.goal_targets.copy()
            for name, raw in targets.items():
                if name not in MANIPULATION_JOINT_NAMES:
                    raise ValueError(f"unowned_joint:{name}")
                value = float(raw)
                index = self.joint_to_index[name]
                lower, upper = self.limits[index]
                if not math.isfinite(value) or value < lower or value > upper:
                    raise ValueError(f"target_out_of_limits:{name}")
                candidate[index] = value
            self.goal_targets[:] = candidate
            self.active_until = expires_at
            result = JointCommandResult(True, sequence, "accepted")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            result = JointCommandResult(False, sequence, str(error))
        self.last_result = result
        self._ack(result, now=timestamp)
        return result

    def update(self, *, dt: float, now: float | None = None) -> None:
        timestamp = time.time() if now is None else float(now)
        if timestamp > self.active_until:
            self.goal_targets[self.indices] = self.neutral[self.indices]
        step = self.max_speed_rad_s * max(0.0, float(dt))
        delta = self.goal_targets[self.indices] - self.current_targets[self.indices]
        self.current_targets[self.indices] += np.clip(delta, -step, step)

    def overlay(self, policy_targets: np.ndarray[Any, Any]) -> None:
        if policy_targets.shape != self.current_targets.shape:
            raise RuntimeError("locomotion policy target shape changed")
        policy_targets[self.indices] = self.current_targets[self.indices]

    def state(self, actual_positions: np.ndarray[Any, Any]) -> dict[str, Any]:
        actual = np.asarray(actual_positions, dtype=np.float64)
        if actual.shape != self.current_targets.shape:
            raise RuntimeError("G1 joint state shape changed")
        return {
            "owned_joint_names": list(MANIPULATION_JOINT_NAMES),
            "last_sequence": self.last_sequence,
            "last_accepted": self.last_result.accepted,
            "last_reason": self.last_result.reason,
            "targets_rad": {
                name: float(self.current_targets[self.joint_to_index[name]])
                for name in MANIPULATION_JOINT_NAMES
            },
            "actual_rad": {
                name: float(actual[self.joint_to_index[name]])
                for name in MANIPULATION_JOINT_NAMES
            },
        }
