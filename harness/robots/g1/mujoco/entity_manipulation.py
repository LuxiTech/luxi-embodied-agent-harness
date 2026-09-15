"""Simulation-only entity manipulation for the pinned G1 MuJoCo worker.

The pinned worker has no scene-entity RPC surface.  Luxi keeps that checkout
immutable and uses the same atomic-file pattern as pose reset: the UI/skills
write one bounded request, while the physics-owning worker performs IK,
contact-gated attachment and placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import mujoco
import numpy as np
from scipy.optimize import least_squares

from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD
from harness.robots.entity_port import (
    EntityOperationResult,
    entity_result,
    normalize_entity_id,
)
from harness.control.simulation_control import _atomic_json, configured_control_path


ENTITY_CONTROL_PATH_ENV = "LUXI_ENTITY_CONTROL_PATH"
MAX_CONTROL_BYTES = 16_384
SUPPORTED_ENTITIES = frozenset({"water_bottle"})
HAND_BODIES = {
    "right": "right_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
    "left": "left_wrist_yaw_link",
    "left_hand": "left_wrist_yaw_link",
}
HAND_COLLISION_GEOMS = {
    "right_wrist_yaw_link": "right_hand_collision",
    "left_wrist_yaw_link": "left_hand_collision",
}
ARM_JOINTS = {
    "right_wrist_yaw_link": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    "left_wrist_yaw_link": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
}
WAIST_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
MANIPULATION_WAIST_PITCH_RAD = 0.35
HAND_GRASP_OFFSETS = {
    # The native rubber-hand collision capsule reaches about 0.15 m from the
    # wrist; placing the 32 mm-radius bottle centre at 0.18 m gives contact
    # without the deep overlap caused by targeting the wrist itself.
    "right_wrist_yaw_link": np.array([0.18, -0.003, 0.0]),
    "left_wrist_yaw_link": np.array([0.18, 0.003, 0.0]),
}
GRIPPER_OPEN_APERTURE_M = 0.09


def configured_entity_control_path() -> Path:
    value = os.environ.get(ENTITY_CONTROL_PATH_ENV, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return (configured_control_path() / "entities").resolve()


class EntityControlChannel:
    """Client side of the worker-owned manipulation channel."""

    backend = "mujoco"

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or configured_entity_control_path()).expanduser().resolve()
        self.request_path = self.root / "request.json"
        self.ack_path = self.root / "ack.json"
        self.state_path = self.root / "state.json"

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if len(raw.encode("utf-8")) > MAX_CONTROL_BYTES:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def state(self) -> dict[str, Any] | None:
        return self._read(self.state_path)

    def request(
        self,
        action: str,
        arguments: dict[str, Any],
        *,
        timeout: float = 8.0,
        cancel=None,
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        if action in {"sim_attach", "sim_release", "sim_place"}:
            prior_cancel = self._read(self.root / "sim-cancel.json") or {}
            arguments = {**arguments, "cancellation_revision": prior_cancel.get("token")}
        if cancel is not None:
            cancel.raise_if_cancelled()
        _atomic_json(
            self.request_path,
            {
                "schema_version": 1,
                "token": token,
                "action": action,
                "arguments": arguments,
                "requested_at": time.time(),
            },
        )
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if cancel is not None and cancel.cancelled:
                _atomic_json(self.root / "sim-cancel.json", {"token": uuid.uuid4().hex})
                return {"ok": False, "token": token, "action": action,
                        "error": "entity control cancelled; side effect unknown"}
            acknowledgement = self._read(self.ack_path)
            if acknowledgement and acknowledgement.get("token") == token:
                return acknowledgement
            time.sleep(0.03)
        acknowledgement = self._read(self.ack_path)
        if acknowledgement and acknowledgement.get("token") == token:
            return acknowledgement
        return {
            "ok": False,
            "token": token,
            "action": action,
            "error": "entity control timed out",
        }

    def _port_request(
        self,
        action: str,
        entity_id: str,
        arguments: dict[str, Any],
        *,
        success_status: str,
        timeout: float,
        cancel=None,
    ) -> EntityOperationResult:
        canonical = normalize_entity_id(entity_id)
        response = self.request(
            action,
            {"entity_id": canonical, **arguments},
            timeout=timeout,
            **({"cancel": cancel} if cancel is not None else {}),
        )
        operation_ok = bool(response.get("ok"))
        error = response.get("error")
        timed_out = error in {"entity control timed out", "entity control cancelled; side effect unknown"}
        failure_status = (
            "entity_backend_timeout"
            if timed_out
            else "contact_required"
            if error == "physical hand/entity contact is required"
            else "entity_unavailable"
            if error == "unknown or unavailable entity"
            else "entity_operation_failed"
        )
        result = entity_result(
            backend=self.backend,
            entity_id=canonical,
            task_status=(
                success_status
                if operation_ok
                else failure_status
            ),
            operation_ok=operation_ok,
            completed=operation_ok,
            tool_ok=not timed_out,
            error=str(error) if error else None,
            details=dict(response),
        )
        applied_at = response.get("applied_at")
        if (
            isinstance(applied_at, (int, float))
            and not isinstance(applied_at, bool)
            and math.isfinite(float(applied_at))
        ):
            result["evidence_timestamp"] = float(applied_at)
        for field in ("hand", "pose", "contact", "attached"):
            if field in response:
                result[field] = response[field]
        return result

    def attach_at_pose(self, entity_id, pickup_pose, *, timeout=2.0, cancel=None, hand="right"):
        return self._port_request(
            "sim_attach", entity_id,
            {"pickup_pose": list(pickup_pose), "expires_at": time.time() + timeout, "hand": hand},
            success_status="sim_attachment_verified", timeout=timeout, cancel=cancel)

    def place_attachment(self, entity_id, surface, *, timeout=2.0, cancel=None):
        return self._port_request("sim_place", entity_id,
            {"surface": surface, "expires_at": time.time()+timeout},
            success_status="sim_placement_applied", timeout=timeout, cancel=cancel)

    def release_attachment(self, entity_id, *, timeout=2.0, cancel=None):
        return self._port_request(
            "sim_release", entity_id, {"expires_at": time.time() + timeout},
            success_status="sim_attachment_released", timeout=timeout, cancel=cancel)

    def entity_state(
        self,
        entity_id: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "status",
            entity_id,
            {},
            success_status="entity_state_available",
            timeout=timeout,
        )

    def contact_state(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "contact",
            entity_id,
            {"hand": hand},
            success_status="contact_observed",
            timeout=timeout,
        )

    def approach(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "approach",
            entity_id,
            {"hand": hand},
            success_status="approach_completed",
            timeout=timeout,
        )

    def grasp(
        self,
        entity_id: str,
        hand: str,
        *,
        evidence_timestamp: float | None = None,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        arguments: dict[str, Any] = {"hand": hand}
        if evidence_timestamp is not None:
            arguments["evidence_timestamp"] = float(evidence_timestamp)
        return self._port_request(
            "grasp",
            entity_id,
            arguments,
            success_status="grasp_completed",
            timeout=timeout,
        )

    def carry(
        self,
        entity_id: str,
        hand: str,
        target_pose: Any = None,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        if target_pose is not None:
            return entity_result(
                backend=self.backend,
                entity_id=normalize_entity_id(entity_id),
                task_status="carry_target_unsupported",
                operation_ok=False,
                error="MuJoCo adapter currently supports its verified carry posture only",
            )
        return self._port_request(
            "carry_pose",
            entity_id,
            {"hand": hand},
            success_status="carry_pose_completed",
            timeout=timeout,
        )

    def place(
        self,
        entity_id: str,
        hand: str,
        target_position: Any,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "place",
            entity_id,
            {"hand": hand, "target_position": list(target_position)},
            success_status="place_completed",
            timeout=timeout,
        )

    def release(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "release",
            entity_id,
            {"hand": hand},
            success_status="release_completed",
            timeout=timeout,
        )

    def reset_entity(
        self,
        entity_id: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult:
        return self._port_request(
            "reset",
            entity_id,
            {},
            success_status="entity_reset",
            timeout=timeout,
        )


@dataclass
class _Attachment:
    child_body_id: int
    child_qpos_adr: int
    child_qvel_adr: int
    parent_body_id: int
    local_position: np.ndarray[Any, Any]
    local_quaternion: np.ndarray[Any, Any]
    geom_ids: np.ndarray[Any, Any]
    geom_contype: np.ndarray[Any, Any]
    geom_conaffinity: np.ndarray[Any, Any]
    hand: str


@dataclass
class _ArmMotion:
    token: str
    action: str
    entity_id: str
    hand: str
    qpos_adrs: tuple[int, ...]
    qvel_adrs: tuple[int, ...]
    ctrl_adrs: tuple[int, ...]
    start: np.ndarray[Any, Any]
    target: np.ndarray[Any, Any]
    started_at: float
    duration: float
    place_target: tuple[float, float, float] | None = None


class EntityManipulationController:
    """Physics-owner controller polled once per MuJoCo display frame."""

    def __init__(
        self,
        model: Any,
        data: Any,
        channel: EntityControlChannel | None = None,
        *, allow_sim_attachment: bool | None = None,
    ) -> None:
        self.allow_sim_attachment = (os.environ.get("LUXI_SIM_ATTACHMENT_DEV") == "1"
                                     if allow_sim_attachment is None else allow_sim_attachment)
        self._sim_attachments: set[str] = set()
        self._sim_placements: dict[str, dict] = {}
        self._sim_stationary_samples = 0
        self._sim_last_sample_time = None
        self.model = model
        self.data = data
        self.channel = channel or EntityControlChannel()
        self._last_token: str | None = None
        self._motion: _ArmMotion | None = None
        self._held_joints: dict[int, float] = {}
        self._held_qvels: dict[int, float] = {}
        self._held_controls: dict[int, float] = {}
        self._attachments: dict[str, _Attachment] = {}
        self._resting_qpos: dict[str, np.ndarray[Any, Any]] = {}
        self._last_state_at = 0.0
        self.channel.root.mkdir(parents=True, exist_ok=True)
        for path in (self.channel.request_path, self.channel.ack_path, self.channel.state_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._restore_entity_poses()
        self._remember_initial_resting_poses()

    def _restore_entity_poses(self) -> None:
        """Restore dynamic entities omitted by the pinned robot keyframe."""
        for entity_id in SUPPORTED_ENTITIES:
            body_id = self._body_id(self._entity_body_name(entity_id))
            if body_id < 0 or int(self.model.body_jntnum[body_id]) != 1:
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            qpos = int(self.model.jnt_qposadr[joint_id])
            qvel = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos : qpos + 7] = self.model.qpos0[qpos : qpos + 7]
            self.data.qvel[qvel : qvel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _repair_keyframe_omissions(self) -> None:
        """Repair zero translation written by the robot-only keyframe."""
        repaired = False
        for entity_id in SUPPORTED_ENTITIES:
            if entity_id in self._attachments:
                continue
            body_id = self._body_id(self._entity_body_name(entity_id))
            if body_id < 0 or int(self.model.body_jntnum[body_id]) != 1:
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            qpos = int(self.model.jnt_qposadr[joint_id])
            current_xy = self.data.qpos[qpos : qpos + 2]
            initial_xy = self.model.qpos0[qpos : qpos + 2]
            if float(np.linalg.norm(current_xy)) >= 0.10 or float(np.linalg.norm(initial_xy)) <= 0.50:
                continue
            self.data.qpos[qpos : qpos + 7] = self.model.qpos0[qpos : qpos + 7]
            qvel = int(self.model.jnt_dofadr[joint_id])
            self.data.qvel[qvel : qvel + 6] = 0.0
            repaired = True
        if repaired:
            mujoco.mj_forward(self.model, self.data)

    def _freejoint_addresses(self, entity_id: str) -> tuple[int, int] | None:
        body_id = self._body_id(self._entity_body_name(entity_id))
        if body_id < 0 or int(self.model.body_jntnum[body_id]) != 1:
            return None
        joint_id = int(self.model.body_jntadr[body_id])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            return None
        return int(self.model.jnt_qposadr[joint_id]), int(
            self.model.jnt_dofadr[joint_id]
        )

    def _remember_initial_resting_poses(self) -> None:
        self._resting_qpos.clear()
        for entity_id in SUPPORTED_ENTITIES:
            addresses = self._freejoint_addresses(entity_id)
            if addresses is None:
                continue
            qpos, _qvel = addresses
            self._resting_qpos[entity_id] = self.data.qpos[qpos : qpos + 7].copy()

    def _hold_resting_entities(self) -> None:
        """Emulate MuJoCo body sleep for props awaiting manipulation.

        The worker keeps solving robot contacts continuously and MuJoCo does
        not sleep this free body.  Tiny independent support impulses therefore
        accumulate into visible prop motion.  An unattached prop stays at its
        measured tabletop pose while the arm approaches, just as a sleeping
        rigid body would until contact wakes it.  The final hand/bottle overlap
        is still resolved by MuJoCo geometry and grasp remains contact-gated.
        """
        changed = False
        for entity_id, resting in self._resting_qpos.items():
            if entity_id in self._attachments:
                continue
            addresses = self._freejoint_addresses(entity_id)
            if addresses is None:
                continue
            qpos, qvel = addresses
            if not np.allclose(self.data.qpos[qpos : qpos + 7], resting, atol=1e-10):
                self.data.qpos[qpos : qpos + 7] = resting
                changed = True
            if np.any(np.abs(self.data.qvel[qvel : qvel + 6]) > 1e-10):
                self.data.qvel[qvel : qvel + 6] = 0.0
                changed = True
        if changed:
            mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _entity_body_name(entity_id: str) -> str:
        return f"entity:{entity_id}"

    def _body_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def _entity_pose(self, entity_id: str) -> list[float] | None:
        body_id = self._body_id(self._entity_body_name(entity_id))
        if body_id < 0:
            return None
        return [
            *[float(value) for value in self.data.xpos[body_id]],
            *[float(value) for value in self.data.xquat[body_id]],
        ]

    def attached_body_ids(self) -> set[int]:
        """Return entity bodies currently carried as part of the robot."""

        return {
            int(attachment.child_body_id)
            for attachment in self._attachments.values()
        }

    def _hand_point(self, hand_body: str) -> np.ndarray[Any, Any] | None:
        body_id = self._body_id(hand_body)
        if body_id < 0:
            return None
        return self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ HAND_GRASP_OFFSETS[hand_body]

    def _entity_in_grasp_volume(self, entity_id: str, hand_body: str) -> bool:
        pose = self._entity_pose(entity_id)
        hand_id = self._body_id(hand_body)
        if pose is None or hand_id < 0:
            return False
        local = self.data.xmat[hand_id].reshape(3, 3).T @ (
            np.asarray(pose[:3]) - self.data.xpos[hand_id]
        )
        return bool(
            # The distal pads extend to x~=0.233 m.  Keep a few millimetres
            # of contact margin while rejecting anything beyond the fingers.
            0.07 <= float(local[0]) <= 0.24
            and abs(float(local[1])) <= 0.065
            and abs(float(local[2])) <= 0.120
        )

    def _approach_target(
        self,
        entity_id: str,
        hand_body: str,
    ) -> np.ndarray[Any, Any] | None:
        """Return a side-contact target instead of driving through the entity."""
        pose = self._entity_pose(entity_id)
        hand_point = self._hand_point(hand_body)
        if pose is None or hand_point is None:
            return None
        target = np.asarray(pose[:3], dtype=float)
        return target

    def _contact(self, entity_id: str, hand_body: str) -> bool:
        first = self._body_id(self._entity_body_name(entity_id))
        second = self._body_id(hand_body)
        if first < 0 or second < 0:
            return False

        def belongs_to_hand(body_id: int) -> bool:
            while body_id > 0:
                if body_id == second:
                    return True
                body_id = int(self.model.body_parentid[body_id])
            return False

        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if float(contact.dist) > 0.0:
                continue
            first_body = int(self.model.geom_bodyid[int(contact.geom1)])
            second_body = int(self.model.geom_bodyid[int(contact.geom2)])
            if (
                first_body == first
                and belongs_to_hand(second_body)
            ) or (
                second_body == first
                and belongs_to_hand(first_body)
            ):
                return True
        # The pinned G1 MJCF keeps general robot collisions disabled and lists
        # only a few explicit pairs.  Use MuJoCo's signed geom distance for the
        # named palm collision shape so contact with a newly injected dynamic
        # entity is still physical geometry evidence, not centre-distance
        # "magnetic" attachment.
        hand_geom = int(
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                HAND_COLLISION_GEOMS[hand_body],
            )
        )
        if hand_geom < 0:
            return False
        for entity_geom in np.flatnonzero(self.model.geom_bodyid == first):
            if not (
                int(self.model.geom_contype[entity_geom])
                or int(self.model.geom_conaffinity[entity_geom])
            ):
                continue
            signed_distance = mujoco.mj_geomDistance(
                self.model,
                self.data,
                hand_geom,
                int(entity_geom),
                0.01,
                None,
            )
            if math.isfinite(float(signed_distance)) and float(signed_distance) <= 0.001:
                return True
        return False

    def _joint_addresses(self, hand_body: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        qpos: list[int] = []
        qvel: list[int] = []
        for name in (*ARM_JOINTS[hand_body], "waist_pitch_joint"):
            joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                return None
            qpos.append(int(self.model.jnt_qposadr[joint_id]))
            qvel.append(int(self.model.jnt_dofadr[joint_id]))
        return tuple(qpos), tuple(qvel)

    def _actuator_addresses(self, hand_body: str) -> tuple[int, ...] | None:
        addresses: list[int] = []
        for name in (*ARM_JOINTS[hand_body], "waist_pitch_joint"):
            joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                return None
            addresses.append(int(matches[0]))
        return tuple(addresses)

    def _hold_waist_posture(self) -> bool:
        """Keep the shoulder reference frame used by the arm IK stationary."""
        for name in WAIST_JOINTS:
            joint_id = int(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            )
            if joint_id < 0:
                return False
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                return False
            qpos = int(self.model.jnt_qposadr[joint_id])
            qvel = int(self.model.jnt_dofadr[joint_id])
            ctrl = int(matches[0])
            value = float(self.data.qpos[qpos])
            self._held_joints[qpos] = value
            self._held_qvels[qvel] = 0.0
            self._held_controls[ctrl] = value
            self.data.ctrl[ctrl] = value
        return True

    def _solve_ik(
        self,
        hand_body: str,
        target: np.ndarray[Any, Any],
        *,
        tolerance: float = 0.035,
        gripper_upright: bool = False,
    ) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray[Any, Any]] | None:
        self._last_ik_diagnostics = {
            "root_qpos": [round(float(value), 5) for value in self.data.qpos[:7]],
            "attempts": [],
        }
        addresses = self._joint_addresses(hand_body)
        body_id = self._body_id(hand_body)
        if addresses is None or body_id < 0:
            return None
        qpos_adrs, qvel_adrs = addresses
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos
        initial = np.asarray([ik_data.qpos[address] for address in qpos_adrs])
        joint_ids = [
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            for name in (*ARM_JOINTS[hand_body], "waist_pitch_joint")
        ]
        lower = np.full(len(joint_ids), -np.inf)
        upper = np.full(len(joint_ids), np.inf)
        for index, joint_id in enumerate(joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                lower[index], upper[index] = self.model.jnt_range[joint_id]
        if gripper_upright:
            # Lean the native torso by a bounded amount so the stock rubber
            # hand reaches the bottle while the base stays clear of the island.
            lower[-1] = MANIPULATION_WAIST_PITCH_RAD - 1e-6
            upper[-1] = MANIPULATION_WAIST_PITCH_RAD + 1e-6

        desired_x = np.array([1.0, 0.0, 0.0])
        desired_z = np.array([0.0, 0.0, 1.0])
        if gripper_upright:
            desired_x = np.asarray(target, dtype=float) - np.asarray(
                [self.data.qpos[0], self.data.qpos[1], target[2]],
                dtype=float,
            )
            desired_norm = float(np.linalg.norm(desired_x))
            if desired_norm <= 1e-6:
                return None
            desired_x /= desired_norm

        def residual(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            ik_data.qpos[list(qpos_adrs)] = values
            mujoco.mj_forward(self.model, ik_data)
            point = ik_data.xpos[body_id] + ik_data.xmat[body_id].reshape(3, 3) @ HAND_GRASP_OFFSETS[hand_body]
            terms = [point - target]
            if gripper_upright:
                rotation = ik_data.xmat[body_id].reshape(3, 3)
                terms.extend(
                    (
                        (rotation[:, 0] - desired_x) * 0.08,
                        (rotation[:, 2] - desired_z) * 0.08,
                    )
                )
            terms.append((values - initial) * 1e-3)
            return np.concatenate(terms)

        # The live arm posture is not guaranteed to be a good local IK seed.
        # In particular, the idle stabilizer may leave the shoulder/elbow in a
        # basin from which a single least-squares run reports an otherwise
        # reachable bottle as unreachable. Retry from the model's neutral arm
        # posture while keeping the small regularizer anchored to the actual
        # posture, so the accepted solution still avoids needless motion.
        neutral = np.asarray([self.model.qpos0[address] for address in qpos_adrs])
        seeds = (
            np.clip(initial, lower, upper),
            np.clip(neutral, lower, upper),
        )
        best: tuple[float, np.ndarray[Any, Any]] | None = None
        best_oriented: tuple[float, np.ndarray[Any, Any]] | None = None
        for seed in seeds:
            result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=600)
            ik_data.qpos[list(qpos_adrs)] = result.x
            mujoco.mj_forward(self.model, ik_data)
            solved = (
                ik_data.xpos[body_id]
                + ik_data.xmat[body_id].reshape(3, 3) @ HAND_GRASP_OFFSETS[hand_body]
            )
            error = float(np.linalg.norm(solved - target))
            orientation_ok = True
            x_alignment = None
            z_alignment = None
            if gripper_upright:
                rotation = ik_data.xmat[body_id].reshape(3, 3)
                x_alignment = float(np.dot(rotation[:, 0], desired_x))
                z_alignment = float(np.dot(rotation[:, 2], desired_z))
                orientation_ok = bool(
                    x_alignment >= 0.95
                    and z_alignment >= 0.95
                )
            self._last_ik_diagnostics["attempts"].append(
                {
                    "position_error_m": round(error, 5),
                    "x_alignment": (
                        None if x_alignment is None else round(x_alignment, 5)
                    ),
                    "z_alignment": (
                        None if z_alignment is None else round(z_alignment, 5)
                    ),
                    "orientation_ok": orientation_ok,
                }
            )
            if best is None or error < best[0]:
                best = error, np.asarray(result.x).copy()
            if orientation_ok and (
                best_oriented is None or error < best_oriented[0]
            ):
                best_oriented = error, np.asarray(result.x).copy()
        # Do not return the first merely-tolerable seed. In live runs the
        # current gait posture can stop about 38 mm short while still satisfying
        # the deliberately broad 65 mm reachability envelope. The neutral seed
        # reaches actual native-hand contact, so evaluate both and select the
        # lowest-error orientation-valid solution.
        if best_oriented is not None and best_oriented[0] <= tolerance:
            return qpos_adrs, qvel_adrs, best_oriented[1]
        if gripper_upright or best is None or best[0] > tolerance:
            return None
        return qpos_adrs, qvel_adrs, best[1]

    def _ack(self, token: str, action: str, **payload: Any) -> None:
        _atomic_json(
            self.channel.ack_path,
            {"token": token, "action": action, "applied_at": time.time(), **payload},
        )

    def _start_arm_motion(
        self,
        token: str,
        action: str,
        entity_id: str,
        hand_body: str,
        target: np.ndarray[Any, Any],
        *,
        place_target: tuple[float, float, float] | None = None,
        duration: float = 2.0,
        tolerance: float = 0.035,
        gripper_upright: bool = False,
    ) -> bool:
        solution = self._solve_ik(
            hand_body,
            target,
            tolerance=tolerance,
            gripper_upright=gripper_upright,
        )
        if solution is None:
            self._ack(
                token,
                action,
                ok=False,
                error="IK target is unreachable",
                ik_diagnostics=getattr(self, "_last_ik_diagnostics", None),
            )
            return False
        qpos_adrs, qvel_adrs, joint_target = solution
        ctrl_adrs = self._actuator_addresses(hand_body)
        if ctrl_adrs is None:
            self._ack(token, action, ok=False, error="arm actuator mapping unavailable")
            return False
        if not self._hold_waist_posture():
            self._ack(token, action, ok=False, error="waist actuator mapping unavailable")
            return False
        self._motion = _ArmMotion(
            token=token,
            action=action,
            entity_id=entity_id,
            hand=hand_body,
            qpos_adrs=qpos_adrs,
            qvel_adrs=qvel_adrs,
            ctrl_adrs=ctrl_adrs,
            start=np.asarray([self.data.qpos[address] for address in qpos_adrs]),
            target=joint_target,
            started_at=float(self.data.time),
            duration=max(0.2, float(duration)),
            place_target=place_target,
        )
        return True

    def _start_carry_pose_motion(
        self,
        token: str,
        entity_id: str,
        hand_body: str,
    ) -> bool:
        """Retract to the model's gait posture before releasing joint holds."""

        addresses = self._joint_addresses(hand_body)
        controls = self._actuator_addresses(hand_body)
        if addresses is None or controls is None:
            self._ack(
                token,
                "carry_pose",
                ok=False,
                error="carry posture joint mapping unavailable",
            )
            return False
        qpos_adrs, qvel_adrs = addresses
        if not self._hold_waist_posture():
            self._ack(
                token,
                "carry_pose",
                ok=False,
                error="waist actuator mapping unavailable",
            )
            return False
        self._motion = _ArmMotion(
            token=token,
            action="carry_pose",
            entity_id=entity_id,
            hand=hand_body,
            qpos_adrs=qpos_adrs,
            qvel_adrs=qvel_adrs,
            ctrl_adrs=controls,
            start=np.asarray(
                [self.data.qpos[address] for address in qpos_adrs]
            ),
            target=np.asarray(
                [self.model.qpos0[address] for address in qpos_adrs]
            ),
            started_at=float(self.data.time),
            duration=1.2,
        )
        return True

    def _release_attachment(self, entity_id: str) -> _Attachment | None:
        self._sim_attachments.discard(entity_id)
        attachment = self._attachments.pop(entity_id, None)
        if attachment is None:
            return None
        self.model.geom_contype[attachment.geom_ids] = attachment.geom_contype
        self.model.geom_conaffinity[attachment.geom_ids] = (
            attachment.geom_conaffinity
        )
        qpos, qvel = attachment.child_qpos_adr, attachment.child_qvel_adr
        self.data.qvel[qvel : qvel + 6] = 0.0
        self._resting_qpos[entity_id] = self.data.qpos[qpos : qpos + 7].copy()
        mujoco.mj_forward(self.model, self.data)
        return attachment

    def _reset_entity(self, entity_id: str) -> bool:
        self._sim_placements.pop(entity_id, None)
        self._release_attachment(entity_id)
        addresses = self._freejoint_addresses(entity_id)
        if addresses is None:
            return False
        qpos, qvel = addresses
        self.data.qpos[qpos : qpos + 7] = self.model.qpos0[qpos : qpos + 7]
        self.data.qvel[qvel : qvel + 6] = 0.0
        self._resting_qpos[entity_id] = self.data.qpos[qpos : qpos + 7].copy()
        if self._motion is not None and self._motion.entity_id == entity_id:
            self._motion = None
            self._held_joints.clear()
            self._held_qvels.clear()
            self._held_controls.clear()
        mujoco.mj_forward(self.model, self.data)
        return True

    def _handle_request(self, payload: dict[str, Any]) -> None:
        token = payload.get("token")
        action = payload.get("action")
        arguments = payload.get("arguments")
        if not isinstance(token, str) or not token or token == self._last_token:
            return
        self._last_token = token
        if not isinstance(action, str) or not isinstance(arguments, dict):
            self._ack(token, str(action), ok=False, error="invalid entity request")
            return
        entity_id = normalize_entity_id(arguments.get("entity_id"))
        if entity_id not in SUPPORTED_ENTITIES or self._entity_pose(str(entity_id)) is None:
            self._ack(token, action, ok=False, error="unknown or unavailable entity")
            return
        entity_id = str(entity_id)
        if action in {"sim_attach", "sim_release", "sim_place"}:
            self._sim_attachment_request(token, action, entity_id, arguments)
            return
        if action == "locate":
            self._ack(token, action, ok=True, entity_id=entity_id, pose=self._entity_pose(entity_id))
            return
        if action == "status":
            from .composed_placement import evidence
            placement = evidence(self, entity_id)
            attachment = self._attachments.get(entity_id)
            self._ack(
                token,
                action,
                ok=True,
                entity_id=entity_id,
                interaction_model="sim_attachment" if entity_id in self._sim_attachments else "contact_grasp",
                placement=placement,
                grasped=attachment is not None and entity_id not in self._sim_attachments,
                attached=attachment is not None,
                hand=attachment.hand if attachment is not None else None,
                pose=self._entity_pose(entity_id),
            )
            return
        if action == "reset":
            reset = self._reset_entity(entity_id)
            self._ack(
                token,
                action,
                ok=reset,
                entity_id=entity_id,
                pose=self._entity_pose(entity_id),
                attached=False,
                error=None if reset else "entity reset is unavailable",
            )
            return
        hand_body = HAND_BODIES.get(str(arguments.get("hand", "right")).casefold())
        if hand_body is None:
            self._ack(token, action, ok=False, error="hand must be left or right")
            return
        if action == "contact":
            self._ack(
                token,
                action,
                ok=True,
                entity_id=entity_id,
                hand=hand_body,
                contact=self._contact(entity_id, hand_body),
                pose=self._entity_pose(entity_id),
            )
            return
        if action == "approach":
            target = self._approach_target(entity_id, hand_body)
            if target is None:
                self._ack(token, action, ok=False, error="grasp surface target unavailable")
                return
            self._start_arm_motion(
                token,
                action,
                entity_id,
                hand_body,
                target,
                duration=3.0,
                # Use the native rubber-hand reference point. The following
                # grasp still requires current MuJoCo hand/entity contact.
                tolerance=0.065,
                gripper_upright=True,
            )
            return
        if action == "grasp":
            self._grasp(token, action, entity_id, hand_body)
            return
        if action == "carry_pose":
            attachment = self._attachments.get(entity_id)
            if attachment is None or attachment.hand != hand_body:
                self._ack(
                    token,
                    action,
                    ok=False,
                    error="entity is not attached to the selected hand",
                )
                return
            self._start_carry_pose_motion(token, entity_id, hand_body)
            return
        if action == "place":
            target = arguments.get("target_position")
            if (
                not isinstance(target, list)
                or len(target) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in target)
                or not all(math.isfinite(float(value)) for value in target)
            ):
                self._ack(token, action, ok=False, error="target_position must contain x,y,z")
                return
            attachment = self._attachments.get(entity_id)
            if attachment is None or attachment.hand != hand_body:
                self._ack(token, action, ok=False, error="entity is not attached")
                return
            point = tuple(float(value) for value in target)
            self._start_arm_motion(
                token,
                action,
                entity_id,
                hand_body,
                np.asarray(point),
                place_target=point,
            )
            return
        if action == "release":
            attachment = self._attachments.get(entity_id)
            if attachment is None or attachment.hand != hand_body:
                self._ack(
                    token,
                    action,
                    ok=False,
                    error="entity is not attached to the selected hand",
                )
                return
            self._release_attachment(entity_id)
            self._ack(
                token,
                action,
                ok=True,
                entity_id=entity_id,
                hand=hand_body,
                attached=False,
                pose=self._entity_pose(entity_id),
            )
            return
        self._ack(token, action, ok=False, error="unsupported entity action")

    def _grasp(self, token: str, action: str, entity_id: str, hand_body: str) -> None:
        entity_body = self._body_id(self._entity_body_name(entity_id))
        hand_id = self._body_id(hand_body)
        pose = self._entity_pose(entity_id)
        point = self._hand_point(hand_body)
        if entity_body < 0 or hand_id < 0 or pose is None or point is None:
            self._ack(token, action, ok=False, error="entity or hand pose unavailable")
            return
        distance = float(np.linalg.norm(np.asarray(pose[:3]) - point))
        if not self._contact(entity_id, hand_body):
            self._ack(
                token,
                action,
                ok=False,
                error="physical hand/entity contact is required",
                distance_m=distance,
            )
            return
        self._bind_attachment(token, action, entity_id, hand_body, entity_body, hand_id,
                              contact_confirmed=True)

    def _sim_attachment_request(self, token, action, entity_id, arguments):
        cancellation = self.channel._read(self.channel.root / "sim-cancel.json") or {}
        if arguments.get("cancellation_revision") != cancellation.get("token"):
            self._ack(token, action, ok=False, error="simulation attachment request cancelled")
            return
        expiry = arguments.get("expires_at")
        if (not self.allow_sim_attachment or not isinstance(expiry, (int, float))
                or isinstance(expiry, bool) or not math.isfinite(expiry)
                or not 0 < expiry - time.time() <= 3.0):
            self._ack(token, action, ok=False, error="simulation attachment disabled or request expired")
            return
        if self._sim_stationary_samples < 2 or np.linalg.norm(self.data.qvel[:2]) > 0.025:
            self._ack(token, action, ok=False, error="fresh stationary samples required")
            return
        if action == "sim_place":
            from .composed_placement import request
            request(self, token, entity_id, arguments)
            return
        if action == "sim_release":
            if entity_id not in self._sim_attachments:
                self._ack(token, action, ok=False, error="entity has no simulation attachment")
                return
            self._release_attachment(entity_id)
            self._ack(token, action, ok=True, entity_id=entity_id, attached=False,
                      interaction_model="sim_attachment", pose=self._entity_pose(entity_id))
            return
        pickup = arguments.get("pickup_pose")
        if (not isinstance(pickup, list) or len(pickup) != 3
                or any(isinstance(x, bool) or not isinstance(x, (float, int))
                       or not math.isfinite(x) for x in pickup)):
            self._ack(token, action, ok=False, error="finite pickup x/y/yaw required")
            return
        q = self.data.qpos[3:7]
        yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))
        yaw_error = abs(math.atan2(math.sin(yaw-pickup[2]), math.cos(yaw-pickup[2])))
        entity_body = self._body_id(self._entity_body_name(entity_id))
        root_id = int(self.model.jnt_bodyid[0])
        pose = self._entity_pose(entity_id)
        if (self._attachments or math.dist(self.data.qpos[:2], pickup[:2]) > POSITION_TOLERANCE_M
                or yaw_error > YAW_TOLERANCE_RAD or pose is None
                or math.dist(pose[:3], self.data.xpos[root_id]) > 1.0):
            self._ack(token, action, ok=False, error="attachment precondition failed: occupied, not at pickup or out of range")
            return
        hand = arguments.get("hand", "right")
        hand_body = HAND_BODIES.get(hand) if isinstance(hand, str) else None
        hand_id = self._body_id(hand_body) if hand_body is not None else -1
        if hand_id < 0:
            self._ack(token, action, ok=False, error="supported hand body is required for simulation attachment")
            return
        # 仿真吸附通过原前置检查后，将瓶心移到已有手部握持点，再随手运动。
        # 这是显式的吸附简化，不声明发生了物理接触。
        self._bind_attachment(token, action, entity_id, hand_body, entity_body, hand_id,
                              contact_confirmed=False, snap_to_hand=True)

    def _bind_attachment(self, token, action, entity_id, hand_body, entity_body, hand_id,
                         *, contact_confirmed, snap_to_hand=False):
        joint_adr = int(self.model.body_jntadr[entity_body])
        if (
            int(self.model.body_jntnum[entity_body]) != 1
            or int(self.model.jnt_type[joint_adr])
            != int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            self._ack(
                token,
                action,
                ok=False,
                error="entity is not a dynamic free body",
            )
            return

        inverse = np.empty(4)
        mujoco.mju_negQuat(inverse, self.data.xquat[hand_id])
        local_position = np.empty(3)
        mujoco.mju_rotVecQuat(
            local_position,
            self.data.xpos[entity_body] - self.data.xpos[hand_id],
            inverse,
        )
        if snap_to_hand:
            # HAND_GRASP_OFFSETS 与预设流程的 _hand_point 使用同一握持点定义。
            # 只改变初始位置，保留瓶子此刻的世界朝向，随后按相对手部旋转跟随。
            local_position = HAND_GRASP_OFFSETS[hand_body].copy()
        local_quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            local_quaternion,
            inverse,
            self.data.xquat[entity_body],
        )
        geom_ids = np.flatnonzero(
            self.model.geom_bodyid == entity_body
        ).astype(np.int32)
        attachment = _Attachment(
            child_body_id=entity_body,
            child_qpos_adr=int(self.model.jnt_qposadr[joint_adr]),
            child_qvel_adr=int(self.model.jnt_dofadr[joint_adr]),
            parent_body_id=hand_id,
            local_position=local_position,
            local_quaternion=local_quaternion,
            geom_ids=geom_ids,
            geom_contype=self.model.geom_contype[geom_ids].copy(),
            geom_conaffinity=self.model.geom_conaffinity[geom_ids].copy(),
            hand=hand_body,
        )
        self.model.geom_contype[geom_ids] = 0
        self.model.geom_conaffinity[geom_ids] = 0
        self._sim_placements.pop(entity_id, None)
        self._resting_qpos.pop(entity_id, None)
        self._attachments[entity_id] = attachment
        if action == "sim_attach":
            self._sim_attachments.add(entity_id)
        self._apply_attachment(attachment)
        self._ack(
            token,
            action,
            ok=True,
            entity_id=entity_id,
            hand=hand_body,
            contact_confirmed=contact_confirmed,
            interaction_model="contact_grasp" if contact_confirmed else "sim_attachment",
            grasped=contact_confirmed,
            attached=True,
        )

    def _apply_attachment(self, attachment: _Attachment) -> None:
        parent_position = self.data.xpos[attachment.parent_body_id]
        parent_quaternion = self.data.xquat[attachment.parent_body_id]
        rotated = np.empty(3)
        mujoco.mju_rotVecQuat(
            rotated,
            attachment.local_position,
            parent_quaternion,
        )
        quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            quaternion,
            parent_quaternion,
            attachment.local_quaternion,
        )
        qpos = attachment.child_qpos_adr
        self.data.qpos[qpos : qpos + 3] = parent_position + rotated
        self.data.qpos[qpos + 3 : qpos + 7] = quaternion
        qvel = attachment.child_qvel_adr
        self.data.qvel[qvel : qvel + 6] = 0.0

    def _finish_motion(self, motion: _ArmMotion) -> None:
        if motion.action == "carry_pose":
            attachment = self._attachments.get(motion.entity_id)
            retained = bool(
                attachment is not None and attachment.hand == motion.hand
            )
            # The gait policy must own every arm and waist actuator while
            # walking. The bottle remains attached to the native wrist and is
            # updated independently on every simulation frame.
            self._held_joints.clear()
            self._held_qvels.clear()
            self._held_controls.clear()
            self._ack(
                motion.token,
                motion.action,
                ok=retained,
                entity_id=motion.entity_id,
                hand=motion.hand,
                attached=retained,
                gait_ownership_restored=True,
                error=None if retained else "attachment lost during retraction",
            )
            return
        if motion.action == "place":
            attachment = self._release_attachment(motion.entity_id)
            if attachment is None:
                self._ack(
                    motion.token,
                    motion.action,
                    ok=False,
                    error="entity detached before placement",
                )
                return
            self._ack(
                motion.token,
                motion.action,
                ok=True,
                entity_id=motion.entity_id,
                placed=True,
                target_position=list(motion.place_target or ()),
                pose=self._entity_pose(motion.entity_id),
            )
            return
        pose = self._entity_pose(motion.entity_id)
        point = self._hand_point(motion.hand)
        distance = None if pose is None or point is None else float(np.linalg.norm(np.asarray(pose[:3]) - point))
        live_joint_values = np.asarray(
            [self.data.qpos[address] for address in motion.qpos_adrs]
        )
        motion_diagnostics = dict(getattr(self, "_last_ik_diagnostics", {}) or {})
        motion_diagnostics.update(
            {
                "live_joint_error_rad": round(
                    float(np.max(np.abs(live_joint_values - motion.target))),
                    6,
                ),
                "target_point": [
                    round(float(value), 6)
                    for value in (
                        np.asarray(pose[:3]) if pose is not None else np.full(3, np.nan)
                    )
                ],
                "live_hand_point": [
                    round(float(value), 6)
                    for value in (
                        np.asarray(point) if point is not None else np.full(3, np.nan)
                    )
                ],
            }
        )
        contact = self._contact(motion.entity_id, motion.hand)
        reached = distance is not None and distance <= GRIPPER_OPEN_APERTURE_M
        self._ack(
            motion.token,
            motion.action,
            ok=bool(reached and contact),
            entity_id=motion.entity_id,
            hand=motion.hand,
            distance_m=distance,
            contact=contact,
            ik_diagnostics=motion_diagnostics,
            error=(
                None
                if reached and contact
                else "hand reached entity envelope without physical contact"
                if reached
                else "hand did not reach entity"
            ),
        )
    def reset(self) -> None:
        for attachment in self._attachments.values():
            self.model.geom_contype[attachment.geom_ids] = attachment.geom_contype
            self.model.geom_conaffinity[attachment.geom_ids] = (
                attachment.geom_conaffinity
            )
        self._attachments.clear()
        self._sim_attachments.clear()
        self._sim_placements.clear()
        self._motion = None
        self._held_joints.clear()
        self._held_qvels.clear()
        self._held_controls.clear()
        self._restore_entity_poses()
        self._remember_initial_resting_poses()

    def poll(self) -> None:
        self._repair_keyframe_omissions()
        self._hold_resting_entities()
        for address, value in self._held_joints.items():
            self.data.qpos[address] = value
        for address, value in self._held_qvels.items():
            self.data.qvel[address] = value
        for address, value in self._held_controls.items():
            self.data.ctrl[address] = value
        if self._held_joints:
            # Physics steps between the approach acknowledgement and the
            # immediately following grasp request can leave xpos/xmat showing
            # the policy posture even though the held manipulation qpos has
            # just been restored above. Refresh geometry before contact-gated
            # request handling.
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        if self._sim_attachments:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
        for attachment in self._attachments.values():
            self._apply_attachment(attachment)
        if self._sim_attachments:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        sample_time = float(self.data.time)
        if sample_time != self._sim_last_sample_time:
            self._sim_last_sample_time = sample_time
            stationary = (np.linalg.norm(self.data.qvel[:2]) <= 0.025
                          and abs(float(self.data.qvel[5])) <= 0.025)
            self._sim_stationary_samples = self._sim_stationary_samples + 1 if stationary else 0

        payload = self.channel._read(self.channel.request_path)
        if payload is not None and self._motion is None:
            self._handle_request(payload)

        motion = self._motion
        if motion is not None:
            progress = min(
                1.0,
                max(0.0, (float(self.data.time) - motion.started_at) / motion.duration),
            )
            blend = progress * progress * (3.0 - 2.0 * progress)
            blend_rate = (
                0.0
                if progress >= 1.0
                else 6.0 * progress * (1.0 - progress) / motion.duration
            )
            values = motion.start + (motion.target - motion.start) * blend
            velocities = (motion.target - motion.start) * blend_rate
            for qpos, qvel, ctrl, value, velocity, target_value in zip(
                motion.qpos_adrs,
                motion.qvel_adrs,
                motion.ctrl_adrs,
                values,
                velocities,
                motion.target,
                strict=True,
            ):
                self.data.qpos[qpos] = value
                # The arm path is kinematically commanded, but contacts still
                # need its true path velocity to produce tangential friction.
                self.data.qvel[qvel] = velocity
                control_value = float(value)
                self.data.ctrl[ctrl] = control_value
                self._held_joints[qpos] = float(value)
                self._held_qvels[qvel] = 0.0
                self._held_controls[ctrl] = control_value
            # Only refresh body/geom transforms here. A full mj_forward can
            # re-enter the policy/constraint pipeline and replace the
            # kinematically commanded manipulation joints before this same
            # frame verifies the native-hand geometry.
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            for attachment in self._attachments.values():
                self._apply_attachment(attachment)
            if progress >= 1.0:
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_comPos(self.model, self.data)
                self._motion = None
                self._finish_motion(motion)

        now = time.monotonic()
        if now - self._last_state_at >= 0.1:
            self._last_state_at = now
            entities = {
                entity_id: {
                    "pose": self._entity_pose(entity_id),
                    "interaction_model": "sim_attachment" if entity_id in self._sim_attachments else "contact_grasp",
                    "grasped": entity_id in self._attachments and entity_id not in self._sim_attachments,
                    "attached": entity_id in self._attachments,
                    "hand": (
                        self._attachments[entity_id].hand
                        if entity_id in self._attachments
                        else None
                    ),
                }
                for entity_id in SUPPORTED_ENTITIES
                if self._entity_pose(entity_id) is not None
            }
            _atomic_json(
                self.channel.state_path,
                {
                    "schema_version": 1,
                    "available": True,
                    "written_at": time.time(),
                    "busy": self._motion is not None,
                    "robot_root_qpos": [
                        round(float(value), 6) for value in self.data.qpos[:7]
                    ],
                    "robot_root_qvel": [
                        round(float(value), 6) for value in self.data.qvel[:6]
                    ],
                    "entities": entities,
                },
            )
