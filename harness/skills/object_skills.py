"""Composable semantic navigation and entity-manipulation skills."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
from typing import Any
import uuid

import numpy as np
from reactivex.disposable import Disposable

from dimos_lcm.std_msgs import Bool
from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.base import NavigationState
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.navigation.visual.query import get_object_bbox_from_image
from dimos.perception.spatial_memory_spec import SpatialMemorySpec
from pydantic import Field

from harness.robots.entity_port import (
    EntityManipulationPort,
    EntityOperationResult,
    normalize_entity_id,
)
from harness.robots.world_adapter import get_world_adapter
from harness.control.long_task_control import LongTaskControlChannel
from harness.skills.rgbd_skills import (
    localize_person_in_camera,
    point_to_parent,
    verify_world_target_with_depth,
)


SEMANTIC_GOALS: dict[str, tuple[float, float, float]] = {
    # Stop at a sensor-visible observation pose just inside the kitchen.  It
    # is inside the living-room spawn costmap; RGB-D then determines the
    # precise manipulation approach.
    # Stop at the already validated room-entry observation pose. Precise bottle
    # approach is a later RGB-D stage with its own table-side affordance.
    "kitchen": (0.80, 0.20, 0.0),
    "厨房": (0.80, 0.20, 0.0),
    "living_room": (-3.20, -2.05, -math.pi / 2.0),
    "living room": (-3.20, -2.05, -math.pi / 2.0),
    "客厅": (-3.20, -2.05, -math.pi / 2.0),
    "bedroom": (-2.45, 1.45, math.pi / 2.0),
    "卧室": (-2.45, 1.45, math.pi / 2.0),
    "dining": (1.55, -2.10, -math.pi / 2.0),
    "餐厅": (1.55, -2.10, -math.pi / 2.0),
    "餐区": (1.55, -2.10, -math.pi / 2.0),
}

ENTITY_VISUAL_QUERIES = {
    "water_bottle": "the small clear mineral water bottle with a blue label and cap",
}

# Known semantic-map affordances constrain only the direction from which the
# RGB-D-localized entity is approached.  They do not supply its metric pose.
ENTITY_APPROACH_YAWS = {
    "water_bottle": math.pi / 2.0,
}


class ObjectTaskConfig(ModuleConfig):
    """Robot-neutral sensing and manipulation approach parameters."""

    camera_info: CameraInfo
    world_adapter: str = Field(default="mujoco", pattern="^(mujoco|isaac-g1)$")
    # Transitional config alias; task code resolves it through WorldAdapter.
    entity_backend: str | None = Field(default=None, pattern="^(mujoco|isaac-g1)$")
    global_semantic_map: bool = False
    sync_tolerance_s: float = Field(default=0.25, gt=0.0, le=1.0)
    min_depth_points: int = Field(default=15, ge=10)
    # A 0.66 m centre standoff plus a bounded native-waist lean gives the
    # safety relay stopping margin at the table edge while keeping the stock
    # rubber hand within reach.
    manipulation_standoff_m: float = Field(default=0.66, ge=0.45, le=0.75)
    standoff_tolerance_m: float = Field(default=0.10, ge=0.005, le=0.20)
    hand_lateral_offset_m: float = Field(default=0.14, ge=0.0, le=0.25)
    max_heading_error_degrees: float = Field(default=35.0, gt=5.0, le=60.0)
    manipulation_settle_s: float = Field(default=0.8, ge=0.0, le=3.0)
    manipulation_takeover_radius_m: float = Field(default=0.18, ge=0.08, le=0.30)
    manipulation_refine_timeout_s: float = Field(default=8.0, ge=2.0, le=15.0)
    manipulation_contact_margin_m: float = Field(default=0.006, ge=0.0, le=0.01)
    visual_evidence_max_age_s: float = Field(default=12.0, gt=1.0, le=30.0)
    require_visual_manipulation: bool = True
    allow_backend_seeded_depth_verification: bool = False


def _result(status: str, operation_ok: bool, **payload: Any) -> dict[str, Any]:
    return {
        "tool_ok": True,
        "operation_ok": operation_ok,
        "task_status": status,
        "completed": False,
        **payload,
    }


def _entity_value(
    result: Mapping[str, Any],
    field: str,
    default: Any = None,
) -> Any:
    """Read common or adapter-specific evidence from a Port result."""

    if field in result:
        return result[field]
    details = result.get("details")
    if isinstance(details, Mapping):
        return details.get(field, default)
    return default


class ObjectTaskSkillContainer(Module):
    """Atomic tools that a planner or behaviour tree can compose."""

    config: ObjectTaskConfig
    _navigation: NavigationInterfaceSpec
    _spatial_memory: SpatialMemorySpec
    color_image: In[Image]
    depth_image: In[Image]
    odom: In[PoseStamped]
    goal_reached: In[Bool]
    stop_movement: Out[Bool]
    nav_cmd_vel: Out[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_odom: PoseStamped | None = None
        self._odom_history: deque[tuple[float, float, float, float]] = deque(
            maxlen=40
        )
        adapter_id = self.config.entity_backend or self.config.world_adapter
        self._world_adapter = get_world_adapter(adapter_id)
        self._entities: EntityManipulationPort = (
            self._world_adapter.create_entity_port()
        )
        self._color_frames: deque[Image] = deque(maxlen=20)
        self._depth_frames: deque[Image] = deque(maxlen=20)
        self._vision_lock = RLock()
        self._last_visual_entities: dict[str, dict[str, Any]] = {}
        self._recent_contact_approaches: dict[str, dict[str, Any]] = {}
        self._last_visual_error = ""
        self._planner_goal_reached = False
        self._long_task_control = LongTaskControlChannel()
        self._active_long_task_id: str | None = None
        from dimos.models.vl import qwen as qwen_module

        self._vl_model = qwen_module.QwenVlModel()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.odom.subscribe(self._on_odom)))
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color)))
        self.register_disposable(Disposable(self.depth_image.subscribe(self._on_depth)))
        self.register_disposable(
            Disposable(self.goal_reached.subscribe(self._on_goal_reached))
        )

    @rpc
    def stop(self) -> None:
        self._vl_model.stop()
        super().stop()

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom
        self._odom_history.append(
            (
                time.monotonic(),
                float(odom.position.x),
                float(odom.position.y),
                float(odom.orientation.to_euler().yaw),
            )
        )

    def _on_color(self, image: Image) -> None:
        with self._vision_lock:
            self._color_frames.append(image)

    def _on_depth(self, image: Image) -> None:
        with self._vision_lock:
            self._depth_frames.append(image)

    def _on_goal_reached(self, reached: Bool) -> None:
        self._planner_goal_reached = bool(reached.data)

    def _long_task_cancel_reason(self) -> str | None:
        composition_cancel = getattr(self, "_composition_cancel", None)
        if composition_cancel is not None:
            channel, revision, expiry = composition_cancel
            if time.time() >= expiry or channel.changed_since(revision):
                return "composition_cancelled_or_expired"
        channel = getattr(self, "_long_task_control", None)
        if channel is None:
            return None
        return channel.cancellation(getattr(self, "_active_long_task_id", None))

    def _long_task_stage(self, stage: str) -> None:
        channel = getattr(self, "_long_task_control", None)
        job_id = getattr(self, "_active_long_task_id", None)
        if channel is not None and job_id is not None:
            channel.update(
                job_id,
                state="running",
                stage=stage,
            )

    def _confirm_stationary_after_stop(
        self,
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Publish the terminal stop and verify two fresh low-speed samples."""

        stop_started = time.monotonic()
        try:
            self._navigation.cancel_goal()
            self.nav_cmd_vel.publish(Twist.zero())
            self.stop_movement.publish(Bool(data=True))
        except Exception as exc:  # noqa: BLE001 - stop evidence must fail closed
            return {
                "stationary_confirmed": False,
                "stop_command_completed_at": None,
                "error": f"terminal stop publish failed: {exc}",
            }
        stop_command_completed_at = time.time()

        deadline = stop_started + max(0.1, float(timeout))
        while time.monotonic() < deadline:
            samples = [
                sample
                for sample in self._odom_history
                if sample[0] > stop_started
            ]
            if len(samples) >= 3:
                speeds: list[float] = []
                for previous, current in zip(samples[-3:-1], samples[-2:]):
                    elapsed = current[0] - previous[0]
                    if elapsed <= 1e-4:
                        speeds = []
                        break
                    speeds.append(
                        math.hypot(
                            current[1] - previous[1],
                            current[2] - previous[2],
                        )
                        / elapsed
                    )
                if len(speeds) == 2 and all(speed <= 0.025 for speed in speeds):
                    return {
                        "stationary_confirmed": True,
                        "stop_command_completed_at": stop_command_completed_at,
                        "stationary_confirmed_at": time.time(),
                        "stationary_samples": 2,
                        "maximum_planar_speed_mps": round(max(speeds), 4),
                    }
            time.sleep(0.05)
        return {
            "stationary_confirmed": False,
            "stop_command_completed_at": stop_command_completed_at,
            "stationary_samples": 0,
            "error": "two fresh stationary odometry samples were not observed",
        }

    def _semantic_map_available(self) -> bool:
        config = getattr(self, "config", None)
        if config is None or not config.global_semantic_map:
            return False
        # The fixed coordinates below are authored for home_complex.  The
        # allow-listed bottle is currently that scene's runtime marker; a
        # different MuJoCo scene must use observed semantic memory instead of
        # silently inheriting Brownstone/home coordinates.
        try:
            state = self._entities.entity_state("water_bottle", timeout=0.25)
        except Exception:  # noqa: BLE001 - unavailable backend fails closed
            return False
        return bool(
            state.get("operation_ok")
            and isinstance(state.get("pose"), list)
            and len(state["pose"]) >= 3
        )

    def _synced_rgbd(self, *, after: float = 0.0, timeout: float = 3.0) -> tuple[Image, Image] | None:
        if not hasattr(self, "_vision_lock"):
            return None
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._long_task_cancel_reason() is not None:
                return None
            with self._vision_lock:
                colors = [frame for frame in self._color_frames if float(frame.ts) > after]
                depths = [frame for frame in self._depth_frames if float(frame.ts) > after]
            if colors and depths:
                pair = min(
                    ((color, depth) for color in colors for depth in depths),
                    key=lambda item: abs(float(item[0].ts) - float(item[1].ts)),
                )
                if abs(float(pair[0].ts) - float(pair[1].ts)) <= self.config.sync_tolerance_s:
                    return pair
            time.sleep(0.05)
        return None

    def _visual_entity_pose(
        self,
        entity_id: str,
        *,
        after: float = 0.0,
    ) -> dict[str, Any] | None:
        """Localize an entity from one synchronized robot RGB-D observation."""

        entity_id = normalize_entity_id(entity_id)
        query = ENTITY_VISUAL_QUERIES.get(entity_id)
        if query is None or not hasattr(self, "_vl_model"):
            return None
        from harness.skills.object_perception import localize_object_rgbd
        outcome = localize_object_rgbd(entity_id=entity_id, query=query,
            acquire=self._synced_rgbd, tf=self.tf, camera_info=self.config.camera_info,
            model=self._vl_model, bbox_query=get_object_bbox_from_image,
            min_points=self.config.min_depth_points, after=after, sync_tolerance_s=self.config.sync_tolerance_s)
        if not outcome['operation_ok']:
            self._last_visual_error = outcome['error']
            return None
        evidence = outcome['visual']
        self._last_visual_entities[str(entity_id)] = evidence
        self._last_visual_error = ""
        return evidence

    def _depth_verify_backend_pose(
        self,
        entity_id: str,
        pose: list[float],
        *,
        after: float = 0.0,
    ) -> dict[str, Any] | None:
        """Verify a backend identity seed against fresh robot depth geometry."""

        if not self.config.allow_backend_seeded_depth_verification or len(pose) < 3:
            return None
        pair = self._synced_rgbd(after=after)
        if pair is None:
            return None
        _color, depth = pair
        timestamp = float(depth.ts)
        world_from_camera = self.tf.get(
            "world",
            depth.frame_id or "camera_optical",
            timestamp,
            time_tolerance=self.config.sync_tolerance_s,
            forward_tolerance=0.5,
        )
        robot_at_frame = self.tf.get(
            "world",
            "base_link",
            timestamp,
            time_tolerance=max(0.3, self.config.sync_tolerance_s),
            forward_tolerance=0.5,
        )
        if world_from_camera is None or robot_at_frame is None:
            return None
        diagnostics: dict[str, Any] = {}
        verified = verify_world_target_with_depth(
            Vector3(float(pose[0]), float(pose[1]), float(pose[2])),
            depth,
            self.config.camera_info,
            world_from_camera,
            robot_at_frame,
            # The adapter seed names the bottle's rigid-body centre while the
            # depth image observes its front surface (and, for this small prop,
            # occasionally the shoulder/base inside the same locked patch).
            # Keep this equal to the existing backend/visual disagreement
            # ceiling: the projection and fresh depth are still mandatory.
            depth_tolerance_m=0.35,
            patch_radius_pixels=12,
            min_points=self.config.min_depth_points,
            # A 500 ml bottle's rigid-body centre can project onto the last
            # few image rows while its shoulder/cap remain visible above it.
            # Search only along the already locked horizontal bearing; depth
            # and world-position checks below still have to pass.
            allow_vertical_surface=True,
            diagnostics=diagnostics,
        )
        if verified is None:
            self._last_visual_error = f"backend-seeded depth verification failed: {diagnostics}"
            return None
        evidence = {
            "source": "robot_depth_backend_identity",
            "frame_timestamp": verified.frame_timestamp,
            "depth_m": round(float(verified.estimate.depth_m), 3),
            "valid_depth_points": int(verified.estimate.valid_points),
            "pose": [
                float(verified.world_point.x),
                float(verified.world_point.y),
                float(verified.world_point.z),
            ],
            "backend_pose": [float(value) for value in pose[:3]],
            "verification": diagnostics,
            "observed_at": time.time(),
        }
        self._last_visual_entities[str(entity_id)] = evidence
        self._last_visual_error = ""
        return evidence

    def _locate_entity_evidence(
        self,
        entity_id: str,
        *,
        after: float = 0.0,
        reuse_recent_visual: bool = False,
    ) -> tuple[
        list[float] | None,
        dict[str, Any] | None,
        EntityOperationResult,
    ]:
        adapter = self._entities.entity_state(str(entity_id), timeout=2.0)
        visual: dict[str, Any] | None = None
        recent = getattr(self, "_last_visual_entities", {}).get(str(entity_id))
        if reuse_recent_visual and after <= 0.0 and isinstance(recent, dict):
            try:
                recent_age = time.time() - float(recent["observed_at"])
            except (KeyError, TypeError, ValueError, OverflowError):
                recent_age = math.inf
            if 0.0 <= recent_age <= min(
                5.0,
                float(self.config.visual_evidence_max_age_s),
            ):
                # A terminal fetch invokes locate and approach back-to-back.
                # Reuse the already cross-checked world-space RGB-D evidence
                # instead of asking a stochastic VLM for a second box on the
                # same frame.  The current backend pose is still checked below,
                # and post-navigation approach obtains a newer RGB-D frame.
                visual = deepcopy(recent)
                visual["reused_for_atomic_chain"] = True
        if visual is None:
            visual = self._visual_entity_pose(entity_id, after=after)
        adapter_pose = adapter.get("pose")
        if (
            visual is None
            and adapter.get("operation_ok")
            and isinstance(adapter_pose, list)
        ):
            visual = self._depth_verify_backend_pose(
                entity_id,
                adapter_pose,
                after=after,
            )
        if visual is not None:
            pose = list(visual["pose"])
            if (
                adapter.get("operation_ok")
                and isinstance(adapter_pose, list)
                and len(adapter_pose) >= 3
            ):
                observed = np.asarray(pose[:3], dtype=float)
                seeded = np.asarray(adapter_pose[:3], dtype=float)
                vertical_seed_verification = (
                    visual.get("source") == "robot_depth_backend_identity"
                    and (visual.get("verification") or {}).get("verification_mode")
                    == "vertical_surface"
                )
                # Vertical-surface verification deliberately proves the
                # locked x/y bearing and depth using any visible point on the
                # same narrow object column.  Comparing z again here would
                # reject the bottle shoulder/base that enabled this mode.
                disagreement = float(
                    np.linalg.norm(
                        observed[:2] - seeded[:2]
                        if vertical_seed_verification
                        else observed - seeded
                    )
                )
                visual["adapter_disagreement_m"] = round(disagreement, 3)
                if disagreement > 0.35:
                    rejected_visual = deepcopy(visual)
                    verified = self._depth_verify_backend_pose(
                        entity_id,
                        adapter_pose,
                        after=after,
                    )
                    if verified is not None:
                        verified_pose = np.asarray(
                            list(verified["pose"])[:3],
                            dtype=float,
                        )
                        verified_vertical_surface = (
                            (verified.get("verification") or {}).get(
                                "verification_mode"
                            )
                            == "vertical_surface"
                        )
                        verified_disagreement = float(
                            np.linalg.norm(
                                verified_pose[:2] - seeded[:2]
                                if verified_vertical_surface
                                else verified_pose - seeded
                            )
                        )
                        verified["adapter_disagreement_m"] = round(
                            verified_disagreement,
                            3,
                        )
                        verified["rejected_visual_candidate"] = rejected_visual
                        if verified_disagreement <= 0.35:
                            visual = verified
                            pose = list(verified["pose"])
                        else:
                            return None, verified, {
                                "tool_ok": True,
                                "operation_ok": False,
                                "completed": False,
                                "task_status": "entity_pose_disagreement",
                                "backend": str(adapter.get("backend", "unknown")),
                                "entity_id": str(entity_id),
                                "error": (
                                    "depth-verified backend projection and "
                                    "manipulation adapter poses disagree"
                                ),
                            }
                    else:
                        return None, visual, {
                            "tool_ok": True,
                            "operation_ok": False,
                            "completed": False,
                            "task_status": "entity_pose_disagreement",
                            "backend": str(adapter.get("backend", "unknown")),
                            "entity_id": str(entity_id),
                            "error": "visual and manipulation adapter poses disagree",
                        }
                if visual.get("source") == "robot_depth_backend_identity":
                    # In normal MuJoCo operation the fresh depth frame proves
                    # that the allow-listed backend identity is physically at
                    # this projection.  The depth estimate is a visible surface
                    # point, while the arm backend targets the entity centre;
                    # retain the sensor evidence but use the verified centre for
                    # manipulation geometry.  Real RGB-D/VLM localization never
                    # enters this simulation-only branch.
                    pose = [float(value) for value in adapter_pose[:3]]
                    visual["manipulation_pose"] = pose
                    visual["manipulation_pose_source"] = (
                        "depth_verified_simulation_entity_center"
                    )
            return pose, visual, adapter
        if adapter.get("operation_ok") and isinstance(adapter_pose, list):
            # The normal-mode MuJoCo adapter is a simulation backend, not visual
            # evidence.  It keeps unit tests and operator simulation observable;
            # a real backend must supply RGB-D or fail closed here.
            return list(adapter_pose), {
                "source": "simulation_entity_adapter",
                "visual_error": getattr(self, "_last_visual_error", ""),
            }, adapter
        return None, None, adapter

    def _manipulation_goal(
        self,
        pose: list[float],
        hand: str = "right",
        *,
        standoff_m: float | None = None,
        approach_yaw: float | None = None,
    ) -> PoseStamped | None:
        odom = self._latest_odom
        if odom is None or len(pose) < 2:
            return None
        dx = float(pose[0]) - float(odom.position.x)
        dy = float(pose[1]) - float(odom.position.y)
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return None
        standoff = float(
            self.config.manipulation_standoff_m if standoff_m is None else standoff_m
        )
        lateral = min(float(self.config.hand_lateral_offset_m), standoff * 0.8)
        if str(hand).strip().casefold() in {"left", "left_hand"}:
            lateral = -lateral
        forward = math.sqrt(max(0.0, standoff * standoff - lateral * lateral))
        if approach_yaw is None:
            unit_x, unit_y = dx / distance, dy / distance
        else:
            unit_x, unit_y = math.cos(approach_yaw), math.sin(approach_yaw)
        right_x, right_y = unit_y, -unit_x
        x = float(pose[0]) - unit_x * forward - right_x * lateral
        y = float(pose[1]) - unit_y * forward - right_y * lateral
        # Keep the torso aligned with the approach axis.  The target remains
        # laterally aligned with the selected shoulder instead of being placed
        # on the robot centreline, which materially improves arm reachability.
        yaw = math.atan2(unit_y, unit_x)
        return self._goal((x, y, yaw))

    def _manipulation_pose_verified(
        self,
        pose: list[float],
        *,
        requested_standoff_m: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        odom = self._latest_odom
        if odom is None:
            return False, {"error": "odometry unavailable"}
        dx = float(pose[0]) - float(odom.position.x)
        dy = float(pose[1]) - float(odom.position.y)
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        actual_yaw = float(odom.orientation.to_euler().yaw)
        heading_error = abs(math.atan2(math.sin(target_yaw - actual_yaw), math.cos(target_yaw - actual_yaw)))
        requested = float(
            self.config.manipulation_standoff_m
            if requested_standoff_m is None
            else requested_standoff_m
        )
        tolerance = float(self.config.standoff_tolerance_m)
        verified = (
            abs(distance - requested) <= tolerance
            and math.degrees(heading_error) <= float(self.config.max_heading_error_degrees)
        )
        return verified, {
            "target_distance_m": round(distance, 3),
            "heading_error_degrees": round(math.degrees(heading_error), 1),
            "requested_standoff_m": requested,
        }

    def _contact_refinement_goal(self, goal: PoseStamped) -> PoseStamped:
        """Add a millimetre-scale reach margin after the safe global approach."""

        refined = deepcopy(goal)
        yaw = float(goal.orientation.to_euler().yaw)
        margin = float(self.config.manipulation_contact_margin_m)
        refined.position.x = float(goal.position.x) + math.cos(yaw) * margin
        refined.position.y = float(goal.position.y) + math.sin(yaw) * margin
        return refined

    def _memory_goal(self, target: str) -> PoseStamped | None:
        """Resolve a pose from the semantic memory built during exploration."""
        try:
            results = self._spatial_memory.query_by_text(target, limit=1)
        except Exception:
            return None
        if not results or not isinstance(results[0], dict):
            return None
        result = results[0]
        distance = result.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            return None
        if 1.0 - float(distance) < 0.23:
            return None
        metadata = result.get("metadata")
        if not isinstance(metadata, list) or not metadata or not isinstance(metadata[0], dict):
            return None
        first = metadata[0]
        try:
            values = (
                float(first["pos_x"]),
                float(first["pos_y"]),
                float(first.get("rot_z", 0.0)),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        return self._goal(values)

    @staticmethod
    def _goal(values: tuple[float, float, float]) -> PoseStamped:
        x, y, yaw = values
        return PoseStamped(
            frame_id="world",
            position=Vector3(x, y, 0.0),
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, yaw)),
        )

    def _navigate_pose(
        self,
        goal: PoseStamped,
        timeout: float,
        *,
        arrival_tolerance_m: float | None = None,
        heading_tolerance_degrees: float | None = None,
        allow_odometry_arrival: bool = False,
        takeover_radius_m: float | None = None,
    ) -> dict[str, Any]:
        self._planner_goal_reached = False
        if not self._navigation.set_goal(goal):
            return _result("planner_failed", False, error="planner rejected goal")
        # One immediate RPC handles planners that complete synchronously (and
        # keeps lightweight test doubles compatible). Subsequent completion
        # comes from the planner's goal_reached stream: repeatedly nesting an
        # is_goal_reached RPC inside the long-running fetch_object RPC can
        # consume the full 120-second transport timeout after motion starts.
        if self._navigation.is_goal_reached():
            return _result("arrived", True, planner_goal_reached=True)
        deadline = time.monotonic() + timeout
        observed_motion = False
        physical_motion_seen = False
        following_since: float | None = None
        idle_since: float | None = None
        stationary_since: float | None = None
        verified_pose_since: float | None = None
        previous_position: tuple[float, float] | None = None
        previous_yaw: float | None = None
        while time.monotonic() < deadline:
            cancel_reason = self._long_task_cancel_reason()
            if cancel_reason is not None:
                self._navigation.cancel_goal()
                return _result(
                    "cancelled",
                    False,
                    cancel_reason=cancel_reason,
                )
            if self._planner_goal_reached:
                return _result("arrived", True, planner_goal_reached=True)
            if observed_motion:
                odom = getattr(self, "_latest_odom", None)
                if odom is not None:
                    position = (float(odom.position.x), float(odom.position.y))
                    distance = math.hypot(
                        float(goal.position.x) - position[0],
                        float(goal.position.y) - position[1],
                    )
                    actual_yaw = float(odom.orientation.to_euler().yaw)
                    goal_yaw = float(goal.orientation.to_euler().yaw)
                    heading_error = abs(
                        math.atan2(
                            math.sin(goal_yaw - actual_yaw),
                            math.cos(goal_yaw - actual_yaw),
                        )
                    )
                    heading_error_degrees = math.degrees(heading_error)
                    tolerance = 0.10 if arrival_tolerance_m is None else arrival_tolerance_m
                    heading_ok = (
                        heading_tolerance_degrees is None
                        or heading_error_degrees <= heading_tolerance_degrees
                    )
                    if (
                        takeover_radius_m is not None
                        and distance <= takeover_radius_m
                    ):
                        # A global path follower is useful for reaching the
                        # table, but its final-pose controller can orbit a
                        # centimetre-scale manipulation goal while repeatedly
                        # correcting yaw. Hand the bounded residual to the
                        # odometry controller below instead of resubmitting the
                        # same A* goal.
                        self._navigation.cancel_goal()
                        return _result(
                            "near_goal",
                            True,
                            planner_goal_reached=False,
                            local_takeover_required=True,
                            final_position_error_m=round(distance, 3),
                            final_heading_error_degrees=round(
                                heading_error_degrees,
                                1,
                            ),
                        )
                    if allow_odometry_arrival and distance <= tolerance and heading_ok:
                        verified_pose_since = verified_pose_since or time.monotonic()
                        if time.monotonic() - verified_pose_since >= 0.5:
                            self._navigation.cancel_goal()
                            return _result(
                                "arrived",
                                True,
                                planner_goal_reached=False,
                                odometry_goal_verified=True,
                                final_position_error_m=round(distance, 3),
                                final_heading_error_degrees=round(
                                    heading_error_degrees,
                                    1,
                                ),
                            )
                    else:
                        verified_pose_since = None
                    yaw_progress = (
                        previous_yaw is not None
                        and abs(
                            math.atan2(
                                math.sin(actual_yaw - previous_yaw),
                                math.cos(actual_yaw - previous_yaw),
                            )
                        )
                        >= math.radians(1.0)
                    )
                    translation_progress = (
                        previous_position is not None
                        and math.dist(position, previous_position) >= 0.01
                    )
                    physical_motion_seen = bool(
                        physical_motion_seen
                        or translation_progress
                        or yaw_progress
                    )
                    if (
                        previous_position is not None
                        and math.dist(position, previous_position) < 0.01
                        and not yaw_progress
                    ):
                        stationary_since = stationary_since or time.monotonic()
                    else:
                        stationary_since = None
                    previous_position = position
                    previous_yaw = actual_yaw
                    # A* may replace an obstructed request with a nearby safe
                    # goal and clear its goal flag immediately on arrival.
                    # Fresh odometry avoids a second nested planner-state RPC,
                    # which can deadlock inside the outer object-skill RPC.
                    if (
                        physical_motion_seen
                        and stationary_since is not None
                        and time.monotonic() - stationary_since >= 2.5
                    ):
                        self._navigation.cancel_goal()
                        # A planner-adjusted endpoint is useful only when it is
                        # still a local substitute for the requested pose.  A
                        # cold map previously stopped several metres away and
                        # this branch falsely promoted that stall to arrival.
                        adjustment_limit = min(0.30, tolerance + 0.12)
                        if distance > adjustment_limit or not heading_ok:
                            return _result(
                                "planner_failed",
                                False,
                                planner_safe_goal_adjustment_rejected=True,
                                maximum_adjustment_m=round(adjustment_limit, 3),
                                final_position_error_m=round(distance, 3),
                                final_heading_error_degrees=round(heading_error_degrees, 1),
                                error=(
                                    "planner stopped with incorrect goal heading"
                                    if distance <= adjustment_limit and not heading_ok
                                    else "planner stopped too far from requested goal"
                                ),
                            )
                        return _result(
                            "planner_failed",
                            False,
                            planner_goal_reached=False,
                            planner_safe_goal_adjustment_rejected=True,
                            maximum_adjustment_m=round(adjustment_limit, 3),
                            final_position_error_m=round(distance, 3),
                            final_heading_error_degrees=round(
                                heading_error_degrees,
                                1,
                            ),
                            error="planner stopped without goal-reached evidence",
                        )
                    if (
                        not physical_motion_seen
                        and following_since is not None
                        and time.monotonic() - following_since >= 8.0
                    ):
                        self._navigation.cancel_goal()
                        adjustment_limit = min(0.30, tolerance + 0.12)
                        return _result(
                            "planner_failed",
                            False,
                            planner_goal_reached=False,
                            planner_safe_goal_adjustment_rejected=True,
                            maximum_adjustment_m=round(adjustment_limit, 3),
                            final_position_error_m=round(distance, 3),
                            final_heading_error_degrees=round(
                                heading_error_degrees,
                                1,
                            ),
                            error=(
                                "planner stopped with incorrect goal heading"
                                if distance <= adjustment_limit and not heading_ok
                                else "planner stopped without goal-reached evidence"
                                if distance <= adjustment_limit
                                else "planner did not start physical motion"
                            ),
                        )
                time.sleep(0.1)
                continue
            state = self._navigation.get_state()
            if state == NavigationState.FOLLOWING_PATH:
                observed_motion = True
                following_since = following_since or time.monotonic()
                idle_since = None
            elif state == NavigationState.IDLE:
                idle_since = idle_since or time.monotonic()
                idle_limit = 0.75 if observed_motion else 3.0
                if time.monotonic() - idle_since >= idle_limit:
                    self._navigation.cancel_goal()
                    error = (
                        "planner became idle before arrival"
                        if observed_motion
                        else "planner did not produce a path"
                    )
                    return _result("planner_failed", False, error=error)
            time.sleep(0.1)
        self._navigation.cancel_goal()
        return _result("navigation_timeout", False, error="semantic navigation timed out")

    def _refine_manipulation_pose(
        self,
        goal: PoseStamped,
        *,
        timeout: float | None = None,
        position_tolerance_m: float = 0.003,
        heading_tolerance_degrees: float = 5.0,
        require_heading: bool = True,
        translation_guard=None,
    ) -> dict[str, Any]:
        """Finish a nearby manipulation pose without another global A* goal."""

        publisher = getattr(self, "nav_cmd_vel", None)
        if publisher is None or not hasattr(publisher, "publish"):
            return _result(
                "planner_failed",
                False,
                error="local manipulation velocity channel unavailable",
            )

        # Cancel inside the navigation process and publish zero before taking
        # over its velocity topic. The bounded relay deliberately rejects late
        # planner commands for 500 ms after this transition.
        publisher.publish(Twist.zero())
        stop = getattr(self, "stop_movement", None)
        if stop is not None and hasattr(stop, "publish"):
            stop.publish(Bool(data=True))
        time.sleep(0.6)

        deadline = time.monotonic() + float(
            self.config.manipulation_refine_timeout_s
            if timeout is None
            else timeout
        )
        stable_since: float | None = None
        last_distance = math.inf
        last_heading_error = math.inf
        try:
            while time.monotonic() < deadline:
                cancel_reason = self._long_task_cancel_reason()
                if cancel_reason is not None:
                    return _result(
                        "cancelled",
                        False,
                        cancel_reason=cancel_reason,
                    )
                odom = getattr(self, "_latest_odom", None)
                if odom is None:
                    time.sleep(0.05)
                    continue
                if translation_guard is not None and not translation_guard(odom):
                    return _result("risk_blocked", False, error="fresh refinement corridor unavailable")
                x = float(odom.position.x)
                y = float(odom.position.y)
                yaw = float(odom.orientation.to_euler().yaw)
                dx = float(goal.position.x) - x
                dy = float(goal.position.y) - y
                distance = math.hypot(dx, dy)
                goal_yaw = float(goal.orientation.to_euler().yaw)
                yaw_error = math.atan2(
                    math.sin(goal_yaw - yaw),
                    math.cos(goal_yaw - yaw),
                )
                if not require_heading:
                    yaw_error = 0.0
                heading_error_degrees = abs(math.degrees(yaw_error))
                last_distance = distance
                last_heading_error = heading_error_degrees
                if (
                    distance <= position_tolerance_m
                    and heading_error_degrees <= heading_tolerance_degrees
                ):
                    stable_since = stable_since or time.monotonic()
                    publisher.publish(Twist.zero())
                    if time.monotonic() - stable_since >= 0.4:
                        return _result(
                            "arrived",
                            True,
                            planner_goal_reached=False,
                            local_odometry_refined=True,
                            final_position_error_m=round(distance, 3),
                            final_heading_error_degrees=round(
                                heading_error_degrees,
                                1,
                            ),
                        )
                    time.sleep(0.05)
                    continue
                stable_since = None

                # Commands are expressed in the robot frame and remain below
                # the normal navigation relay limits. Reduce translation while
                # badly misaligned so the G1 does not circle the target.
                local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
                local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                translation_scale = max(
                    0.0,
                    min(1.0, 1.0 - abs(yaw_error) / math.radians(75.0)),
                )
                max_speed = 0.08
                requested_speed = min(max_speed, max(0.025, 0.9 * distance))
                if distance > 1e-6:
                    vx = requested_speed * local_x / distance
                    vy = requested_speed * local_y / distance
                else:
                    vx = 0.0
                    vy = 0.0
                vx *= translation_scale
                vy *= translation_scale
                yaw_rate = max(-0.25, min(0.25, 1.2 * yaw_error))
                publisher.publish(
                    Twist(
                        linear=Vector3(vx, vy, 0.0),
                        angular=Vector3(0.0, 0.0, yaw_rate),
                    )
                )
                time.sleep(0.05)
        finally:
            publisher.publish(Twist.zero())
            if stop is not None and hasattr(stop, "publish"):
                stop.publish(Bool(data=True))

        return _result(
            "navigation_timeout",
            False,
            final_position_error_m=(
                None if not math.isfinite(last_distance) else round(last_distance, 3)
            ),
            final_heading_error_degrees=(
                None
                if not math.isfinite(last_heading_error)
                else round(last_heading_error, 1)
            ),
            error="local manipulation pose refinement timed out",
        )

    def _global_semantic_route(
        self, normalized: str, final: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], ...]:
        """Return one semantic goal per navigation RPC.

        Navigation proxy calls are synchronous.  Submitting multiple waypoint
        goals from one outer MCP request can leave the second proxy call
        waiting for re-entry even though the first goal completed.  Semantic
        room goals therefore name a single, sensor-visible observation pose;
        manipulation performs its later visual approach as a separate phase.
        """
        del normalized
        return (final,)

    @skill()
    def locate_composed_object(self, entity_id: str, expires_at: float,
                               cancel_revision: str = "") -> dict[str, Any]:
        """独立组合感知 RPC：只读本轮 RGB-D，不查询实体坐标、不调用预设取物流程。"""
        import os
        from harness.control.terminal_cancellation import TerminalCancellationChannel
        from harness.skills.object_perception import localize_object_rgbd
        if os.environ.get("LUXI_COMPOSED_DEV") != "1":
            return _result("tool_denied", False)
        if (entity_id not in ENTITY_VISUAL_QUERIES or type(expires_at) not in (float, int)
                or not 0 < expires_at-time.time() <= 120):
            return _result("invalid_input", False)
        channel = TerminalCancellationChannel()
        revision = cancel_revision or None
        if channel.changed_since(revision):
            return _result("cancelled", False)
        self._composition_cancel = (channel, revision, expires_at)
        try:
            if not hasattr(self, '_vl_model'):
                return _result('observation_unavailable', False)
            outcome = localize_object_rgbd(entity_id=entity_id, query=ENTITY_VISUAL_QUERIES[entity_id],
                acquire=self._synced_rgbd, tf=self.tf, camera_info=self.config.camera_info,
                model=self._vl_model, bbox_query=get_object_bbox_from_image,
                min_points=self.config.min_depth_points, after=time.time(), compact=True,
                sync_tolerance_s=self.config.sync_tolerance_s)
            if self._long_task_cancel_reason() is not None:
                return _result('cancelled', False)
            return outcome
        finally:
            self._composition_cancel = None

    @skill(uses=[CAP_MOVEMENT])
    def navigate_composed_pose(self, x: float, y: float, yaw: float,
                               expires_at: float, cancel_revision: str = "", require_heading: bool = True) -> dict[str, Any]:
        """Candidate-only metric navigation; same planner, cancellation and stop path."""
        import os
        from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD
        from harness.control.terminal_cancellation import TerminalCancellationChannel
        if os.environ.get("LUXI_COMPOSED_DEV") != "1":
            return _result("tool_denied", False, error="composition candidate is disabled")
        values = (x, y, yaw, expires_at)
        if (type(require_heading) is not bool or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values)
                or not 0 < expires_at - time.time() <= 120):
            return _result("invalid_input", False)
        channel = TerminalCancellationChannel()
        revision = cancel_revision or None
        if channel.changed_since(revision):
            return _result("cancelled", False)
        self._composition_cancel = (channel, revision, expires_at)
        try:
            if not require_heading:
                odom = getattr(self, '_latest_odom', None)
                if odom is None:
                    return _result('observation_unavailable', False)
                dx, dy = x-float(odom.position.x), y-float(odom.position.y)
                # A path-following hint, never a user terminal-heading condition.
                yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > .10 else float(odom.orientation.to_euler().yaw)
            outcome = self._navigate_pose(self._goal((x, y, yaw)), expires_at-time.time(),
                                           arrival_tolerance_m=POSITION_TOLERANCE_M,
                                           heading_tolerance_degrees=math.degrees(YAW_TOLERANCE_RAD) if require_heading else None)
            if outcome.get("operation_ok") and outcome.get("planner_goal_reached"):
                settled = self._confirm_stationary_after_stop()
                if not settled.get("stationary_confirmed"):
                    return {**settled, "operation_ok": False, "task_status": "verification_failed"}
                odom = getattr(self, "_latest_odom", None)
                error = math.inf if odom is None else math.hypot(x-float(odom.position.x), y-float(odom.position.y))
                actual_yaw = 0 if odom is None else float(odom.orientation.to_euler().yaw)
                yaw_error = abs(math.atan2(math.sin(yaw-actual_yaw), math.cos(yaw-actual_yaw)))
                # 精调只补足任务到位要求；已经达标不能被额外精度门槛阻断。
                if error > POSITION_TOLERANCE_M or (require_heading and yaw_error > YAW_TOLERANCE_RAD):
                    from harness.robots.g1.mujoco.composed_navigation import refine_arrival
                    refined = refine_arrival(self, self._goal((x, y, yaw)), expires_at, require_heading=require_heading)
                    outcome = {**outcome, **refined, "planner_goal_reached": True}
        finally:
            stop = self._confirm_stationary_after_stop()
            self._composition_cancel = None
        return {**outcome, **stop, "operation_ok": bool(outcome.get("operation_ok") and stop.get("stationary_confirmed")),
                "interaction_model": "sim_attachment", "candidate": True}

    @skill(uses=[CAP_MOVEMENT])
    def navigate_to(self, target: str, timeout: float = 60.0) -> dict[str, Any]:
        """Navigate to a semantic destination.

        When a global semantic map is available, this submits its metric pose.
        If no global semantic map is available, it returns
        ``semantic_map_unavailable`` so the planner can explore while building
        one.  A ``planner_failed`` result with a known semantic goal instead
        means the local obstacle map may be cold: acquire a 360-degree in-place
        sensor sweep before retrying, without translating into unseen space.
        """
        normalized = str(target).strip().casefold()
        if not 1 <= len(normalized) <= 80:
            return _result("invalid_input", False, error="target must be 1-80 characters")
        if self._semantic_map_available() and normalized in SEMANTIC_GOALS:
            route = self._global_semantic_route(normalized, SEMANTIC_GOALS[normalized])
            navigation_mode = "global_semantic_map"
        else:
            goal = self._memory_goal(normalized)
            route = (goal,) if goal is not None else ()
            navigation_mode = "explored_semantic_map"
        if not route:
            return _result(
                "semantic_map_unavailable",
                False,
                navigation_mode="explore_and_build",
                exploration_required=True,
                target=normalized,
            )
        timeout = min(120.0, max(3.0, float(timeout)))
        deadline = time.monotonic() + timeout
        route_steps: list[dict[str, Any]] = []
        outcome: dict[str, Any] = _result("planner_failed", False)
        for index, route_goal in enumerate(route):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                outcome = _result("navigation_timeout", False, error="semantic route timed out")
                break
            goal = route_goal if isinstance(route_goal, PoseStamped) else self._goal(route_goal)
            # Room-level semantic navigation does not need manipulation-grade
            # 6 cm convergence.  Stopping inside an 18 cm semantic region
            # avoids the gait controller orbiting a point while it tries to
            # satisfy an unnecessarily tight room pose; precise RGB-D approach
            # below intentionally keeps the strict planner tolerance.
            outcome = self._navigate_pose(
                goal,
                remaining,
                arrival_tolerance_m=0.18 if index == len(route) - 1 else None,
            )
            route_steps.append(
                {
                    "index": index,
                    "x": round(float(goal.position.x), 3),
                    "y": round(float(goal.position.y), 3),
                    "task_status": outcome.get("task_status"),
                }
            )
            if not outcome.get("operation_ok"):
                break
        outcome.update(navigation_mode=navigation_mode, target=normalized)
        if len(route) > 1:
            outcome["route_steps"] = route_steps
        return outcome

    @skill
    def locate_entity(self, entity_id: str) -> dict[str, Any]:
        """Locate an entity from fresh RGB-D, with a backend cross-check when available."""
        entity_id = normalize_entity_id(entity_id)
        pose, visual, response = self._locate_entity_evidence(entity_id)
        return _result(
            "entity_located" if pose is not None else "entity_not_found",
            pose is not None,
            entity_id=entity_id,
            pose=pose,
            visual_evidence=visual,
            error=response.get("error"),
        )

    @skill(uses=[CAP_MOVEMENT])
    def approach_entity(self, entity_id: str, hand: str = "right") -> dict[str, Any]:
        """Use RGB-D to reach a target-facing standoff, reobserve, then run arm IK."""
        entity_id = normalize_entity_id(entity_id)
        pose, visual, located = self._locate_entity_evidence(
            entity_id,
            reuse_recent_visual=True,
        )
        base_navigation: dict[str, Any] | None = None
        base_navigation_attempts: list[dict[str, Any]] = []
        if pose is None:
            return _result(
                "entity_not_found",
                False,
                entity_id=entity_id,
                hand=str(hand),
                visual_evidence=visual,
                error=located.get("error") or "fresh visual entity pose unavailable",
            )
        if (
            self.config.require_visual_manipulation
            and (visual or {}).get("source") == "simulation_entity_adapter"
        ):
            # The backend pose is not manipulation evidence, but in simulation
            # it can safely supply a viewpoint direction when the object sits
            # just outside the head-camera FOV.  Rotate in place, acquire a new
            # RGB-D frame, then still require depth verification before any
            # base approach or arm motion.  Real hardware never enters this
            # adapter-only branch.
            for camera_margin in (0.0, math.radians(-10.0), math.radians(10.0)):
                odom = getattr(self, "_latest_odom", None)
                if odom is None or pose is None or len(pose) < 2:
                    break
                # The head camera is narrower than the room planner's 15° yaw
                # tolerance.  Bias the object into the image instead of merely
                # accepting it at the navigation heading boundary.
                view_yaw = math.atan2(
                    float(pose[1]) - float(odom.position.y),
                    float(pose[0]) - float(odom.position.x),
                ) + camera_margin
                view_goal = self._goal(
                    (float(odom.position.x), float(odom.position.y), view_yaw)
                )
                observation_cutoff = time.time()
                view_navigation = self._navigate_pose(
                    view_goal,
                    15.0,
                    heading_tolerance_degrees=5.0,
                )
                if not view_navigation.get("operation_ok"):
                    break
                pose, visual, located = self._locate_entity_evidence(
                    entity_id,
                    after=observation_cutoff,
                )
                if pose is not None and (visual or {}).get("source") != "simulation_entity_adapter":
                    break
            # If the distant observation pose still cannot see this small,
            # low bottle, the simulation adapter may seed only the safe base
            # standoff below.  The post-navigation branch must obtain fresh
            # RGB-D evidence before it can invoke arm IK.

        nominal_standoff = float(self.config.manipulation_standoff_m)
        verified, verification = self._manipulation_pose_verified(
            pose,
            requested_standoff_m=nominal_standoff,
        )

        def reposition(requested_standoff: float) -> tuple[bool, str | None]:
            nonlocal pose, visual, verification, base_navigation
            goal = self._manipulation_goal(
                pose,
                str(hand),
                standoff_m=requested_standoff,
                approach_yaw=(
                    ENTITY_APPROACH_YAWS.get(entity_id)
                    if self._semantic_map_available()
                    else None
                ),
            )
            if goal is None:
                return False, "could not construct manipulation standoff"
            base_navigation = self._navigate_pose(
                goal,
                45.0,
                arrival_tolerance_m=0.04,
                heading_tolerance_degrees=min(
                    20.0,
                    float(self.config.max_heading_error_degrees),
                ),
                allow_odometry_arrival=True,
                takeover_radius_m=float(
                    self.config.manipulation_takeover_radius_m
                ),
            )
            base_navigation_attempts.append(
                {"requested_standoff_m": requested_standoff, "result": base_navigation}
            )
            if not base_navigation.get("operation_ok"):
                return False, str(base_navigation.get("error") or "base navigation failed")
            local_goal = self._contact_refinement_goal(goal)
            local_refinement = self._refine_manipulation_pose(local_goal)
            local_refinement["contact_margin_m"] = float(
                self.config.manipulation_contact_margin_m
            )
            base_navigation["local_refinement"] = local_refinement
            if not local_refinement.get("operation_ok"):
                return False, str(
                    local_refinement.get("error")
                    or "local manipulation pose refinement failed"
                )
            # Navigation cancellation and the worker's idle-pose stabilizer
            # are asynchronous.  Let the floating base and torso return to a
            # reproducible stationary posture before the fresh RGB-D frame and
            # arm IK are evaluated.
            time.sleep(float(self.config.manipulation_settle_s))
            observation_cutoff = time.time()
            refreshed_pose, refreshed_visual, refreshed = self._locate_entity_evidence(
                entity_id,
                after=observation_cutoff,
            )
            visual = refreshed_visual
            if refreshed_pose is None or (
                self.config.require_visual_manipulation
                and (refreshed_visual or {}).get("source")
                == "simulation_entity_adapter"
            ):
                return False, str(
                    refreshed.get("error")
                    or "post-navigation RGB-D verification failed"
                )
            pose = refreshed_pose
            verified_after, verification = self._manipulation_pose_verified(
                pose,
                requested_standoff_m=requested_standoff,
            )
            if not verified_after:
                return False, "post-navigation manipulation pose was not verified"
            return True, None

        if not verified:
            verified, error = reposition(nominal_standoff)
            if not verified:
                return _result(
                    "approach_failed",
                    False,
                    entity_id=entity_id,
                    hand=str(hand),
                    base_navigation=base_navigation,
                    base_navigation_attempts=base_navigation_attempts,
                    visual_evidence=visual,
                    manipulation_pose=verification,
                    error=error,
                )

        response = self._entities.approach(
            entity_id,
            str(hand),
            timeout=8.0,
        )
        contact_approaches = getattr(self, "_recent_contact_approaches", None)
        if contact_approaches is None:
            contact_approaches = {}
            self._recent_contact_approaches = contact_approaches
        if response.get("operation_ok") and response.get("contact"):
            # Once the hand reaches the bottle it can occlude the head camera.
            # Preserve this short-lived, same-hand physical-contact proof for
            # the immediately following grasp.  The worker still performs its
            # own live contact check, so this cannot enable a remote grasp.
            contact_approaches[entity_id] = {
                "hand": str(hand),
                "confirmed_at": time.time(),
                "visual_evidence": visual,
            }
        else:
            contact_approaches.pop(entity_id, None)
        return _result(
            "entity_approached"
            if response.get("operation_ok")
            else "approach_failed",
            bool(response.get("operation_ok")),
            entity_id=entity_id,
            hand=str(hand),
            distance_m=_entity_value(response, "distance_m"),
            contact=response.get("contact"),
            ik_diagnostics=_entity_value(response, "ik_diagnostics"),
            base_navigation=base_navigation,
            base_navigation_attempts=base_navigation_attempts,
            visual_evidence=visual,
            manipulation_pose=verification,
            error=response.get("error"),
        )

    @skill(uses=[CAP_MOVEMENT])
    def grasp_entity(self, entity_id: str, hand: str = "right") -> dict[str, Any]:
        """Require recent robot-vision evidence and backend physical contact."""
        entity_id = normalize_entity_id(entity_id)
        hand = str(hand)
        visual = getattr(self, "_last_visual_entities", {}).get(entity_id)
        if visual is not None:
            age = time.time() - float(visual.get("observed_at", 0.0))
            if age > float(self.config.visual_evidence_max_age_s):
                visual = self._visual_entity_pose(entity_id)
        elif hasattr(self, "_vision_lock"):
            visual = self._visual_entity_pose(entity_id)
        contact_approach = getattr(self, "_recent_contact_approaches", {}).get(entity_id)
        contact_age = math.inf
        if isinstance(contact_approach, dict):
            contact_age = time.time() - float(contact_approach.get("confirmed_at", 0.0))
        recent_same_hand_contact = bool(
            isinstance(contact_approach, dict)
            and contact_approach.get("hand") == hand
            and 0.0 <= contact_age <= 5.0
        )
        live_contact: EntityOperationResult | None = None
        live_contact_confirmed = False
        if (
            hasattr(self, "_vision_lock")
            and visual is None
            and not recent_same_hand_contact
        ):
            # Atomic MCP calls may be separated by an LLM round that exceeds
            # the short approach cache.  The hand can also occlude the bottle
            # after a successful approach.  Re-check backend physical contact
            # instead of either trusting stale evidence or rejecting a still
            # valid grasp.  The grasp operation performs its own contact check
            # again, so a moved hand/entity still fails closed.
            try:
                live_contact = self._entities.contact_state(
                    entity_id,
                    hand,
                    timeout=2.0,
                )
            except Exception:  # noqa: BLE001 - unavailable evidence fails closed
                live_contact = None
            live_contact_confirmed = bool(
                live_contact is not None
                and live_contact.get("operation_ok")
                and live_contact.get("contact")
            )
        if (
            hasattr(self, "_vision_lock")
            and visual is None
            and not recent_same_hand_contact
            and not live_contact_confirmed
        ):
            return _result(
                "grasp_failed",
                False,
                entity_id=entity_id,
                hand=str(hand),
                error=(
                    "fresh visual entity evidence or current physical contact "
                    "is required before grasp"
                ),
            )
        evidence_timestamp = None
        if isinstance(visual, Mapping):
            timestamp = visual.get("frame_timestamp", visual.get("observed_at"))
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                evidence_timestamp = float(timestamp)
        elif live_contact is not None:
            timestamp = live_contact.get("evidence_timestamp")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                evidence_timestamp = float(timestamp)
        response = self._entities.grasp(
            entity_id,
            hand,
            evidence_timestamp=evidence_timestamp,
            timeout=7.0,
        )
        getattr(self, "_recent_contact_approaches", {}).pop(entity_id, None)
        return _result(
            "entity_grasped"
            if response.get("operation_ok")
            else "grasp_failed",
            bool(response.get("operation_ok")),
            entity_id=entity_id,
            hand=hand,
            visual_evidence=(
                visual
                if visual is not None
                else contact_approach.get("visual_evidence")
                if isinstance(contact_approach, dict)
                else None
            ),
            contact_approach_confirmed=recent_same_hand_contact,
            pregrasp_contact_confirmed=bool(
                recent_same_hand_contact or live_contact_confirmed
            ),
            live_contact_evidence=(
                {
                    "backend": live_contact.get("backend"),
                    "hand": live_contact.get("hand"),
                    "contact": live_contact.get("contact"),
                    "evidence_timestamp": live_contact.get(
                        "evidence_timestamp"
                    ),
                }
                if live_contact is not None
                else None
            ),
            contact_confirmed=_entity_value(
                response,
                "contact_confirmed",
                False,
            ),
            grasped=_entity_value(response, "grasped", False),
            attached=response.get("attached", False),
            error=response.get("error"),
        )

    @skill(uses=[CAP_MOVEMENT])
    def prepare_carry_entity(
        self,
        entity_id: str,
        hand: str = "right",
    ) -> dict[str, Any]:
        """Retract from the grasp pose and return arm/waist ownership to gait."""

        entity_id = normalize_entity_id(entity_id)
        response = self._entities.carry(
            entity_id,
            str(hand),
            timeout=8.0,
        )
        return _result(
            "carry_pose_ready"
            if response.get("operation_ok")
            else "carry_pose_failed",
            bool(response.get("operation_ok")),
            entity_id=entity_id,
            hand=str(hand),
            attachment_retained=response.get("attached", False),
            gait_ownership_restored=_entity_value(
                response,
                "gait_ownership_restored",
                False,
            ),
            error=response.get("error"),
        )

    @skill(uses=[CAP_MOVEMENT])
    def carry_entity(self, entity_id: str, destination: str) -> dict[str, Any]:
        """Verify an attachment, navigate, and verify attachment retention."""
        entity_id = normalize_entity_id(entity_id)
        before = self._entities.entity_state(entity_id, timeout=2.0)
        if not before.get("operation_ok") or not before.get("attached"):
            return _result("carry_failed", False, error="entity is not attached")
        navigation = self.navigate_to(destination)
        after = self._entities.entity_state(entity_id, timeout=2.0)
        retained = bool(after.get("operation_ok") and after.get("attached"))
        ok = bool(navigation.get("operation_ok") and retained)
        return _result(
            "entity_carried" if ok else "carry_failed",
            ok,
            entity_id=entity_id,
            destination=str(destination),
            attachment_retained=retained,
            navigation=navigation,
        )

    @skill(uses=[CAP_MOVEMENT])
    def place_entity(
        self,
        entity_id: str,
        target: str = "nearby_surface",
        hand: str = "right",
    ) -> dict[str, Any]:
        """Place a held entity without teleporting it away from the hand.

        The initial adapter supports ``nearby_surface``: the bottle is lowered
        12 cm from its current pose through arm control, then the attachment is
        released and collisions are restored. Semantic surface placement can
        be added without changing the task API.
        """
        entity_id = normalize_entity_id(entity_id)
        status = self._entities.entity_state(entity_id, timeout=2.0)
        pose = status.get("pose")
        if (
            not status.get("operation_ok")
            or not status.get("attached")
            or not isinstance(pose, list)
        ):
            return _result("place_failed", False, error="entity is not attached")
        if str(target).strip().casefold() not in {"nearby_surface", "附近台面", "附近表面"}:
            return _result("place_failed", False, error="only nearby_surface is currently supported")
        resting_height = _entity_value(status, "resting_height_m")
        if (
            isinstance(resting_height, (int, float))
            and not isinstance(resting_height, bool)
            and math.isfinite(float(resting_height))
        ):
            target_position = [
                float(pose[0]),
                float(pose[1]),
                float(resting_height),
            ]
        else:
            target_position = [
                float(pose[0]),
                float(pose[1]),
                max(0.20, float(pose[2]) - 0.12),
            ]
        response = self._entities.place(
            entity_id,
            str(hand),
            target_position,
            timeout=8.0,
        )
        return _result(
            "entity_placed"
            if response.get("operation_ok")
            else "place_failed",
            bool(response.get("operation_ok")),
            entity_id=entity_id,
            target=str(target),
            placed=_entity_value(response, "placed", False),
            pose=response.get("pose"),
            error=response.get("error"),
        )

    def _fetch_object_pipeline(
        self,
        object_id: str,
        pickup_pose: str,
        destination: str = "start",
        hand: str = "right",
    ) -> dict[str, Any]:
        """Fetch one object through a single terminal closed-loop pipeline.

        The caller must invoke this tool at most once.  A failed result may
        already include physical progress and must not be retried or completed
        by issuing lower-level movement tools in the same Agent turn.
        """
        object_id = normalize_entity_id(object_id)
        start = deepcopy(self._latest_odom)
        if start is None:
            return {"tool_ok": False, "completed": False, "task_status": "pose_unavailable"}
        steps: list[dict[str, Any]] = []

        def incomplete(task_status: str) -> dict[str, Any]:
            stationary = self._confirm_stationary_after_stop()
            steps.append({"step": "confirm_stationary", "result": stationary})
            try:
                retained = (
                    self._entities.entity_state(object_id, timeout=2.0).get(
                        "attached"
                    )
                    is True
                )
            except Exception:  # noqa: BLE001 - status evidence is optional on failure
                retained = False
            result = {
                "tool_ok": True,
                "completed": False,
                "task_status": str(task_status),
                "object_id": object_id,
                "destination": destination,
                "attachment_retained": retained,
                **stationary,
                "steps": steps,
            }
            cancel_reason = self._long_task_cancel_reason()
            if cancel_reason is not None:
                result["cancel_reason"] = cancel_reason
            return result

        cancel_reason = self._long_task_cancel_reason()
        if cancel_reason is not None:
            return incomplete("cancelled")
        self._long_task_stage("navigate_to_pickup")
        navigation = self.navigate_to(pickup_pose)
        steps.append({"step": "navigate_to", "result": navigation})
        if not navigation.get("operation_ok"):
            return incomplete(str(navigation["task_status"]))
        for name, invoke in (
            ("locate_entity", lambda: self.locate_entity(object_id)),
            ("approach_entity", lambda: self.approach_entity(object_id, hand)),
            ("grasp_entity", lambda: self.grasp_entity(object_id, hand)),
            (
                "prepare_carry_entity",
                lambda: self.prepare_carry_entity(object_id, hand),
            ),
        ):
            cancel_reason = self._long_task_cancel_reason()
            if cancel_reason is not None:
                return incomplete("cancelled")
            self._long_task_stage(name)
            outcome = invoke()
            steps.append({"step": name, "result": outcome})
            if self._long_task_cancel_reason() is not None:
                return incomplete("cancelled")
            if not outcome.get("operation_ok"):
                return incomplete(str(outcome["task_status"]))
        cancel_reason = self._long_task_cancel_reason()
        if cancel_reason is not None:
            return incomplete("cancelled")
        self._long_task_stage("carry_entity")
        if str(destination).strip().casefold() in {"start", "起点", "原地"}:
            # Odometry is a floating-base pose whose z is the G1 pelvis height
            # (~0.755 m). The navigation planner accepts ground-plane goals;
            # passing odometry through unchanged makes it report a permanent
            # 0.755 m goal error even after x/y arrival.
            start_goal = self._goal(
                (
                    float(start.position.x),
                    float(start.position.y),
                    float(start.orientation.to_euler().yaw),
                )
            )
            carried = self._navigate_pose(start_goal, 90.0)
            retained = (
                self._entities.entity_state(object_id, timeout=2.0).get("attached")
                is True
            )
            carried = _result("entity_carried" if carried.get("operation_ok") and retained else "carry_failed", bool(carried.get("operation_ok") and retained), attachment_retained=retained, navigation=carried)
        else:
            carried = self.carry_entity(object_id, destination)
        steps.append({"step": "carry_entity", "result": carried})
        if self._long_task_cancel_reason() is not None:
            return incomplete("cancelled")
        stationary = self._confirm_stationary_after_stop()
        steps.append({"step": "confirm_stationary", "result": stationary})
        completed = bool(
            carried.get("operation_ok")
            and carried.get("attachment_retained")
            and stationary.get("stationary_confirmed")
        )
        task_status = (
            "object_fetched"
            if completed
            else "verification_failed"
            if carried.get("operation_ok")
            else "carry_failed"
        )
        carry_navigation = carried.get("navigation")
        return {
            "tool_ok": True,
            "completed": completed,
            "task_status": task_status,
            "object_id": object_id,
            "destination": destination,
            "attachment_retained": carried.get("attachment_retained", False),
            "planner_goal_reached": bool(
                isinstance(carry_navigation, Mapping)
                and carry_navigation.get("planner_goal_reached") is True
            ),
            **stationary,
            "steps": steps,
        }

    @skill(uses=[CAP_MOVEMENT])
    def fetch_object(
        self,
        object_id: str,
        pickup_pose: str,
        destination: str = "start",
        hand: str = "right",
        job_id: str = "",
    ) -> dict[str, Any]:
        """Run the terminal fetch pipeline as one observable, cancellable job."""

        try:
            normalized_job_id = (
                str(uuid.UUID(str(job_id))) if str(job_id).strip() else str(uuid.uuid4())
            )
        except (ValueError, TypeError, AttributeError):
            return {
                "tool_ok": False,
                "completed": False,
                "task_status": "invalid_input",
                "error": "job_id must be a UUID when provided",
            }
        arguments = {
            "object_id": str(object_id),
            "pickup_pose": str(pickup_pose),
            "destination": str(destination),
            "hand": str(hand),
        }
        channel = getattr(self, "_long_task_control", None)
        if channel is None:
            return self._fetch_object_pipeline(
                object_id,
                pickup_pose,
                destination,
                hand,
            )
        self._active_long_task_id = normalized_job_id
        channel.begin(
            normalized_job_id,
            tool="fetch_object",
            owner="ui" if str(job_id).strip() else "skill",
            arguments=arguments,
        )
        try:
            result = self._fetch_object_pipeline(
                object_id,
                pickup_pose,
                destination,
                hand,
            )
            cancelled = self._long_task_cancel_reason()
            terminal_state = (
                "cancelled"
                if cancelled is not None or result.get("task_status") == "cancelled"
                else "completed"
                if result.get("completed") is True
                else "failed"
            )
            channel.update(
                normalized_job_id,
                state=terminal_state,
                stage=terminal_state,
                result=result,
                cancel_reason=cancelled,
            )
            return result
        finally:
            self._active_long_task_id = None


object_task_skills = ObjectTaskSkillContainer.blueprint
