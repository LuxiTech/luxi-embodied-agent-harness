"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import base64
from dataclasses import dataclass
import math
import re
import time
from typing import Any, Callable

from harness.skills.tool_results import (
    ISAAC_PURE_YAW_SETTLE_INTERVAL_SECONDS,
    ISAAC_PURE_YAW_SETTLE_SECONDS,
    _structured_mcp_payload,
)

ROOM_LOOP_MAP_MAX_AGE_SECONDS = 1.50

ROOM_LOOP_ROUTE_CLEARANCE_M = 0.75

ROOM_LOOP_MIN_SIDE_M = 1.00

ROOM_LOOP_MAX_SIDE_M = 2.00

ROOM_LOOP_MAX_TRANSLATION_PULSE_M = 0.16

ROOM_LOOP_DYNAMIC_LOOKAHEAD_M = 0.02

ROOM_LOOP_ROBUST_CLEARANCES_M = (0.85, 0.95, 1.05)

ROOM_LOOP_STABLE_REVISIONS = 3

ROOM_LOOP_STABILITY_TIMEOUT_SECONDS = 6.0

ROOM_LOOP_STABILITY_POLL_SECONDS = 0.10

@dataclass(frozen=True)
class _OnlineCostmap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    cells: bytes

    @classmethod
    def from_payload(cls, payload: Any) -> _OnlineCostmap:
        if not isinstance(payload, dict):
            raise ValueError("实时 costmap 不是 object")
        try:
            available = payload.get("available") is True
            source = str(payload.get("source") or "")
            frame_id = str(payload.get("frame_id") or "")
            age = float(payload["age_seconds"])
            width = int(payload["width"])
            height = int(payload["height"])
            resolution = float(payload["resolution"])
            origin = payload["origin"]
            origin_x = float(origin["x"])
            origin_y = float(origin["y"])
            origin_yaw = float(origin.get("yaw", 0.0))
            cells = base64.b64decode(payload["data"], validate=True)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("实时 costmap 字段无效") from error
        if not available or source != "live" or frame_id != "world":
            raise ValueError("实时 world costmap 不可用")
        if (
            not all(
                math.isfinite(value)
                for value in (age, resolution, origin_x, origin_y, origin_yaw)
            )
            or age < 0.0
            or age > ROOM_LOOP_MAP_MAX_AGE_SECONDS
            or width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(cells) != width * height
        ):
            raise ValueError("实时 costmap 过期或尺寸无效")
        return cls(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_yaw=origin_yaw,
            cells=cells,
        )

    def clearance_offsets(self, clearance_m: float) -> tuple[tuple[int, int], ...]:
        radius = max(0, math.ceil(clearance_m / self.resolution))
        return tuple(
            (row_offset, column_offset)
            for row_offset in range(-radius, radius + 1)
            for column_offset in range(-radius, radius + 1)
            if math.hypot(
                row_offset * self.resolution,
                column_offset * self.resolution,
            )
            <= clearance_m
        )

    def _cell(self, world_x: float, world_y: float) -> tuple[int, int]:
        relative_x = world_x - self.origin_x
        relative_y = world_y - self.origin_y
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = cos_yaw * relative_x + sin_yaw * relative_y
        local_y = -sin_yaw * relative_x + cos_yaw * relative_y
        return (
            math.floor(local_y / self.resolution),
            math.floor(local_x / self.resolution),
        )

    def with_verified_free_disk(
        self,
        center: tuple[float, float],
        radius_m: float,
    ) -> _OnlineCostmap:
        """Refresh the local map disk proven empty by current physical lidar.

        HeightCost is accumulated planning memory and can retain a robot/self
        artifact behind the current pose.  A fresh, identity-filtered 360-degree
        lidar frame is stronger evidence only inside its measured empty radius;
        outside that disk, unknown and cost cells remain fail-closed.
        """

        radius = max(0.0, float(radius_m))
        if radius <= 0.0:
            return self
        center_row, center_column = self._cell(*center)
        radius_cells = math.ceil(radius / self.resolution)
        cells = bytearray(self.cells)
        for row_offset in range(-radius_cells, radius_cells + 1):
            for column_offset in range(-radius_cells, radius_cells + 1):
                if math.hypot(
                    row_offset * self.resolution,
                    column_offset * self.resolution,
                ) > radius:
                    continue
                row = center_row + row_offset
                column = center_column + column_offset
                if not (0 <= row < self.height and 0 <= column < self.width):
                    continue
                index = row * self.width + column
                cells[index] = 0
        return _OnlineCostmap(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            origin_yaw=self.origin_yaw,
            cells=bytes(cells),
        )

    def point_is_clear(
        self,
        world_x: float,
        world_y: float,
        offsets: tuple[tuple[int, int], ...],
    ) -> bool:
        row, column = self._cell(world_x, world_y)
        for row_offset, column_offset in offsets:
            nearby_row = row + row_offset
            nearby_column = column + column_offset
            if not (
                0 <= nearby_row < self.height
                and 0 <= nearby_column < self.width
            ):
                return False
            if self.cells[nearby_row * self.width + nearby_column] >= 50:
                return False
        return True

    def segment_is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        clearance_m: float,
    ) -> bool:
        offsets = self.clearance_offsets(clearance_m)
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(1, math.ceil(distance / max(0.025, self.resolution * 0.75)))
        return all(
            self.point_is_clear(
                start[0] + (end[0] - start[0]) * index / samples,
                start[1] + (end[1] - start[1]) * index / samples,
                offsets,
            )
            for index in range(samples + 1)
        )

    def swept_pulse_is_clear(
        self,
        start: tuple[float, float],
        *,
        heading: float,
        travel_m: float,
        clearance_m: float,
        lookahead_m: float = ROOM_LOOP_DYNAMIC_LOOKAHEAD_M,
    ) -> bool:
        """Check only the next bounded command envelope on a preplanned route.

        The complete loop is validated before motion begins. During execution,
        a freshly rebuilt HeightCost map may gain unknown/self-artifact cells
        far down the same segment. Requiring the entire remaining segment to
        stay byte-identical makes a safe loop fail nondeterministically. This
        method still fails closed over the command's swept footprint plus a
        small forward margin; later space is checked again from later, fresher
        observations before another pulse can start.
        """

        bounded_travel = max(0.0, float(travel_m)) + max(
            0.0, float(lookahead_m)
        )
        end = (
            start[0] + math.cos(heading) * bounded_travel,
            start[1] + math.sin(heading) * bounded_travel,
        )
        return self.segment_is_clear(start, end, clearance_m=clearance_m)

def _plan_safe_costmap_loop(
    payload: Any,
    *,
    start_x: float,
    start_y: float,
    start_yaw: float,
    verified_start_clearance_m: float = 0.0,
) -> dict[str, Any]:
    """Select a large square, preferring the route with most mapped clearance."""

    start = (start_x, start_y)
    grid = _OnlineCostmap.from_payload(payload).with_verified_free_disk(
        start,
        verified_start_clearance_m,
    )
    relative_headings = (0, 45, -45, 90, -90, 135, -135, 180)
    side_steps = round(
        (ROOM_LOOP_MAX_SIDE_M - ROOM_LOOP_MIN_SIDE_M) / 0.25
    )
    side_lengths = tuple(
        ROOM_LOOP_MAX_SIDE_M - 0.25 * index
        for index in range(side_steps + 1)
    )
    best_candidate: dict[str, Any] | None = None
    for side_length in side_lengths:
        candidates: list[dict[str, Any]] = []
        for relative_heading_deg in relative_headings:
            heading = start_yaw + math.radians(relative_heading_deg)
            forward = (math.cos(heading), math.sin(heading))
            left = (-math.sin(heading), math.cos(heading))
            for turn_sign in (1.0, -1.0):
                lateral = (turn_sign * left[0], turn_sign * left[1])
                waypoints = [
                    (
                        start_x + side_length * forward[0],
                        start_y + side_length * forward[1],
                    ),
                    (
                        start_x + side_length * (forward[0] + lateral[0]),
                        start_y + side_length * (forward[1] + lateral[1]),
                    ),
                    (
                        start_x + side_length * lateral[0],
                        start_y + side_length * lateral[1],
                    ),
                    start,
                ]
                points = [start, *waypoints]
                if all(
                    grid.segment_is_clear(
                        segment_start,
                        segment_end,
                        clearance_m=ROOM_LOOP_ROUTE_CLEARANCE_M,
                    )
                    for segment_start, segment_end in zip(points, points[1:])
                ):
                    robust_clearance = ROOM_LOOP_ROUTE_CLEARANCE_M
                    for candidate_clearance in ROOM_LOOP_ROBUST_CLEARANCES_M:
                        if not all(
                            grid.segment_is_clear(
                                segment_start,
                                segment_end,
                                clearance_m=candidate_clearance,
                            )
                            for segment_start, segment_end in zip(
                                points, points[1:]
                            )
                        ):
                            break
                        robust_clearance = candidate_clearance
                    candidates.append(
                        {
                            "waypoints": waypoints,
                            "side_length_m": side_length,
                            "planned_perimeter_m": side_length * 4.0,
                            # The enforced execution clearance is unchanged.
                            "route_clearance_m": ROOM_LOOP_ROUTE_CLEARANCE_M,
                            "selection_clearance_m": robust_clearance,
                            "relative_heading_deg": relative_heading_deg,
                            "turn_direction": (
                                "left" if turn_sign > 0 else "right"
                            ),
                        }
                    )
        if candidates:
            # Candidate enumeration is deterministic, so max() also preserves
            # the existing heading preference when robustness is tied. Prefer
            # mapped clearance across different side lengths; a smaller loop
            # with more margin is safer than the largest marginally valid one.
            side_best = max(
                candidates,
                key=lambda candidate: float(candidate["selection_clearance_m"]),
            )
            if best_candidate is None or (
                float(side_best["selection_clearance_m"]),
                float(side_best["side_length_m"]),
            ) > (
                float(best_candidate["selection_clearance_m"]),
                float(best_candidate["side_length_m"]),
            ):
                best_candidate = side_best
            if float(side_best["selection_clearance_m"]) >= max(
                ROOM_LOOP_ROBUST_CLEARANCES_M
            ):
                return side_best
    if best_candidate is not None:
        return best_candidate
    raise ValueError(
        "在线地图中找不到边长至少 "
        f"{ROOM_LOOP_MIN_SIDE_M:.2f} m、净空 "
        f"{ROOM_LOOP_ROUTE_CLEARANCE_M:.2f} m 的闭合路线"
    )

def _room_loop_plan_signature(plan: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(plan["side_length_m"]), 3),
        int(plan["relative_heading_deg"]),
        str(plan["turn_direction"]),
        round(float(plan.get("selection_clearance_m", 0.0)), 3),
    )

def _wait_for_stable_room_loop_plan(
    provider: Callable[[], Any],
    initial_payload: Any,
    *,
    start_x: float,
    start_y: float,
    start_yaw: float,
    verified_start_clearance_m: float,
    cancelled: Callable[[], bool],
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Require independent live-map revisions to agree before first motion."""

    payload = initial_payload
    revision = payload.get("revision") if isinstance(payload, dict) else None
    if isinstance(revision, bool) or not isinstance(revision, int):
        # Unit-test and legacy providers without revision metadata retain the
        # original fail-closed planner behavior. Production CostmapMonitor
        # always supplies a positive, monotonically increasing revision.
        return _plan_safe_costmap_loop(
            payload,
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            verified_start_clearance_m=verified_start_clearance_m,
        )

    deadline = clock() + ROOM_LOOP_STABILITY_TIMEOUT_SECONDS
    last_revision: int | None = None
    stable_signature: tuple[Any, ...] | None = None
    stable_count = 0
    last_error: ValueError | None = None
    while clock() < deadline:
        if cancelled():
            raise ValueError("地图稳定等待期间任务已取消")
        try:
            current_revision = payload.get("revision")
            if (
                isinstance(current_revision, bool)
                or not isinstance(current_revision, int)
                or current_revision < 1
            ):
                raise ValueError("实时 costmap revision 无效")
            if current_revision != last_revision:
                plan = _plan_safe_costmap_loop(
                    payload,
                    start_x=start_x,
                    start_y=start_y,
                    start_yaw=start_yaw,
                    verified_start_clearance_m=verified_start_clearance_m,
                )
                signature = _room_loop_plan_signature(plan)
                stable_count = stable_count + 1 if signature == stable_signature else 1
                stable_signature = signature
                last_revision = current_revision
                last_error = None
                if stable_count >= ROOM_LOOP_STABLE_REVISIONS:
                    return {
                        **plan,
                        "stable_map_revisions": stable_count,
                        "stable_map_revision": current_revision,
                    }
        except ValueError as error:
            last_error = error
            stable_signature = None
            stable_count = 0
        sleeper(ROOM_LOOP_STABILITY_POLL_SECONDS)
        payload = provider()
    detail = str(last_error) if last_error is not None else "路线决策持续变化"
    raise ValueError(
        f"实时地图未在 {ROOM_LOOP_STABILITY_TIMEOUT_SECONDS:.1f}s 内形成 "
        f"{ROOM_LOOP_STABLE_REVISIONS} 个一致 revision：{detail}"
    )

class G1MotionSkills:
    """Existing bounded G1 motion and measured verification; no model loop."""

    @staticmethod
    def _number(arguments: dict[str, Any], name: str, lower: float, upper: float) -> float:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是数字")
        number = float(value)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{name} 必须在 {lower} 到 {upper} 之间")
        return number

    def _motion_guard(self) -> str | None:
        if not self._observed_this_turn:
            return "本轮必须先调用 observe_environment"
        snapshot = self.monitor.agent_snapshot()
        recovery = snapshot.get("safety_recovery") or {}
        if recovery.get("safety_hold"):
            return (
                "本地安全恢复锁定了运动控制："
                f"state={recovery.get('state')}, reason={recovery.get('reason')}；"
                "只允许观察或 stop_robot"
            )
        if self.backend == "isaac-g1":
            if snapshot.get("pose") is None or not snapshot.get("camera_available"):
                return "Isaac 新鲜位姿或第一人称相机不可用，只允许 stop_robot"
            return None
        metrics = snapshot.get("metrics") or {}
        risk = metrics.get("risk", "unknown")
        if risk in {"unknown", "critical"}:
            return f"当前风险为 {risk}，只允许 stop_robot"
        return None

    def _move_robot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        guard = self._motion_guard()
        if guard:
            return {"ok": False, "error": guard}
        try:
            x_limit = 0.30 if self.backend == "isaac-g1" else 0.18
            x = self._number(arguments, "x", -x_limit, x_limit)
            y = self._number(arguments, "y", -0.18, 0.18)
            yaw = self._number(arguments, "yaw", -0.45, 0.45)
            duration = self._number(arguments, "duration", 0.1, 2.5)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        if self.backend == "isaac-g1" and (
            abs(x) > 0.30
            or abs(y) > 0.18
            or abs(yaw) > 0.30
            or duration > 1.0
        ):
            return {
                "ok": False,
                "error": (
                    "Isaac 手动脉冲限值为 |x|<=0.30、|y|<=0.18、"
                    "|yaw|<=0.30、duration<=1.0"
                ),
            }

        risk = (self.monitor.agent_snapshot().get("metrics") or {}).get("risk")
        if risk == "warning" and (
            abs(x) > 0.10 or abs(y) > 0.10 or abs(yaw) > 0.25 or duration > 1.0
        ):
            return {
                "ok": False,
                "error": "warning 风险下限值为 |x/y|<=0.10、|yaw|<=0.25、duration<=1.0",
            }
        observation_before = self._observation()
        result = self._command_runner(
            [
                str(self.project_root / "scripts/dimos.sh"),
                "move",
                "--x",
                str(x),
                "--y",
                str(y),
                "--yaw",
                str(yaw),
                "--duration",
                str(duration),
            ],
            duration + 6.0,
        )
        time.sleep(0.25)
        immediate_observation = self._observation()
        observation_after = immediate_observation
        pure_isaac_yaw = (
            self.backend == "isaac-g1"
            and math.hypot(x, y) <= 1e-6
            and abs(yaw) > 1e-6
        )
        if pure_isaac_yaw:
            settle_samples = max(
                1,
                round(
                    ISAAC_PURE_YAW_SETTLE_SECONDS
                    / ISAAC_PURE_YAW_SETTLE_INTERVAL_SECONDS
                ),
            )
            for _ in range(settle_samples):
                time.sleep(ISAAC_PURE_YAW_SETTLE_INTERVAL_SECONDS)
                observation_after = self._observation()
        before_pose = self._xy_yaw_from_observation(observation_before)
        immediate_pose = self._xy_yaw_from_observation(immediate_observation)
        after_pose = self._xy_yaw_from_observation(observation_after)
        if before_pose is None or after_pose is None:
            motion_evidence = {
                "available": False,
                "planar_displacement_m": None,
                "yaw_change_rad": None,
                "motion_observed": False,
            }
        else:
            planar_displacement = math.hypot(
                after_pose[0] - before_pose[0],
                after_pose[1] - before_pose[1],
            )
            yaw_change = self._wrap_angle(after_pose[2] - before_pose[2])
            translation_requested = math.hypot(x, y) > 1e-6
            rotation_requested = abs(yaw) > 1e-6
            motion_evidence = {
                "available": True,
                "planar_displacement_m": round(planar_displacement, 6),
                "yaw_change_rad": round(yaw_change, 6),
                "motion_observed": (
                    (translation_requested and planar_displacement >= 0.01)
                    or (rotation_requested and abs(yaw_change) >= 0.02)
                ),
            }
        if pure_isaac_yaw:
            immediate_yaw_change = (
                self._wrap_angle(immediate_pose[2] - before_pose[2])
                if immediate_pose is not None and before_pose is not None
                else None
            )
            motion_evidence.update(
                {
                    "immediate_yaw_change_rad": (
                        round(immediate_yaw_change, 6)
                        if immediate_yaw_change is not None
                        else None
                    ),
                    "settle_verification_s": ISAAC_PURE_YAW_SETTLE_SECONDS,
                }
            )
        result["motion_evidence"] = motion_evidence
        result["observation_after"] = observation_after
        self._observed_this_turn = True
        return result

    def _wait_for_stationary(
        self,
        *,
        timeout_seconds: float = 6.0,
    ) -> tuple[dict[str, Any], int, bool]:
        """Wait for two consecutive low-speed observations after a zero command."""

        consecutive = 0
        observation = self._observation()
        sample_count = max(2, math.ceil(timeout_seconds / 0.25))
        for sample_index in range(sample_count):
            observation = self._observation()
            motion = observation.get("motion") or {}
            planar_speed = motion.get("planar_speed")
            yaw_rate = motion.get("yaw_rate")
            stationary = (
                isinstance(planar_speed, (int, float))
                and not isinstance(planar_speed, bool)
                and math.isfinite(float(planar_speed))
                and abs(float(planar_speed)) <= 0.025
                and isinstance(yaw_rate, (int, float))
                and not isinstance(yaw_rate, bool)
                and math.isfinite(float(yaw_rate))
                and abs(float(yaw_rate)) <= 0.08
            )
            consecutive = consecutive + 1 if stationary else 0
            if consecutive >= 2:
                return observation, consecutive, True
            if self._cancel.is_set() or sample_index + 1 >= sample_count:
                break
            time.sleep(0.25)
        return observation, consecutive, False

    @classmethod
    def _relative_distance_progress(
        cls,
        start_pose: tuple[float, float, float],
        current_pose: tuple[float, float, float],
        direction: float,
    ) -> tuple[float, float, float]:
        start_x, start_y, start_yaw = start_pose
        current_x, current_y, current_yaw = current_pose
        dx = current_x - start_x
        dy = current_y - start_y
        forward_x = math.cos(start_yaw)
        forward_y = math.sin(start_yaw)
        along_track = dx * forward_x + dy * forward_y
        cross_track = abs(-dx * forward_y + dy * forward_x)
        yaw_error = abs(cls._wrap_angle(current_yaw - start_yaw))
        return direction * along_track, cross_track, yaw_error

    def _move_distance(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one explicit relative-distance request without model round trips."""

        if self.backend != "isaac-g1":
            return {"ok": False, "error": "move_distance 仅支持 Isaac G1 后端"}
        if set(arguments) != {"distance_m"}:
            return {"ok": False, "error": "move_distance 只接受 distance_m"}
        try:
            requested_distance = self._number(arguments, "distance_m", -3.0, 3.0)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if abs(requested_distance) < 0.10:
            return {"ok": False, "error": "distance_m 的绝对值必须至少为 0.10 米"}
        guard = self._motion_guard()
        if guard:
            return {"ok": False, "error": guard}

        started_at = time.monotonic()
        initial_observation = self._observation()
        start_pose = self._xy_yaw_from_observation(initial_observation)
        if start_pose is None:
            return {"ok": False, "error": "当前观测缺少有效位姿，无法执行距离闭环"}

        direction = 1.0 if requested_distance > 0.0 else -1.0
        target = abs(requested_distance)
        tolerance = max(0.08, min(0.15, target * 0.05))
        braking_allowance = max(0.10, min(0.22, target * 0.10))
        max_pulses = 18
        max_cross_track = 0.30
        max_yaw_error = math.radians(25.0)
        pulses: list[dict[str, Any]] = []
        stalled_pulses = 0
        last_progress = 0.0

        def progress_from(
            observation: dict[str, Any],
        ) -> tuple[float, float, float] | None:
            pose = self._xy_yaw_from_observation(observation)
            if pose is None:
                return None
            return self._relative_distance_progress(start_pose, pose, direction)

        def stop_and_measure() -> tuple[
            dict[str, Any],
            tuple[float, float, float] | None,
            int,
            bool,
            dict[str, Any],
        ]:
            stop = self._stop_robot({})
            final_observation, stationary_samples, stationary = self._wait_for_stationary()
            return (
                final_observation,
                progress_from(final_observation),
                stationary_samples,
                stationary,
                stop,
            )

        def failure(task_status: str, error: str) -> dict[str, Any]:
            observation, measured, stationary_samples, stationary, stop = stop_and_measure()
            progress, cross_track, yaw_error = measured or (0.0, 0.0, 0.0)
            signed_actual = direction * progress
            return {
                "ok": False,
                "completed": False,
                "task_status": task_status,
                "error": error,
                "target_distance_m": round(requested_distance, 3),
                "actual_distance_m": round(signed_actual, 3),
                "distance_error_m": round(requested_distance - signed_actual, 3),
                "cross_track_error_m": round(cross_track, 3),
                "final_yaw_error_deg": round(math.degrees(yaw_error), 2),
                "pulse_count": len(pulses),
                "pulses": pulses,
                "stationary_samples": stationary_samples,
                "stationary_confirmed": stationary,
                "elapsed_s": round(time.monotonic() - started_at, 3),
                "stop": stop,
                "observation_after": observation,
            }

        reached_braking_zone = False
        while len(pulses) < max_pulses:
            if self._cancel.is_set():
                return failure("distance_cancelled", "Agent turn 已取消，机器人已停车")
            if time.monotonic() - started_at > 50.0:
                return failure("distance_timeout", "距离闭环超过 50 秒，机器人已停车")

            observation = self._observation()
            measured = progress_from(observation)
            if measured is None:
                return failure("distance_observation_failed", "距离闭环丢失有效位姿")
            progress, cross_track, yaw_error = measured
            if not observation.get("camera_available"):
                return failure("distance_observation_failed", "第一人称相机失效，机器人已停车")
            if progress < -0.05:
                return failure("distance_wrong_direction", "实测运动方向与请求相反")
            if cross_track > max_cross_track:
                return failure(
                    "distance_drifted",
                    f"横向漂移达到 {cross_track:.2f} 米，超过 0.30 米限值",
                )
            if yaw_error > max_yaw_error:
                return failure(
                    "distance_drifted",
                    f"航向偏差达到 {math.degrees(yaw_error):.1f} 度，超过 25 度限值",
                )

            remaining = target - progress
            if remaining <= braking_allowance:
                reached_braking_zone = True
                break
            duration = min(
                1.0,
                max(0.25, (remaining - braking_allowance) / 0.30),
            )
            result = self._move_robot(
                {
                    "x": direction * 0.30,
                    "y": 0.0,
                    "yaw": 0.0,
                    "duration": duration,
                }
            )
            if not result.get("ok"):
                return failure(
                    "distance_motion_failed",
                    f"距离子脉冲执行失败：{result.get('error', 'unknown error')}",
                )
            after = result.get("observation_after") or self._observation()
            after_measured = progress_from(after)
            if after_measured is None:
                return failure("distance_observation_failed", "子脉冲后丢失有效位姿")
            after_progress, after_cross_track, after_yaw_error = after_measured
            pulses.append(
                {
                    "x": round(direction * 0.30, 3),
                    "duration": round(duration, 3),
                    "progress_m": round(after_progress, 3),
                    "cross_track_m": round(after_cross_track, 3),
                    "yaw_error_deg": round(math.degrees(after_yaw_error), 2),
                    "immediate_motion_evidence": result.get("motion_evidence"),
                }
            )
            if after_progress - last_progress < 0.01:
                stalled_pulses += 1
                if stalled_pulses >= 3:
                    return failure(
                        "distance_stalled",
                        "连续三个距离子脉冲没有产生有效前进，机器人已停车",
                    )
            else:
                stalled_pulses = 0
            last_progress = max(last_progress, after_progress)

        if not reached_braking_zone:
            return failure("distance_pulse_limit", "达到距离子脉冲上限，机器人已停车")

        final_observation, measured, stationary_samples, stationary, stop = stop_and_measure()
        if measured is None:
            return failure("distance_observation_failed", "停车后丢失有效位姿")

        # At most three small settled corrections handle the learned gait's
        # variable coast distance without returning to Qwen between pulses.
        for _ in range(3):
            progress, cross_track, yaw_error = measured
            error = target - progress
            if abs(error) <= tolerance:
                break
            if error < -tolerance:
                return failure(
                    "distance_overshot",
                    f"停车后超出目标 {-error:.2f} 米，未宣称距离完成",
                )
            if not stationary:
                return failure("distance_stop_unverified", "零速后未确认机器人物理静止")
            if len(pulses) >= max_pulses:
                return failure("distance_pulse_limit", "修正前达到距离子脉冲上限")
            correction_duration = min(0.8, max(0.20, (error - 0.03) / 0.30))
            correction = self._move_robot(
                {
                    "x": direction * 0.30,
                    "y": 0.0,
                    "yaw": 0.0,
                    "duration": correction_duration,
                }
            )
            if not correction.get("ok"):
                return failure(
                    "distance_motion_failed",
                    f"距离修正脉冲失败：{correction.get('error', 'unknown error')}",
                )
            correction_after = correction.get("observation_after") or self._observation()
            correction_measured = progress_from(correction_after)
            if correction_measured is None:
                return failure("distance_observation_failed", "修正脉冲后丢失有效位姿")
            pulses.append(
                {
                    "x": round(direction * 0.30, 3),
                    "duration": round(correction_duration, 3),
                    "progress_m": round(correction_measured[0], 3),
                    "cross_track_m": round(correction_measured[1], 3),
                    "yaw_error_deg": round(math.degrees(correction_measured[2]), 2),
                    "correction": True,
                    "immediate_motion_evidence": correction.get("motion_evidence"),
                }
            )
            final_observation, measured, stationary_samples, stationary, stop = stop_and_measure()
            if measured is None:
                return failure("distance_observation_failed", "修正停车后丢失有效位姿")

        progress, cross_track, yaw_error = measured
        signed_actual = direction * progress
        distance_error = requested_distance - signed_actual
        if not stationary:
            return failure("distance_stop_unverified", "最终零速后未确认机器人物理静止")
        if abs(distance_error) > tolerance:
            return failure(
                "distance_tolerance_failed",
                f"最终距离误差 {abs(distance_error):.2f} 米超过 {tolerance:.2f} 米容差",
            )
        if cross_track > max_cross_track or yaw_error > max_yaw_error:
            return failure("distance_drifted", "最终横向或航向偏差超过闭环限值")
        return {
            "ok": True,
            "completed": True,
            "task_status": "distance_verified",
            "target_distance_m": round(requested_distance, 3),
            "actual_distance_m": round(signed_actual, 3),
            "distance_error_m": round(distance_error, 3),
            "distance_tolerance_m": round(tolerance, 3),
            "cross_track_error_m": round(cross_track, 3),
            "final_yaw_error_deg": round(math.degrees(yaw_error), 2),
            "pulse_count": len(pulses),
            "pulses": pulses,
            "stationary_samples": stationary_samples,
            "stationary_confirmed": True,
            "elapsed_s": round(time.monotonic() - started_at, 3),
            "stop": stop,
            "observation_after": final_observation,
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _yaw_from_observation(observation: dict[str, Any]) -> float | None:
        yaw = (observation.get("pose") or {}).get("yaw")
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
            return None
        yaw = float(yaw)
        return yaw if math.isfinite(yaw) else None

    def _turn_around_failure(
        self,
        error: str,
        progress: float,
        pulses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stop = self._stop_robot({})
        return {
            "ok": False,
            "completed": False,
            "task_status": "turn_failed",
            "error": error,
            "turned_degrees": round(math.degrees(progress), 1),
            "pulses": pulses,
            "stop": stop,
            "observation_after": self._observation(),
        }

    def _turn_around(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"ok": False, "error": "turn_around 不接受参数"}
        guard = self._motion_guard()
        if guard:
            return {"ok": False, "error": guard}

        if self.backend == "isaac-g1":
            transport = self._mcp_call("turn_around", timeout=75.0)
            structured = _structured_mcp_payload(transport)
            raw_result = transport.get("result")
            result_text = ""
            if isinstance(structured, dict):
                native = structured.get("native_skill_result")
                if isinstance(native, str):
                    result_text = native
            if not result_text:
                result_text = (
                    raw_result
                    if isinstance(raw_result, str)
                    else transport.get("output")
                    if isinstance(transport.get("output"), str)
                    else ""
                )
            verified_line = next(
                (
                    line.strip()
                    for line in result_text.splitlines()
                    if line.strip().startswith("turn_verified:")
                ),
                "",
            )
            rotation_match = re.search(
                r"\brotation=(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)deg\b",
                verified_line,
            )
            measured_degrees = (
                float(rotation_match.group(1)) if rotation_match else float("nan")
            )
            target_degrees = (
                float(rotation_match.group(2)) if rotation_match else float("nan")
            )
            verified = bool(
                transport.get("ok")
                and (
                    structured is None
                    or (
                        structured.get("tool_ok") is True
                        and structured.get("completed") is True
                        and structured.get("task_status") == "turn_verified"
                        and structured.get("turn_verified") is True
                    )
                )
                and "stationary_confirmed=true" in verified_line
                and math.isfinite(measured_degrees)
                and math.isfinite(target_degrees)
                and abs(target_degrees - 180.0) <= 0.1
                and abs(measured_degrees - target_degrees) <= 12.0
            )
            self._observed_this_turn = True
            if not verified:
                stop = self._stop_robot({})
                return {
                    "ok": False,
                    "completed": False,
                    "task_status": "turn_failed",
                    "error": (
                        "Isaac turn_around 未返回 turn_verified，已停车"
                    ),
                    "mcp_result": transport,
                    "stop": stop,
                    "observation_after": self._observation(),
                }
            return {
                "ok": True,
                "completed": True,
                "task_status": "turn_verified",
                "target_degrees": target_degrees,
                "turned_degrees": measured_degrees,
                "direction": "left",
                "mcp_result": transport,
                "observation_after": self._observation(),
            }

        initial = self._observation()
        previous_yaw = self._yaw_from_observation(initial)
        if previous_yaw is None:
            return {"ok": False, "error": "当前观测缺少有效 yaw，无法闭环转身"}

        direction = 1.0
        target = math.pi
        tolerance = math.radians(10.0)
        progress = 0.0
        stalled_pulses = 0
        pulses: list[dict[str, Any]] = []

        for _ in range(20):
            if self._cancel.is_set():
                return self._turn_around_failure(
                    "agent turn 已取消，转身已停车",
                    progress,
                    pulses,
                )

            snapshot = self._observation()
            risk = (snapshot.get("metrics") or {}).get("risk", "unknown")
            if risk in {"unknown", "critical"}:
                return self._turn_around_failure(
                    f"转身过程中风险变为 {risk}，已停车",
                    progress,
                    pulses,
                )

            remaining = target - progress
            if remaining <= tolerance:
                break
            if risk == "warning":
                yaw_speed = 0.25
                max_duration = 1.0
            else:
                yaw_speed = 0.45
                max_duration = 2.5
            duration = min(max_duration, max(0.4, remaining / yaw_speed))
            result = self._move_robot(
                {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw": direction * yaw_speed,
                    "duration": duration,
                }
            )
            if not result.get("ok"):
                return self._turn_around_failure(
                    f"转身脉冲执行失败：{result.get('error', 'unknown error')}",
                    progress,
                    pulses,
                )

            observation_after = result.get("observation_after") or self._observation()
            current_yaw = self._yaw_from_observation(observation_after)
            if current_yaw is None:
                return self._turn_around_failure(
                    "转身后观测缺少有效 yaw，已停车",
                    progress,
                    pulses,
                )
            observed_delta = direction * self._wrap_angle(current_yaw - previous_yaw)
            pulses.append(
                {
                    "risk": risk,
                    "yaw": direction * yaw_speed,
                    "duration": round(duration, 3),
                    "observed_degrees": round(math.degrees(observed_delta), 1),
                }
            )
            previous_yaw = current_yaw
            if observed_delta < -math.radians(5.0):
                return self._turn_around_failure(
                    "实测航向与命令方向相反，已停车",
                    progress,
                    pulses,
                )
            if observed_delta < math.radians(1.0):
                stalled_pulses += 1
                if stalled_pulses >= 3:
                    return self._turn_around_failure(
                        "连续三个转身脉冲没有产生有效航向变化，已停车",
                        progress,
                        pulses,
                    )
            else:
                stalled_pulses = 0
                progress += observed_delta

        completed = progress >= target - tolerance
        if not completed:
            return self._turn_around_failure(
                "达到转身脉冲上限但尚未完成 180 度，已停车",
                progress,
                pulses,
            )
        return {
            "ok": True,
            "completed": True,
            "task_status": "turn_verified",
            "target_degrees": 180.0,
            "turned_degrees": round(math.degrees(progress), 1),
            "direction": "left" if direction > 0 else "right",
            "pulses": pulses,
            "observation_after": self._observation(),
        }

    @staticmethod
    def _xy_yaw_from_observation(
        observation: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        pose = observation.get("pose")
        if not isinstance(pose, dict):
            return None
        try:
            values = (float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    def _room_loop_failure(
        self,
        *,
        task_status: str,
        error: str,
        path_length_m: float,
        pulse_count: int,
        reached_waypoints: int,
        route_plan: dict[str, Any] | None = None,
        motion_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stop = self._stop_robot({})
        observation_after = self._observation()
        metrics = observation_after.get("metrics") or {}
        recovery = observation_after.get("safety_recovery") or {}
        return {
            "ok": False,
            "completed": False,
            "task_status": task_status,
            "error": error,
            "path_length_m": round(path_length_m, 3),
            "pulse_count": pulse_count,
            "reached_waypoints": reached_waypoints,
            "route_plan": route_plan,
            "motion_audit": motion_audit,
            "stop": stop,
            "failure_context": {
                "physical_risk": metrics.get("risk"),
                "physical_risk_source": metrics.get("risk_source"),
                "nearest_obstacle_distance_m": metrics.get(
                    "nearest_obstacle_distance"
                ),
                "recovery_state": recovery.get("state"),
                "recovery_reason": recovery.get("reason"),
            },
            "observation_after": observation_after,
        }

    def _current_costmap_grid(
        self,
        *,
        wait_timeout_seconds: float = 2.5,
    ) -> _OnlineCostmap:
        provider = getattr(self.monitor, "costmap_payload", None)
        if not callable(provider):
            raise ValueError("Agent 运行时没有实时 costmap provider")
        deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
        last_error = "实时 costmap 不可用"
        while True:
            try:
                return _OnlineCostmap.from_payload(provider())
            except ValueError as error:
                last_error = str(error)
            if self._cancel.is_set():
                raise ValueError("等待实时 costmap 时任务被取消")
            if time.monotonic() >= deadline:
                raise ValueError(
                    f"连续 {wait_timeout_seconds:.1f} 秒没有新鲜 costmap：{last_error}"
                )
            # Every manual pulse has already published zero before this wait;
            # retaining that zero command is the bounded safe response to a
            # short mapper/LCM publication gap.
            time.sleep(0.10)

    def _begin_room_loop_audit(self) -> Any | None:
        begin = getattr(self.monitor, "begin_motion_audit", None)
        if not callable(begin):
            return None
        try:
            return begin()
        except Exception:  # noqa: BLE001 - unavailable evidence fails verification
            return None

    def _read_room_loop_audit(
        self,
        token: Any | None,
        *,
        finish: bool = False,
    ) -> dict[str, Any]:
        if token is None:
            return {"available": False, "reason": "motion_audit_unavailable"}
        method_name = "finish_motion_audit" if finish else "motion_audit"
        method = getattr(self.monitor, method_name, None)
        if not callable(method):
            if not finish:
                return {"available": True, "samples": 0}
            return {"available": False, "reason": "motion_audit_unavailable"}
        try:
            result = method(token)
        except Exception as error:  # noqa: BLE001 - evidence is untrusted on failure
            return {
                "available": False,
                "reason": f"motion_audit_error:{type(error).__name__}",
            }
        return result if isinstance(result, dict) else {"available": False}

    @staticmethod
    def _room_loop_clearance_violation(
        motion_audit: dict[str, Any],
    ) -> str | None:
        consecutive = motion_audit.get("max_consecutive_critical_raw_samples")
        immediate = motion_audit.get("immediate_critical_samples")
        if (
            isinstance(consecutive, bool)
            or not isinstance(consecutive, int)
            or consecutive < 0
            or isinstance(immediate, bool)
            or not isinstance(immediate, int)
            or immediate < 0
        ):
            return "全程物理净空审计缺少连续帧或立即急停证据"
        surface = motion_audit.get("minimum_robot_surface_clearance_m")
        surface_detail = (
            f"；本轮最小机器人表面净空 {float(surface):.3f} m"
            if (
                isinstance(surface, (int, float))
                and not isinstance(surface, bool)
                and math.isfinite(float(surface))
            )
            else ""
        )
        if immediate > 0:
            return (
                "平移期间出现机器人表面净空不大于 0.10 m 的立即急停帧"
                + surface_detail
            )
        if consecutive >= 2:
            return (
                "连续两个独立 lidar 帧的机器人表面净空不大于 0.15 m"
                + surface_detail
            )
        return None

    def _walk_room_loop(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Plan, walk, and verify one bounded loop on the live online map."""

        if arguments:
            return {
                "ok": False,
                "completed": False,
                "task_status": "invalid_input",
                "error": "walk_room_loop 不接受参数",
            }
        guard = self._motion_guard()
        if guard:
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_risk_blocked",
                "error": guard,
            }

        initial_observation = self._observation()
        initial_pose = self._xy_yaw_from_observation(initial_observation)
        if initial_pose is None:
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_observation_failed",
                "error": "当前观测缺少有效 x/y/yaw，无法闭环行走",
            }
        initial_metrics = initial_observation.get("metrics") or {}
        if initial_metrics.get("lidar_sequence") is None:
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_observation_failed",
                "error": "当前帧物理雷达不可用，拒绝开始闭环行走",
            }

        start_x, start_y, start_yaw = initial_pose
        provider = getattr(self.monitor, "costmap_payload", None)
        if not callable(provider):
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_route_unavailable",
                "error": "Agent 运行时没有实时 costmap provider，拒绝盲走固定路线",
            }
        try:
            initial_map_payload = provider()
            _OnlineCostmap.from_payload(initial_map_payload)
        except ValueError as error:
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_route_unavailable",
                "error": str(error),
            }

        audit_token = self._begin_room_loop_audit()
        if audit_token is None:
            return {
                "ok": False,
                "completed": False,
                "task_status": "room_loop_observation_failed",
                "error": "全程物理净空审计不可用，拒绝开始闭环行走",
            }

        def verified_clearance(metrics: dict[str, Any]) -> float:
            nearest = metrics.get("nearest_obstacle_distance")
            if (
                isinstance(nearest, (int, float))
                and not isinstance(nearest, bool)
                and math.isfinite(float(nearest))
            ):
                return min(1.0, max(0.0, float(nearest) - 0.15))
            return 0.0

        route_plan: dict[str, Any] | None = None
        scan_pulses = 0
        try:
            route_plan = _wait_for_stable_room_loop_plan(
                provider,
                initial_map_payload,
                start_x=start_x,
                start_y=start_y,
                start_yaw=start_yaw,
                verified_start_clearance_m=verified_clearance(initial_metrics),
                cancelled=self._cancel.is_set,
            )
        except ValueError:
            # A cold online map can leave the robot footprint unknown even
            # though current 360-degree lidar proves the pose safe.  Acquire
            # more sensor viewpoints by two bounded 180-degree turns; this
            # sweeps no new circular-footprint area and never reads a saved or
            # simulator-authored map.
            for scan_index in range(2):
                scan = self._turn_around({})
                pulses = scan.get("pulses")
                if isinstance(pulses, list):
                    scan_pulses += len(pulses)
                audit = self._read_room_loop_audit(audit_token)
                violation = self._room_loop_clearance_violation(audit)
                if not scan.get("completed") or violation is not None:
                    return self._room_loop_failure(
                        task_status=(
                            "room_loop_clearance_violation"
                            if violation is not None
                            else "room_loop_map_acquisition_failed"
                        ),
                        error=(
                            violation
                            or f"第 {scan_index + 1} 次地图采集转身失败："
                            f"{scan.get('error', 'unknown error')}"
                        ),
                        path_length_m=0.0,
                        pulse_count=scan_pulses,
                        reached_waypoints=0,
                        route_plan=None,
                        motion_audit=self._read_room_loop_audit(
                            audit_token,
                            finish=True,
                        ),
                    )
            scanned_observation = self._observation()
            scanned_pose = self._xy_yaw_from_observation(scanned_observation)
            if scanned_pose is None:
                return self._room_loop_failure(
                    task_status="room_loop_observation_failed",
                    error="地图采集转身后缺少有效里程计",
                    path_length_m=0.0,
                    pulse_count=scan_pulses,
                    reached_waypoints=0,
                    route_plan=None,
                    motion_audit=self._read_room_loop_audit(
                        audit_token,
                        finish=True,
                    ),
                )
            start_x, start_y, start_yaw = scanned_pose
            scanned_metrics = scanned_observation.get("metrics") or {}
            try:
                scanned_map_payload = provider()
                route_plan = _wait_for_stable_room_loop_plan(
                    provider,
                    scanned_map_payload,
                    start_x=start_x,
                    start_y=start_y,
                    start_yaw=start_yaw,
                    verified_start_clearance_m=verified_clearance(
                        scanned_metrics
                    ),
                    cancelled=self._cancel.is_set,
                )
            except ValueError as error:
                return self._room_loop_failure(
                    task_status="room_loop_route_unavailable",
                    error=f"360 度在线建图后仍无安全闭环：{error}",
                    path_length_m=0.0,
                    pulse_count=scan_pulses,
                    reached_waypoints=0,
                    route_plan=None,
                    motion_audit=self._read_room_loop_audit(
                        audit_token,
                        finish=True,
                    ),
                )
        assert route_plan is not None
        waypoints = list(route_plan["waypoints"])
        side_length = float(route_plan["side_length_m"])
        planned_perimeter = float(route_plan["planned_perimeter_m"])
        audit_finished = False

        path_length = 0.0
        pulse_count = scan_pulses
        reached_waypoints = 0
        last_pose = (start_x, start_y, start_yaw)

        def read_audit(*, finish: bool = False) -> dict[str, Any]:
            nonlocal audit_finished
            if finish and audit_finished:
                return {"available": False, "reason": "motion_audit_already_finished"}
            result = self._read_room_loop_audit(audit_token, finish=finish)
            if finish:
                audit_finished = True
            return result

        def fail(status: str, message: str) -> dict[str, Any]:
            return self._room_loop_failure(
                task_status=status,
                error=message,
                path_length_m=path_length,
                pulse_count=pulse_count,
                reached_waypoints=reached_waypoints,
                route_plan=route_plan,
                motion_audit=read_audit(finish=True),
            )

        def confirm_swept_lidar_after_stop(
            required_clearance: float,
            initial_sequence: Any,
        ) -> tuple[bool, str]:
            """Stop, then debounce a marginal lidar frame without lowering it."""

            stop_barrier = self._stop_robot({})
            if stop_barrier.get("ok") is not True:
                return False, "低净空后的零速屏障失败"
            last_sequence = (
                initial_sequence
                if isinstance(initial_sequence, int)
                and not isinstance(initial_sequence, bool)
                else None
            )
            clear_frames = 0
            blocked_frames = 0
            last_detail = "没有获得两个不同的新鲜雷达帧"
            for _ in range(30):
                if self._cancel.is_set():
                    return False, "净空复核期间任务已取消"
                observation = self._observation()
                metrics = observation.get("metrics") or {}
                sequence = metrics.get("lidar_sequence")
                if (
                    isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence == last_sequence
                ):
                    time.sleep(0.10)
                    continue
                last_sequence = sequence
                sectors = metrics.get("lidar_sectors_m") or {}
                clearances: list[tuple[str, float]] = []
                for sector_name in ("front", "front_left", "front_right"):
                    raw = sectors.get(sector_name)
                    if raw is None:
                        continue
                    try:
                        value = float(raw)
                    except (TypeError, ValueError, OverflowError):
                        return False, f"{sector_name} 雷达距离无效"
                    if not math.isfinite(value):
                        return False, f"{sector_name} 雷达距离无效"
                    clearances.append((sector_name, value))
                risk = metrics.get("risk", "unknown")
                if not clearances or risk in {"unknown", "critical"}:
                    clear_frames = 0
                    blocked_frames += 1
                    last_detail = f"复核帧风险为 {risk} 或缺少前向扇区"
                else:
                    limiting_sector, measured = min(
                        clearances, key=lambda item: item[1]
                    )
                    last_detail = (
                        f"{limiting_sector} {measured:.2f} m / "
                        f"要求 {required_clearance:.2f} m"
                    )
                    if measured >= required_clearance:
                        clear_frames += 1
                        blocked_frames = 0
                        if clear_frames >= 2:
                            route_plan["clearance_revalidations"] = int(
                                route_plan.get("clearance_revalidations", 0)
                            ) + 1
                            return True, last_detail
                    else:
                        clear_frames = 0
                        blocked_frames += 1
                if blocked_frames >= 2:
                    return False, last_detail
                time.sleep(0.10)
            return False, last_detail

        def confirm_costmap_pulse_after_stop(
            travel_m: float,
            initial_sequence: Any,
        ) -> tuple[bool, str]:
            """Reconcile accumulated map cost with two fresh physical frames."""

            stop_barrier = self._stop_robot({})
            if stop_barrier.get("ok") is not True:
                return False, "地图冲突后的零速屏障失败"
            last_sequence = (
                initial_sequence
                if isinstance(initial_sequence, int)
                and not isinstance(initial_sequence, bool)
                else None
            )
            clear_frames = 0
            blocked_frames = 0
            last_detail = "没有获得两个不同的新鲜雷达帧"
            for _ in range(30):
                if self._cancel.is_set():
                    return False, "地图复核期间任务已取消"
                observation = self._observation()
                pose = self._xy_yaw_from_observation(observation)
                metrics = observation.get("metrics") or {}
                sequence = metrics.get("lidar_sequence")
                if (
                    pose is None
                    or isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence == last_sequence
                ):
                    time.sleep(0.10)
                    continue
                last_sequence = sequence
                risk = metrics.get("risk", "unknown")
                nearest = metrics.get("nearest_obstacle_distance")
                if (
                    risk in {"unknown", "critical"}
                    or isinstance(nearest, bool)
                    or not isinstance(nearest, (int, float))
                    or not math.isfinite(float(nearest))
                ):
                    pulse_clear = False
                    last_detail = f"复核帧风险为 {risk} 或物理距离无效"
                else:
                    verified_radius = min(
                        1.0, max(0.0, float(nearest) - 0.15)
                    )
                    try:
                        grid = self._current_costmap_grid().with_verified_free_disk(
                            (pose[0], pose[1]), verified_radius
                        )
                        pulse_clear = grid.swept_pulse_is_clear(
                            (pose[0], pose[1]),
                            heading=pose[2],
                            travel_m=travel_m,
                            clearance_m=ROOM_LOOP_ROUTE_CLEARANCE_M,
                        )
                    except ValueError as error:
                        pulse_clear = False
                        last_detail = str(error)
                    else:
                        last_detail = (
                            f"revision={self.monitor.costmap_payload().get('revision')}; "
                            f"verified_radius={verified_radius:.2f} m"
                        )
                if pulse_clear:
                    clear_frames += 1
                    blocked_frames = 0
                    if clear_frames >= 2:
                        route_plan["costmap_revalidations"] = int(
                            route_plan.get("costmap_revalidations", 0)
                        ) + 1
                        return True, last_detail
                else:
                    clear_frames = 0
                    blocked_frames += 1
                    if blocked_frames >= 2:
                        return False, last_detail
                time.sleep(0.10)
            return False, last_detail

        for waypoint_index, (target_x, target_y) in enumerate(waypoints, start=1):
            stalled_forward_pulses = 0
            previous_distance: float | None = None
            for _ in range(36):
                if self._cancel.is_set():
                    return fail("room_loop_cancelled", "Agent turn 已取消，机器人已停车")
                observation = self._observation()
                pose = self._xy_yaw_from_observation(observation)
                if pose is None:
                    return fail(
                        "room_loop_observation_failed",
                        "行走过程中丢失有效里程计，机器人已停车",
                    )
                current_x, current_y, current_yaw = pose
                dx = target_x - current_x
                dy = target_y - current_y
                distance = math.hypot(dx, dy)
                if distance <= 0.08:
                    reached_waypoints += 1
                    last_pose = pose
                    break

                metrics = observation.get("metrics") or {}
                risk = metrics.get("risk", "unknown")
                if risk in {"unknown", "critical"}:
                    return fail(
                        "room_loop_risk_blocked",
                        f"第 {waypoint_index} 段风险变为 {risk}，机器人已停车",
                    )
                desired_yaw = math.atan2(dy, dx)
                heading_error = self._wrap_angle(desired_yaw - current_yaw)
                if abs(heading_error) > math.radians(7.0):
                    yaw_speed = 0.25 if risk == "warning" else 0.42
                    max_duration = 1.0 if risk == "warning" else 2.0
                    duration = min(
                        max_duration,
                        max(0.20, abs(heading_error) / yaw_speed),
                    )
                    command = {
                        "x": 0.0,
                        "y": 0.0,
                        "yaw": math.copysign(yaw_speed, heading_error),
                        "duration": duration,
                    }
                else:
                    speed = 0.08 if risk == "warning" else 0.16
                    duration = min(
                        (
                            1.0
                            if risk == "warning"
                            else ROOM_LOOP_MAX_TRANSLATION_PULSE_M / speed
                        ),
                        max(0.25, (distance - 0.055) / speed),
                    )
                    try:
                        nearest_now = metrics.get("nearest_obstacle_distance")
                        verified_now = 0.0
                        if (
                            isinstance(nearest_now, (int, float))
                            and not isinstance(nearest_now, bool)
                            and math.isfinite(float(nearest_now))
                        ):
                            verified_now = min(
                                1.0,
                                max(0.0, float(nearest_now) - 0.15),
                            )
                        live_grid = self._current_costmap_grid().with_verified_free_disk(
                            (current_x, current_y),
                            verified_now,
                        )
                    except ValueError as error:
                        return fail(
                            "room_loop_route_unavailable",
                            f"第 {waypoint_index} 段实时地图不可用：{error}",
                        )
                    if not live_grid.swept_pulse_is_clear(
                        (current_x, current_y),
                        heading=current_yaw,
                        travel_m=speed * duration,
                        clearance_m=ROOM_LOOP_ROUTE_CLEARANCE_M,
                    ):
                        confirmed, confirmation_detail = (
                            confirm_costmap_pulse_after_stop(
                                speed * duration,
                                metrics.get("lidar_sequence"),
                            )
                        )
                        if not confirmed:
                            return fail(
                                "room_loop_path_blocked",
                                (
                                    f"第 {waypoint_index} 段下一脉冲被新 costmap 障碍或未知区域阻断；"
                                    f"停车复核仍被阻断：{confirmation_detail}"
                                ),
                            )
                    lidar_sectors = metrics.get("lidar_sectors_m") or {}
                    swept_clearances: list[tuple[str, float]] = []
                    for sector_name in ("front", "front_left", "front_right"):
                        raw_clearance = lidar_sectors.get(sector_name)
                        if raw_clearance is None:
                            continue
                        try:
                            sector_clearance = float(raw_clearance)
                        except (TypeError, ValueError, OverflowError):
                            return fail(
                                "room_loop_observation_failed",
                                f"{sector_name} 雷达距离无效，机器人已停车",
                            )
                        if not math.isfinite(sector_clearance):
                            return fail(
                                "room_loop_observation_failed",
                                f"{sector_name} 雷达距离无效，机器人已停车",
                            )
                        swept_clearances.append((sector_name, sector_clearance))
                    if swept_clearances:
                        limiting_sector, swept_clearance = min(
                            swept_clearances,
                            key=lambda item: item[1],
                        )
                        required_clearance = 0.55 + speed * duration
                        if swept_clearance < required_clearance:
                            confirmed, confirmation_detail = (
                                confirm_swept_lidar_after_stop(
                                    required_clearance,
                                    metrics.get("lidar_sequence"),
                                )
                            )
                            if not confirmed:
                                return fail(
                                    "room_loop_path_blocked",
                                    (
                                        f"第 {waypoint_index} 段 {limiting_sector} 仅 "
                                        f"{swept_clearance:.2f} m，不满足 "
                                        f"{required_clearance:.2f} m 扫掠安全余量；"
                                        f"停车复核仍被阻断：{confirmation_detail}"
                                    ),
                                )
                    command = {
                        "x": speed,
                        "y": 0.0,
                        "yaw": 0.0,
                        "duration": duration,
                    }

                result = self._move_robot(command)
                if not result.get("ok"):
                    error = str(result.get("error", "unknown error"))
                    if "warning 风险下限值" in error:
                        # Lidar can cross clear→warning between this loop's
                        # observation and _move_robot's final guard.  No command
                        # was sent, so recompute a warning-safe pulse instead of
                        # aborting the whole closed loop.
                        time.sleep(0.05)
                        continue
                    return fail(
                        "room_loop_motion_failed",
                        f"第 {waypoint_index} 段脉冲失败：{error}",
                    )
                pulse_count += 1
                after = result.get("observation_after") or self._observation()
                live_audit = read_audit()
                clearance_violation = self._room_loop_clearance_violation(
                    live_audit
                )
                if clearance_violation is not None:
                    return fail(
                        "room_loop_clearance_violation",
                        clearance_violation + "；闭环任务不得报成功",
                    )
                after_pose = self._xy_yaw_from_observation(after)
                if after_pose is None:
                    return fail(
                        "room_loop_observation_failed",
                        "运动后没有有效里程计，机器人已停车",
                    )
                path_length += math.hypot(
                    after_pose[0] - last_pose[0],
                    after_pose[1] - last_pose[1],
                )
                last_pose = after_pose
                if abs(command["x"]) > 0.01:
                    new_distance = math.hypot(
                        target_x - after_pose[0],
                        target_y - after_pose[1],
                    )
                    if (
                        previous_distance is not None
                        and previous_distance - new_distance < 0.01
                    ):
                        stalled_forward_pulses += 1
                    else:
                        stalled_forward_pulses = 0
                    previous_distance = new_distance
                    if stalled_forward_pulses >= 3:
                        return fail(
                            "room_loop_motion_failed",
                            f"第 {waypoint_index} 段连续三个平移脉冲无有效进展",
                        )
            else:
                return fail(
                    "room_loop_motion_failed",
                    f"第 {waypoint_index} 段达到脉冲上限",
                )

        # Restore the starting heading, still using bounded sensor-gated pulses.
        for _ in range(10):
            observation = self._observation()
            pose = self._xy_yaw_from_observation(observation)
            if pose is None:
                return fail(
                    "room_loop_observation_failed",
                    "闭环后缺少航向观测，机器人已停车",
                )
            yaw_error = self._wrap_angle(start_yaw - pose[2])
            if abs(yaw_error) <= math.radians(7.0):
                break
            risk = (observation.get("metrics") or {}).get("risk", "unknown")
            if risk in {"unknown", "critical"}:
                return fail(
                    "room_loop_risk_blocked",
                    f"恢复初始航向时风险变为 {risk}，机器人已停车",
                )
            yaw_speed = 0.25 if risk == "warning" else 0.42
            duration = min(
                1.0 if risk == "warning" else 2.0,
                max(0.20, abs(yaw_error) / yaw_speed),
            )
            result = self._move_robot(
                {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw": math.copysign(yaw_speed, yaw_error),
                    "duration": duration,
                }
            )
            pulse_count += 1
            if not result.get("ok"):
                return fail(
                    "room_loop_motion_failed",
                    f"恢复初始航向失败：{result.get('error', 'unknown error')}",
                )
            clearance_violation = self._room_loop_clearance_violation(read_audit())
            if clearance_violation is not None:
                return fail(
                    "room_loop_clearance_violation",
                    clearance_violation + "；恢复航向期间验收失败",
                )
        else:
            return fail("room_loop_motion_failed", "恢复初始航向达到脉冲上限")

        stop = self._stop_robot({})
        stationary_samples = 0
        final_observation = stop.get("observation_after") or self._observation()
        for _ in range(16):
            final_observation = self._observation()
            motion = final_observation.get("motion") or {}
            planar = motion.get("planar_speed")
            yaw_rate = motion.get("yaw_rate")
            if (
                isinstance(planar, (int, float))
                and not isinstance(planar, bool)
                and isinstance(yaw_rate, (int, float))
                and not isinstance(yaw_rate, bool)
                and abs(float(planar)) <= 0.025
                and abs(float(yaw_rate)) <= 0.08
            ):
                stationary_samples += 1
                if stationary_samples >= 2:
                    break
            else:
                stationary_samples = 0
            time.sleep(0.25)

        motion_audit = read_audit(finish=True)
        final_pose = self._xy_yaw_from_observation(final_observation)
        if final_pose is None:
            return self._room_loop_failure(
                task_status="room_loop_verification_failed",
                error="停车后缺少最终里程计",
                path_length_m=path_length,
                pulse_count=pulse_count,
                reached_waypoints=reached_waypoints,
                route_plan=route_plan,
                motion_audit=motion_audit,
            )
        closure_error = math.hypot(final_pose[0] - start_x, final_pose[1] - start_y)
        final_yaw_error = math.degrees(self._wrap_angle(final_pose[2] - start_yaw))
        clearance_violation = self._room_loop_clearance_violation(motion_audit)
        audit_samples = motion_audit.get("samples")
        audit_verified = bool(
            motion_audit.get("available") is True
            and isinstance(audit_samples, int)
            and not isinstance(audit_samples, bool)
            and audit_samples > 0
            and clearance_violation is None
        )
        completed = bool(
            stop.get("ok")
            and stationary_samples >= 2
            and reached_waypoints == len(waypoints)
            and path_length >= planned_perimeter * 0.85
            and closure_error <= 0.18
            and abs(final_yaw_error) <= 12.0
            and audit_verified
        )
        if not completed:
            task_status = (
                "room_loop_clearance_violation"
                if clearance_violation is not None
                else "room_loop_verification_failed"
            )
            return {
                "ok": False,
                "completed": False,
                "task_status": task_status,
                "error": (
                    clearance_violation
                    or "闭环行走完成了运动，但最终位姿、静止或全程安全审计未通过"
                ),
                "path_length_m": round(path_length, 3),
                "planned_perimeter_m": round(planned_perimeter, 3),
                "closure_error_m": round(closure_error, 3),
                "final_yaw_error_deg": round(final_yaw_error, 2),
                "stationary_samples": stationary_samples,
                "pulse_count": pulse_count,
                "reached_waypoints": reached_waypoints,
                "route_plan": route_plan,
                "motion_audit": motion_audit,
                "stop": stop,
                "observation_after": final_observation,
            }
        return {
            "ok": True,
            "completed": True,
            "task_status": "room_loop_verified",
            "path_length_m": round(path_length, 3),
            "side_length_m": side_length,
            "planned_perimeter_m": round(planned_perimeter, 3),
            "route_clearance_m": route_plan["route_clearance_m"],
            "selection_clearance_m": route_plan["selection_clearance_m"],
            "stable_map_revisions": route_plan.get("stable_map_revisions"),
            "stable_map_revision": route_plan.get("stable_map_revision"),
            "closure_error_m": round(closure_error, 3),
            "final_yaw_error_deg": round(final_yaw_error, 2),
            "stationary_samples": stationary_samples,
            "pulse_count": pulse_count,
            "reached_waypoints": reached_waypoints,
            "turn_direction": route_plan["turn_direction"],
            "route_heading_deg": route_plan["relative_heading_deg"],
            "motion_audit": motion_audit,
            "stop": stop,
            "observation_after": final_observation,
        }

    def _stop_robot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return {"ok": False, "error": "stop_robot 不接受参数"}
        # Emergency zero must never inherit the Turn's cancellation/deadline.
        # The backend-local primitive is bounded and repeats zero independently.
        zero_command = [
            str(self.project_root / "scripts/dimos.sh"),
            "move",
            "--x",
            "0",
            "--y",
            "0",
            "--yaw",
            "0",
            "--duration",
            "0.2",
        ]
        if self._custom_command_runner:
            # Deterministic injected runners remain observable to unit and
            # Adapter contract tests; production uses the local direct port.
            zero = self._command_runner(zero_command, 6.0)
        else:
            zero_ok = self._publish_emergency_zero(zero_command[0])
            zero = {
                "ok": zero_ok,
                "output": "backend-local emergency zero published",
            }
        results = (
            [zero]
            if self.backend == "isaac-g1"
            else [
                self._mcp_call("end_exploration"),
                self._mcp_call("stop_navigation"),
                zero,
            ]
        )
        self._observed_this_turn = True
        return {
            "ok": bool(results[-1].get("ok")),
            "results": results,
            "observation_after": self._observation(),
        }
