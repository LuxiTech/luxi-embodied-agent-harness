"""Narrow compatibility guards for the pinned DimOS A* planner."""

from __future__ import annotations

import math
import os
from collections import deque
from threading import RLock
import time
from typing import Any, Callable

_ORIGINAL_PLAN_PATH: Callable[[Any], None] | None = None
_ORIGINAL_PLANNER_INIT: Callable[..., None] | None = None
_ORIGINAL_LOCAL_PLANNER_CHANGE_STATE: Callable[[Any, str], None] | None = None
_ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING: Callable[[Any], Any] | None = None
_ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION: Callable[[Any], Any] | None = None
_ORIGINAL_PATH_CLEARANCE_UPDATE_COSTMAP: Callable[[Any, Any], None] | None = None


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


class IsaacInitialRotationProgressTracker:
    """Use angular progress instead of XY motion during Isaac start alignment."""

    def __init__(
        self,
        planner: Any,
        delegate: Any,
        *,
        time_window: float,
        minimum_progress_radians: float,
        maximum_duration: float,
        maximum_drift: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._planner = planner
        self._delegate = delegate
        self._time_window = float(time_window)
        self._minimum_progress_radians = float(minimum_progress_radians)
        self._maximum_duration = float(maximum_duration)
        self._maximum_drift = float(maximum_drift)
        self._clock = clock
        self._lock = RLock()
        self._state_key: tuple[str, int] | None = None
        self._state_started_at: float | None = None
        self._samples: deque[tuple[float, float, float, float]] = deque()

    def _local_state_key(self) -> tuple[str, int]:
        local_planner = self._planner._local_planner
        reader = getattr(local_planner, "get_unique_state", None)
        if callable(reader):
            state, unique_id = reader()
            return str(state), int(unique_id)
        state = str(getattr(local_planner, "_state", "idle"))
        return state, id(getattr(local_planner, "_path", None))

    def _reset_rotation_samples(self) -> None:
        self._state_key = None
        self._state_started_at = None
        self._samples.clear()

    def reset_data(self) -> None:
        reset = getattr(self._delegate, "reset_data", None)
        if callable(reset):
            reset()
        with self._lock:
            self._reset_rotation_samples()

    def add_position(self, pose: Any) -> None:
        add = getattr(self._delegate, "add_position", None)
        if callable(add):
            add(pose)
        now = self._clock()
        try:
            state_key = self._local_state_key()
            sample = (
                now,
                float(pose.position.x),
                float(pose.position.y),
                float(pose.orientation.euler[2]),
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(value) for value in sample):
            return
        with self._lock:
            if state_key[0] != "initial_rotation":
                self._reset_rotation_samples()
                return
            if state_key != self._state_key:
                self._state_key = state_key
                self._state_started_at = now
                self._samples.clear()
            self._samples.append(sample)
            cutoff = now - self._time_window
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def _initial_target_yaw(self) -> float | None:
        try:
            local_planner = self._planner._local_planner
            path = local_planner._path
            if path is None or not path.poses:
                return None
            yaw = float(path.poses[0].orientation.euler[2])
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        return yaw if math.isfinite(yaw) else None

    @staticmethod
    def _yaw_error(target: float, actual: float) -> float:
        return abs(math.atan2(math.sin(target - actual), math.cos(target - actual)))

    def is_stuck(self) -> bool:
        delegate_reader = getattr(self._delegate, "is_stuck", None)
        delegate_stuck = bool(delegate_reader()) if callable(delegate_reader) else False
        now = self._clock()
        try:
            state_key = self._local_state_key()
        except (AttributeError, TypeError, ValueError, OverflowError):
            return delegate_stuck
        if state_key[0] != "initial_rotation":
            with self._lock:
                self._reset_rotation_samples()
            return delegate_stuck

        target_yaw = self._initial_target_yaw()
        with self._lock:
            if state_key != self._state_key:
                self._state_key = state_key
                self._state_started_at = now
                self._samples.clear()
                return False
            samples = tuple(self._samples)
            started_at = self._state_started_at
        if target_yaw is None or started_at is None or len(samples) < 2:
            return delegate_stuck

        origin = samples[0]
        maximum_drift = max(
            math.hypot(sample[1] - origin[1], sample[2] - origin[2])
            for sample in samples
        )
        if maximum_drift > self._maximum_drift:
            return True
        if now - started_at >= self._maximum_duration:
            return True

        initial_error = self._yaw_error(target_yaw, origin[3])
        current_error = self._yaw_error(target_yaw, samples[-1][3])
        angular_progress = initial_error - current_error
        return angular_progress < self._minimum_progress_radians


def tune_simulation_stuck_detection(
    planner: Any,
    *,
    tracker_type: Callable[[float, float], Any] | None = None,
) -> bool:
    """Tune simulated G1 navigation, including the isolated Isaac backend.

    Isaac deliberately uses the hardware-shaped primitive blueprint and is
    therefore not launched through DimOS' ``--simulation mujoco`` switch.  It
    is still a simulated G1 with the same slow-turn/stuck-detector mismatch,
    and ``LUXI_SIM_BACKEND`` is the authoritative project backend boundary.
    """

    simulation = bool(getattr(planner._global_config, "simulation", False))
    backend = os.environ.get("LUXI_SIM_BACKEND", "").strip().casefold()
    if not simulation and backend != "isaac-g1":
        return False
    if tracker_type is None:
        from dimos.navigation.replanning_a_star.position_tracker import PositionTracker

        tracker_type = PositionTracker

    time_window = _bounded_float("LUXI_NAV_STUCK_TIME_WINDOW", 15.0, 8.0, 60.0)
    distance = _bounded_float("LUXI_NAV_STUCK_DISTANCE", 0.15, 0.05, 1.0)
    planner._stuck_time_window = time_window
    planner._stuck_threshold = distance
    position_tracker = tracker_type(time_window, distance)
    local_planner = getattr(planner, "_local_planner", None)
    if backend == "isaac-g1":
        position_tracker = IsaacInitialRotationProgressTracker(
            planner,
            position_tracker,
            time_window=time_window,
            minimum_progress_radians=math.radians(
                _bounded_float(
                    "LUXI_NAV_INITIAL_ROTATION_MIN_PROGRESS_DEGREES",
                    3.0,
                    1.0,
                    20.0,
                )
            ),
            maximum_duration=_bounded_float(
                "LUXI_NAV_INITIAL_ROTATION_MAX_SECONDS",
                30.0,
                15.0,
                60.0,
            ),
            maximum_drift=_bounded_float(
                "LUXI_NAV_INITIAL_ROTATION_MAX_DRIFT",
                0.25,
                0.10,
                0.50,
            ),
        )
        controller = getattr(local_planner, "_controller", None)
        if controller is not None and hasattr(controller, "_rotation_threshold"):
            # Mixed translation is intentionally limited to 0.08 rad/s for
            # the reference G1 gait.  The upstream 90-degree rotate/drive
            # boundary therefore produces a large orbit around short goals.
            # Correct heading first at a tighter threshold while retaining
            # the same planner, relay, speed limits and command publisher.
            controller._rotation_threshold = math.radians(
                _bounded_float(
                    "LUXI_NAV_ISAAC_ROTATE_THEN_DRIVE_DEGREES",
                    35.0,
                    20.0,
                    75.0,
                )
            )
            local_planner._luxi_rotate_then_drive_threshold = (
                controller._rotation_threshold
            )
    planner._position_tracker = position_tracker
    if local_planner is not None and hasattr(local_planner, "_orientation_tolerance"):
        final_tolerance = float(local_planner._orientation_tolerance)
        initial_degrees = _bounded_float(
            "LUXI_NAV_INITIAL_ORIENTATION_TOLERANCE_DEGREES",
            75.0,
            20.0,
            80.0,
        )
        local_planner._luxi_final_orientation_tolerance = final_tolerance
        local_planner._luxi_initial_orientation_tolerance = math.radians(
            initial_degrees
        )
        local_planner._orientation_tolerance = (
            local_planner._luxi_initial_orientation_tolerance
        )
        # The learned Isaac gait can keep translating for roughly one second
        # after its command target becomes zero.  Delay the planner's terminal
        # goal signal until odometry proves that this coast has finished; the
        # visual skill's independent 500 ms stop contract then starts from an
        # already stationary pose instead of from the first zero publication.
        local_planner._luxi_settle_before_arrival = backend == "isaac-g1"
    return True


def _reset_arrival_settle(planner: Any, phase: str | None = None) -> None:
    planner._luxi_arrival_settle_phase = phase
    planner._luxi_arrival_settle_pose = None
    planner._luxi_arrival_settle_count = 0


def _arrival_stationary_sample(planner: Any, odom: Any, phase: str) -> bool:
    """Require two fresh low-speed odometry intervals before goal success."""

    if getattr(planner, "_luxi_arrival_settle_phase", None) != phase:
        _reset_arrival_settle(planner, phase)
    try:
        sample = (
            float(odom.position.x),
            float(odom.position.y),
            float(odom.ts),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        _reset_arrival_settle(planner, phase)
        return False
    if not all(math.isfinite(value) for value in sample):
        _reset_arrival_settle(planner, phase)
        return False
    previous = getattr(planner, "_luxi_arrival_settle_pose", None)
    if previous is None or sample[2] <= previous[2]:
        planner._luxi_arrival_settle_pose = sample
        return False
    elapsed = sample[2] - previous[2]
    speed = math.hypot(sample[0] - previous[0], sample[1] - previous[1]) / elapsed
    planner._luxi_arrival_settle_pose = sample
    threshold = _bounded_float(
        "LUXI_NAV_ARRIVAL_STATIONARY_SPEED",
        0.025,
        0.005,
        0.05,
    )
    if speed <= threshold:
        planner._luxi_arrival_settle_count += 1
    else:
        planner._luxi_arrival_settle_count = 0
    required = int(
        _bounded_float("LUXI_NAV_ARRIVAL_STATIONARY_SAMPLES", 2.0, 2.0, 5.0)
    )
    return planner._luxi_arrival_settle_count >= required


def local_planner_path_following_with_arrival_settle(planner: Any) -> Any:
    """Brake inside the Isaac goal tolerance before final rotation."""

    if _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    if not getattr(planner, "_luxi_settle_before_arrival", False):
        return _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING(planner)
    with planner._lock:
        path = planner._path
        current_odom = planner._current_odom
    if path is None or current_odom is None or not path.poses:
        _reset_arrival_settle(planner)
        return _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING(planner)
    goal = path.poses[-1]
    distance = math.hypot(
        float(goal.position.x) - float(current_odom.position.x),
        float(goal.position.y) - float(current_odom.position.y),
    )
    if distance >= float(planner._goal_tolerance):
        _reset_arrival_settle(planner)
        return _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING(planner)

    from dimos.msgs.geometry_msgs.Twist import Twist

    if not _arrival_stationary_sample(planner, current_odom, "position"):
        return Twist.zero()
    with planner._lock:
        planner._change_state("final_rotation")
    return local_planner_final_rotation_with_arrival_settle(planner)


def local_planner_final_rotation_with_arrival_settle(planner: Any) -> Any:
    """Publish arrival only after the final turning gait has settled."""

    if _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    if not getattr(planner, "_luxi_settle_before_arrival", False):
        return _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION(planner)
    with planner._lock:
        path = planner._path
        current_odom = planner._current_odom
    if path is None or current_odom is None or not path.poses:
        _reset_arrival_settle(planner)
        return _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION(planner)
    goal_yaw = float(path.poses[-1].orientation.euler[2])
    robot_yaw = float(current_odom.orientation.euler[2])
    yaw_error = math.atan2(
        math.sin(goal_yaw - robot_yaw),
        math.cos(goal_yaw - robot_yaw),
    )
    if abs(yaw_error) >= float(planner._orientation_tolerance):
        _reset_arrival_settle(planner)
        return _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION(planner)

    from dimos.msgs.geometry_msgs.Twist import Twist

    if not _arrival_stationary_sample(planner, current_odom, "orientation"):
        return Twist.zero()
    with planner._lock:
        planner._change_state("arrived")
    _reset_arrival_settle(planner)
    return Twist.zero()


def local_planner_change_state_with_simulation_tuning(
    planner: Any,
    new_state: str,
) -> None:
    """Use a wide Isaac start tolerance, then restore strict final rotation."""

    if _ORIGINAL_LOCAL_PLANNER_CHANGE_STATE is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    _ORIGINAL_LOCAL_PLANNER_CHANGE_STATE(planner, new_state)
    initial = getattr(planner, "_luxi_initial_orientation_tolerance", None)
    final = getattr(planner, "_luxi_final_orientation_tolerance", None)
    if initial is None or final is None:
        return
    planner._orientation_tolerance = (
        float(initial) if new_state in {"idle", "initial_rotation"} else float(final)
    )


def path_clearance_update_rolling_costmap(clearance: Any, costmap: Any) -> None:
    """Invalidate DimOS' cached path mask when a rolling map origin changes."""

    if _ORIGINAL_PATH_CLEARANCE_UPDATE_COSTMAP is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    origin = getattr(costmap, "origin", None)
    position = getattr(origin, "position", origin)
    grid = getattr(costmap, "grid", None)
    key = (
        getattr(grid, "shape", None),
        float(getattr(costmap, "resolution", 0.0)),
        float(getattr(position, "x", 0.0)),
        float(getattr(position, "y", 0.0)),
    )
    previous_key = getattr(clearance, "_luxi_costmap_frame_key", None)
    _ORIGINAL_PATH_CLEARANCE_UPDATE_COSTMAP(clearance, costmap)
    clearance._luxi_costmap_frame_key = key
    if previous_key is not None and previous_key != key:
        clearance._last_mask = None
        clearance._last_used_pose = None
        clearance._last_used_shape = None


def planner_init_with_simulation_tuning(
    planner: Any, *args: Any, **kwargs: Any
) -> None:
    if _ORIGINAL_PLANNER_INIT is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    _ORIGINAL_PLANNER_INIT(planner, *args, **kwargs)
    tune_simulation_stuck_detection(planner)


def plan_path_with_arrival_race_guard(planner: Any) -> None:
    """Accept an assertion only when another thread already completed the goal.

    DimOS' planner deliberately releases its lock between cancelling the old
    path and reading the current goal.  If the monitoring thread notices that
    the robot is already at the requested pose in that window, it marks the
    goal reached and clears ``_current_goal``.  The planning thread then hits
    an assertion even though navigation has completed successfully.
    """
    if _ORIGINAL_PLAN_PATH is None:
        raise RuntimeError("navigation compatibility layer is not installed")
    try:
        _ORIGINAL_PLAN_PATH(planner)
    except AssertionError:
        with planner._lock:
            arrived_elsewhere = planner._current_goal is None and planner._goal_reached
        if not arrived_elsewhere:
            raise


def install_navigation_compat() -> bool:
    """Install narrow planner guards without changing the DimOS checkout."""
    if not _enabled("LUXI_NAVIGATION_COMPAT"):
        return False

    from dimos.navigation.replanning_a_star.global_planner import GlobalPlanner

    if getattr(GlobalPlanner, "_luxi_arrival_race_guard", False):
        return True

    from dimos.navigation.replanning_a_star.local_planner import LocalPlanner
    from dimos.navigation.replanning_a_star.path_clearance import PathClearance

    global _ORIGINAL_PLAN_PATH, _ORIGINAL_PLANNER_INIT
    global _ORIGINAL_LOCAL_PLANNER_CHANGE_STATE
    global _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING
    global _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION
    global _ORIGINAL_PATH_CLEARANCE_UPDATE_COSTMAP
    _ORIGINAL_PLAN_PATH = GlobalPlanner._plan_path
    _ORIGINAL_PLANNER_INIT = GlobalPlanner.__init__
    _ORIGINAL_LOCAL_PLANNER_CHANGE_STATE = LocalPlanner._change_state
    _ORIGINAL_LOCAL_PLANNER_PATH_FOLLOWING = LocalPlanner._compute_path_following
    _ORIGINAL_LOCAL_PLANNER_FINAL_ROTATION = LocalPlanner._compute_final_rotation
    _ORIGINAL_PATH_CLEARANCE_UPDATE_COSTMAP = PathClearance.update_costmap
    GlobalPlanner._plan_path = plan_path_with_arrival_race_guard  # type: ignore[method-assign]
    GlobalPlanner.__init__ = planner_init_with_simulation_tuning  # type: ignore[method-assign]
    LocalPlanner._change_state = local_planner_change_state_with_simulation_tuning  # type: ignore[method-assign]
    LocalPlanner._compute_path_following = local_planner_path_following_with_arrival_settle  # type: ignore[method-assign]
    LocalPlanner._compute_final_rotation = local_planner_final_rotation_with_arrival_settle  # type: ignore[method-assign]
    PathClearance.update_costmap = path_clearance_update_rolling_costmap  # type: ignore[method-assign]
    GlobalPlanner._luxi_arrival_race_guard = True
    return True
