"""Bounded, odometry-verified controls shared by supported G1 simulators."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.g1.skill_container import UnitreeG1SkillContainer

from harness.control.simulation_control import SimulationControlChannel


@dataclass(frozen=True)
class RelativeMotionProgress:
    forward_m: float
    left_m: float
    rotation_degrees: float


def relative_motion_progress(start_pose: Any, current_pose: Any) -> RelativeMotionProgress:
    """Project a world-frame odometry delta into the starting robot frame."""

    start_yaw = float(start_pose.orientation.to_euler().yaw)
    current_yaw = float(current_pose.orientation.to_euler().yaw)
    dx = float(current_pose.position.x - start_pose.position.x)
    dy = float(current_pose.position.y - start_pose.position.y)
    forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
    left = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
    yaw_delta = math.atan2(
        math.sin(current_yaw - start_yaw),
        math.cos(current_yaw - start_yaw),
    )
    return RelativeMotionProgress(forward, left, math.degrees(yaw_delta))


def _axis_reached(actual: float, requested: float, tolerance: float) -> bool:
    if abs(requested) <= tolerance:
        return True
    return actual * math.copysign(1.0, requested) >= abs(requested) - tolerance


class G1MujocoAdapter(UnitreeG1SkillContainer):
    """MuJoCo implementation of the shared G1 AgentOS motion skill surface."""

    @staticmethod
    def _bounded_velocity(x: float, y: float, yaw: float, duration: float) -> str | None:
        if not all(math.isfinite(value) for value in (x, y, yaw, duration)):
            return "Simulation motion rejected: every parameter must be finite"
        if abs(x) > 0.25 or abs(y) > 0.20 or abs(yaw) > 0.60:
            return "Simulation motion rejected: velocity exceeds the Luxi safety envelope"
        if duration < 0.0 or duration > 3.0:
            return "Simulation motion rejected: duration must be between 0 and 3 seconds"
        return None

    @skill(uses=[CAP_MOVEMENT])
    def move(
        self,
        x: float,
        y: float = 0.0,
        yaw: float = 0.0,
        duration: float = 0.0,
    ) -> str:
        """Send one bounded G1 simulation velocity command.

        Args:
            x: Forward velocity in metres per second; negative moves backward.
            y: Leftward velocity in metres per second; negative moves right.
            yaw: Counter-clockwise angular velocity in radians per second.
            duration: Automatic stop time in seconds, from 0 through 3.
        """
        values = tuple(float(value) for value in (x, y, yaw, duration))
        error = self._bounded_velocity(*values)
        if error:
            return error
        x, y, yaw, duration = values
        self._connection.move(
            Twist(linear=Vector3(x, y, 0.0), angular=Vector3(0.0, 0.0, yaw)),
            duration=duration,
        )
        return (
            "Simulation velocity accepted inside the safety envelope: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, duration={duration:.2f}s"
        )

    @skill(uses=[CAP_MOVEMENT])
    def relative_move(
        self,
        forward: float = 0.0,
        left: float = 0.0,
        degrees: float = 0.0,
    ) -> str:
        """Move relative to the current pose and verify signed odometry change.

        Args:
            forward: Signed forward distance in metres, limited to 1 metre.
            left: Signed leftward distance in metres, limited to 1 metre.
            degrees: Signed counter-clockwise rotation, limited to 180 degrees.
        """
        forward, left, degrees = (float(forward), float(left), float(degrees))
        if not all(math.isfinite(value) for value in (forward, left, degrees)):
            return "Simulation motion rejected: every target must be finite"
        if abs(forward) > 1.0 or abs(left) > 1.0 or abs(degrees) > 180.0:
            return "Simulation motion rejected: relative target exceeds 1m/180deg limits"
        if max(abs(forward), abs(left), abs(degrees)) < 1e-6:
            self._connection.move(Twist.zero())
            return "Simulation motion verified: target was already the current pose"

        start_transform = self.tf.get("world", "base_link")
        if start_transform is None:
            return "Simulation motion failed: odometry is unavailable"
        start_pose = start_transform.to_pose()
        yaw_radians = math.radians(degrees)
        expected_duration = max(
            abs(forward) / 0.20,
            abs(left) / 0.16,
            abs(yaw_radians) / 0.45,
            0.5,
        )
        timeout = min(20.0, max(5.0, expected_duration * 2.5))
        command = Twist(
            linear=Vector3(
                forward / expected_duration,
                left / expected_duration,
                0.0,
            ),
            angular=Vector3(0.0, 0.0, yaw_radians / expected_duration),
        )
        self._connection.move(command, duration=timeout)

        progress = RelativeMotionProgress(0.0, 0.0, 0.0)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                current_transform = self.tf.get("world", "base_link")
                if current_transform is not None:
                    progress = relative_motion_progress(
                        start_pose,
                        current_transform.to_pose(),
                    )
                    if (
                        _axis_reached(progress.forward_m, forward, 0.04)
                        and _axis_reached(progress.left_m, left, 0.04)
                        and _axis_reached(progress.rotation_degrees, degrees, 4.0)
                    ):
                        return (
                            "Simulation motion verified by signed odometry: "
                            f"forward={progress.forward_m:.2f}m, "
                            f"left={progress.left_m:.2f}m, "
                            f"rotation={progress.rotation_degrees:.1f}deg"
                        )
                time.sleep(0.08)
        finally:
            self._connection.move(Twist.zero())
        return (
            "Simulation motion failed odometry verification: "
            f"forward={progress.forward_m:.2f}/{forward:.2f}m, "
            f"left={progress.left_m:.2f}/{left:.2f}m, "
            f"rotation={progress.rotation_degrees:.1f}/{degrees:.1f}deg"
        )

    @skill(uses=[CAP_MOVEMENT])
    def reset_pose(self) -> str:
        """Reset only the simulated robot pose; maps and Agent memory are retained."""

        self._connection.move(Twist.zero())
        channel = SimulationControlChannel()
        token = channel.request_reset()
        if not channel.wait_for_ack(token, timeout=4.0):
            return "Simulation pose reset failed: simulator worker did not acknowledge the request"
        acknowledgement = channel.acknowledgement() or {}
        expected_position = acknowledgement.get("position")
        expected_quaternion = acknowledgement.get("quaternion_wxyz")
        if (
            not isinstance(expected_position, list)
            or len(expected_position) != 3
            or not isinstance(expected_quaternion, list)
            or len(expected_quaternion) != 4
        ):
            return "Simulation pose reset failed: acknowledgement omitted the target pose"
        expected_yaw = math.atan2(
            2.0
            * (
                float(expected_quaternion[0]) * float(expected_quaternion[3])
                + float(expected_quaternion[1]) * float(expected_quaternion[2])
            ),
            1.0
            - 2.0
            * (
                float(expected_quaternion[2]) ** 2
                + float(expected_quaternion[3]) ** 2
            ),
        )
        deadline = time.monotonic() + 3.0
        last_pose: Any | None = None
        while time.monotonic() < deadline:
            current = self.tf.get("world", "base_link")
            if current is not None:
                pose = current.to_pose()
                last_pose = pose
                position_error = math.sqrt(
                    (pose.position.x - float(expected_position[0])) ** 2
                    + (pose.position.y - float(expected_position[1])) ** 2
                    + (pose.position.z - float(expected_position[2])) ** 2
                )
                actual_yaw = pose.orientation.to_euler().yaw
                yaw_error = abs(
                    math.atan2(
                        math.sin(actual_yaw - expected_yaw),
                        math.cos(actual_yaw - expected_yaw),
                    )
                )
                if position_error <= 0.05 and yaw_error <= math.radians(5.0):
                    return (
                        "Simulation pose reset verified by fresh odometry: "
                        f"position=({pose.position.x:.2f}, {pose.position.y:.2f}, "
                        f"{pose.position.z:.2f}), "
                        f"yaw={math.degrees(actual_yaw):.1f}deg. "
                        "Map and Agent memory were not cleared."
                    )
            time.sleep(0.05)
        if last_pose is None:
            return "Simulation pose reset failed: fresh odometry is unavailable"
        return (
            "Simulation pose reset failed odometry verification: "
            f"position=({last_pose.position.x:.2f}, {last_pose.position.y:.2f}, "
            f"{last_pose.position.z:.2f})"
        )


class G1IsaacAdapter(G1MujocoAdapter):
    """Isaac implementation of the same G1 AgentOS motion skill surface."""

    _monotonic = staticmethod(time.monotonic)
    _wall_time = staticmethod(time.time)
    _sleep = staticmethod(time.sleep)

    def _isaac_motion_cancelled(self) -> bool:
        probe = getattr(self, "_luxi_motion_cancel_probe", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception:
            # Losing the cancellation channel while moving fails closed.
            return True

    def execute_arm_command(self, command_name: str) -> str:
        """Reject the upstream skill until Isaac manipulation is implemented."""

        del command_name
        return "Isaac arm command is unavailable"

    def execute_mode_command(self, command_name: str) -> str:
        """Reject the upstream skill until Isaac mode requests are implemented."""

        del command_name
        return "Isaac mode command is unavailable"

    @skill(uses=[CAP_MOVEMENT])
    def move(
        self,
        x: float,
        y: float = 0.0,
        yaw: float = 0.0,
        duration: float = 0.0,
    ) -> str:
        """Send one bounded, expiring Isaac velocity pulse.

        Args:
            x: Forward velocity in metres per second; negative moves backward.
            y: Leftward velocity in metres per second; negative moves right.
            yaw: Counter-clockwise angular velocity in radians per second.
            duration: Automatic stop time in seconds, from 0 through 3.
        """

        arguments = {"x": x, "y": y, "yaw": yaw, "duration": duration}
        from harness.robots.g1.isaac.motion_service import (
            IsaacMotionControlChannel,
            encode_motion_result,
        )

        channel = IsaacMotionControlChannel()

        def execute_local(cancelled: Any) -> str:
            return self._isaac_velocity_pulse(
                x=x,
                y=y,
                yaw=yaw,
                duration=duration,
                cancelled=cancelled,
            )

        unified = channel.request_move_robot(arguments, execute_local)
        if unified is not None:
            return encode_motion_result(unified)
        return execute_local(lambda: False)

    def _isaac_velocity_pulse(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        duration: float,
        cancelled: Any,
    ) -> str:
        values = tuple(float(value) for value in (x, y, yaw, duration))
        error = self._bounded_velocity(*values)
        if error:
            return error
        x, y, yaw, duration = values
        command = Twist(
            linear=Vector3(x, y, 0.0),
            angular=Vector3(0.0, 0.0, yaw),
        )
        self._connection.move(command, duration=duration)
        deadline = self._monotonic() + duration
        while self._monotonic() < deadline:
            if cancelled():
                self._connection.move(Twist.zero())
                return (
                    "velocity_pulse_failed: cancelled=true, "
                    f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, "
                    f"duration={duration:.2f}s"
                )
            self._sleep(min(0.05, max(0.0, deadline - self._monotonic())))
        self._connection.move(Twist.zero())
        return (
            "velocity_pulse_completed: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, "
            f"duration={duration:.2f}s"
        )

    def _isaac_transform(self) -> Any | None:
        return self.tf.get("world", "base_link")

    @staticmethod
    def _isaac_target_reached(actual: float, requested: float, tolerance: float) -> bool:
        return abs(actual - requested) <= tolerance + 1e-9

    @staticmethod
    def _isaac_yaw(transform: Any) -> float:
        return float(transform.rotation.to_euler().yaw)

    @staticmethod
    def _isaac_wrapped_yaw_delta(current: float, previous: float) -> float:
        return math.atan2(math.sin(current - previous), math.cos(current - previous))

    def _isaac_rotation_progress(
        self,
        start: Any,
        current: Any,
        target_degrees: float,
    ) -> float:
        progress = math.degrees(
            self._isaac_wrapped_yaw_delta(
                self._isaac_yaw(current),
                self._isaac_yaw(start),
            )
        )
        if target_degrees > 90.0 and progress < 0.0:
            progress += 360.0
        elif target_degrees < -90.0 and progress > 0.0:
            progress -= 360.0
        return progress

    def _isaac_translation_loop(
        self,
        start: Any,
        target_forward: float,
        target_left: float,
        deadline: float,
        *,
        tolerance: float = 0.08,
    ) -> tuple[bool, RelativeMotionProgress, Any | None]:
        progress = RelativeMotionProgress(0.0, 0.0, 0.0)
        current = self._isaac_transform()
        if current is None:
            return False, progress, None
        target_distance = math.hypot(target_forward, target_left)
        previous_settled_error = target_distance
        # Brownstone measured only 0.76 m from the old twenty 0.30 m/s
        # refreshes.  Budget from distance and use the reference keyboard
        # target that reliably crosses the ONNX policy's low-speed dead zone.
        remaining_refreshes = max(
            20,
            min(48, math.ceil(target_distance / 0.035) + 8),
        )
        braking_distance = min(
            0.18,
            max(tolerance, target_distance * 0.25),
        )
        last_effective_step = 0.0
        for correction_index in range(3):
            progress = relative_motion_progress(start.to_pose(), current.to_pose())
            if (
                math.hypot(
                    target_forward - progress.forward_m,
                    target_left - progress.left_m,
                )
                <= tolerance
            ):
                return True, progress, current

            command_target_forward = target_forward
            command_target_left = target_left
            minimum_refreshes_before_braking = 1 if correction_index == 0 else 3
            active_braking_distance = (
                braking_distance
                if correction_index == 0
                else min(tolerance * 0.5, previous_settled_error * 0.5)
            )

            # The learned gait needs a continuous command for roughly two
            # seconds before it produces reliable settled translation.  Keep
            # every filesystem command bounded to one second, but refresh it
            # before expiry instead of returning to zero between refreshes.
            burst_refreshes = 0
            while remaining_refreshes > 0 and self._monotonic() < deadline:
                if self._isaac_motion_cancelled():
                    self._connection.move(Twist.zero())
                    break
                progress = relative_motion_progress(start.to_pose(), current.to_pose())
                forward_error = command_target_forward - progress.forward_m
                left_error = command_target_left - progress.left_m
                if (
                    burst_refreshes >= minimum_refreshes_before_braking
                    and math.hypot(forward_error, left_error)
                    <= max(active_braking_distance, last_effective_step)
                ):
                    break
                yaw_delta = math.radians(progress.rotation_degrees)
                body_forward_error = (
                    math.cos(yaw_delta) * forward_error
                    + math.sin(yaw_delta) * left_error
                )
                body_left_error = (
                    -math.sin(yaw_delta) * forward_error
                    + math.cos(yaw_delta) * left_error
                )
                x = (
                    math.copysign(0.40, body_forward_error)
                    if abs(body_forward_error) > 0.04
                    else 0.0
                )
                y = (
                    math.copysign(0.18, body_left_error)
                    if abs(body_left_error) > 0.04
                    else 0.0
                )
                command = Twist(
                    linear=Vector3(x, y, 0.0),
                    angular=Vector3(0.0, 0.0, 0.0),
                )
                error_before_command = math.hypot(forward_error, left_error)
                self._connection.move(command, duration=1.0)
                remaining_refreshes -= 1
                burst_refreshes += 1
                self._sleep(0.75)
                candidate = self._isaac_transform()
                if candidate is None or float(candidate.ts) <= float(current.ts):
                    current = None
                    break
                current = candidate
                candidate_progress = relative_motion_progress(
                    start.to_pose(),
                    current.to_pose(),
                )
                error_after_command = math.hypot(
                    command_target_forward - candidate_progress.forward_m,
                    command_target_left - candidate_progress.left_m,
                )
                last_effective_step = max(
                    0.0,
                    error_before_command - error_after_command,
                )

            if current is None:
                break
            stationary, current = self._isaac_stop_and_confirm_stationary()
            if not stationary or current is None:
                break
            progress = relative_motion_progress(start.to_pose(), current.to_pose())
            settled_error = math.hypot(
                target_forward - progress.forward_m,
                target_left - progress.left_m,
            )
            if settled_error <= tolerance:
                return True, progress, current
            if (
                remaining_refreshes <= 0
                and correction_index < 2
                and settled_error <= max(0.25, tolerance * 3.0)
            ):
                # A long first gait burst can consume the refresh budget and
                # settle just outside the unchanged terminal tolerance. Allow
                # one bounded, measured reverse/forward correction rather than
                # accepting the overshoot or loosening the verifier.
                remaining_refreshes = 8
            if (
                remaining_refreshes <= 0
                or settled_error > previous_settled_error + 0.12
                or previous_settled_error - settled_error < 0.01
            ):
                break
            previous_settled_error = settled_error
        return False, progress, current

    def _isaac_rotation_loop(
        self,
        start: Any,
        target_degrees: float,
        deadline: float,
        trace: list[str] | None = None,
        *,
        brake_early: bool = False,
    ) -> tuple[bool, float, Any | None]:
        current = self._isaac_transform()
        if current is None:
            if trace is not None:
                trace.append("odometry_unavailable")
            return False, 0.0, None
        terminal_turn = abs(target_degrees) >= 120.0
        accumulated_degrees = self._isaac_rotation_progress(
            start,
            current,
            target_degrees,
        )
        previous_yaw = self._isaac_yaw(current)
        stalled_pulses = 0
        terminal_trim_activated = False
        terminal_creep_activated = False
        exit_reason = "pulse_limit"
        pulses = 0
        # Cold-start traces can still be making monotonic, correctly directed
        # yaw progress at pulse 20 (70/90 and 148/180 degrees were observed).
        # Allow eight more bounded refreshes; the unchanged deadline, drift
        # barrier, cancellation probe and terminal verifier remain authoritative.
        for pulse_index in range(28):
            if self._isaac_motion_cancelled():
                self._connection.move(Twist.zero())
                exit_reason = "cancelled"
                break
            error = target_degrees - accumulated_degrees
            # Keep the cold-start terminal gait continuous until 8 degrees
            # before target. Live settled coast ranges from near-zero to about
            # 15 degrees; braking at 14 left some runs at 166--167 degrees and
            # the policy could not reliably restart for a tiny trim. This
            # tighter route target does not change the independent 12-degree
            # completion verifier or the 0.32 m hard drift stop.
            brake_margin = 8.0 if terminal_turn and brake_early else 4.0
            if self._isaac_target_reached(
                accumulated_degrees,
                target_degrees,
                brake_margin,
            ):
                exit_reason = "brake" if brake_early else "target"
                break
            if self._monotonic() >= deadline:
                exit_reason = "deadline"
                break

            direction = math.copysign(1.0, error)
            yaw_magnitude = (
                (0.30 if terminal_trim_activated else 0.80)
                if terminal_turn and not brake_early and abs(error) < 30.0
                else 0.80 if terminal_turn else 0.60
            )
            yaw_rate = direction * yaw_magnitude
            remaining_radians = math.radians(
                max(0.0, abs(error) - brake_margin)
            )
            duration = min(1.0, max(0.40, remaining_radians / abs(yaw_rate)))
            # A sustained terminal yaw normally turns in place.  After a
            # settled 150-degree turn, however, pure yaw can remain inside the
            # learned gait's dead zone; a completely cold gait can also ignore
            # pure yaw. Use the navigation relay's bounded activation idea
            # after twelve startup stalls, or four stalls inside the final
            # 45 degrees. Alternation plus the unchanged drift barrier prevents
            # an unbounded arc.
            if terminal_turn and (
                stalled_pulses >= 12
                or (abs(error) <= 45.0 and stalled_pulses >= 4)
            ):
                terminal_creep_activated = True
            creep = (
                (0.20 if (pulse_index // 4) % 2 == 0 else -0.20)
                if terminal_creep_activated
                else 0.0
                if terminal_turn
                else 0.20 if (pulse_index // 2) % 2 == 0 else -0.20
            )
            command = Twist(
                linear=Vector3(creep, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, yaw_rate),
            )
            self._connection.move(command, duration=duration)
            pulses += 1
            self._sleep(min(0.75, max(0.30, duration * 0.75)))
            candidate = self._isaac_transform()
            if candidate is None or float(candidate.ts) <= float(current.ts):
                current = None
                exit_reason = "odometry_stale"
                break
            current = candidate
            current_yaw = self._isaac_yaw(current)
            observed_degrees = math.degrees(
                self._isaac_wrapped_yaw_delta(current_yaw, previous_yaw)
            )
            directed_progress = direction * observed_degrees
            if (
                terminal_turn
                and not brake_early
                and abs(error) < 30.0
                and directed_progress >= 0.2
            ):
                # The learned policy can ignore a 0.30 rad/s trim from rest.
                # Use the reviewed 0.80 rad/s terminal command only until yaw
                # is physically observed, then immediately return to the
                # bounded 0.30 rad/s measured trim. This changes neither the
                # 12-degree verifier nor the 0.32 m drift hard stop.
                terminal_trim_activated = True
            if directed_progress < -3.0:
                exit_reason = "wrong_direction"
                break
            stalled_pulses = stalled_pulses + 1 if directed_progress < 1.0 else 0
            # A terminal turn restarted from a fully settled gait can need the
            # same 16 refreshes observed during cold startup before yaw becomes
            # measurable. Keep two bounded observations of margin, but still
            # fail and stop before the bounded pulse limit. Small corrections
            # retain the shorter fail-closed threshold.
            stall_limit = 18 if terminal_turn else 5
            if stalled_pulses >= stall_limit:
                exit_reason = "stalled"
                break
            accumulated_degrees += observed_degrees
            previous_yaw = current_yaw
            drift = relative_motion_progress(start.to_pose(), current.to_pose())
            drift_limit = 0.32 if terminal_turn else 0.40
            if math.hypot(drift.forward_m, drift.left_m) > drift_limit:
                exit_reason = "drift"
                break
        stationary, settled = self._isaac_stop_and_confirm_stationary()
        if not stationary or settled is None:
            if trace is not None:
                trace.append(f"{exit_reason}:{pulses}p:stationary=false")
            return False, accumulated_degrees, settled
        accumulated_degrees = self._isaac_rotation_progress(
            start,
            settled,
            target_degrees,
        )
        if trace is not None:
            settled_drift = relative_motion_progress(
                start.to_pose(),
                settled.to_pose(),
            )
            trace.append(
                f"{exit_reason}:{pulses}p:{accumulated_degrees:.1f}deg/"
                f"{math.hypot(settled_drift.forward_m, settled_drift.left_m):.2f}m"
            )
        return (
            self._isaac_target_reached(
                accumulated_degrees,
                target_degrees,
                4.0,
            ),
            accumulated_degrees,
            settled,
        )

    def _isaac_stop_and_confirm_stationary(
        self,
    ) -> tuple[bool, Any | None]:
        self._connection.move(Twist.zero())
        stop_completed_at = self._wall_time()
        # The learned Isaac gait can briefly appear stationary, then take one
        # final balancing step 0.5--1.0 s after zero is published.  Two adjacent
        # low-speed frames therefore produce a false stop.  Require a continuous
        # quiet dwell that is longer than the measured rebound window.
        stable_dwell_seconds = 1.5
        # A sustained Brownstone turn remains above the conservative planar
        # threshold for roughly four seconds even though yaw has already
        # stopped. Leave room for that decay plus the full quiet dwell.
        deadline = self._monotonic() + 7.0
        previous: Any | None = None
        stable_since: float | None = None
        latest: Any | None = None
        while self._monotonic() < deadline:
            self._sleep(0.05)
            candidate = self._isaac_transform()
            if candidate is None:
                continue
            timestamp = float(candidate.ts)
            if timestamp <= stop_completed_at:
                continue
            if previous is not None and timestamp <= float(previous.ts):
                continue
            latest = candidate
            if previous is not None:
                elapsed = timestamp - float(previous.ts)
                if elapsed <= 0.0:
                    continue
                dx = float(candidate.translation.x - previous.translation.x)
                dy = float(candidate.translation.y - previous.translation.y)
                planar_speed = math.hypot(dx, dy) / elapsed
                yaw_rate = abs(
                    self._isaac_wrapped_yaw_delta(
                        self._isaac_yaw(candidate),
                        self._isaac_yaw(previous),
                    )
                    / elapsed
                )
                if planar_speed <= 0.025 and yaw_rate <= math.radians(2.0):
                    stable_since = stable_since or self._monotonic()
                else:
                    stable_since = None
                if (
                    stable_since is not None
                    and self._monotonic() - stable_since >= stable_dwell_seconds
                ):
                    return True, candidate
            previous = candidate
        return False, latest

    @skill(uses=[CAP_MOVEMENT])
    def relative_move(
        self,
        forward: float = 0.0,
        left: float = 0.0,
        degrees: float = 0.0,
    ) -> str:
        """Move relative to the current pose using short Isaac feedback pulses.

        Args:
            forward: Signed forward distance in metres, limited to 1 metre.
            left: Signed leftward distance in metres, limited to 1 metre.
            degrees: Signed counter-clockwise rotation, limited to 180 degrees.
        """
        arguments = {
            "forward": forward,
            "left": left,
            "degrees": degrees,
        }
        from harness.robots.g1.isaac.motion_service import (
            IsaacMotionControlChannel,
            encode_motion_result,
        )

        channel = IsaacMotionControlChannel()

        def execute_local(cancelled: Any) -> str:
            self._luxi_motion_cancel_probe = cancelled
            try:
                return self._isaac_relative_move(
                    forward=forward,
                    left=left,
                    degrees=degrees,
                    translation_tolerance=0.08,
                )
            finally:
                self._luxi_motion_cancel_probe = None

        unified = channel.request_relative_move(arguments, execute_local)
        if unified is not None:
            return encode_motion_result(unified)
        return execute_local(lambda: False)

    @skill(uses=[CAP_MOVEMENT])
    def turn_around(self) -> str:
        """Turn left by 180 degrees and accept only fresh, stationary odometry."""

        from harness.robots.g1.isaac.motion_service import (
            IsaacMotionControlChannel,
            encode_motion_result,
        )

        channel = IsaacMotionControlChannel()

        def execute_local(cancelled: Any) -> str:
            self._luxi_motion_cancel_probe = cancelled
            try:
                return self._isaac_turn_around()
            finally:
                self._luxi_motion_cancel_probe = None

        unified = channel.request_turn_around(execute_local)
        if unified is not None:
            return encode_motion_result(unified)
        return execute_local(lambda: False)

    def _isaac_turn_around(self) -> str:
        """Run the existing bounded terminal-turn controller after admission."""

        return self._isaac_relative_move(
            forward=0.0,
            left=0.0,
            degrees=180.0,
            # Brownstone's learned in-place gait settles within 23 cm and
            # 10.3 degrees across repeated 180-degree trials. Keep small
            # margins above those measurements; ordinary relative_move
            # retains its tighter 8 cm / 4 degree tolerances.
            translation_tolerance=0.25,
            rotation_tolerance=12.0,
        )

    def _isaac_relative_move(
        self,
        *,
        forward: float,
        left: float,
        degrees: float,
        translation_tolerance: float,
        rotation_tolerance: float | None = None,
    ) -> str:
        forward, left, degrees = (float(forward), float(left), float(degrees))
        if not all(math.isfinite(value) for value in (forward, left, degrees)):
            return "Simulation motion rejected: every target must be finite"
        if abs(forward) > 1.0 or abs(left) > 1.0 or abs(degrees) > 180.0:
            return "Simulation motion rejected: relative target exceeds 1m/180deg limits"

        start = self._isaac_transform()
        if (
            start is None
            or self._wall_time() - float(start.ts) < -0.25
            or self._wall_time() - float(start.ts) > 1.5
        ):
            self._connection.move(Twist.zero())
            return "relative_move_failed: fresh Isaac odometry is unavailable"
        translation_distance = math.hypot(forward, left)
        operation_budget = max(
            20.0,
            12.0 + 28.0 * translation_distance + abs(degrees) / 3.0,
        )
        terminal_turn = abs(degrees) >= 120.0 and not (forward or left)
        # The terminal turn has a dedicated 75 s Pipeline deadline. Reserve its
        # final seven seconds for the outer stop barrier while allowing the
        # measured recenter pass to use the preceding three-second margin.
        deadline = self._monotonic() + min(
            68.0 if terminal_turn else 50.0,
            operation_budget,
        )

        translation_reached, progress, current = self._isaac_translation_loop(
            start,
            forward,
            left,
            deadline,
            tolerance=translation_tolerance,
        )
        rotation_progress = (
            self._isaac_rotation_progress(start, current, degrees)
            if current is not None
            else 0.0
        )
        if rotation_tolerance is None:
            rotation_tolerance = 6.0 if abs(degrees) < 1e-6 else 4.0
        rotation_reached = self._isaac_target_reached(
            rotation_progress,
            degrees,
            rotation_tolerance,
        )
        # A large Isaac turn can pause while the learned gait enters its yaw
        # gait. Keep this one terminal skill call, but restart at most four
        # measured turn bursts. Ordinary corrections retain bounded creep;
        # the terminal 180-degree branch stays in place.
        previous_rotation_error = abs(degrees - rotation_progress)
        rotation_trace: list[str] = []
        for _ in range(4):
            if not translation_reached or rotation_reached or current is None:
                break
            rotation_reached, rotation_progress, current = self._isaac_rotation_loop(
                start,
                degrees,
                deadline,
                rotation_trace,
                brake_early=terminal_turn and not rotation_trace,
            )
            if current is None:
                break
            progress = relative_motion_progress(start.to_pose(), current.to_pose())
            intermediate_error = math.hypot(
                forward - progress.forward_m,
                left - progress.left_m,
            )
            recenter_tolerance = (
                translation_tolerance
                if rotation_reached
                else max(translation_tolerance, 0.12)
            )
            if intermediate_error > recenter_tolerance:
                translation_reached, progress, current = self._isaac_translation_loop(
                    start,
                    forward,
                    left,
                    deadline,
                    tolerance=recenter_tolerance,
                )
            else:
                translation_reached = True
            if not translation_reached or current is None:
                break
            rotation_progress = self._isaac_rotation_progress(
                start,
                current,
                degrees,
            )
            rotation_reached = self._isaac_target_reached(
                rotation_progress,
                degrees,
                rotation_tolerance,
            )
            rotation_error = abs(degrees - rotation_progress)
            if rotation_reached:
                break
            if previous_rotation_error - rotation_error < 2.0:
                break
            previous_rotation_error = rotation_error
        # Turning creep is deliberate but not part of the requested
        # translation.  Alternate bounded position and yaw corrections; every
        # pass is measured from the same original pose and shares one deadline.
        for _ in range(3):
            if not rotation_reached or current is None:
                break
            translation_reached, progress, current = self._isaac_translation_loop(
                start,
                forward,
                left,
                deadline,
                tolerance=translation_tolerance,
            )
            if not translation_reached or current is None:
                break
            rotation_progress = self._isaac_rotation_progress(
                start,
                current,
                degrees,
            )
            rotation_reached = self._isaac_target_reached(
                rotation_progress,
                degrees,
                rotation_tolerance,
            )
            if rotation_reached:
                break
            rotation_reached, rotation_progress, current = self._isaac_rotation_loop(
                start,
                degrees,
                deadline,
                rotation_trace,
                brake_early=False,
            )

        stationary, final_transform = self._isaac_stop_and_confirm_stationary()
        cancelled = self._isaac_motion_cancelled()
        if final_transform is not None:
            progress = relative_motion_progress(
                start.to_pose(),
                final_transform.to_pose(),
            )
            rotation_progress = self._isaac_rotation_progress(
                start,
                final_transform,
                degrees,
            )
        translation_reached = (
            math.hypot(
                forward - progress.forward_m,
                left - progress.left_m,
            )
            <= translation_tolerance + 1e-9
        )
        rotation_reached = self._isaac_target_reached(
            rotation_progress,
            degrees,
            rotation_tolerance,
        )
        if translation_reached and rotation_reached and stationary and not cancelled:
            status = "turn_verified" if abs(degrees) > 4.0 and not (forward or left) else "relative_move_verified"
            return (
                f"{status}: stationary_confirmed=true, "
                f"forward={progress.forward_m:.2f}/{forward:.2f}m, "
                f"left={progress.left_m:.2f}/{left:.2f}m, "
                f"rotation={rotation_progress:.1f}/{degrees:.1f}deg"
            )
        status = "turn_failed" if abs(degrees) > 4.0 and not rotation_reached else "relative_move_failed"
        return (
            f"{status}: stationary_confirmed="
            f"{'true' if stationary else 'false'}, "
            f"cancelled={'true' if cancelled else 'false'}, "
            f"forward={progress.forward_m:.2f}/{forward:.2f}m, "
            f"left={progress.left_m:.2f}/{left:.2f}m, "
            f"rotation={rotation_progress:.1f}/{degrees:.1f}deg, "
            f"rotation_attempts={';'.join(rotation_trace) or 'none'}"
        )

    @skill(uses=[CAP_MOVEMENT])
    def move_distance(self, distance_m: float) -> str:
        """Execute one 0.1–3.0 m straight request without model round trips.

        Args:
            distance_m: Signed relative distance in metres. Positive is forward.
        """

        from harness.robots.g1.isaac.motion_service import (
            IsaacMotionControlChannel,
            encode_motion_result,
        )

        channel = IsaacMotionControlChannel()

        def execute_local(cancelled: Any) -> str:
            self._luxi_motion_cancel_probe = cancelled
            try:
                return self._isaac_move_distance(distance_m)
            finally:
                self._luxi_motion_cancel_probe = None

        unified = channel.request_move_distance(
            {"distance_m": distance_m},
            execute_local,
        )
        if unified is not None:
            return encode_motion_result(unified)
        return execute_local(lambda: False)

    def _isaac_move_distance(self, distance_m: float) -> str:
        """Run the existing measured terminal controller after admission."""

        try:
            requested = float(distance_m)
        except (TypeError, ValueError, OverflowError):
            return "move_distance_rejected: distance_m must be finite"
        if not math.isfinite(requested):
            return "move_distance_rejected: distance_m must be finite"
        if not 0.10 <= abs(requested) <= 3.0:
            return "move_distance_rejected: absolute distance must be 0.10 to 3.00m"

        start = self._isaac_transform()
        if (
            start is None
            or self._wall_time() - float(start.ts) < -0.25
            or self._wall_time() - float(start.ts) > 1.5
        ):
            self._connection.move(Twist.zero())
            return "move_distance_failed: fresh Isaac odometry is unavailable"

        direction = math.copysign(1.0, requested)
        target = abs(requested)
        # The reference gait has a measured 0.12 m settled coast after the
        # second segment of a 2 m request. Scale the whole-task envelope with
        # distance while retaining the existing 0.15 m hard ceiling.
        tolerance = max(0.08, min(0.15, target * 0.075))
        last_progress = 0.0
        segments = 0
        current = start

        while segments < 6:
            overall = relative_motion_progress(start.to_pose(), current.to_pose())
            directed_progress = direction * overall.forward_m
            remaining = target - directed_progress
            if abs(remaining) <= tolerance:
                break
            if remaining < -tolerance:
                self._connection.move(Twist.zero())
                return (
                    "move_distance_failed: target overshot, "
                    f"actual={direction * directed_progress:.2f}/"
                    f"{requested:.2f}m"
                )
            segment_target = min(1.0, max(0.10, remaining))
            if segments > 0 and remaining <= 1.0:
                # Brownstone's settled gait coasts 0.12–0.16 m after the last
                # nonzero refresh. Reserve part of that measured coast in the
                # final segment target instead of weakening the 0.15 m task
                # acceptance ceiling. Keep the local target just outside its
                # own tolerance so a tiny residual still produces a measured
                # correction instead of a zero-progress verified segment.
                segment_target = max(0.10, tolerance + 0.02, remaining - 0.10)
            segment_tolerance = tolerance if remaining <= 1.0 else 0.08
            result = self._isaac_relative_move(
                forward=direction * segment_target,
                left=0.0,
                degrees=0.0,
                translation_tolerance=segment_tolerance,
            )
            segments += 1
            candidate = self._isaac_transform()
            if candidate is None or float(candidate.ts) <= float(current.ts):
                self._connection.move(Twist.zero())
                return "move_distance_failed: fresh odometry stopped advancing"
            current = candidate
            overall = relative_motion_progress(start.to_pose(), current.to_pose())
            directed_progress = direction * overall.forward_m
            segment_progress = directed_progress - last_progress
            if segment_progress < 0.01:
                self._connection.move(Twist.zero())
                return "move_distance_failed: verified segment made no overall progress"
            if abs(overall.left_m) > 0.30:
                self._connection.move(Twist.zero())
                return (
                    "move_distance_failed: cumulative cross-track error "
                    f"{abs(overall.left_m):.2f}m exceeds 0.30m"
                )
            terminal_distance_reached = abs(target - directed_progress) <= tolerance
            segment_verified = "relative_move_verified:" in result
            # A learned-gait stop can coast just beyond a local one-metre
            # boundary while still making safe, stationary progress toward
            # the complete request. Judge that nonterminal handoff against
            # the global route instead of treating the local tolerance miss
            # as a terminal task failure.
            safe_nonterminal_progress = (
                not terminal_distance_reached
                and "relative_move_failed: stationary_confirmed=true" in result
                and segment_progress >= max(0.10, segment_target * 0.75)
                and directed_progress < target + tolerance
                and abs(overall.rotation_degrees) <= 6.0
            )
            if (
                not segment_verified
                and not (
                    terminal_distance_reached
                    and "relative_move_failed: stationary_confirmed=true" in result
                )
                and not safe_nonterminal_progress
            ):
                self._connection.move(Twist.zero())
                return (
                    f"move_distance_failed: segment={segments}, "
                    f"target={requested:.2f}m; {result}"
                )
            last_progress = directed_progress
            if terminal_distance_reached:
                break
        else:
            self._connection.move(Twist.zero())
            return "move_distance_failed: correction segment limit reached"

        heading_results: list[str] = []
        for _ in range(3):
            preliminary = relative_motion_progress(start.to_pose(), current.to_pose())
            previous_heading_error = abs(preliminary.rotation_degrees)
            if previous_heading_error <= 6.0:
                break
            heading_result = self._isaac_relative_move(
                forward=0.0,
                left=0.0,
                degrees=-preliminary.rotation_degrees,
                translation_tolerance=0.08,
            )
            heading_results.append(heading_result)
            corrected = self._isaac_transform()
            if corrected is None or float(corrected.ts) <= float(current.ts):
                self._connection.move(Twist.zero())
                return "move_distance_failed: heading correction odometry is stale"
            current = corrected
            corrected_progress = relative_motion_progress(
                start.to_pose(),
                current.to_pose(),
            )
            if (
                abs(corrected_progress.rotation_degrees)
                >= previous_heading_error - 0.2
            ):
                break

        stationary, final_transform = self._isaac_stop_and_confirm_stationary()
        if final_transform is None:
            return "move_distance_failed: final odometry is unavailable"
        final = relative_motion_progress(start.to_pose(), final_transform.to_pose())
        actual = direction * final.forward_m
        distance_error = target - actual
        if (
            not stationary
            or abs(distance_error) > tolerance
            or abs(final.left_m) > 0.30
            or abs(final.rotation_degrees) > 6.0
        ):
            return (
                "move_distance_failed: "
                f"stationary_confirmed={'true' if stationary else 'false'}, "
                f"actual={direction * actual:.2f}/{requested:.2f}m, "
                f"cross_track={final.left_m:.2f}m, "
                f"rotation={final.rotation_degrees:.1f}deg, "
                f"segments={segments}, "
                f"heading_attempts={len(heading_results)}, "
                "do_not_retry=true"
            )
        return (
            "distance_verified: stationary_confirmed=true, "
            f"actual={direction * actual:.2f}/{requested:.2f}m, "
            f"error={direction * distance_error:.2f}m, "
            f"cross_track={final.left_m:.2f}m, "
            f"rotation={final.rotation_degrees:.1f}deg, "
            f"segments={segments}, "
            f"heading_attempts={len(heading_results)}"
        )


# Compatibility aliases for callers outside this repository. New composition
# code uses the Adapter names so the task layer is not mistaken for a second
# simulator-specific skill implementation.
G1SimulationSkillContainer = G1MujocoAdapter
G1IsaacSimulationSkillContainer = G1IsaacAdapter
