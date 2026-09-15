"""Bounded person-follow state machine shared by simulator adapters.

The execution model follows the ``ansheng`` branch: one semantic acquisition,
task-local CSRT tracking, real RGB-D cadence accounting, fail-closed risk
handling, and post-stop verification. Simulator details live behind
``PersonFollowIO``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Protocol

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image

from harness.skills.task_primitives import (
    CsrtTargetTracker,
    FollowDistanceAcquisition,
    FollowControlConfig,
    PersonFollowResult,
    TrackingMeasurement,
    compute_follow_twist,
    stable_person_tracking_bbox,
)


@dataclass(frozen=True)
class FollowRuntimeConfig:
    control_hz: float = 10.0
    frame_hold_s: float = 0.35
    tracking_lost_s: float = 0.75
    sensor_gap_timeout_s: float = 3.0
    unknown_risk_timeout_s: float = 2.0
    verification_timeout_s: float = 1.0
    distance_tolerance_m: float = 0.35
    max_final_angle_degrees: float = 30.0
    min_tracking_coverage: float = 0.9
    max_stop_latency_s: float = 0.5
    # The reference Isaac learned gait can coast for roughly 1.8 s after its
    # last forward command. Continue identity tracking while holding zero long
    # enough that the *subsequent* final stop barrier can still collect two
    # new stationary odometry frames inside 500 ms.
    stop_settle_lead_s: float = 2.25


@dataclass(frozen=True)
class FollowAcquisition:
    image: Image
    bbox: tuple[float, float, float, float]
    frame_timestamp: float
    initial_distance_m: float


@dataclass(frozen=True)
class FollowStopEvidence:
    stop_command_publish_latency_ms: float
    physical_stop_latency_ms: float | None
    stop_command_completed_at: float | None
    stationary_confirmed_at: float | None


class PersonFollowIO(Protocol):
    def follow_risk_state(self, *, translating: bool) -> str: ...

    def follow_stop_evidence(self) -> FollowStopEvidence: ...

    def acquire_follow_target(
        self,
        query: str,
        *,
        after: float,
        deadline: float,
    ) -> FollowAcquisition | None: ...

    def vision_request_metadata(self) -> dict[str, Any]: ...

    def follow_viewpoint_search(self, *, deadline: float) -> float | None: ...

    def follow_cancelled(self) -> bool: ...

    def wait_for_follow_pair(
        self,
        *,
        after: float,
        timeout: float,
    ) -> tuple[Image, Image] | None: ...

    def follow_measurement(
        self,
        bbox: tuple[float, float, float, float],
        pair: tuple[Image, Image],
    ) -> TrackingMeasurement | None: ...

    def publish_follow_command(self, command: Twist) -> None: ...


class PersonFollowExecutor:
    """Run one terminal follow task using the reference branch state machine."""

    def __init__(
        self,
        io: PersonFollowIO,
        *,
        runtime: FollowRuntimeConfig = FollowRuntimeConfig(),
        control: FollowControlConfig = FollowControlConfig(),
        tracker_factory: Callable[[], CsrtTargetTracker] = CsrtTargetTracker,
    ) -> None:
        self.io = io
        self.runtime = runtime
        self.control = control
        self.tracker_factory = tracker_factory

    def run(
        self,
        query: str,
        *,
        follow_distance: float,
        duration: float,
        timeout: float,
    ) -> PersonFollowResult:
        started = time.monotonic()
        deadline = started + timeout
        tracked_frames = 0
        verified_duration = 0.0
        max_gap = 0.0
        tracking_coverage = 0.0
        last_measurement: TrackingMeasurement | None = None
        last_valid_at: float | None = None
        last_processed_frame_at: float | None = None
        last_command = Twist()
        tracking_started: float | None = None
        tracking_finished_at: float | None = None
        last_accounted_at: float | None = None
        stop_evidence: FollowStopEvidence | None = None

        def refresh_tracking_metrics(now: float) -> None:
            nonlocal last_accounted_at
            nonlocal tracking_coverage
            nonlocal verified_duration
            if tracking_started is None or last_accounted_at is None:
                return
            metric_now = (
                min(now, tracking_finished_at)
                if tracking_finished_at is not None
                else now
            )
            if metric_now <= last_accounted_at:
                return
            if last_valid_at is not None:
                fresh_until = last_valid_at + self.runtime.frame_hold_s
                verified_duration += max(
                    0.0,
                    min(metric_now, fresh_until) - last_accounted_at,
                )
            last_accounted_at = metric_now
            tracking_elapsed = max(0.0, metric_now - tracking_started)
            tracking_coverage = (
                min(1.0, verified_duration / tracking_elapsed)
                if tracking_elapsed > 0.0
                else 1.0
            )

        def stop() -> FollowStopEvidence:
            nonlocal stop_evidence
            if stop_evidence is None:
                stop_evidence = self.io.follow_stop_evidence()
            return stop_evidence

        def finish(
            status: str,
            reason: str,
            message: str,
            *,
            tool_ok: bool = True,
            verification: TrackingMeasurement | None = None,
        ) -> PersonFollowResult:
            refresh_tracking_metrics(time.monotonic())
            evidence = stop()
            final_measurement = verification or last_measurement
            return PersonFollowResult(
                tool_ok=tool_ok,
                task_status=status,
                completed=status == "follow_verified",
                requested_follow_duration_s=duration,
                verified_tracking_duration_s=verified_duration,
                elapsed_s=max(0.0, time.monotonic() - started),
                requested_follow_distance_m=follow_distance,
                target_distance_m=(
                    final_measurement.distance_m
                    if final_measurement is not None
                    else None
                ),
                target_bearing_degrees=(
                    math.degrees(final_measurement.bearing_radians)
                    if final_measurement is not None
                    else None
                ),
                tracked_frames=tracked_frames,
                tracking_coverage=tracking_coverage,
                max_tracking_gap_s=max_gap,
                stop_command_publish_latency_ms=(
                    evidence.stop_command_publish_latency_ms
                ),
                physical_stop_latency_ms=evidence.physical_stop_latency_ms,
                stop_command_completed_at=evidence.stop_command_completed_at,
                stationary_confirmed_at=evidence.stationary_confirmed_at,
                verification_frame_timestamp=(
                    verification.frame_timestamp
                    if verification is not None
                    else None
                ),
                termination_reason=reason,
                message=message,
            )

        initial_risk = self.io.follow_risk_state(translating=False)
        if initial_risk not in {"clear", "warning"}:
            return finish(
                "risk_blocked",
                f"risk_{initial_risk}",
                f"跟随只允许在 clear 或 warning 风险下启动，当前为 {initial_risk}",
            )

        acquisition_deadline = deadline - duration - 2.0
        if acquisition_deadline - time.monotonic() < 3.0:
            return finish(
                "follow_timeout",
                "insufficient_acquisition_budget",
                "整段时间预算不足以完成识别、跟随和停车验证",
            )
        acquired = self.io.acquire_follow_target(
            query,
            after=0.0,
            deadline=acquisition_deadline,
        )
        if acquired is None:
            stage = str(
                self.io.vision_request_metadata().get("localization_stage") or ""
            )
            if stage not in {"bbox_not_found", "depth_localization_failed"}:
                return finish(
                    "follow_timeout" if stage == "time_budget" else "target_not_found",
                    stage or "initial_acquisition_failed",
                    "没有取得可用于跟随的人物 RGB-D 定位",
                )
            stationary_at = self.io.follow_viewpoint_search(
                deadline=acquisition_deadline - 3.0,
            )
            if stationary_at is None:
                return finish(
                    "target_not_found",
                    "viewpoint_search_failed",
                    "首次未发现人物，受控转向也未取得新的安全视角",
                )
            acquired = self.io.acquire_follow_target(
                query,
                after=stationary_at,
                deadline=acquisition_deadline,
            )
            if acquired is None:
                stage = str(
                    self.io.vision_request_metadata().get("localization_stage")
                    or ""
                )
                return finish(
                    "follow_timeout" if stage == "time_budget" else "target_not_found",
                    stage or "second_acquisition_failed",
                    "转向后的第二次有界识别仍未找到人物",
                )

        if self.io.follow_cancelled():
            return finish("cancelled", "cancelled_before_tracking", "跟随已取消")

        tracking_bbox = stable_person_tracking_bbox(acquired.bbox)
        tracker = self.tracker_factory()
        if not tracker.initialize(acquired.image, tracking_bbox):
            return finish(
                "tracking_init_failed",
                "csrt_initialization_failed",
                "人物已定位，但 CSRT 无法在识别帧上建立跟踪",
            )
        current_pair = self.io.wait_for_follow_pair(
            after=acquired.frame_timestamp,
            timeout=self.runtime.tracking_lost_s,
        )
        if current_pair is None:
            return finish(
                "tracking_init_failed",
                "fresh_rgbd_unavailable",
                "跟踪初始化后没有取得更新的 RGB-D 帧",
            )
        current_bbox = tracker.update(current_pair[0])
        measurement = (
            self.io.follow_measurement(current_bbox, current_pair)
            if current_bbox is not None
            else None
        )
        if (
            measurement is None
            or abs(measurement.distance_m - acquired.initial_distance_m) > 0.75
        ):
            return finish(
                "tracking_init_failed",
                "detection_frame_continuity_failed",
                "无法证明识别帧与当前人物位置连续，拒绝启动运动",
            )

        period = 1.0 / self.runtime.control_hz
        now = time.monotonic()
        distance_acquisition = FollowDistanceAcquisition(
            target_distance_m=follow_distance,
            tolerance_m=min(
                self.runtime.distance_tolerance_m,
                self.control.distance_deadband_m,
            ),
            required_samples=3,
        )
        distance_acquired = distance_acquisition.observe(measurement, now=now)
        tracking_started = now if distance_acquired else None
        last_valid_at = now
        last_accounted_at = now if distance_acquired else None
        last_processed_frame_at = measurement.frame_timestamp
        last_measurement = measurement
        tracked_frames = 1
        tracking_coverage = 1.0 if distance_acquired else 0.0
        unknown_since: float | None = None
        next_tick = now
        last_continuity_failure = "tracker_lost"

        while time.monotonic() < deadline:
            now = time.monotonic()
            refresh_tracking_metrics(now)
            if self.io.follow_cancelled():
                return finish("cancelled", "cancelled", "跟随已取消")
            translating = abs(float(last_command.linear.x)) > 1e-6
            risk = self.io.follow_risk_state(translating=translating)
            if risk == "critical":
                return finish(
                    "risk_blocked",
                    "risk_critical",
                    "当前雷达确认 critical，跟随已终止",
                )
            if risk in {"unknown", "transient"}:
                self.io.publish_follow_command(Twist())
                last_command = Twist()
                if risk == "unknown":
                    unknown_since = unknown_since or now
                    if now - unknown_since >= self.runtime.unknown_risk_timeout_s:
                        return finish(
                            "risk_blocked",
                            "risk_unknown_timeout",
                            "当前雷达证据持续 unknown，跟随失败关闭",
                        )
                else:
                    unknown_since = None
                time.sleep(min(period, 0.05))
                continue
            unknown_since = None

            wait_timeout = min(period, max(0.0, deadline - now))
            if last_valid_at is not None:
                gap = max(0.0, now - last_valid_at)
                next_boundary = (
                    self.runtime.frame_hold_s
                    if gap < self.runtime.frame_hold_s
                    else self.runtime.sensor_gap_timeout_s
                )
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, next_boundary - gap),
                )
            pair = self.io.wait_for_follow_pair(
                after=last_processed_frame_at or 0.0,
                timeout=wait_timeout,
            )
            now = time.monotonic()
            refresh_tracking_metrics(now)
            new_valid_frame = False
            if pair is None:
                gap = max(0.0, now - (last_valid_at or now))
                max_gap = max(max_gap, gap)
                if gap >= self.runtime.frame_hold_s:
                    self.io.publish_follow_command(Twist())
                    last_command = Twist()
                if gap >= self.runtime.sensor_gap_timeout_s:
                    return finish(
                        "observation_unavailable",
                        "rgbd_stream_timeout",
                        "第一人称 RGB-D 数据流超时，已停车；没有把传感器中断判成人物丢失",
                    )
            else:
                last_processed_frame_at = float(pair[1].ts)
                bbox = tracker.update(pair[0])
                last_continuity_failure = (
                    "tracker_lost" if bbox is None else "depth_localization_lost"
                )
                candidate = (
                    self.io.follow_measurement(bbox, pair)
                    if bbox is not None
                    else None
                )
                if (
                    candidate is not None
                    and last_measurement is not None
                    and abs(candidate.distance_m - last_measurement.distance_m)
                    > 0.75
                ):
                    last_continuity_failure = "depth_distance_discontinuity"
                    candidate = None
                if (
                    candidate is None
                    and bbox is None
                    and last_measurement is not None
                    and last_valid_at is not None
                    and now - last_valid_at < self.runtime.tracking_lost_s
                ):
                    # A gait animation or one dropped frame can invalidate
                    # CSRT's appearance model while the same person remains
                    # at the last proven image location. Reinitialize only
                    # when current aligned depth at that exact prior box is
                    # still continuous; this never runs a detector or selects
                    # another person.
                    recovery_bbox = last_measurement.bbox
                    recovery = self.io.follow_measurement(recovery_bbox, pair)
                    if (
                        recovery is not None
                        and abs(
                            recovery.distance_m - last_measurement.distance_m
                        )
                        <= 0.35
                    ):
                        replacement = self.tracker_factory()
                        if replacement.initialize(pair[0], recovery_bbox):
                            tracker = replacement
                            bbox = recovery_bbox
                            candidate = recovery
                            last_continuity_failure = "tracker_reinitialized"
                if candidate is None:
                    self.io.publish_follow_command(Twist())
                    last_command = Twist()
                    gap = max(0.0, now - (last_valid_at or now))
                    max_gap = max(max_gap, gap)
                    if gap >= self.runtime.tracking_lost_s:
                        return finish(
                            "tracking_lost",
                            last_continuity_failure,
                            (
                                "同一人物的 CSRT 框或深度连续性丢失，"
                                "已停车且没有重新选择人物"
                            ),
                        )
                else:
                    if last_valid_at is not None:
                        max_gap = max(max_gap, max(0.0, now - last_valid_at))
                    last_valid_at = now
                    last_measurement = candidate
                    tracked_frames += 1
                    new_valid_frame = True
                    if (
                        tracking_started is None
                        and distance_acquisition.observe(candidate, now=now)
                    ):
                        tracking_started = now
                        last_accounted_at = now
                        tracking_coverage = 1.0
                    # First acquire the requested distance band. Then keep
                    # collecting fresh identity/distance evidence while
                    # settling the learned gait near the end of the requested
                    # duration. This preserves the 500 ms final-stop boundary.
                    command = (
                        Twist()
                        if tracking_started is not None
                        and duration - verified_duration
                        <= self.runtime.stop_settle_lead_s
                        else compute_follow_twist(
                            candidate,
                            follow_distance_m=follow_distance,
                            risk=risk,
                            config=self.control,
                        )
                    )
                    self.io.publish_follow_command(command)
                    last_command = command

            if (
                new_valid_frame
                and tracking_started is not None
                and verified_duration >= duration
                and tracking_coverage >= self.runtime.min_tracking_coverage
            ):
                tracking_finished_at = now
                break
            next_tick += period
            remaining_sleep = next_tick - time.monotonic()
            if remaining_sleep > 0.0:
                time.sleep(remaining_sleep)
            else:
                next_tick = time.monotonic()
        else:
            return finish(
                "follow_timeout",
                "whole_skill_deadline",
                "整段跟随时间预算已耗尽",
            )

        evidence = stop()
        if (
            evidence.stationary_confirmed_at is None
            or evidence.physical_stop_latency_ms is None
            or evidence.physical_stop_latency_ms
            > self.runtime.max_stop_latency_s * 1_000.0
        ):
            return finish(
                "verification_failed",
                "stationary_confirmation_failed",
                "停车命令已发布，但没有在平台时限内确认连续静止",
            )
        verification_pair = self.io.wait_for_follow_pair(
            after=evidence.stationary_confirmed_at,
            timeout=self.runtime.verification_timeout_s,
        )
        verification_bbox = (
            tracker.update(verification_pair[0])
            if verification_pair is not None
            else None
        )
        verification = (
            self.io.follow_measurement(verification_bbox, verification_pair)
            if verification_bbox is not None and verification_pair is not None
            else None
        )
        if verification is None:
            return finish(
                "verification_failed",
                "post_stop_tracking_unavailable",
                "静止后没有取得更新的人物 RGB-D 跟踪证据",
            )
        distance_ok = (
            abs(verification.distance_m - follow_distance)
            <= self.runtime.distance_tolerance_m
        )
        angle_ok = (
            abs(math.degrees(verification.bearing_radians))
            <= self.runtime.max_final_angle_degrees
        )
        if not distance_ok or not angle_ok:
            return finish(
                "verification_failed",
                "final_distance_or_bearing",
                (
                    f"最终距离 {verification.distance_m:.2f}m 或朝向误差 "
                    f"{abs(math.degrees(verification.bearing_radians)):.1f}deg 未通过"
                ),
                verification=verification,
            )
        return finish(
            "follow_verified",
            "duration_and_post_stop_rgbd_verified",
            (
                f"已连续跟随 {verified_duration:.1f}s，最终距离 "
                f"{verification.distance_m:.2f}m，并取得停车后新 RGB-D 证据"
            ),
            verification=verification,
        )
