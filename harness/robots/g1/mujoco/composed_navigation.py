"""Candidate-only settled-pose refinement over a freshly observed free corridor."""
import math
import json
import os
from pathlib import Path
import uuid
import time
from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD
from harness.robots.g1.safety_geometry import NAVIGATION_RADIUS_M


MAX_REFINEMENT_DISTANCE_M = 0.35


def settled_pose_evidence(owner, goal, stop, expires_at, *, require_heading):
    """Read a fresh pose after stationarity, bounded by the original deadline."""
    after = stop.get('stationary_confirmed_at')
    if (stop.get('stationary_confirmed') is not True or isinstance(after, bool)
            or not isinstance(after, (int, float)) or not math.isfinite(after)):
        return {'observation_valid': False, 'reason': 'stationarity_evidence_unavailable'}
    until = min(expires_at, time.time() + .75)
    while time.time() < until:
        if owner._long_task_cancel_reason() is not None:
            return {'observation_valid': False, 'reason': 'cancelled'}
        odom = getattr(owner, '_latest_odom', None)
        stamp = getattr(odom, 'ts', None)
        if (isinstance(stamp, (int, float)) and not isinstance(stamp, bool)
                and math.isfinite(stamp) and after < stamp <= time.time()
                and time.time() - stamp <= .75):
            x, y = float(odom.position.x), float(odom.position.y)
            yaw = float(odom.orientation.to_euler().yaw)
            if all(math.isfinite(v) for v in (x, y, yaw)):
                distance = math.hypot(float(goal.position.x)-x, float(goal.position.y)-y)
                target_yaw = float(goal.orientation.to_euler().yaw)
                yaw_error = abs(math.atan2(math.sin(target_yaw-yaw), math.cos(target_yaw-yaw)))
                return {'observation_valid': True, 'pose_timestamp': stamp,
                        'position_error_m': distance, 'heading_error_rad': yaw_error,
                        'within_tolerance': distance <= POSITION_TOLERANCE_M and
                            (not require_heading or yaw_error <= YAW_TOLERANCE_RAD)}
        time.sleep(.02)
    return {'observation_valid': False, 'reason': 'post_stop_odometry_unavailable'}


def refine_arrival(owner, goal, expires_at, *, require_heading=True):
    from harness.skills.rgbd_skills import LiveCostmapRiskFeed, LiveCostmapRiskMonitor
    monitor = LiveCostmapRiskMonitor(max_age_seconds=1.0)
    feed = LiveCostmapRiskFeed(monitor)

    last_failure = {'reason': 'odom_unavailable', 'captured_at': time.time(),
                    'goal': [float(goal.position.x), float(goal.position.y),
                             float(goal.orientation.to_euler().yaw)],
                    'require_heading': require_heading}

    def corridor_clear(odom):
        nonlocal last_failure
        now = time.time()
        stamp = getattr(odom, 'ts', None)
        detail = {'captured_at': now, 'captured_monotonic': time.monotonic(),
                  'goal': [float(goal.position.x), float(goal.position.y),
                           float(goal.orientation.to_euler().yaw)],
                  'require_heading': require_heading, 'max_odom_age_s': .75,
                  'position_tolerance_m': POSITION_TOLERANCE_M,
                  'heading_tolerance_rad': YAW_TOLERANCE_RAD}
        if odom is not None:
            x, y = float(odom.position.x), float(odom.position.y)
            yaw = float(odom.orientation.to_euler().yaw)
            dx, dy = float(goal.position.x)-x, float(goal.position.y)-y
            distance = math.hypot(dx, dy)
            yaw_error = abs(math.atan2(math.sin(detail['goal'][2]-yaw),
                                       math.cos(detail['goal'][2]-yaw)))
            detail.update(pose=[x, y, yaw], position_error_m=distance,
                          heading_error_rad=yaw_error,
                          within_tolerance=distance <= POSITION_TOLERANCE_M and
                          (not require_heading or yaw_error <= YAW_TOLERANCE_RAD))
        if not isinstance(stamp, (float, int)) or not 0 <= now-stamp <= .75:
            last_failure = {**detail, 'reason': 'odom_stale_or_missing',
                            'odom_age_s': now-stamp if isinstance(stamp, (float, int)) else None}
            return False
        detail['odom_age_s'] = now-stamp
        if not math.isfinite(distance) or distance > MAX_REFINEMENT_DISTANCE_M:
            last_failure = {**detail, 'reason': 'refinement_distance_exceeded'}
            return False
        # Same checks and order; capture the first failing point, not a later map.
        count = max(1, math.ceil(distance/.05))
        for i in range(count+1):
            region = {}
            if monitor.goal_region_state(x=x+dx*i/count, y=y+dy*i/count,
                                         clearance_m=NAVIGATION_RADIUS_M, diagnostics=region) != 'clear':
                last_failure = {**detail, 'reason': region.get('reason', 'corridor_unavailable'),
                                'corridor_index': i, 'corridor_segments': count, 'map': region}
                return False
        return True

    def record_failure(outcome, phase):
        if outcome.get('task_status') != 'risk_blocked':
            return outcome
        detail = {**last_failure, 'schema_version': 1, 'phase': phase,
                  'error': outcome.get('error')}
        summary = {'reason': detail['reason'], 'phase': phase,
                   'within_tolerance': detail.get('within_tolerance')}
        try:
            root = Path(os.environ.get('DIMOS_RUNTIME_DIR',
                        str(Path.home() / 'work/Asset/dimos/runtime')))
            directory = root / 'luxi-ui/diagnostics/refinement'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json')
            # Exclusive creation; one small record per failed invocation, never per tick.
            with path.open('x', encoding='utf-8') as stream:
                json.dump(detail, stream, ensure_ascii=False, indent=2)
            summary['path'] = str(path)
        except (OSError, TypeError, ValueError) as error:
            # Diagnostic I/O must not turn a safety stop into an exception.
            summary['write_error'] = str(error)
        return {**outcome, 'refinement_diagnostic': summary}

    feed.start()
    try:
        # Receive a new map under a zero command, without manufacturing free cells.
        until = min(expires_at, time.time()+2)
        while time.time() < until:
            if owner._long_task_cancel_reason() is not None:
                return {'operation_ok': False, 'task_status': 'cancelled'}
            odom = getattr(owner, '_latest_odom', None)
            if odom is not None and corridor_clear(odom):
                outcome = owner._refine_manipulation_pose(
                    goal, timeout=max(.01, min(20, expires_at-time.time())),
                    position_tolerance_m=POSITION_TOLERANCE_M,
                    heading_tolerance_degrees=math.degrees(YAW_TOLERANCE_RAD),
                    translation_guard=corridor_clear,
                    translation_guard_retryable=lambda: last_failure.get('reason') == 'map_stale',
                    require_heading=require_heading)
                return record_failure(outcome, "refinement")
            if last_failure.get('reason') == 'refinement_distance_exceeded':
                break
            time.sleep(.05)
        return record_failure({'operation_ok': False, 'task_status': 'risk_blocked',
                'error': 'settled-pose refinement lacks a fresh footprint-clear corridor'}, 'admission')
    finally:
        feed.stop()
