"""Candidate-only settled-pose refinement over a freshly observed free corridor."""
import math
import time
from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD


def refine_arrival(owner, goal, expires_at, *, require_heading=True):
    from harness.skills.rgbd_skills import LiveCostmapRiskFeed, LiveCostmapRiskMonitor
    monitor = LiveCostmapRiskMonitor(max_age_seconds=.75)
    feed = LiveCostmapRiskFeed(monitor)

    def corridor_clear(odom):
        stamp = getattr(odom, 'ts', None)
        if not isinstance(stamp, (float, int)) or not 0 <= time.time()-stamp <= .75:
            return False
        x, y = float(odom.position.x), float(odom.position.y)
        dx, dy = float(goal.position.x)-x, float(goal.position.y)-y
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance) or distance > .35:
            return False
        # Robot radius 0.30 m plus 0.10 m margin; recheck the whole remaining segment every tick.
        count = max(1, math.ceil(distance/.05))
        return all(monitor.goal_region_state(x=x+dx*i/count, y=y+dy*i/count,
                                             clearance_m=.40) == 'clear' for i in range(count+1))

    feed.start()
    try:
        # Receive a new map under a zero command, without manufacturing free cells.
        until = min(expires_at, time.time()+2)
        while time.time() < until:
            odom = getattr(owner, '_latest_odom', None)
            if odom is not None and corridor_clear(odom):
                return owner._refine_manipulation_pose(
                    goal, timeout=max(.01, min(20, expires_at-time.time())),
                    position_tolerance_m=POSITION_TOLERANCE_M,
                    heading_tolerance_degrees=math.degrees(YAW_TOLERANCE_RAD),
                    translation_guard=corridor_clear, require_heading=require_heading)
            if owner._long_task_cancel_reason() is not None:
                return {'operation_ok': False, 'task_status': 'cancelled'}
            time.sleep(.05)
        return {'operation_ok': False, 'task_status': 'risk_blocked',
                'error': 'settled-pose refinement lacks a fresh footprint-clear corridor'}
    finally:
        feed.stop()
