"""Runtime-only visual bindings, separate from confirmed semantic goals."""
import math
import time
import uuid
from harness.runtime.composition_goals import pose_value

VISUAL_MAX_AGE_S = 12.0
BINDING_MAX_AGE_S = 120.0


def is_visual(condition):
    return condition.get('target_source') == 'visual'


def binding_for(task, condition, state):
    binding = state.get('object_bindings', {}).get(condition['id'])
    if not binding or binding.get('invalidated') or binding.get('world_revision') != task.world_revision:
        return None
    if binding.get('entity_id') != condition.get('entity_id') or binding.get('target') != condition['target']:
        return None
    if not 0 <= time.monotonic()-binding['bound_at'] <= BINDING_MAX_AGE_S:
        return None
    return binding


def target_pose(task, condition, state):
    if not is_visual(condition):
        return task.references[condition['target']]
    binding = binding_for(task, condition, state)
    return binding['pickup_pose'] if binding else None


def validate_visual(task, condition, raw, *, after_wall):
    if raw.get('operation_ok') is not True:
        raise ValueError(raw.get('error', raw.get('task_status', '视觉定位失败')))
    visual = raw.get('visual', {})
    if visual.get('source') != 'robot_rgbd' or visual.get('entity_id') != condition['entity_id']:
        raise ValueError('缺少指定对象的独立 RGB-D 证据')
    stamp, color = visual.get('frame_timestamp'), visual.get('color_timestamp')
    if (type(stamp) not in (int, float) or type(color) not in (int, float)
            or not after_wall < min(stamp, color) <= max(stamp, color) <= time.time()
            or time.time()-min(stamp, color) > VISUAL_MAX_AGE_S or abs(stamp-color) > .25):
        raise ValueError('RGB-D 帧过期、不同步或早于本次定位请求')
    point = pose_value(visual.get('pose'))  # xyz, not a robot navigation pose.
    if (type(visual.get('valid_depth_points')) is not int or visual['valid_depth_points'] < 15
            or type(visual.get('depth_m')) not in (int, float) or not .3 <= visual['depth_m'] <= 8
            or type(visual.get('depth_spread_m')) not in (int, float) or not 0 <= visual['depth_spread_m'] <= .06):
        raise ValueError('没有可靠的目标对象深度证据')
    region = task.visual_regions[condition['target']]
    if math.dist(point[:2], task.references[condition['target']][:2]) > region['search_radius_m']:
        raise ValueError('视觉目标在已确认搜索区域之外')
    yaw = region['approach_yaw']
    # Reviewed right-hand standoff and approach direction; all XY comes from RGB-D.
    forward = math.sqrt(.66**2-.14**2)
    pickup = (point[0]-math.cos(yaw)*forward-math.sin(yaw)*.14,
              point[1]-math.sin(yaw)*forward+math.cos(yaw)*.14, yaw)
    return {'binding_id': uuid.uuid4().hex, 'goal_id': condition['id'],
            'entity_id': condition['entity_id'], 'target': condition['target'],
            'world_revision': task.world_revision, 'bound_at': time.monotonic(),
            'pickup_pose': pickup, 'visual': visual}
