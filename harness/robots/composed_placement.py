"""Backend-neutral, annotated surface and simulated placement evidence contracts."""
import math
import time


def vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def validate_surface(surface):
    """A support point is XYZ on the surface, distinct from robot XY/yaw."""
    if not isinstance(surface, dict):
        raise ValueError('放置表面必须为对象')
    required = {'canonical_name', 'source', 'revision', 'scene_id', 'frame',
                'robot_pose', 'place_point', 'bounds_xy'}
    if set(surface) != required or surface['source'] != 'operator_annotation' or surface['frame'] != 'world':
        raise ValueError('放置表面需要完整可信标注及 world 坐标系')
    if any(not isinstance(surface[k], str) or not surface[k] for k in ('canonical_name','revision','scene_id')):
        raise ValueError('放置表面标识或版本无效')
    if not vector(surface['robot_pose'], 3) or not vector(surface['place_point'], 3) or not vector(surface['bounds_xy'], 4):
        raise ValueError('放置位姿、桌面放物点及边界必须为有限坐标')
    x, y, z = surface['place_point']
    xmin, xmax, ymin, ymax = surface['bounds_xy']
    if not (xmin < x < xmax and ymin < y < ymax and 0 < z <= 2.0):
        raise ValueError('放物点必须位于合法桌面区域内')
    if math.dist(surface['robot_pose'][:2], (x,y)) > 1.0:
        raise ValueError('放物点超出机器人放置位姿的操作范围')
    return surface


def placement_matches(evidence, surface, entity_id):
    """Current geometry and consecutive simulation samples, never just a release ack."""
    return (isinstance(evidence, dict)
            and evidence.get('source') == 'simulation_placement_adapter'
            and type(evidence.get('evidence_timestamp')) in (int, float)
            and 0 <= time.time()-evidence['evidence_timestamp'] <= 1.0
            and evidence.get('entity_id') == entity_id
            and evidence.get('surface') == surface
            and evidence.get('position_valid') is True
            and evidence.get('upright') is True
            and evidence.get('supported') is True
            and evidence.get('clear') is True
            and evidence.get('stationary') is True
            and type(evidence.get('stable_samples')) is int and evidence['stable_samples'] >= 2)
