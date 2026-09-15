"""Reviewed place annotations; never obtain object coordinates from scene geometry."""
import json
import math
from .composition_goals import pose_value


def location_catalog(path, scene_id):
    data = json.loads(path.read_text())
    if data['source'] != 'operator_annotation' or data['scene_id'] != scene_id:
        return {'references': {}, 'visual_regions': {}}
    references, regions = {}, {}
    for name, location in data['locations'].items():
        pose = pose_value(location['observation_pose'])
        radius, yaw = location['search_radius_m'], location['approach_yaw']
        if (type(radius) not in (float, int) or not 0 < radius <= 3
                or type(yaw) not in (float, int) or not math.isfinite(yaw)):
            raise ValueError('Invalid annotated search region')
        for alias in (name, *location['aliases']):
            if alias in references or alias == 'start':
                raise ValueError('Duplicate or reserved annotated location')
            references[alias] = pose
            regions[alias] = {'canonical_name': name, 'search_radius_m': radius,
                             'approach_yaw': yaw, 'scene_id': scene_id,
                             'source': data['source'], 'revision': data['revision']}
    from harness.robots.composed_placement import validate_surface
    surfaces = {}
    for name, raw in data.get('surfaces', {}).items():
        surface = validate_surface({k: v for k, v in {**raw, 'canonical_name': name,
            'scene_id': scene_id, 'source': data['source'], 'revision': data['revision']}.items() if k != 'aliases'})
        for alias in (name, *raw.get('aliases', [])):
            if alias in references or alias == 'start':
                raise ValueError('Duplicate or reserved surface reference')
            references[alias] = pose_value(surface['robot_pose'])
            surfaces[alias] = surface
    return {'references': references, 'visual_regions': regions, 'placement_surfaces': surfaces}
