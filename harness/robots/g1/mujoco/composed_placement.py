"""Physics-owner simulated placement. Geometry stays inside the adapter.

Annotated coordinates are inputs, never scene-derived navigation goals. Placement
uses the existing sleeping-prop constraint, not a claim of contact manipulation.
"""
import copy
import math
import time
import mujoco
import numpy as np
from harness.robots.composed_placement import validate_surface
from harness.robots.composed_pose import POSITION_TOLERANCE_M, YAW_TOLERANCE_RAD


def geom_bounds(owner, geom, *, local=False):
    center, half = owner.model.geom_aabb[geom, :3], owner.model.geom_aabb[geom, 3:]
    if local:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, owner.model.geom_quat[geom])
        rotation, position = matrix.reshape(3,3), owner.model.geom_pos[geom]
    else:
        rotation, position = owner.data.geom_xmat[geom].reshape(3,3), owner.data.geom_xpos[geom]
    center = position + rotation @ center
    half = np.abs(rotation) @ half
    return center-half, center+half


def environment(owner, low, high, entity_body, surface):
    """Require a static horizontal box under the entire footprint and clear volume."""
    support = False
    clear = True
    z = surface['place_point'][2]
    for geom in range(owner.model.ngeom):
        body = int(owner.model.geom_bodyid[geom])
        if body == entity_body or not (owner.model.geom_contype[geom] or owner.model.geom_conaffinity[geom]):
            continue
        a, b = geom_bounds(owner, geom)
        is_support = (owner.model.body_weldid[body] == 0
            and owner.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
            and np.allclose(owner.data.geom_xmat[geom].reshape(3,3), np.eye(3), atol=1e-6)
            and abs(b[2]-z) <= .003
            and np.all(low[:2] >= a[:2]) and np.all(high[:2] <= b[:2]))
        support = support or bool(is_support)
        if not is_support and np.all(np.minimum(high,b)-np.maximum(low,a) > .003):
            clear = False
    return support, clear


def request(owner, token, entity_id, arguments):
    """Called only after candidate, expiry, cancellation and stationarity checks."""
    def deny(reason):
        owner._ack(token, 'sim_place', ok=False, error=reason)
    try:
        surface = validate_surface(arguments.get('surface'))
    except (ValueError, TypeError):
        deny('invalid annotated placement surface')
        return
    attachment = owner._attachments.get(entity_id)
    if entity_id not in owner._sim_attachments or attachment is None:
        deny('entity has no simulation attachment')
        return
    target = surface['robot_pose']
    q = owner.data.qpos[3:7]
    yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
    error = abs(math.atan2(math.sin(yaw-target[2]), math.cos(yaw-target[2])))
    if (math.dist(owner.data.qpos[:2], target[:2]) > POSITION_TOLERANCE_M or error > YAW_TOLERANCE_RAD
            or math.dist(owner.data.qpos[:2], surface['place_point'][:2]) > 1.0):
        deny('not at placement pose or point out of range')
        return
    bounds = [geom_bounds(owner, int(g), local=True) for g in attachment.geom_ids]
    if not bounds:
        deny('entity geometry unavailable')
        return
    low = np.min([a for a,b in bounds], axis=0)
    high = np.max([b for a,b in bounds], axis=0)
    # Upright body frame; compensate the lowest geometry point, not a hardcoded bottle height.
    position = np.asarray(surface['place_point'], dtype=float).copy()
    position[2] -= low[2]
    low, high = low+position, high+position
    xmin,xmax,ymin,ymax = surface['bounds_xy']
    if not (low[0] >= xmin and high[0] <= xmax and low[1] >= ymin and high[1] <= ymax):
        deny('entity footprint exceeds annotated placement region')
        return
    supported, clear = environment(owner, low, high, attachment.child_body_id, surface)
    if not supported or not clear:
        deny('placement support unavailable or occupied')
        return
    # All checks precede mutation. Move the free body, then restore collision and pin at the new support pose.
    adr = attachment.child_qpos_adr
    owner.data.qpos[adr:adr+7] = [*position,1,0,0,0]
    owner._release_attachment(entity_id)
    owner._sim_placements[entity_id] = {'surface':copy.deepcopy(surface), 'samples':0,
                                      'last_time':float(owner.data.time)}
    owner._ack(token, 'sim_place', ok=True, entity_id=entity_id, attached=False,
               interaction_model='sim_placement', pose=owner._entity_pose(entity_id))


def evidence(owner, entity_id):
    record = owner._sim_placements.get(entity_id)
    if record is None:
        return None
    body = owner._body_id(owner._entity_body_name(entity_id))
    geoms = np.flatnonzero(owner.model.geom_bodyid == body)
    bounds = [geom_bounds(owner, int(g)) for g in geoms]
    low, high = np.min([a for a,b in bounds],axis=0), np.max([b for a,b in bounds],axis=0)
    surface = record['surface']
    x,y,z = surface['place_point']
    xmin,xmax,ymin,ymax = surface['bounds_xy']
    pose = owner._entity_pose(entity_id)
    position_valid = (math.dist(pose[:2], (x,y)) <= .02 and abs(low[2]-z) <= .005
        and low[0] >= xmin and high[0] <= xmax and low[1] >= ymin and high[1] <= ymax)
    upright = owner.data.xmat[body].reshape(3,3)[2,2] >= math.cos(.15)
    _, vel = owner._freejoint_addresses(entity_id)
    stationary = np.linalg.norm(owner.data.qvel[vel:vel+3]) <= .01 and np.linalg.norm(owner.data.qvel[vel+3:vel+6]) <= .025
    supported, clear = environment(owner, low, high, body, surface)
    valid = position_valid and upright and stationary and supported and clear and entity_id not in owner._attachments
    sample_time = float(owner.data.time)
    if sample_time != record['last_time']:
        record['last_time'] = sample_time
        record['samples'] = record['samples']+1 if valid else 0
    if not valid:
        record['samples'] = 0
    return {'source':'simulation_placement_adapter', 'interaction_model':'sim_placement',
            'stability_model':'sleeping_prop_constraint', 'entity_id':entity_id,
            'surface':copy.deepcopy(surface), 'position_valid':bool(position_valid),
            'upright':bool(upright), 'supported':bool(supported), 'clear':bool(clear),
            'stationary':bool(stationary), 'stable_samples':record['samples'],
            'evidence_timestamp':time.time(), 'simulation_time':sample_time}
