"""Operator-scene navigation prior, fused with live costs (never risk evidence)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

from dimos.utils.logging_config import setup_logger
from dimos.mapping.costmapper import CostMapper
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from harness.robots.g1.mujoco.operator_scenes import (
    OPERATOR_SCENE_PAYLOAD_ENV, load_operator_scene_payload,
)


def known_map_enabled(environment: Mapping[str, str]) -> bool:
    if environment.get('LUXI_BLIND_MODE', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        return False
    return (environment.get('LUXI_KNOWN_GLOBAL_MAP', '1').strip().lower()
            not in {'0', 'false', 'no', 'off'}
            and bool(environment.get(OPERATOR_SCENE_PAYLOAD_ENV)))


def scene_costmap(scene_xml: str, resolution: float = 0.05) -> OccupancyGrid:
    """Rasterize collision geometry at the scene's initial pose, excluding props.

    Conservative projected AABBs include furniture/tabletops and door leaves.
    Free-jointed entities are observed at runtime, not baked into the floor plan.
    Robot footprint inflation remains the planner's responsibility.
    """
    import mujoco

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('invalid known-map resolution')
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        body = int(model.geom_bodyid[geom])
        movable = False
        while body:
            first = int(model.body_jntadr[body])
            for joint in range(first, first + int(model.body_jntnum[body])):
                movable |= model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            body = int(model.body_parentid[body])
        if movable:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        extent = np.abs(rotation) @ model.geom_aabb[geom, 3:]
        low, high = center - extent, center + extent
        if high[2] <= 0.10 or low[2] >= 1.5:
            continue
        boxes.append((low[:2], high[:2]))
    if not boxes:
        raise ValueError('scene has no bounded collision geometry for a known map')
    low = np.floor(np.min([b[0] for b in boxes], axis=0) / resolution) * resolution
    high = np.ceil(np.max([b[1] for b in boxes], axis=0) / resolution) * resolution
    size = np.ceil((high - low) / resolution).astype(int) + 1
    if np.prod(size) > 4_000_000:
        raise ValueError('known map exceeds cell limit')
    cells = np.zeros((size[1], size[0]), dtype=np.int8)
    for minimum, maximum in boxes:
        start = np.maximum(0, np.floor((minimum - low) / resolution).astype(int))
        end = np.minimum(size - 1, np.floor((maximum - low) / resolution).astype(int))
        cells[start[1]:end[1]+1, start[0]:end[0]+1] = 100
    # Do not create routes around the outside of the scene boundary.
    cells[[0, -1], :] = 100
    cells[:, [0, -1]] = 100
    return OccupancyGrid(grid=cells, resolution=resolution,
                         origin=Pose(position=Vector3(float(low[0]), float(low[1]), 0)))


def fuse_costmap(prior: OccupancyGrid, live: OccupancyGrid) -> OccupancyGrid:
    """Keep static obstacles and overlay live costs without clearing either."""
    if live.frame_id != prior.frame_id:
        raise ValueError('known and live maps must share the world frame')
    cells = prior.grid.copy()
    rows, cols = np.indices(cells.shape)
    x = prior.origin.position.x + (cols + 0.5) * prior.resolution
    y = prior.origin.position.y + (rows + 0.5) * prior.resolution
    lc = np.floor((x - live.origin.position.x) / live.resolution).astype(int)
    lr = np.floor((y - live.origin.position.y) / live.resolution).astype(int)
    valid = (lc >= 0) & (lr >= 0) & (lc < live.grid.shape[1]) & (lr < live.grid.shape[0])
    cells[valid] = np.maximum(cells[valid], live.grid[lr[valid], lc[valid]])
    return OccupancyGrid(grid=cells, resolution=prior.resolution, origin=prior.origin,
                         frame_id=prior.frame_id, ts=live.ts)


class KnownSceneCostMapper(CostMapper):
    """Single planner-map publisher; sensor costs continue to refresh normally."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scene_prior = None
        if known_map_enabled(os.environ):
            payload = load_operator_scene_payload(Path(os.environ[OPERATOR_SCENE_PAYLOAD_ENV]))
            # The legacy office asset has no procedural scene XML; retain online mapping.
            if payload.get('scene_xml'):
                self._scene_prior = scene_costmap(payload['scene_xml'], self.config.config.resolution)
                setup_logger().info('Known scene navigation map enabled',
                                    shape=str(self._scene_prior.grid.shape),
                                    resolution=self._scene_prior.resolution)

    def _calculate_costmap(self, msg):
        live = super()._calculate_costmap(msg)
        return fuse_costmap(self._scene_prior, live) if self._scene_prior is not None else live
