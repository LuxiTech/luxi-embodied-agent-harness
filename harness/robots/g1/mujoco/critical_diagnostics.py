"""Operator-only, bounded lidar evidence capture. Never an input to safety decisions."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import uuid

import numpy as np


def trigger_identity(model, data, point, camera_ids, self_body_ids, carried_body_ids):
    """Audit the retained point against the same physics state used by ray filtering.

    These are ray matches, not invented labels for an unmatched/ambiguous voxel.
    Downsampling means a point may not coincide with any original surface hit.
    """
    import mujoco
    matches = []
    for camera in camera_ids:
        origin = np.asarray(data.cam_xpos[camera], dtype=float)
        delta = np.asarray(point, dtype=float) - origin
        expected = float(np.linalg.norm(delta))
        if expected <= 1e-9:
            continue
        geom = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(model, data, origin, delta / expected,
                                      None, 1, -1, geom))
        gid = int(geom[0])
        body = int(model.geom_bodyid[gid]) if gid >= 0 else None
        matches.append({
            'camera_id': int(camera),
            'camera_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera),
            'geom_id': gid,
            'geom_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid >= 0 else None,
            'body_id': body,
            'body_name': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) if body is not None else None,
            'expected_distance_m': expected, 'ray_distance_m': distance,
            'matches_point': gid >= 0 and abs(distance - expected) <= .08,
            'is_self': body in self_body_ids, 'is_carried': body in carried_body_ids,
        })
    matched = [m for m in matches if m['matches_point']]
    return {'method': 'same_state_camera_ray', 'tolerance_m': .08,
            'classification': ('unmatched' if not matched else
                               'ambiguous' if any(m['is_self'] for m in matched) and any(not m['is_self'] for m in matched)
                               else 'self' if all(m['is_self'] for m in matched) else 'external'),
            'rays': matches}


class CriticalFrameRecorder:
    """Keep at most 32 candidate frames; UI pins the exact confirmed critical frame."""
    def __init__(self, root: Path, limit: int = 32):
        self.root = root
        self.limit = limit
        self.paths = deque()
        self.run_id = uuid.uuid4().hex
        self.initialized = False

    def capture(self, payload, raw_points, filtered_points, *, model, data,
                camera_ids, self_body_ids, carried_body_ids):
        nearest = payload.get('nearest_obstacle_distance')
        # Include the critical release band so hysteresis transitions still have evidence.
        if nearest is None or nearest > .60:
            return None
        pending = self.root / 'pending'
        if not self.initialized:
            pending.mkdir(parents=True, exist_ok=True)
            # Previous worker is stopped before a new owned simulation starts.
            for old in pending.glob('*.npz'):
                old.unlink(missing_ok=True)
            self.initialized = True
        points = np.asarray(filtered_points)
        base = np.asarray(data.qpos[:3]).copy()
        # Match summarize_lidar_proximity's exact unrounded height limits.
        eligible = np.flatnonzero(np.isfinite(points).all(axis=1) &
                                 (points[:, 2] >= base[2]-.55) & (points[:, 2] <= base[2]+.75))
        index = int(eligible[np.argmin(np.linalg.norm(points[eligible, :2]-base[:2], axis=1))])
        point = points[index].copy()
        detail = {'schema_version': 1, 'run_id': self.run_id, 'lidar': payload,
                  'position': base.tolist(), 'quaternion_wxyz': np.asarray(data.qpos[3:7]).tolist(),
                  'simulation_time': float(data.time), 'trigger_point_index': index,
                  'trigger_point_world_xyz': point.tolist(),
                  'input_cloud_stage': 'after_segmentation_before_nearfield_ray_filter',
                  'output_cloud_stage': 'after_nearfield_ray_filter',
                  'self_body_ids': sorted(self_body_ids), 'carried_body_ids': sorted(carried_body_ids),
                  'identity': trigger_identity(model, data, point, camera_ids, self_body_ids, carried_body_ids)}
        target = pending / f'{self.run_id}-{payload["sequence"]}.npz'
        temporary = target.with_suffix('.tmp')
        try:
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, input_points=np.asarray(raw_points),
                                    filtered_points=points,
                                    metadata=np.array(json.dumps(detail, allow_nan=False)))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.paths.append(target)
        while len(self.paths) > self.limit:
            self.paths.popleft().unlink(missing_ok=True)
        return str(target)


def pin_critical_frame(candidate: str, sequence: int, timestamp: float):
    """Pin an immutable inode before the producer rotates its bounded buffer."""
    path = Path(candidate)
    target = path.parent.parent / path.name
    try:
        if path.parent.name != 'pending' or path.suffix != '.npz':
            raise ValueError('invalid diagnostic candidate path')
        try:
            os.link(path, target)
        except FileExistsError:
            pass
        with np.load(target, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
        if (metadata['lidar']['sequence'] != sequence or
                metadata['lidar']['frame_timestamp'] != timestamp):
            raise ValueError('diagnostic frame identity mismatch')
        return {'path': str(target), 'sequence': sequence, 'frame_timestamp': timestamp}
    except Exception as error:
        # Corrupt diagnostic archives must not interrupt proximity safety handling.
        return {'sequence': sequence, 'error': str(error)}
