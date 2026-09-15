"""Pure RTX lidar geometry and identity filtering helpers.

This module intentionally has no Isaac Sim imports.  The container runtime can
feed it GMO arrays while host-side unit tests exercise the safety-critical
stable-ID decisions without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np


class UnverifiedLidarIdentity(ValueError):
    """Raised when a lidar frame cannot prove the identity of every return."""


@dataclass(frozen=True)
class LidarIdentityAudit:
    raw_return_count: int
    resolved_return_count: int
    self_return_count: int
    retained_return_count: int
    invalid_return_count: int
    identity_verified: bool = True
    identity_source: str = "rtx_stable_id"


def _normalized_prim_path(value: str) -> str:
    path = value.strip()
    if not path.startswith("/"):
        raise UnverifiedLidarIdentity("stable ID label is not an absolute prim path")
    return path.rstrip("/") or "/"


def _is_prim_or_descendant(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _stable_id_label(object_id: int, stable_id_map: Mapping[int, str]) -> str:
    try:
        return stable_id_map[object_id]
    except (KeyError, TypeError, ValueError, OverflowError) as exact_error:
        # Isaac Sim 5.1 may put an instance discriminator in the upper uint32
        # of a GMO uint128 object ID while StableIdMap publishes its uint96
        # prim identity. Only resolve this form when that exact base identity
        # is present in the audited map.
        instance_discriminator = object_id >> 96
        base_object_id = object_id & ((1 << 96) - 1)
        if instance_discriminator:
            try:
                return stable_id_map[base_object_id]
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        raise UnverifiedLidarIdentity(
            f"unresolved stable object ID: {object_id!r}"
        ) from exact_error


def merge_stable_id_mapping(
    cached: Mapping[int, str],
    current: Mapping[int, str],
) -> dict[int, str]:
    """Merge RTX StableIdMap deltas without accepting identity reassignment."""

    merged: dict[int, str] = {}
    for source in (cached, current):
        for raw_object_id, raw_path in source.items():
            if isinstance(raw_object_id, bool):
                raise UnverifiedLidarIdentity("invalid stable object ID")
            try:
                object_id = int(raw_object_id)
            except (TypeError, ValueError, OverflowError) as error:
                raise UnverifiedLidarIdentity("invalid stable object ID") from error
            if not isinstance(raw_path, str):
                raise UnverifiedLidarIdentity("stable ID label is not a prim path")
            path = _normalized_prim_path(raw_path)
            previous = merged.get(object_id)
            if previous is not None and previous != path:
                raise UnverifiedLidarIdentity(
                    f"stable object ID changed identity: {object_id!r}"
                )
            merged[object_id] = path
    return merged


def filter_lidar_returns(
    points: Any,
    object_ids: Sequence[int],
    stable_id_map: Mapping[int, str],
    *,
    robot_prim_path: str,
    carried_prim_paths: Sequence[str] = (),
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], LidarIdentityAudit]:
    """Remove robot and physically attached entity returns by stable identity.

    Unknown IDs fail the entire frame.  Keeping an unresolved point would risk
    treating a moving robot link as an external obstacle; dropping it would
    risk hiding a real obstacle.  A fail-closed frame is the only safe result.
    """

    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("lidar points must be an Nx3 array")
    if len(object_ids) != len(xyz):
        raise UnverifiedLidarIdentity("point and stable object ID count mismatch")
    robot_root = _normalized_prim_path(robot_prim_path)
    self_roots = (
        robot_root,
        *tuple(_normalized_prim_path(path) for path in carried_prim_paths),
    )
    if len(self_roots) != len(set(self_roots)):
        raise UnverifiedLidarIdentity("self-filter prim paths must be unique")

    raw_ids = np.asarray(object_ids)
    if raw_ids.ndim != 1:
        raise UnverifiedLidarIdentity("stable object IDs must be one-dimensional")
    if np.issubdtype(raw_ids.dtype, np.bool_):
        raise UnverifiedLidarIdentity("invalid stable object ID")
    if np.issubdtype(raw_ids.dtype, np.integer):
        normalized_ids = raw_ids
    else:
        normalized_values: list[int] = []
        for object_id in object_ids:
            if isinstance(object_id, (bool, np.bool_)):
                raise UnverifiedLidarIdentity("invalid stable object ID")
            try:
                normalized_values.append(int(object_id))
            except (TypeError, ValueError, OverflowError) as error:
                raise UnverifiedLidarIdentity("invalid stable object ID") from error
        # RTX object IDs are stable hashes and may exceed signed or unsigned
        # 64-bit ranges. Keep arbitrary-precision Python integers while still
        # resolving only the small set of unique IDs.
        normalized_ids = np.asarray(normalized_values, dtype=object)

    self_ids: list[int] = []
    for object_id in np.unique(normalized_ids):
        normalized_id = int(object_id)
        label = _stable_id_label(normalized_id, stable_id_map)
        if not isinstance(label, str):
            raise UnverifiedLidarIdentity("stable ID label is not a prim path")
        if any(
            _is_prim_or_descendant(_normalized_prim_path(label), root)
            for root in self_roots
        ):
            self_ids.append(normalized_id)

    self_mask = np.isin(normalized_ids, np.asarray(self_ids))
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    retained_mask = ~self_mask & finite_mask
    invalid_mask = ~self_mask & ~finite_mask
    filtered = np.ascontiguousarray(xyz[retained_mask], dtype=np.float32)
    audit = LidarIdentityAudit(
        raw_return_count=len(xyz),
        resolved_return_count=len(normalized_ids),
        self_return_count=int(np.count_nonzero(self_mask)),
        retained_return_count=len(filtered),
        invalid_return_count=int(np.count_nonzero(invalid_mask)),
    )
    return filtered, audit


def spherical_returns_to_cartesian(
    *,
    azimuth_degrees: Any,
    elevation_degrees: Any,
    ranges_m: Any,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert Isaac GMO spherical returns into sensor-frame XYZ."""

    azimuth = np.asarray(azimuth_degrees, dtype=np.float64).reshape(-1)
    elevation = np.asarray(elevation_degrees, dtype=np.float64).reshape(-1)
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    if not (len(azimuth) == len(elevation) == len(ranges)):
        raise ValueError("spherical lidar arrays must have equal lengths")
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    horizontal = ranges * np.cos(elevation)
    return np.ascontiguousarray(
        np.column_stack(
            (
                horizontal * np.cos(azimuth),
                horizontal * np.sin(azimuth),
                ranges * np.sin(elevation),
            )
        ),
        dtype=np.float32,
    )


def transform_points_wxyz(
    points: Any,
    *,
    position: Any,
    quaternion_wxyz: Any,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Transform sensor-frame XYZ into world coordinates."""

    xyz = np.asarray(points, dtype=np.float64)
    origin = np.asarray(position, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or origin.size != 3 or quaternion.size != 4:
        raise ValueError("invalid lidar points or sensor pose")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid lidar quaternion")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return np.ascontiguousarray(xyz @ rotation.T + origin, dtype=np.float32)


def _direction_label(bearing_radians: float) -> str:
    degrees = math.degrees(bearing_radians)
    absolute = abs(degrees)
    if absolute <= 22.5:
        return "front"
    if absolute <= 67.5:
        return "front_left" if degrees > 0 else "front_right"
    if absolute <= 112.5:
        return "left" if degrees > 0 else "right"
    if absolute <= 157.5:
        return "rear_left" if degrees > 0 else "rear_right"
    return "rear"


def summarize_lidar_proximity(
    points: Any,
    *,
    frame_timestamp: float,
    scan_started_at: float,
    pose_timestamp: float,
    position: Any,
    quaternion_wxyz: Any,
    sequence: int,
    audit: LidarIdentityAudit,
    written_at: float | None = None,
) -> dict[str, Any]:
    """Build the current-lidar sidecar consumed by the local safety monitor."""

    xyz = np.asarray(points, dtype=np.float64)
    base = np.asarray(position, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or base.size != 3 or quaternion.size != 4:
        raise ValueError("invalid lidar proximity inputs")
    lower_z = float(base[2]) - 0.55
    upper_z = float(base[2]) + 0.75
    candidates = xyz[
        np.all(np.isfinite(xyz), axis=1)
        & (xyz[:, 2] >= lower_z)
        & (xyz[:, 2] <= upper_z)
    ]
    qw, qx, qy, qz = (float(value) for value in quaternion)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    nearest_distance: float | None = None
    nearest_bearing: float | None = None
    nearest_direction: str | None = None
    sectors: dict[str, float] = {}
    if len(candidates):
        offsets = candidates[:, :2] - base[:2]
        distances = np.linalg.norm(offsets, axis=1)
        bearings = np.arctan2(offsets[:, 1], offsets[:, 0]) - yaw
        bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
        bearing_degrees = np.degrees(bearings)
        sector_indices = np.zeros(len(bearings), dtype=np.uint8)
        sector_indices[(bearing_degrees > 22.5) & (bearing_degrees <= 67.5)] = 1
        sector_indices[(bearing_degrees > 67.5) & (bearing_degrees <= 112.5)] = 2
        sector_indices[(bearing_degrees > 112.5) & (bearing_degrees <= 157.5)] = 3
        sector_indices[np.abs(bearing_degrees) > 157.5] = 4
        sector_indices[(bearing_degrees < -112.5) & (bearing_degrees >= -157.5)] = 5
        sector_indices[(bearing_degrees < -67.5) & (bearing_degrees >= -112.5)] = 6
        sector_indices[(bearing_degrees < -22.5) & (bearing_degrees >= -67.5)] = 7
        sector_names = (
            "front",
            "front_left",
            "left",
            "rear_left",
            "rear",
            "rear_right",
            "right",
            "front_right",
        )
        sector_minima = np.full(len(sector_names), np.inf, dtype=np.float64)
        np.minimum.at(sector_minima, sector_indices, distances)
        sectors = {
            name: float(distance)
            for name, distance in zip(sector_names, sector_minima, strict=True)
            if np.isfinite(distance)
        }
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        nearest_bearing = float(bearings[nearest_index])
        nearest_direction = _direction_label(nearest_bearing)

    return {
        "schema_version": 1,
        "available": True,
        "source": "current_lidar",
        "frame_id": "world",
        "sequence": int(sequence),
        "frame_timestamp": float(frame_timestamp),
        "scan_started_at": float(scan_started_at),
        "pose_timestamp": float(pose_timestamp),
        "written_at": time.time() if written_at is None else float(written_at),
        "nearest_obstacle_distance": nearest_distance,
        "nearest_obstacle_bearing_deg": (
            None if nearest_bearing is None else round(math.degrees(nearest_bearing), 3)
        ),
        "nearest_obstacle_direction": nearest_direction,
        "sectors_m": {name: round(value, 4) for name, value in sectors.items()},
        "candidate_points": int(len(candidates)),
        "self_filter": {
            "identity_verified": bool(audit.identity_verified),
            "mode": audit.identity_source,
            "frames": 1,
            "applied": 1,
            "missing": 0,
            "invalid": 0,
            "raw_returns": audit.raw_return_count,
            "resolved_returns": audit.resolved_return_count,
            "self_returns_removed": audit.self_return_count,
            "invalid_returns": audit.invalid_return_count,
        },
    }
