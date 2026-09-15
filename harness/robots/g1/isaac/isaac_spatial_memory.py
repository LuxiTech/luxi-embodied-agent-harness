"""Persistent first-person CLIP spatial memory for the Isaac G1 backend."""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
from pydantic import Field

from dimos.perception.spatial_perception import SpatialConfig, SpatialMemory
from dimos.utils.logging_config import setup_logger

from harness.robots.g1.isaac.isaac_protocol import IsaacRuntimePaths


SPATIAL_MEMORY_PATH_ENV = "LUXI_ISAAC_SPATIAL_MEMORY_PATH"
logger = setup_logger()


def configured_spatial_memory_path() -> Path:
    """Resolve the experiment-owned semantic-memory directory."""

    configured = os.getenv(SPATIAL_MEMORY_PATH_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return IsaacRuntimePaths.configured().root / "spatial-memory"


def _db_path() -> str:
    return str(configured_spatial_memory_path() / "chromadb")


def _visual_memory_path() -> str:
    return str(configured_spatial_memory_path() / "visual-memory.pkl")


def _output_path() -> str:
    return str(configured_spatial_memory_path() / "frames")


class IsaacSpatialMemoryConfig(SpatialConfig):
    """Keep one CLIP index across process restarts within an experiment."""

    collection_name: str = "isaac_g1_first_person"
    embedding_model: str = "clip"
    embedding_dimensions: int = 512
    min_distance_threshold: float = 0.20
    min_yaw_threshold_rad: float = math.radians(15.0)
    min_time_threshold: float = 1.0
    db_path: str | None = Field(default_factory=_db_path)
    visual_memory_path: str | None = Field(default_factory=_visual_memory_path)
    output_dir: str | None = Field(default_factory=_output_path)
    new_memory: bool = False


class IsaacPersistentSpatialMemory(SpatialMemory):
    """DimOS SpatialMemory with Isaac-scoped, non-destructive persistence.

    The inherited producer consumes only the connected first-person color
    stream and the measured ``world -> base_link`` transform.  No USD scene
    coordinates or operator-camera frames enter this index.
    """

    config: IsaacSpatialMemoryConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_stored_yaw: float | None = None

    def _process_frame(self) -> None:
        """Store translated or rotated first-person views.

        Upstream SpatialMemory samples translation only, which drops every new
        view from an in-place scan. Isaac exploration intentionally rotates at
        reached viewpoints, so angular diversity is part of the sensor-memory
        contract here.
        """

        tf = self.tf.get("world", "base_link")
        frame = self._latest_video_frame
        if tf is None or frame is None:
            return
        pose = tf.to_pose()
        yaw = float(tf.rotation.to_euler().z)
        now = time.time()
        if self.last_record_time is not None:
            if now - self.last_record_time < self.min_time_threshold:
                return
        distance = math.inf
        if self.last_position is not None:
            distance = float(
                np.linalg.norm(
                    [
                        pose.position.x - self.last_position.x,
                        pose.position.y - self.last_position.y,
                        pose.position.z - self.last_position.z,
                    ]
                )
            )
        yaw_change = math.inf
        if self._last_stored_yaw is not None:
            yaw_change = abs(
                math.atan2(
                    math.sin(yaw - self._last_stored_yaw),
                    math.cos(yaw - self._last_stored_yaw),
                )
            )
        self.frame_count += 1
        if (
            distance < self.min_distance_threshold
            and yaw_change < self.config.min_yaw_threshold_rad
        ):
            return
        try:
            embedding = np.asarray(
                self.embedding_provider.get_embedding(frame),
                dtype=np.float32,
            )
            if (
                embedding.shape != (self.embedding_dimensions,)
                or not np.isfinite(embedding).all()
                or float(np.linalg.norm(embedding)) < 0.99
            ):
                raise RuntimeError("CLIP produced an invalid embedding")
            frame_id = (
                f"isaac_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            euler = tf.rotation.to_euler()
            self.vector_db.add_image_vector(
                vector_id=frame_id,
                image=frame,
                embedding=embedding,
                metadata={
                    "pos_x": float(pose.position.x),
                    "pos_y": float(pose.position.y),
                    "pos_z": float(pose.position.z),
                    "rot_x": float(euler.x),
                    "rot_y": float(euler.y),
                    "rot_z": yaw,
                    "timestamp": now,
                    "frame_id": frame_id,
                    "backend": "isaac-g1",
                    "source": "first_person_color",
                },
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill mapping
            logger.error(
                "Isaac CLIP spatial-memory frame rejected: "
                f"{type(exc).__name__}"
            )
            return
        self.last_position = pose.position
        self._last_stored_yaw = yaw
        self.last_record_time = now
        self.stored_frame_count += 1
        if self.stored_frame_count % 25 == 0:
            self.save()
