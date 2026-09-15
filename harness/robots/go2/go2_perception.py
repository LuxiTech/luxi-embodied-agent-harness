"""HIKROBOT RGB semantic tracking with MID-360 metric association."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import cv2
import numpy as np

from harness.integrations.qwen.qwen_vl_compat import (
    build_qwen_vl_model_class,
    normalize_qwen_bbox_payload,
)


@dataclass(frozen=True)
class RgbLidarDetection:
    label: str
    bbox: tuple[int, int, int, int]
    distance_m: float
    bearing_rad: float
    frame_time: float
    source: str


TARGET_DESCRIPTIONS = {
    "red_cube": "the red cube / 红色方块",
    "blue_ball": "the blue ball / 蓝色球",
    "bottle": "the water bottle / 水瓶",
    "person": "the visible person / 人",
}


def _target_color_mask(color: np.ndarray[Any, Any], target: str) -> np.ndarray[Any, Any] | None:
    """Return a strict RGB mask only for targets with an explicit color name."""

    pixels = np.asarray(color, dtype=np.int16)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    if target == "red_cube":
        return (red >= 145) & (red >= green + 55) & (red >= blue + 55)
    if target == "blue_ball":
        return (blue >= 125) & (blue >= red + 45) & (blue >= green + 25)
    return None


def refine_colored_target_bbox(
    *,
    target: str,
    color: np.ndarray[Any, Any],
    proposal: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Tighten a semantic proposal with nearby RGB evidence, never scene IDs."""

    mask = _target_color_mask(color, target)
    if mask is None:
        return proposal
    height, width = color.shape[:2]
    x0, y0, x1, y1 = proposal
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < 3.0 or box_height < 3.0:
        return None
    margin_x = max(12, int(round(0.55 * box_width)))
    margin_y = max(12, int(round(0.70 * box_height)))
    sx0, sx1 = max(0, int(x0) - margin_x), min(width, int(math.ceil(x1)) + margin_x)
    sy0, sy1 = max(0, int(y0) - margin_y), min(height, int(math.ceil(y1)) + margin_y)
    region = mask[sy0:sy1, sx0:sx1].astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(region, 8)
    proposal_area = max(1.0, box_width * box_height)
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for label in range(1, count):
        left, top, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if area < max(20, int(0.004 * proposal_area)):
            continue
        cx, cy = float(centroids[label][0] + sx0), float(centroids[label][1] + sy0)
        bx0, by0 = float(left + sx0), float(top + sy0)
        bx1, by1 = bx0 + component_width, by0 + component_height
        overlap = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(
            0.0, min(y1, by1) - max(y0, by0)
        )
        if overlap <= 0.0:
            continue
        distance = math.hypot(cx - 0.5 * (x0 + x1), cy - 0.5 * (y0 + y1))
        score = 3.0 * overlap + float(area) - 0.2 * distance
        if best is None or score > best[0]:
            best = (score, (bx0, by0, bx1, by1))
    if best is None:
        return None
    refined = best[1]
    refined_width = refined[2] - refined[0]
    refined_height = refined[3] - refined[1]
    aspect = refined_width / max(refined_height, 1.0)
    component_area = float(np.count_nonzero(mask[int(refined[1]):int(refined[3]), int(refined[0]):int(refined[2])]))
    fill = component_area / max(1.0, refined_width * refined_height)
    if target == "red_cube" and not (0.45 <= aspect <= 2.2 and fill >= 0.55):
        return None
    if target == "blue_ball" and not (0.65 <= aspect <= 1.35 and fill >= 0.45):
        return None
    return refined


def metric_detection_from_rgb_and_lidar(
    *,
    target: str,
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    mid360_points: np.ndarray[Any, Any],
    frame_time: float,
    horizontal_fov_degrees: float,
    source: str,
) -> RgbLidarDetection | None:
    """Associate an RGB bbox bearing with current sensor-frame MID-360 returns."""

    height, width = image_shape[:2]
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    fx = 0.5 * width / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    centre_x = 0.5 * (x0 + x1)
    image_bearing = math.atan2(centre_x - 0.5 * width, fx)
    left_bearing = math.atan2(x0 - 0.5 * width, fx)
    right_bearing = math.atan2(x1 - 0.5 * width, fx)
    half_window = max(math.radians(1.5), 0.5 * abs(right_bearing - left_bearing))

    points = np.asarray(mid360_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or not len(points):
        return None
    planar_ranges = np.linalg.norm(points[:, :2], axis=1)
    lidar_bearings = np.arctan2(points[:, 1], points[:, 0])
    # Camera image-right is body -Y, while MID-360 positive bearing is body +Y.
    expected_lidar_bearing = -image_bearing
    errors = np.arctan2(
        np.sin(lidar_bearings - expected_lidar_bearing),
        np.cos(lidar_bearings - expected_lidar_bearing),
    )
    valid = (
        np.isfinite(planar_ranges)
        & (planar_ranges > 0.10)
        & (planar_ranges < 20.0)
        & (np.abs(errors) <= half_window)
    )
    ranges = planar_ranges[valid]
    if ranges.size < 2:
        return None
    # A low target may occupy only a few vertical channels while most rays
    # continue to the wall behind it. Select the robust near-return decile,
    # then still require multiple geometrically consistent returns.
    near = float(np.quantile(ranges, 0.10))
    cluster = ranges[ranges <= near + 0.20]
    if cluster.size < 2:
        return None

    return RgbLidarDetection(
        label=target,
        bbox=(x0, y0, x1, y1),
        distance_m=float(np.median(cluster)),
        bearing_rad=image_bearing,
        frame_time=float(frame_time),
        source=source,
    )


class HikrobotRgbSemanticDetector:
    """One-shot Qwen RGB bbox acquisition followed by RGB CSRT tracking."""

    def __init__(
        self,
        *,
        horizontal_fov_degrees: float,
        localizer: Callable[
            [np.ndarray[Any, Any], str, float],
            tuple[float, float, float, float] | None,
        ]
        | None = None,
    ) -> None:
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self._localizer = localizer
        self._model: Any | None = None
        self._tracker: Any | None = None
        self._locked_target: str | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_distance_m: float | None = None
        self._last_range_time: float | None = None
        self._last_failure: str | None = None
        self._semantic_inference_active = False
        self._last_request_metadata: dict[str, Any] = {}

    def _range_is_continuous(self, detection: RgbLidarDetection) -> bool:
        if self._last_distance_m is None or self._last_range_time is None:
            return True
        elapsed = max(0.0, detection.frame_time - self._last_range_time)
        return detection.distance_m <= self._last_distance_m + 0.30 + 0.60 * elapsed

    def _remember_range(self, detection: RgbLidarDetection) -> None:
        self._last_distance_m = detection.distance_m
        self._last_range_time = detection.frame_time

    def _qwen_bbox(
        self, color: np.ndarray[Any, Any], target: str, frame_time: float
    ) -> tuple[float, float, float, float] | None:
        from dimos.models.vl.qwen import QwenVlModel
        from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

        if self._model is None:
            model_type = build_qwen_vl_model_class(QwenVlModel)
            self._model = model_type()
        image = Image.from_numpy(
            color,
            format=ImageFormat.RGB,
            frame_id="hikrobot_mv_cu013_a0uc_color",
            ts=float(frame_time),
        )
        prompt = (
            f"Find {TARGET_DESCRIPTIONS[target]}. Return one tight bbox "
            'as JSON [{"bbox_2d": [x1, y1, x2, y2]}], with top-left and '
            "bottom-right corners in normalized 0-1000 coordinates; return [] if absent."
        )
        self._semantic_inference_active = True
        try:
            response = self._model.query(image, prompt)
        finally:
            metadata = getattr(self._model, "last_request_metadata", None)
            if callable(metadata):
                self._last_request_metadata = dict(metadata())
            self._semantic_inference_active = False
        normalized = normalize_qwen_bbox_payload(
            response,
            width=int(color.shape[1]),
            height=int(color.shape[0]),
            # The shared VLM adapter already emits pixel bbox after its
            # bounded bbox request. Do not scale those pixels a second time.
            model_name=(None if self._last_request_metadata.get("normalized_detection") is True
                        else str(self._model.config.model_name)),
        )
        if normalized is None:
            return None
        bbox = normalized.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]

    def localize_bbox(
        self, *, target: str, color: np.ndarray[Any, Any], frame_time: float
    ) -> tuple[float, float, float, float] | None:
        """Run only semantic localization, safe to dispatch to an I/O worker."""

        localizer = self._localizer or self._qwen_bbox
        return localizer(color, target, frame_time)

    def acquire_from_bbox(
        self,
        *,
        target: str,
        color: np.ndarray[Any, Any],
        mid360_points: np.ndarray[Any, Any],
        frame_time: float,
        semantic_bbox: tuple[float, float, float, float] | None,
    ) -> RgbLidarDetection | None:
        """Finish synchronized RGB/MID-360 acquisition on the simulation thread."""

        if semantic_bbox is None:
            self._last_failure = "semantic_bbox_missing"
            self.clear_lock()
            return None
        bbox = refine_colored_target_bbox(
            target=target,
            color=color,
            proposal=semantic_bbox,
        )
        if bbox is None:
            self._last_failure = "semantic_bbox_rgb_validation_failed"
            self.clear_lock()
            return None
        detection = metric_detection_from_rgb_and_lidar(
            target=target,
            bbox=bbox,
            image_shape=color.shape,
            mid360_points=mid360_points,
            frame_time=frame_time,
            horizontal_fov_degrees=self.horizontal_fov_degrees,
            source="hikrobot_qwen_rgb_bbox+mid360_range",
        )
        if detection is None:
            self._last_failure = "mid360_bbox_association_missing"
            self.clear_lock()
            return None
        if not self._range_is_continuous(detection):
            self._last_failure = "mid360_background_range_jump"
            return None
        tracker = cv2.TrackerCSRT_create()
        x0, y0, x1, y1 = detection.bbox
        tracker.init(color, (x0, y0, x1 - x0, y1 - y0))
        self._tracker = tracker
        self._locked_target = target
        self._last_bbox = tuple(float(value) for value in detection.bbox)
        self._remember_range(detection)
        self._last_failure = None
        return detection

    def acquire(
        self,
        *,
        target: str,
        color: np.ndarray[Any, Any],
        mid360_points: np.ndarray[Any, Any],
        frame_time: float,
    ) -> RgbLidarDetection | None:
        return self.acquire_from_bbox(
            target=target,
            color=color,
            mid360_points=mid360_points,
            frame_time=frame_time,
            semantic_bbox=self.localize_bbox(
                target=target, color=color, frame_time=frame_time
            ),
        )

    def track(
        self,
        *,
        target: str,
        color: np.ndarray[Any, Any],
        mid360_points: np.ndarray[Any, Any],
        frame_time: float,
    ) -> RgbLidarDetection | None:
        if self._tracker is None or self._locked_target != target:
            self._last_failure = "tracker_not_initialized"
            return None
        ok, tracked = self._tracker.update(color)
        recovered = False
        if ok:
            x, y, width, height = (float(value) for value in tracked)
            proposal = (x, y, x + width, y + height)
        elif self._last_bbox is not None and _target_color_mask(color, target) is not None:
            proposal = self._last_bbox
            recovered = True
        else:
            self._last_failure = "csrt_update_failed"
            return None
        bbox = refine_colored_target_bbox(
            target=target,
            color=color,
            proposal=proposal,
        )
        if bbox is None:
            self._last_failure = (
                "csrt_local_rgb_recovery_failed" if recovered else "tracked_bbox_rgb_validation_failed"
            )
            return None
        detection = metric_detection_from_rgb_and_lidar(
            target=target,
            bbox=bbox,
            image_shape=color.shape,
            mid360_points=mid360_points,
            frame_time=frame_time,
            horizontal_fov_degrees=self.horizontal_fov_degrees,
            source=(
                "hikrobot_local_rgb_recovery+mid360_range"
                if recovered
                else "hikrobot_csrt_rgb_track+mid360_range"
            ),
        )
        if detection is not None:
            if not self._range_is_continuous(detection):
                self._last_failure = "mid360_background_range_jump"
                return None
            self._last_bbox = bbox
            self._remember_range(detection)
            self._last_failure = None
            if recovered:
                tracker = cv2.TrackerCSRT_create()
                x0, y0, x1, y1 = detection.bbox
                tracker.init(color, (x0, y0, x1 - x0, y1 - y0))
                self._tracker = tracker
        else:
            self._last_failure = "mid360_bbox_association_missing"
        return detection

    def clear_lock(self) -> None:
        self._tracker = None
        self._locked_target = None
        self._last_bbox = None

    def metadata(self) -> dict[str, Any]:
        request = dict(self._last_request_metadata)
        return {
            "semantic_model": (
                str(self._model.config.model_name) if self._model is not None else "qwen"
            ),
            "semantic_input": "hikrobot_mv_cu013_a0uc_rgb_only",
            "metric_input": "mid360_current_scan_bearing_association",
            "continuous_tracker": "opencv_csrt_rgb",
            "geom_id_input": False,
            "camera_depth_input": False,
            "locked_target": self._locked_target,
            "last_metric_distance_m": self._last_distance_m,
            "last_bbox": self._last_bbox,
            "last_failure": self._last_failure,
            "semantic_inference_active": self._semantic_inference_active,
            "last_request": request,
        }
