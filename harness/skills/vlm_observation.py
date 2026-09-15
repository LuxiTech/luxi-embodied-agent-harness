"""Native DimOS skills for structured visual questions."""

from __future__ import annotations

import json
import math
from threading import RLock
from typing import Any


def _pixel_bbox(value: Any, *, width: int, height: int, qwen_1000: bool) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if qwen_1000:
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    elif max(abs(item) for item in (x1, y1, x2, y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def normalize_visual_result(result: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    """Return a copy with every supported bbox represented in image pixels."""

    normalized = dict(result)

    def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        current = dict(item)
        if "bbox_2d" in current:
            current["bbox"] = _pixel_bbox(
                current.pop("bbox_2d"), width=width, height=height, qwen_1000=True
            )
        elif "bbox" in current:
            current["bbox"] = _pixel_bbox(
                current["bbox"], width=width, height=height, qwen_1000=False
            )
        return current

    normalized = normalize_item(normalized)
    objects = normalized.get("objects")
    if isinstance(objects, list):
        normalized["objects"] = [
            normalize_item(item) if isinstance(item, dict) else item for item in objects
        ]
    return normalized


def _request_metadata(model: Any) -> dict[str, Any]:
    reader = getattr(model, "last_request_metadata", None)
    if not callable(reader):
        return {}
    try:
        metadata = reader()
    except Exception:  # noqa: BLE001 - tracing must never mask a skill result
        return {}
    if not isinstance(metadata, dict):
        return {}
    allowed = (
        "request_id",
        "provider_request_id",
        "frame_timestamp",
        "http_attempts",
        "started_at",
        "finished_at",
        "in_flight",
        "error",
    )
    return {key: metadata[key] for key in allowed if key in metadata}


# DimOS imports stay below the pure helpers so metric/unit tests can still be
# read in environments where only the Luxi repository is available.
try:
    from reactivex.disposable import Disposable

    from dimos.agents.annotation import skill
    from dimos.core.core import rpc
    from dimos.core.module import Module
    from dimos.core.stream import In
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.utils.llm_utils import extract_json
except ImportError:  # pragma: no cover - bootstrap has a dedicated doctor check
    Disposable = None  # type: ignore[assignment, misc]
    Module = object  # type: ignore[assignment, misc]


class VlmObservationSkill(Module):  # type: ignore[misc, valid-type]
    """Analyze the newest robot RGB frame with the configured remote VLM."""

    if Disposable is not None:
        color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_image: Any | None = None
        self._lock = RLock()
        from dimos.models.vl import qwen as qwen_module

        self._vl_model = qwen_module.QwenVlModel()

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_color_image)))

    @rpc
    def stop(self) -> None:
        self._vl_model.stop()
        super().stop()

    def _on_color_image(self, image: Any) -> None:
        with self._lock:
            self._latest_image = image

    def _analyze(self, prompt: str) -> str:
        with self._lock:
            image = self._latest_image
        if image is None:
            return json.dumps({"ok": False, "error": "No camera frame is available"})
        try:
            raw = self._vl_model.query(
                image,
                prompt,
                response_format={"type": "json_object"},
            )
            result = extract_json(raw)
        except Exception as exc:  # noqa: BLE001 - error is returned as a skill result
            failure = {"ok": False, "error": f"VLM API request failed: {exc}"}
            request = _request_metadata(self._vl_model)
            if request:
                failure["request"] = request
            return json.dumps(failure, ensure_ascii=False)
        if not isinstance(result, dict):
            return json.dumps({"ok": False, "error": "VLM returned non-object JSON"})
        result = normalize_visual_result(result, width=image.width, height=image.height)
        response = {
            "ok": True,
            "model": self._vl_model.config.model_name,
            "frame_timestamp": float(image.ts),
            "image_size": [image.width, image.height],
            "bbox_coordinate_system": "pixels",
            **result,
        }
        request = _request_metadata(self._vl_model)
        if request:
            response["request"] = request
        return json.dumps(response, ensure_ascii=False)

    @skill
    def analyze_scene(self, question: str = "Describe the current scene") -> str:
        """Answer a visual question about the latest camera frame.

        Args:
            question: Question grounded only in what the camera currently sees.
        """
        return self._analyze(
            f"""Analyze the robot camera frame and answer: {question}
Return only one JSON object with keys scene, objects, hazards, answer and confidence.
Each object may include label, bbox [x1,y1,x2,y2] and confidence. Use pixel
coordinates and do not invent objects that are not visible."""
        )

    @skill
    def find_visual_target(self, target: str) -> str:
        """Locate one target in the latest camera frame.

        Args:
            target: Object or person description to locate.
        """
        return self._analyze(
            f"""Find this target in the robot camera frame: {target}
Return only one JSON object with keys target, found, bbox, confidence and evidence.
If found, bbox is [x1,y1,x2,y2] in pixels. Use only current visible evidence."""
        )

    @skill
    def verify_visual_condition(self, condition: str) -> str:
        """Verify a condition using only the newest camera frame.

        Args:
            condition: Visual condition that must be checked.
        """
        return self._analyze(
            f"""Verify this condition using only the current robot camera frame: {condition}
Return only one JSON object with keys condition, satisfied, confidence and evidence.
If ambiguous, set satisfied to false and lower confidence."""
        )
