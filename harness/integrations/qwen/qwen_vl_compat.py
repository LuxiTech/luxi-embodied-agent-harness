"""Safely reuse the Luxi Qwen configuration for DimOS vision skills.

The pinned DimOS Qwen VLM only reads a plaintext ``ALIBABA_API_KEY`` and
hard-codes the international DashScope endpoint and an older model.  Luxi's UI
already keeps the key in a mode-0600 file and may use a regional endpoint.  The
launcher installs this compatibility class before DimOS workers are forked so
both agent layers share configuration without exporting the secret.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from functools import cached_property
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, RLock
from typing import Any
from urllib.parse import urlsplit
import uuid


DEFAULT_DIMOS_VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DIMOS_VLM_MODEL = "qwen2.5-vl-72b-instruct"
DEFAULT_DIMOS_VLM_HARD_TIMEOUT_SECONDS = 50.0

ClientFactory = Callable[[str, str], Any]


def _extract_json_payload(response: str) -> Any | None:
    decoder = json.JSONDecoder()
    starts = sorted(
        index for index in (response.find("["), response.find("{")) if index >= 0
    )
    for start in starts:
        try:
            value, _ = decoder.raw_decode(response[start:])
        except json.JSONDecodeError:
            # Qwen occasionally returns Python-style single-quoted mappings
            # despite an explicit JSON-only prompt. ``literal_eval`` accepts
            # only literal containers and keeps this recovery non-executable.
            closing = "]" if response[start] == "[" else "}"
            end = response.rfind(closing)
            if end < start or end - start > 10_000:
                continue
            try:
                value = ast.literal_eval(response[start : end + 1])
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _bbox_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("objects", "detections", "results"):
        nested = payload.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return [payload]


def _is_qwen3_model(model_name: str | None) -> bool:
    normalized = (model_name or "").strip().casefold().replace("_", "-")
    return normalized.startswith("qwen3")


def normalize_qwen_bbox_payload(
    response: str,
    *,
    width: int,
    height: int,
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Convert old/new Qwen bbox JSON into one pixel-coordinate detection."""
    if width <= 0 or height <= 0:
        return None
    candidates: list[tuple[float, dict[str, Any]]] = []
    for item in _bbox_items(_extract_json_payload(response)):
        coordinate_key = "bbox_2d" if "bbox_2d" in item else "bbox"
        raw_bbox = item.get(coordinate_key)
        if raw_bbox is None:
            top_left, bottom_right = item.get("top_left"), item.get("bottom_right")
            if (isinstance(top_left, (list, tuple)) and len(top_left) == 2
                    and isinstance(bottom_right, (list, tuple)) and len(bottom_right) == 2):
                raw_bbox = [*top_left, *bottom_right]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in raw_bbox)
        except (TypeError, ValueError):
            continue
        coordinates = (x1, y1, x2, y2)
        if not all(math.isfinite(value) for value in coordinates):
            continue
        unit_coordinates = all(0.0 <= value <= 1.0 for value in coordinates)
        thousand_coordinates = coordinate_key == "bbox_2d" or _is_qwen3_model(
            model_name
        )
        if unit_coordinates:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        elif thousand_coordinates:
            if any(value < 0.0 or value > 1_000.0 for value in coordinates):
                continue
            x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
            y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
        x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        y1, y2 = sorted(
            (max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2)))
        )
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue
        name = str(item.get("label") or item.get("name") or "object")
        normalized = {
            "name": name,
            "bbox": [round(value, 4) for value in (x1, y1, x2, y2)],
        }
        candidates.append((area, normalized))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _is_bbox_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "bbox" in lowered and "top-left" in lowered and "bottom-right" in lowered


def _conservative_bbox_prompt(
    prompt: str,
    *,
    normalized_coordinates: bool = False,
) -> str:
    """Require positive identity evidence instead of forcing a plausible box."""

    rules = (
        "\nDetection policy: return a tight bounding box only when the requested "
        "object category is clearly recognizable. Never guess or substitute a "
        "plain wall, partition, pillar, table, or unrelated object. A severely "
        "occluded narrow strip is not sufficient evidence. If identity is "
        "ambiguous or the object is not clearly visible, return None."
    )
    lowered = prompt.lower()
    if "door" in lowered or "门" in prompt:
        rules += (
            " A door must have recognizable door evidence such as a panel or "
            "opening together with its surrounding frame; a featureless planar "
            "surface is not a door."
        )
    if normalized_coordinates:
        rules += " Return every bbox coordinate normalized to the range 0 through 999."
    return prompt + rules


def _supports_thinking_toggle(model_name: str) -> bool:
    """Return whether Alibaba documents per-request thinking for this Qwen."""

    return _is_qwen3_model(model_name)


def _configured_value(
    environment: Mapping[str, str],
    dedicated_name: str,
    shared_name: str,
    default: str,
) -> str:
    return (
        environment.get(dedicated_name, "").strip()
        or environment.get(shared_name, "").strip()
        or default
    )


def configured_qwen_vl_base_url(environment: Mapping[str, str] | None = None) -> str:
    """Return a validated VLM endpoint without exposing credentials."""
    values = os.environ if environment is None else environment
    base_url = _configured_value(
        values,
        "LUXI_DIMOS_VLM_BASE_URL",
        "LUXI_QWEN_BASE_URL",
        DEFAULT_DIMOS_VLM_BASE_URL,
    ).rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("DimOS Qwen VLM base URL 必须是无凭据、无查询参数的 HTTPS URL")
    return base_url


def configured_qwen_vl_model(environment: Mapping[str, str] | None = None) -> str:
    """Use a dedicated VLM model when set, otherwise reuse the UI model."""
    values = os.environ if environment is None else environment
    return _configured_value(
        values,
        "LUXI_DIMOS_VLM_MODEL",
        "LUXI_QWEN_MODEL",
        DEFAULT_DIMOS_VLM_MODEL,
    )


def configured_qwen_vl_hard_timeout(
    environment: Mapping[str, str] | None = None,
) -> float:
    values = os.environ if environment is None else environment
    raw = values.get("LUXI_DIMOS_VLM_HARD_TIMEOUT_SECONDS", "").strip()
    try:
        configured = float(raw) if raw else DEFAULT_DIMOS_VLM_HARD_TIMEOUT_SECONDS
    except ValueError as exc:
        raise RuntimeError("LUXI_DIMOS_VLM_HARD_TIMEOUT_SECONDS 必须是数字") from exc
    if not math.isfinite(configured) or not 10.0 <= configured <= 55.0:
        raise RuntimeError("LUXI_DIMOS_VLM_HARD_TIMEOUT_SECONDS 必须在 10–55 秒之间")
    return configured


def isolated_chat_completion(
    *,
    api_key: str,
    base_url: str,
    api_kwargs: Mapping[str, Any],
    hard_timeout_seconds: float,
    runner: Callable[..., Any] | None = None,
) -> tuple[str, str | None]:
    """Run one request in a killable process with a true wall-clock deadline."""

    run = subprocess.run if runner is None else runner
    worker_environment = {
        name: value
        for name, value in os.environ.items()
        if "API_KEY" not in name.upper()
    }
    payload = json.dumps(
        {
            "api_key": api_key,
            "base_url": base_url,
            "request": dict(api_kwargs),
            "http_timeout_seconds": min(48.0, hard_timeout_seconds - 2.0),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        completed = run(
            [sys.executable, "-m", "harness.integrations.qwen.qwen_vl_worker"],
            input=payload,
            text=True,
            capture_output=True,
            timeout=hard_timeout_seconds,
            check=False,
            start_new_session=True,
            env=worker_environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"VLM request exceeded {hard_timeout_seconds:.1f}s hard deadline"
        ) from exc
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("隔离 VLM 进程没有返回有效 JSON") from exc
    if completed.returncode != 0 or not isinstance(result, dict) or not result.get("ok"):
        error_type = str(result.get("error_type", "VlmWorkerError"))[:80]
        error = str(result.get("error", "isolated VLM request failed"))
        # Keep the cross-process error boundary safe even if a provider or
        # client library unexpectedly echoes request credentials.
        error = error.replace(api_key, "[redacted]")[:400]
        if "timeout" in error_type.casefold() or "timeout" in error.casefold():
            raise TimeoutError(error)
        raise RuntimeError(f"{error_type}: {error}")
    content = result.get("content")
    if not isinstance(content, str):
        raise RuntimeError("隔离 VLM 响应缺少文本内容")
    request_id = result.get("request_id")
    return content, str(request_id) if request_id else None


def _validate_api_key(api_key: str) -> str:
    api_key = api_key.strip()
    if not api_key.startswith("sk-") or any(character.isspace() for character in api_key):
        raise RuntimeError("DimOS Qwen VLM API key 文件格式无效")
    return api_key


def _api_key_from_file_content(content: str) -> str:
    """Accept either a one-line key or the protected deployment dotenv."""

    stripped = content.strip()
    if "\n" not in stripped and "=" not in stripped:
        return _validate_api_key(stripped)
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() in {
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "ALIBABA_API_KEY",
            "OPENAI_API_KEY",
        }:
            return _validate_api_key(value.strip().strip("\"'"))
    raise RuntimeError("DimOS Qwen VLM API key 文件格式无效")


def read_qwen_vl_api_key(environment: Mapping[str, str] | None = None) -> str:
    """Read the shared mode-0600 key, falling back to the upstream env name."""
    values = os.environ if environment is None else environment
    configured_path = (
        values.get("LUXI_DIMOS_VLM_API_KEY_FILE", "").strip()
        or values.get("LUXI_QWEN_API_KEY_FILE", "").strip()
    )
    if configured_path:
        key_file = Path(configured_path).expanduser()
        if not key_file.is_file() or not os.access(key_file, os.R_OK):
            raise RuntimeError("DimOS Qwen VLM API key 文件不存在或不可读")
        try:
            mode = key_file.stat().st_mode & 0o777
        except OSError as exc:
            raise RuntimeError("无法检查 DimOS Qwen VLM API key 文件") from exc
        if mode & 0o077:
            raise RuntimeError("DimOS Qwen VLM API key 文件权限过宽；请执行 chmod 600")
        try:
            return _api_key_from_file_content(key_file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError("无法读取 DimOS Qwen VLM API key 文件") from exc

    api_key = values.get("ALIBABA_API_KEY", "")
    if api_key:
        return _validate_api_key(api_key)
    raise RuntimeError(
        "未配置 LUXI_QWEN_API_KEY_FILE、LUXI_DIMOS_VLM_API_KEY_FILE 或 ALIBABA_API_KEY"
    )


def _default_client_factory(api_key: str, base_url: str) -> Any:
    # Desktop tools export a socks:// proxy that HTTPX may reject. DashScope is
    # directly reachable, so isolate this client instead of mutating globals.
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(48.0, connect=15.0),
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )


def build_qwen_vl_model_class(
    original_model_class: type[Any],
    *,
    environment: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
) -> type[Any]:
    """Build a drop-in DimOS Qwen VLM using Luxi's safe configuration."""
    values = os.environ if environment is None else environment
    factory = client_factory or _default_client_factory
    isolate_bbox_requests = client_factory is None
    hard_timeout_seconds = configured_qwen_vl_hard_timeout(values)

    class LuxiQwenVlModel(original_model_class):  # type: ignore[misc, valid-type]
        _luxi_qwen_vl_compat = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("model_name", configured_qwen_vl_model(values))
            super().__init__(*args, **kwargs)
            self._luxi_query_lock = Lock()
            self._luxi_metadata_lock = RLock()
            self._luxi_last_request: dict[str, Any] = {}

        @cached_property
        def _client(self) -> Any:
            return factory(
                self._configured_api_key(),
                configured_qwen_vl_base_url(values),
            )

        def _configured_api_key(self) -> str:
            configured_key = getattr(self.config, "api_key", None)
            return (
                _validate_api_key(configured_key)
                if configured_key
                else read_qwen_vl_api_key(values)
            )

        def last_request_metadata(self) -> dict[str, Any]:
            with self._luxi_metadata_lock:
                return dict(self._luxi_last_request)

        def query(self, image: Any, query: str, **kwargs: Any) -> str:
            if not self._luxi_query_lock.acquire(blocking=False):
                raise RuntimeError("VLM request already in flight for this model instance")

            request_id = uuid.uuid4().hex
            frame_timestamp = getattr(image, "ts", None)
            request_metadata: dict[str, Any] = {
                "request_id": request_id,
                "provider_request_id": None,
                "frame_timestamp": (
                    float(frame_timestamp) if frame_timestamp is not None else None
                ),
                "http_attempts": 1,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "finished_at": None,
                "in_flight": True,
                "error": None,
                "response_excerpt": None,
                "normalized_detection": None,
            }
            with self._luxi_metadata_lock:
                self._luxi_last_request = request_metadata

            completion: Any | None = None
            provider_request_id: str | None = None
            try:
                bbox_prompt = _is_bbox_prompt(query)
                if kwargs or bbox_prompt:
                    # The pinned Qwen implementation predates response_format and
                    # rejects all keyword arguments. Use the same prepared image
                    # and client while forwarding only explicit API options.
                    prepared, _ = self._prepare_image(image)
                    api_kwargs: dict[str, Any] = {
                        "model": self.config.model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": (
                                                "data:image/png;base64,"
                                                f"{prepared.to_base64()}"
                                            )
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": (
                                            _conservative_bbox_prompt(
                                                query,
                                                normalized_coordinates=_is_qwen3_model(
                                                    str(self.config.model_name)
                                                ),
                                            )
                                            if bbox_prompt
                                            else query
                                        ),
                                    },
                                ],
                            }
                        ],
                        **kwargs,
                    }
                    if bbox_prompt:
                        # Bounding-box extraction is a perception primitive,
                        # not a creative response.  Keep exactly one request
                        # and remove provider sampling/reasoning variance.
                        api_kwargs.setdefault("temperature", 0.0)
                        api_kwargs.setdefault("max_tokens", 128)
                        if _supports_thinking_toggle(str(self.config.model_name)):
                            extra_body = dict(api_kwargs.get("extra_body") or {})
                            extra_body.setdefault("enable_thinking", False)
                            api_kwargs["extra_body"] = extra_body
                            request_metadata["thinking_enabled"] = bool(
                                extra_body["enable_thinking"]
                            )
                        request_metadata["max_tokens"] = int(
                            api_kwargs["max_tokens"]
                        )
                    if bbox_prompt and isolate_bbox_requests:
                        request_metadata["execution_isolation"] = "subprocess"
                        request_metadata["hard_timeout_seconds"] = hard_timeout_seconds
                        response, provider_request_id = isolated_chat_completion(
                            api_key=self._configured_api_key(),
                            base_url=configured_qwen_vl_base_url(values),
                            api_kwargs=api_kwargs,
                            hard_timeout_seconds=hard_timeout_seconds,
                        )
                    else:
                        completion = self._client.chat.completions.create(**api_kwargs)
                        provider_request_id = (
                            str(getattr(completion, "id"))
                            if getattr(completion, "id", None)
                            else None
                        )
                        response = completion.choices[0].message.content or ""
                else:
                    response = super().query(image, query)
                request_metadata["response_excerpt"] = response[:500]
                if not bbox_prompt:
                    return response
                normalized = normalize_qwen_bbox_payload(
                    response,
                    width=int(image.width),
                    height=int(image.height),
                    model_name=str(self.config.model_name),
                )
                request_metadata["bbox_coordinate_space"] = (
                    "normalized_0_999"
                    if _is_qwen3_model(str(self.config.model_name))
                    else "legacy_auto"
                )
                request_metadata["normalized_detection"] = normalized is not None
                if normalized is None:
                    return response
                return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
            except BaseException as exc:
                request_metadata["error"] = f"{type(exc).__name__}: {exc}"[:500]
                raise
            finally:
                request_metadata["provider_request_id"] = provider_request_id
                request_metadata["finished_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
                request_metadata["in_flight"] = False
                with self._luxi_metadata_lock:
                    self._luxi_last_request = dict(request_metadata)
                self._luxi_query_lock.release()

    LuxiQwenVlModel.__name__ = original_model_class.__name__
    LuxiQwenVlModel.__qualname__ = original_model_class.__qualname__
    return LuxiQwenVlModel


def install_qwen_vl_compat() -> bool:
    """Install the compatibility model before DimOS constructs its skills."""
    from dimos.models.vl import qwen as qwen_module

    current = qwen_module.QwenVlModel
    if getattr(current, "_luxi_qwen_vl_compat", False):
        return True
    qwen_module.QwenVlModel = build_qwen_vl_model_class(current)
    return True
