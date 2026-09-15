"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import ast
import json
import math
import re
from typing import Any


MAX_TOOL_OUTPUT_CHARS = 8_000

MAX_MODEL_TOOL_OUTPUT_CHARS = 2_400

ISAAC_PURE_YAW_SETTLE_SECONDS = 1.5

ISAAC_PURE_YAW_SETTLE_INTERVAL_SECONDS = 0.25

NATIVE_VLM_TIMEOUT = 55.0

APPROACH_PERSON_TIMEOUT = 150.0

APPROACH_VISUAL_TARGET_TIMEOUT = 150.0

OBJECT_SEARCH_TIMEOUT = 195.0

FOLLOW_PERSON_TIMEOUT = 195.0

FETCH_OBJECT_PROCESS_TIMEOUT = 245.0

SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9._-]{12,}")

BLIND_FORBIDDEN_KEYS = frozenset(
    {
        "scene",
        "scene_id",
        "scene_xml",
        "scene_asset",
        "seed",
        "xml",
        "scorer",
        "scorer_path",
        "simulator_path",
        "run_token",
    }
)

MAX_VISUAL_STANDOFF_INCREASE_M = 0.8

def _redact(value: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", value)

def _json_text(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[truncated]"

def _parse_structured_cli_output(output: str) -> dict[str, Any] | list[Any] | None:
    """Parse bounded JSON or a legacy Python-literal MCP result.

    Some pinned DimOS MCP clients print a native mapping with single quotes
    instead of serializing it as JSON. ``literal_eval`` keeps that compatibility
    boundary data-only; the recursive check rejects tuples, sets, bytes, complex
    numbers, non-string mapping keys, and any other Python-specific value.
    """

    if not output or len(output) > MAX_TOOL_OUTPUT_CHARS:
        return None
    try:
        parsed: Any = json.loads(output)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(output)
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            return None

    def json_compatible(value: Any) -> bool:
        if value is None or isinstance(value, (bool, int, float, str)):
            return not isinstance(value, float) or math.isfinite(value)
        if isinstance(value, list):
            return all(json_compatible(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and json_compatible(item)
                for key, item in value.items()
            )
        return False

    if not isinstance(parsed, (dict, list)) or not json_compatible(parsed):
        return None
    return parsed

def _blind_safe_payload(value: Any) -> Any:
    """Remove layout/scorer artifacts before blind data reaches Qwen or UI state."""

    if isinstance(value, dict):
        return {
            str(key): _blind_safe_payload(item)
            for key, item in value.items()
            if str(key).strip().lower() not in BLIND_FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_blind_safe_payload(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(
            marker in lowered
            for marker in (
                "<mujoco",
                "scorer.json",
                "simulator.json",
                "/scorer/",
                "\\scorer\\",
                "blind-evaluation/runs/",
            )
        ):
            return "[withheld by blind information boundary]"
    return value

def _structured_mcp_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap structured tool state from CLI text or a native MCP envelope."""

    def unwrap(value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, dict):
            return None
        if "task_status" in value:
            # A formally cut-over provider returns the unified ToolResult
            # envelope.  Its physical proof lives under ``evidence`` whereas
            # the legacy MCP result placed those fields at the top level.
            # Flatten only for the compatibility validator; keep the envelope
            # fields authoritative when names overlap.
            evidence = value.get("evidence")
            return (
                {**evidence, **value}
                if isinstance(evidence, dict)
                else value
            )
        nested = value.get("result")
        payload = unwrap(nested)
        if payload is not None:
            return payload
        content = value.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    payload = unwrap(item.get("text"))
                    if payload is not None:
                        return payload
        return None

    return unwrap(result)

def _closed_loop_mcp_result(
    result: dict[str, Any],
    skill_name: str,
) -> dict[str, Any]:
    payload = _structured_mcp_payload(result)
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": f"{skill_name} 未返回结构化任务状态",
            "completed": False,
            "task_status": "invalid_tool_result",
        }
    task_status = str(payload.get("task_status") or "invalid_tool_result")
    planner_goal_reached = payload.get("planner_goal_reached") is True
    completed = bool(
        payload.get("completed") is True
        and planner_goal_reached
        and task_status == "arrived_verified"
    )
    return {
        **payload,
        "ok": bool(result.get("ok") and payload.get("tool_ok") is True),
        "task_status": task_status,
        "planner_goal_reached": planner_goal_reached,
        "completed": completed,
    }

def _isaac_navigation_mcp_result(result: dict[str, Any]) -> dict[str, Any]:
    """Accept arrival only with planner, pose, and stationary evidence."""

    payload = _structured_mcp_payload(result)
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "Isaac 导航未返回结构化任务状态",
            "completed": False,
            "task_status": "invalid_tool_result",
        }
    status = str(payload.get("task_status") or "invalid_tool_result")
    evidence_ok = bool(
        result.get("ok")
        and payload.get("tool_ok") is True
        and payload.get("completed") is True
        and status == "navigation_verified"
        and payload.get("planner_goal_reached") is True
        and payload.get("stationary_confirmed") is True
        and isinstance(payload.get("stationary_samples"), int)
        and payload["stationary_samples"] >= 3
    )
    if evidence_ok:
        return {
            **payload,
            "ok": True,
            "completed": True,
            "task_status": "navigation_verified",
        }
    return {
        **payload,
        "ok": bool(result.get("ok") and payload.get("tool_ok") is True),
        "completed": False,
        "task_status": status,
    }

def _fetch_object_mcp_result(
    result: dict[str, Any],
    *,
    object_id: str,
    destination: str,
) -> dict[str, Any]:
    """Validate terminal fetch success from its structured physical evidence."""

    payload = _structured_mcp_payload(result)
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "fetch_object 未返回结构化任务状态",
            "completed": False,
            "task_status": "invalid_tool_result",
        }
    task_status = str(payload.get("task_status") or "invalid_tool_result")
    transport_ok = bool(result.get("ok") and payload.get("tool_ok") is True)
    reported_completed = payload.get("completed") is True
    if not reported_completed:
        return {
            **payload,
            "ok": transport_ok,
            "completed": False,
            "task_status": task_status,
        }

    steps = payload.get("steps")
    indexed_steps = {
        step.get("step"): step.get("result")
        for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("step"), str)
        and isinstance(step.get("result"), dict)
    } if isinstance(steps, list) else {}
    expected_steps = {
        "navigate_to",
        "locate_entity",
        "approach_entity",
        "grasp_entity",
        "prepare_carry_entity",
        "carry_entity",
        "confirm_stationary",
    }
    grasp = indexed_steps.get("grasp_entity")
    carry = indexed_steps.get("carry_entity")
    stationary = indexed_steps.get("confirm_stationary")
    evidence_ok = bool(
        task_status == "object_fetched"
        and payload.get("object_id") == object_id
        and payload.get("destination") == destination
        and payload.get("attachment_retained") is True
        and payload.get("planner_goal_reached") is True
        and payload.get("stationary_confirmed") is True
        and expected_steps.issubset(indexed_steps)
        and isinstance(grasp, dict)
        and grasp.get("operation_ok") is True
        and grasp.get("contact_confirmed") is True
        and grasp.get("attached") is True
        and isinstance(carry, dict)
        and carry.get("operation_ok") is True
        and carry.get("attachment_retained") is True
        and isinstance(stationary, dict)
        and stationary.get("stationary_confirmed") is True
    )
    if transport_ok and evidence_ok:
        return {
            **payload,
            "ok": True,
            "completed": True,
            "task_status": "object_fetched",
        }
    return {
        **payload,
        "ok": False,
        "reported_task_status": task_status,
        "completed": False,
        "task_status": "invalid_tool_result",
        "error": "fetch_object 成功声明缺少接触、携带、到达或最终静止证据",
    }

def _validate_visual_approach_standoff(
    outcome: dict[str, Any],
    *,
    query: str,
    requested_distance: float,
) -> dict[str, Any]:
    """Fail closed when a skill's success no longer matches user stop intent."""

    if outcome.get("completed") is not True:
        return outcome
    try:
        reported_requested = float(outcome["requested_standoff_distance_m"])
        effective = float(outcome["effective_standoff_distance_m"])
        target_distance = float(outcome["target_distance_m"])
    except (KeyError, TypeError, ValueError, OverflowError):
        reason = "视觉接近成功结果缺少可验证的停距字段"
    else:
        values = (reported_requested, effective, target_distance)
        vertical_target = any(
            term in query.strip().casefold()
            for term in ("door", "gate", "doorway", "门")
        )
        maximum_effective = requested_distance + (
            0.0 if vertical_target else MAX_VISUAL_STANDOFF_INCREASE_M
        )
        if not all(math.isfinite(value) for value in values):
            reason = "视觉接近成功结果包含无效停距"
        elif abs(reported_requested - requested_distance) > 0.01:
            reason = "视觉接近技能返回的请求停距与用户参数不一致"
        elif not requested_distance - 0.01 <= effective <= maximum_effective + 0.01:
            reason = (
                f"视觉接近技能将停距从 {requested_distance:.2f}m "
                f"异常改为 {effective:.2f}m"
            )
        elif abs(target_distance - effective) > 0.31:
            reason = (
                f"最终目标距离 {target_distance:.2f}m 与生效停距 "
                f"{effective:.2f}m 不一致"
            )
        else:
            return outcome
    return {
        **outcome,
        "ok": False,
        "reported_task_status": outcome.get("task_status"),
        "task_status": "invalid_tool_result",
        "completed": False,
        "error": reason,
    }

def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep model-facing outcomes small while the event store retains full traces."""

    noisy_collections = {"observations", "pulses", "results"}

    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            reduced: dict[str, Any] = {}
            for key, item in value.items():
                if key in noisy_collections:
                    if isinstance(item, (list, tuple)):
                        reduced[f"{key}_count"] = len(item)
                    continue
                reduced[key] = compact(item)
            return reduced
        if isinstance(value, (list, tuple)):
            items = [compact(item) for item in value[:8]]
            if len(value) > 8:
                items.append({"truncated_items": len(value) - 8})
            return items
        if isinstance(value, str) and len(value) > 700:
            return value[:680] + "...[truncated]"
        return value

    return compact(result)

def _visual_branch_failed_result(name: str, result: dict[str, Any]) -> bool:
    if not result.get("ok"):
        return True
    if name in {"approach_visual_target", "object_search"}:
        return not bool(
            result.get("completed") is True
            and result.get("task_status") == "arrived_verified"
        )
    if name in {"approach_person", "follow_person"}:
        return result.get("completed") is not True
    if name == "find_visual_target" and result.get("found") is False:
        return True
    return False

def _tool_result_level(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return "danger"
    if result.get("completed") is False and result.get("task_status"):
        return "warning"
    return "info"
