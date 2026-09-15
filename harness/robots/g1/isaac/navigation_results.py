"""Provider-neutral ownership of the native Isaac terminal navigation Skill."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping




NAVIGATE_TO_POSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": -100.0, "maximum": 100.0},
        "y": {"type": "number", "minimum": -100.0, "maximum": 100.0},
        "yaw_degrees": {
            "type": "number",
            "minimum": -360.0,
            "maximum": 360.0,
        },
        "timeout_seconds": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 110.0,
        },
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}

NAVIGATE_TO_TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location_name": {"type": "string", "minLength": 1, "maxLength": 80},
        "timeout_seconds": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 110.0,
        },
    },
    "required": ["location_name"],
    "additionalProperties": False,
}

EXPLORE_FRONTIERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "timeout": {
            "type": "number",
            "minimum": 20.0,
            "maximum": 180.0,
        },
        "max_frontiers": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
        },
    },
    "additionalProperties": False,
}

OBJECT_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "standoff_distance": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 3.0,
        },
        "timeout": {
            "type": "number",
            "minimum": 20.0,
            "maximum": 180.0,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

FOLLOW_PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "follow_distance": {
            "type": "number",
            "minimum": 1.2,
            "maximum": 2.0,
        },
        "duration": {"type": "number", "minimum": 5.0, "maximum": 60.0},
        "timeout": {"type": "number", "minimum": 65.0, "maximum": 180.0},
    },
    "additionalProperties": False,
}

APPROACH_PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "standoff_distance": {
            "type": "number",
            "minimum": 1.0,
            "maximum": 2.0,
        },
        "timeout": {"type": "number", "minimum": 3.0, "maximum": 30.0},
    },
    "additionalProperties": False,
}

NAVIGATE_WITH_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "standoff_distance": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 3.0,
        },
        "timeout": {"type": "number", "minimum": 20.0, "maximum": 180.0},
    },
    "required": ["query"],
    "additionalProperties": False,
}

NAVIGATION_CAPABILITIES = frozenset(
    {
        "navigate_to_pose",
        "navigate_to_tag",
        "explore_frontiers",
        "object_search",
        "follow_person",
        "approach_person",
        "navigate_with_text",
    }
)


def _native_result_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep acknowledgement payloads bounded while preserving outcome fields."""

    names = (
        "tool_ok",
        "task_status",
        "completed",
        "planner_goal_reached",
        "stationary_confirmed",
        "elapsed_s",
        "message",
        "error",
    )
    return {name: value[name] for name in names if name in value}


def parse_navigation_result(raw: str, capability_id: str) -> dict[str, Any]:
    raw_text = str(raw)
    raw_bytes = raw_text.encode("utf-8")
    raw_digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "tool_ok": False,
            "completed": False,
            "task_status": "invalid_tool_result",
            "error": f"{capability_id} returned non-JSON output",
            "evidence": {
                "native_skill_result": raw_text[:4096],
                "native_skill_result_bytes": len(raw_bytes),
                "native_skill_result_sha256": raw_digest,
            },
        }
    value = dict(payload)
    status = str(value.get("task_status", "invalid_tool_result"))
    follow_verified = False
    if capability_id == "explore_frontiers":
        frontiers_reached = value.get("frontiers_reached")
        goals = value.get("goals")
        planner_goal_reached = bool(
            isinstance(goals, list)
            and any(
                isinstance(goal, Mapping)
                and goal.get("navigation_status") == "frontier_reached"
                for goal in goals
            )
        )
        completion_claimed = bool(
            value.get("completed") is True
            and status in {"exploration_complete", "exploration_budget_complete"}
        )
        frontier_exploration_verified = bool(
            completion_claimed
            and value.get("planner_goal_space") == "known_free_only"
            and isinstance(frontiers_reached, int)
            and not isinstance(frontiers_reached, bool)
            and frontiers_reached > 0
            and planner_goal_reached
        )
        completed = completion_claimed
    elif capability_id in {"object_search", "approach_person"}:
        planner_goal_reached = value.get("planner_goal_reached") is True
        verification_frame_timestamp = value.get("verification_frame_timestamp")
        stationary_confirmed_at = value.get("stationary_confirmed_at")
        verification = value.get("verification")
        verification_observed = bool(
            value.get("completed") is True
            and status == "arrived_verified"
            and planner_goal_reached
            and isinstance(verification_frame_timestamp, (int, float))
            and not isinstance(verification_frame_timestamp, bool)
            and isinstance(stationary_confirmed_at, (int, float))
            and not isinstance(stationary_confirmed_at, bool)
            and float(verification_frame_timestamp) > float(stationary_confirmed_at)
            and isinstance(verification, Mapping)
        )
        if capability_id == "approach_person":
            try:
                requested_distance = float(value["requested_standoff_distance_m"])
                effective_distance = float(value["effective_standoff_distance_m"])
                target_distance = float(value["target_distance_m"])
                physical_stop_latency_ms = float(value["physical_stop_latency_ms"])
            except (KeyError, TypeError, ValueError, OverflowError):
                verification_observed = False
            else:
                verification_observed = bool(
                    verification_observed
                    and 1.0 <= requested_distance <= 2.0
                    and math.isclose(
                        effective_distance,
                        requested_distance,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    and abs(target_distance - effective_distance) <= 0.3
                    and 0.0 <= physical_stop_latency_ms <= 500.0
                )
        completed = value.get("completed") is True and status == "arrived_verified"
        frontier_exploration_verified = False
    elif capability_id == "follow_person":
        planner_goal_reached = False
        verification_frame_timestamp = value.get("verification_frame_timestamp")
        stationary_confirmed_at = value.get("stationary_confirmed_at")
        try:
            tracking_coverage = float(value.get("tracking_coverage", 0.0))
            physical_stop_latency_ms = float(
                value.get("physical_stop_latency_ms", math.inf)
            )
        except (TypeError, ValueError, OverflowError):
            tracking_coverage = 0.0
            physical_stop_latency_ms = math.inf
        native_stationary = bool(
            isinstance(stationary_confirmed_at, (int, float))
            and not isinstance(stationary_confirmed_at, bool)
        )
        follow_verified = bool(
            value.get("completed") is True
            and status == "follow_verified"
            and native_stationary
            and isinstance(verification_frame_timestamp, (int, float))
            and not isinstance(verification_frame_timestamp, bool)
            and float(verification_frame_timestamp) > float(stationary_confirmed_at)
            and tracking_coverage >= 0.9
            and 0.0 <= physical_stop_latency_ms <= 500.0
        )
        completed = value.get("completed") is True and status == "follow_verified"
        frontier_exploration_verified = False
    elif capability_id == "navigate_with_text":
        route = value.get("navigate_with_text_route")
        planner_goal_reached = value.get("planner_goal_reached") is True
        verification_frame_timestamp = value.get("verification_frame_timestamp")
        stationary_confirmed_at = value.get("stationary_confirmed_at")
        visual_route = route in {
            "current_rgbd",
            "persistent_clip_memory",
            "frontier_fallback",
        }
        visual_verified = bool(
            visual_route
            and status == "arrived_verified"
            and planner_goal_reached
            and isinstance(value.get("verification"), Mapping)
            and isinstance(verification_frame_timestamp, (int, float))
            and not isinstance(verification_frame_timestamp, bool)
            and isinstance(stationary_confirmed_at, (int, float))
            and not isinstance(stationary_confirmed_at, bool)
            and float(verification_frame_timestamp) > float(stationary_confirmed_at)
        )
        exact_tag_verified = bool(
            route == "exact_tag"
            and status == "navigation_verified"
            and planner_goal_reached
            and value.get("stationary_confirmed") is True
            and isinstance(value.get("position_error_m"), (int, float))
            and not isinstance(value.get("position_error_m"), bool)
            and 0.0 <= float(value["position_error_m"]) <= 0.5
        )
        text_navigation_verified = bool(
            value.get("completed") is True
            and value.get("used_scene_truth") is False
            and (exact_tag_verified or visual_verified)
        )
        completed = value.get("completed") is True
        frontier_exploration_verified = False
    else:
        completed = value.get("completed") is True and status == "navigation_verified"
        frontier_exploration_verified = False
        planner_goal_reached = value.get("planner_goal_reached") is True
    native_summary = _native_result_summary(value)
    evidence = {
        "native_skill_result": json.dumps(
            native_summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "native_skill_result_bytes": len(raw_bytes),
        "native_skill_result_sha256": raw_digest,
        "native_stationary_claim": value.get("stationary_confirmed") is True,
        "planner_goal_reached": planner_goal_reached,
        "frontier_exploration_verified": frontier_exploration_verified,
        "verification_observed": (
            verification_observed
            if capability_id in {"object_search", "approach_person"}
            else False
        ),
        "follow_verified": follow_verified,
        "text_navigation_verified": (
            text_navigation_verified
            if capability_id == "navigate_with_text"
            else False
        ),
    }
    for name in (
        "stationary_samples",
        "position_error_m",
        "yaw_error_rad",
        "distance_m",
        "target",
        "frontiers_attempted",
        "frontiers_reached",
        "known_cells_before",
        "known_cells_after",
        "information_gain_cells",
        "termination_reason",
        "planner_goal_space",
        "goals",
        "map_acquisition",
        "target_distance_m",
        "verification_frame_timestamp",
        "verification",
        "viewpoint_search",
        "vision_requests",
        "requested_standoff_distance_m",
        "effective_standoff_distance_m",
        "requested_follow_duration_s",
        "verified_tracking_duration_s",
        "requested_follow_distance_m",
        "target_bearing_degrees",
        "tracked_frames",
        "tracking_coverage",
        "max_tracking_gap_s",
        "physical_stop_latency_ms",
        "stop_command_completed_at",
        "stationary_confirmed_at",
        "navigate_with_text_route",
        "query",
        "semantic_memory_similarity",
        "used_scene_truth",
    ):
        if name in value:
            evidence[name] = value[name]
    return {
        "ok": bool(value.get("tool_ok") is True),
        "tool_ok": bool(value.get("tool_ok") is True),
        "completed": completed,
        "task_status": status,
        "error": value.get("error"),
        "evidence": evidence,
        # The complete native payload remains the immutable control-channel
        # result artifact.  Do not duplicate its large audit tree in ack.json.
        "native_result": native_summary,
    }
