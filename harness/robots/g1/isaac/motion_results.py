"""Provider-neutral ownership of reviewed native Isaac motion Skills."""

from __future__ import annotations

import re
from typing import Any




RELATIVE_MOVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "forward": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "left": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "degrees": {"type": "number", "minimum": -180.0, "maximum": 180.0},
    },
    "required": [],
    "additionalProperties": False,
}

MOVE_DISTANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "distance_m": {
            "type": "number",
            "anyOf": [
                {"minimum": -3.0, "maximum": -0.1},
                {"minimum": 0.1, "maximum": 3.0},
            ],
        },
    },
    "required": ["distance_m"],
    "additionalProperties": False,
}

TURN_AROUND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

MOVE_ROBOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": -0.25, "maximum": 0.25},
        "y": {"type": "number", "minimum": -0.20, "maximum": 0.20},
        "yaw": {"type": "number", "minimum": -0.60, "maximum": 0.60},
        "duration": {"type": "number", "minimum": 0.0, "maximum": 3.0},
    },
    "required": ["x"],
    "additionalProperties": False,
}


def _parse_native_result(raw: str, capability_id: str = "relative_move") -> dict[str, Any]:
    text = str(raw).strip()
    prefix = text.partition(":")[0].strip()
    cancelled = "cancelled=true" in text.casefold()
    verified = prefix in {"relative_move_verified", "turn_verified", "distance_verified"}
    known_failure = prefix in {
        "relative_move_failed",
        "turn_failed",
        "move_distance_failed",
        "velocity_pulse_failed",
    }
    rejected = (
        text.startswith("Simulation motion rejected")
        or prefix == "move_distance_rejected"
    )
    pulse_completed = prefix == "velocity_pulse_completed"
    status = (
        "cancelled"
        if cancelled
        else prefix
        if verified or known_failure
        else "invalid_input"
        if rejected
        else "invalid_tool_result"
    )
    evidence: dict[str, Any] = {
        "native_skill_result": text,
        "native_stationary_claim": "stationary_confirmed=true" in text,
    }
    if verified:
        evidence[prefix] = True
        evidence["native_target_verified"] = True
    for match in re.finditer(
        r"(?P<name>forward|left|rotation)="
        r"(?P<actual>-?\d+(?:\.\d+)?)/(?P<target>-?\d+(?:\.\d+)?)",
        text,
    ):
        evidence[f"measured_{match.group('name')}"] = float(match.group("actual"))
        evidence[f"requested_{match.group('name')}"] = float(match.group("target"))
    distance = re.search(
        r"actual=(?P<actual>-?\d+(?:\.\d+)?)/(?P<target>-?\d+(?:\.\d+)?)m",
        text,
    )
    if distance:
        evidence["measured_distance"] = float(distance.group("actual"))
        evidence["requested_distance"] = float(distance.group("target"))
    evidence["capability_id"] = capability_id
    return {
        "ok": bool(verified or known_failure or pulse_completed),
        "tool_ok": bool(verified or known_failure or pulse_completed),
        "completed": verified,
        "task_status": "velocity_pulse_completed" if pulse_completed else status,
        "error": None if verified or pulse_completed else text,
        "evidence": evidence,
        "native_result": text,
    }
