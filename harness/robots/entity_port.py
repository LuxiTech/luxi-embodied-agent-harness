"""Backend-neutral contract for manipulable simulation entities.

High-level skills depend on this module only. MuJoCo qpos/geom identifiers and
Isaac USD/PhysX primitives stay behind their respective adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict, runtime_checkable


ENTITY_ALIASES = {
    "water_bottle": "water_bottle",
    "water_cup": "water_bottle",
    "矿泉水瓶": "water_bottle",
    "水瓶": "water_bottle",
    "水杯": "water_bottle",
}


def normalize_entity_id(entity_id: object) -> str:
    """Normalize user-facing aliases without depending on a simulator."""

    value = str(entity_id).strip().casefold()
    return ENTITY_ALIASES.get(value, value)


class EntityOperationResult(TypedDict, total=False):
    """Structured evidence returned by every entity backend operation."""

    tool_ok: bool
    operation_ok: bool
    completed: bool
    task_status: str
    backend: str
    entity_id: str
    hand: str | None
    pose: list[float] | None
    contact: bool
    attached: bool
    evidence_timestamp: float
    error: str | None
    details: dict[str, Any]


def entity_result(
    *,
    backend: str,
    entity_id: str,
    task_status: str,
    operation_ok: bool,
    completed: bool = False,
    tool_ok: bool = True,
    **payload: Any,
) -> EntityOperationResult:
    """Build the common result envelope without inventing success evidence."""

    result: EntityOperationResult = {
        "tool_ok": bool(tool_ok),
        "operation_ok": bool(operation_ok),
        "completed": bool(completed and operation_ok),
        "task_status": str(task_status),
        "backend": str(backend),
        "entity_id": str(entity_id),
    }
    result.update(payload)
    return result


@runtime_checkable
class EntityManipulationPort(Protocol):
    """Common high-level surface implemented by each simulator adapter."""

    backend: str

    def entity_state(
        self,
        entity_id: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult: ...

    def contact_state(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult: ...

    def approach(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult: ...

    def grasp(
        self,
        entity_id: str,
        hand: str,
        *,
        evidence_timestamp: float | None = None,
        timeout: float = 8.0,
    ) -> EntityOperationResult: ...

    def carry(
        self,
        entity_id: str,
        hand: str,
        target_pose: Sequence[float] | None = None,
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult: ...

    def place(
        self,
        entity_id: str,
        hand: str,
        target_position: Sequence[float],
        *,
        timeout: float = 8.0,
    ) -> EntityOperationResult: ...

    def release(
        self,
        entity_id: str,
        hand: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult: ...

    def reset_entity(
        self,
        entity_id: str,
        *,
        timeout: float = 2.0,
    ) -> EntityOperationResult: ...


def validate_entity_result(
    result: Mapping[str, Any],
    *,
    backend: str,
    entity_id: str,
) -> None:
    """Raise when an adapter violates the shared result/evidence contract."""

    required = {
        "tool_ok",
        "operation_ok",
        "completed",
        "task_status",
        "backend",
        "entity_id",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"entity result missing fields: {sorted(missing)}")
    if result["backend"] != backend:
        raise ValueError("entity result backend does not match adapter")
    if result["entity_id"] != entity_id:
        raise ValueError("entity result id does not match request")
    if not isinstance(result["task_status"], str) or not result["task_status"]:
        raise ValueError("entity result task_status must be non-empty")
    for field in ("tool_ok", "operation_ok", "completed"):
        if not isinstance(result[field], bool):
            raise ValueError(f"entity result {field} must be boolean")
    if result["completed"] and not result["operation_ok"]:
        raise ValueError("incomplete operation cannot report completed=true")
