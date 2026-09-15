"""Decision comparison utilities for old/new Loop shadow validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DecisionDiff:
    equivalent: bool
    old: tuple[tuple[str, str], ...]
    new: tuple[tuple[str, str], ...]
    reason: str


def _canonical(
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    values = []
    for decision in decisions:
        name = str(decision.get("name") or decision.get("tool") or "")
        arguments = decision.get("arguments", {})
        values.append(
            (
                name,
                json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(values)


def compare_tool_decisions(
    old: Iterable[Mapping[str, Any]], new: Iterable[Mapping[str, Any]]
) -> DecisionDiff:
    old_value = _canonical(old)
    new_value = _canonical(new)
    if old_value == new_value:
        return DecisionDiff(True, old_value, new_value, "exact_match")
    old_names = tuple(item[0] for item in old_value)
    new_names = tuple(item[0] for item in new_value)
    reason = "tool_route_changed" if old_names != new_names else "arguments_changed"
    return DecisionDiff(False, old_value, new_value, reason)
