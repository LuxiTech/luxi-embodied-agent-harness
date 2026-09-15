"""Shared robot execution boundary, independent of model providers."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

def load_physical_cutover_gate(project_root: Path) -> dict[str, Any]:
    """Load the reviewed production ownership gate, failing closed."""

    path = project_root / "config" / "physical-cutover.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "backends": {}}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("backends"), dict):
        return {"schema_version": 1, "backends": {}}
    return payload

def accepted_cutover_tools(project_root: Path, backend: str) -> frozenset[str]:
    gate = load_physical_cutover_gate(project_root)
    entry = gate["backends"].get(backend, {})
    if not isinstance(entry, Mapping) or entry.get("status") not in {
        "accepted",
        "partial",
    }:
        return frozenset()
    tools = entry.get("accepted_tools", [])
    if not isinstance(tools, list) or not all(isinstance(name, str) for name in tools):
        return frozenset()
    return frozenset(name.strip() for name in tools if name.strip())

def default_cutover_tools(project_root: Path, backend: str) -> frozenset[str]:
    gate = load_physical_cutover_gate(project_root)
    entry = gate["backends"].get(backend, {})
    accepted = accepted_cutover_tools(project_root, backend)
    defaults = entry.get("default_enabled", []) if isinstance(entry, Mapping) else []
    if not isinstance(defaults, list) or not all(isinstance(name, str) for name in defaults):
        return frozenset()
    return frozenset(defaults) & accepted

def configured_cutover_tools(
    project_root: Path,
    backend: str,
    value: str | None,
    *,
    setting: str,
) -> frozenset[str]:
    """Resolve a fail-closed per-process cutover rollback selection.

    An override may only remove reviewed default capabilities. It cannot use
    process configuration to promote a merely accepted/non-default tool.
    """

    defaults = default_cutover_tools(project_root, backend)
    if value is None:
        return defaults
    normalized = value.strip()
    if normalized.casefold() == "disabled":
        return frozenset()
    requested = frozenset(
        name.strip() for name in normalized.split(",") if name.strip()
    )
    unsupported = requested - defaults
    if unsupported:
        raise ValueError(
            f"{setting} may only select reviewed default tools; invalid: "
            f"{sorted(unsupported)}"
        )
    return requested

def runtime_fenced_cutover_tools(
    project_root: Path, backend: str
) -> frozenset[str]:
    """Return only the separately reviewed RuntimeHost ownership subset."""

    gate = load_physical_cutover_gate(project_root)
    entry = gate["backends"].get(backend, {})
    accepted = accepted_cutover_tools(project_root, backend)
    fenced = entry.get("runtime_fenced_tools", []) if isinstance(entry, Mapping) else []
    if not isinstance(fenced, list) or not all(isinstance(name, str) for name in fenced):
        return frozenset()
    # RuntimeHost fencing is a narrower ownership gate, never an implicit
    # promotion beyond the backend's reviewed physical Tool set.
    return frozenset(name.strip() for name in fenced if name.strip()) & accepted
