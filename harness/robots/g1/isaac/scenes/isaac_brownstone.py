"""Brownstone collision policy without Isaac Sim import-time dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class BrownstoneCollisionRole(str, Enum):
    STRUCTURE = "structure"
    NAVIGATION_OBSTACLE = "navigation_obstacle"


@dataclass(frozen=True)
class BrownstoneCollisionTarget:
    group: str
    role: BrownstoneCollisionRole


BROWNSTONE_STRUCTURAL_COLLISION_GROUP_PREFIXES = (
    "Doors",
    "Floors",
    "Landings",
    "Railings",
    "Runs",
    "Stair",
    "Stairs",
    "Structural_Columns",
    "Structural_Foundations",
    "Supports",
    "Top_Rails",
    "Walls",
)

# Brownstone is an AEC visualization asset, not a physics-ready scene. These
# groups are fixed, room-scale obstacles whose omission lets the articulated
# robot enter cabinets, counters and appliances even though they are visible.
# Small wall/ceiling details stay render-only to keep the PhysX scene bounded.
BROWNSTONE_NAVIGATION_OBSTACLE_GROUPS = frozenset(
    {
        "Casework",
        "Generic_Models",
        "Mechanical_Equipment",
        "Plumbing_Fixtures",
        "Specialty_Equipment",
    }
)


def brownstone_collision_target(
    prim_path: str,
) -> BrownstoneCollisionTarget | None:
    """Classify one composed mesh path under Brownstone's Geometry namespace."""

    parts = prim_path.split("/")
    try:
        group = parts[parts.index("Geometry") + 1]
    except (ValueError, IndexError):
        return None
    if group in BROWNSTONE_NAVIGATION_OBSTACLE_GROUPS:
        return BrownstoneCollisionTarget(
            group=group,
            role=BrownstoneCollisionRole.NAVIGATION_OBSTACLE,
        )
    if group.startswith(BROWNSTONE_STRUCTURAL_COLLISION_GROUP_PREFIXES):
        return BrownstoneCollisionTarget(
            group=group,
            role=BrownstoneCollisionRole.STRUCTURE,
        )
    return None


def validate_brownstone_collision_counts(counts: Mapping[str, int]) -> None:
    """Fail closed when a required obstacle group produced no collision mesh."""

    missing = sorted(
        group
        for group in BROWNSTONE_NAVIGATION_OBSTACLE_GROUPS
        if isinstance(counts.get(group), bool)
        or not isinstance(counts.get(group), int)
        or counts[group] <= 0
    )
    if missing:
        raise RuntimeError(
            "Brownstone composed without required navigation collision groups: "
            + ", ".join(missing)
        )
