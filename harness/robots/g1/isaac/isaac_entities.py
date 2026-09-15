"""Backend-owned dynamic entity declarations for the Isaac G1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


ENTITY_ROOT = "/World/LuxiEntities"
WATER_BOTTLE_ID = "water_bottle"


@dataclass(frozen=True)
class IsaacEntitySpec:
    """One deterministic USD/PhysX entity owned by the Luxi adapter."""

    entity_id: str
    prim_path: str
    shape: str
    radius_m: float
    height_m: float
    mass_kg: float
    spawn_offset_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    graspable: bool = True

    def spawn_position(
        self,
        robot_start_position: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return tuple(
            float(origin) + float(offset)
            for origin, offset in zip(
                robot_start_position,
                self.spawn_offset_m,
                strict=True,
            )
        )


WATER_BOTTLE = IsaacEntitySpec(
    entity_id=WATER_BOTTLE_ID,
    prim_path=f"{ENTITY_ROOT}/{WATER_BOTTLE_ID}",
    shape="cylinder",
    radius_m=0.035,
    height_m=0.22,
    mass_kg=0.50,
    # G1 root starts 0.80 m above the local floor. The bottle centre is
    # 0.12 m above that floor and is offset from the robot to avoid startup
    # contact while remaining visible in the acceptance scene.
    spawn_offset_m=(1.20, 0.40, -0.68),
    color_rgb=(0.15, 0.45, 0.95),
)

ENTITY_SPECS = {WATER_BOTTLE.entity_id: WATER_BOTTLE}


def selected_entity_specs(entity_ids: list[str] | tuple[str, ...]) -> tuple[IsaacEntitySpec, ...]:
    """Resolve unique configured IDs and reject an unknown entity."""

    selected: list[IsaacEntitySpec] = []
    seen: set[str] = set()
    for raw in entity_ids:
        entity_id = str(raw).strip()
        if entity_id in seen:
            continue
        try:
            spec = ENTITY_SPECS[entity_id]
        except KeyError as exc:
            raise ValueError(f"unsupported Isaac entity: {entity_id!r}") from exc
        selected.append(spec)
        seen.add(entity_id)
    return tuple(selected)


def valid_entity_snapshot(
    payload: Mapping[str, Any] | None,
    *,
    now: float,
    max_age_s: float = 1.5,
) -> bool:
    """Validate the evidence fields shared by runtime and future entity port."""

    if not isinstance(payload, Mapping):
        return False
    try:
        written_at = float(payload["written_at"])
        sequence = int(payload["sequence"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        payload.get("schema_version") != 1
        or payload.get("backend") != "isaac-g1"
        or sequence < 1
        or not math.isfinite(written_at)
        or now - written_at < -0.25
        or now - written_at > max(0.1, float(max_age_s))
    ):
        return False
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return False
    seen: set[str] = set()
    for item in entities:
        if not isinstance(item, Mapping):
            return False
        entity_id = item.get("entity_id")
        prim_path = item.get("prim_path")
        if (
            not isinstance(entity_id, str)
            or entity_id not in ENTITY_SPECS
            or entity_id in seen
            or prim_path != ENTITY_SPECS[entity_id].prim_path
            or item.get("dynamic") is not True
            or item.get("graspable") is not True
        ):
            return False
        try:
            vectors = (
                tuple(float(value) for value in item["position"]),
                tuple(float(value) for value in item["quaternion_wxyz"]),
                tuple(float(value) for value in item["linear_velocity_world"]),
                tuple(float(value) for value in item["angular_velocity_world"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            tuple(map(len, vectors)) != (3, 4, 3, 3)
            or not all(math.isfinite(value) for vector in vectors for value in vector)
            or not 0.9 <= math.sqrt(sum(value * value for value in vectors[1])) <= 1.1
        ):
            return False
        seen.add(entity_id)
    return True
