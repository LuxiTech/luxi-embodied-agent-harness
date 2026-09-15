"""Versioned identity-scoped messages for the single Go2 ROS boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping




ROBOT_MESSAGE_SCHEMA_VERSION = 2
GO2_ROBOT_IDS = frozenset({"go2-01"})
ROBOT_MESSAGE_KINDS = frozenset(
    {
        "robot_state",
        "local_map",
        "heartbeat",
        "task_result",
    }
)


@dataclass(frozen=True)
class RobotMessageEnvelope:
    """Identity-fenced envelope published by exactly one robot runtime."""

    kind: str
    robot_id: str
    boot_epoch: str
    sequence: int
    sent_at: float
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in ROBOT_MESSAGE_KINDS:
            raise ValueError(f"unsupported robot message kind: {self.kind!r}")
        if self.robot_id not in GO2_ROBOT_IDS:
            raise ValueError(f"unsupported single Go2 robot: {self.robot_id!r}")
        if not self.boot_epoch.strip() or len(self.boot_epoch) > 128:
            raise ValueError("robot boot epoch must be a non-empty bounded string")
        if self.sequence < 0:
            raise ValueError("robot message sequence must be non-negative")
        if not math.isfinite(self.sent_at):
            raise ValueError("robot message timestamp must be finite")

    def as_message(self) -> dict[str, Any]:
        return {
            "schema_version": ROBOT_MESSAGE_SCHEMA_VERSION,
            "kind": self.kind,
            "robot_id": self.robot_id,
            "boot_epoch": self.boot_epoch,
            "sequence": self.sequence,
            "sent_at": self.sent_at,
            "payload": dict(self.payload),
        }

    @classmethod
    def parse(
        cls,
        value: Mapping[str, Any],
        *,
        expected_kind: str | None = None,
        expected_robot_id: str | None = None,
    ) -> "RobotMessageEnvelope":
        if int(value.get("schema_version", -1)) != ROBOT_MESSAGE_SCHEMA_VERSION:
            raise ValueError("unsupported single Go2 robot message schema")
        try:
            envelope = cls(
                kind=str(value["kind"]),
                robot_id=str(value["robot_id"]),
                boot_epoch=str(value["boot_epoch"]),
                sequence=int(value["sequence"]),
                sent_at=float(value["sent_at"]),
                payload=dict(value["payload"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid single Go2 robot message") from error
        if expected_kind is not None and envelope.kind != expected_kind:
            raise ValueError(
                f"expected {expected_kind!r}, received {envelope.kind!r}"
            )
        if expected_robot_id is not None and envelope.robot_id != expected_robot_id:
            raise ValueError(
                f"robot {expected_robot_id!r} cannot publish for {envelope.robot_id!r}"
            )
        return envelope


class RobotMessagePublisher:
    """Create envelopes while enforcing a bridge's immutable robot identity."""

    def __init__(self, robot_id: str, boot_epoch: str) -> None:
        if robot_id not in GO2_ROBOT_IDS:
            raise ValueError(f"unsupported single Go2 robot: {robot_id!r}")
        if not str(boot_epoch).strip():
            raise ValueError("robot boot epoch must not be empty")
        self.robot_id = robot_id
        self.boot_epoch = str(boot_epoch)
        self._sequences = {kind: 0 for kind in ROBOT_MESSAGE_KINDS}

    def create(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> RobotMessageEnvelope:
        if kind not in self._sequences:
            raise ValueError(f"unsupported robot message kind: {kind!r}")
        claimed_robot = payload.get("robot_id")
        if claimed_robot is not None and str(claimed_robot) != self.robot_id:
            raise ValueError(
                f"robot {self.robot_id!r} cannot publish payload for {claimed_robot!r}"
            )
        self._sequences[kind] += 1
        return RobotMessageEnvelope(
            kind,
            self.robot_id,
            self.boot_epoch,
            self._sequences[kind],
            time.time() if now is None else float(now),
            dict(payload),
        )
