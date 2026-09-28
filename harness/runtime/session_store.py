"""SQLite-backed append-only event log and deterministic projections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterable, Mapping

from .contracts import SCHEMA_VERSION, new_id


SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "private_key",
        "scorer_truth",
        "third_person_frame",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_payload(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if normalized in SENSITIVE_KEYS:
                raise ValueError(f"sensitive field is forbidden in event log: {path}.{key}")
            _validate_payload(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_payload(child, f"{path}[{index}]")
    elif isinstance(value, bytes):
        raise ValueError(f"binary payload must be stored as an artifact reference: {path}")


@dataclass(frozen=True)
class SessionEvent:
    event_id: str
    schema_version: int
    session_id: str
    sequence: int
    event_type: str
    source: str
    timestamp: str
    monotonic_ns: int
    turn_id: str | None
    step_id: str | None
    task_id: str | None
    tool_call_id: str | None
    correlation_id: str | None
    payload: Mapping[str, Any]


class LuxiSessionStore:
    """The single persistent fact source for Agent orchestration.

    High-frequency telemetry and image/point-cloud bytes do not belong here;
    events carry immutable artifact references instead.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            with closing(self._connect()) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        next_sequence INTEGER NOT NULL DEFAULT 1
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        session_id TEXT NOT NULL REFERENCES sessions(session_id),
                        sequence INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        monotonic_ns INTEGER NOT NULL,
                        turn_id TEXT,
                        step_id TEXT,
                        task_id TEXT,
                        tool_call_id TEXT,
                        correlation_id TEXT,
                        idempotency_key TEXT,
                        payload_json TEXT NOT NULL,
                        UNIQUE(session_id, sequence),
                        UNIQUE(session_id, idempotency_key)
                    );
                    CREATE INDEX IF NOT EXISTS events_turn
                        ON events(session_id, turn_id, sequence);
                    CREATE INDEX IF NOT EXISTS events_task
                        ON events(session_id, task_id, sequence);
                    CREATE INDEX IF NOT EXISTS events_tool
                        ON events(tool_call_id, sequence);
                    """
                )
            self._initialized = True

    def create_session(
        self,
        session_id: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        value = session_id or new_id("session")
        safe_metadata = dict(metadata or {})
        _validate_payload(safe_metadata, "metadata")
        encoded = json.dumps(safe_metadata, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at, metadata_json) VALUES (?, ?, ?)",
                (value, _utc_now(), encoded),
            )
        return value

    def emit(
        self,
        event_type: str,
        *,
        session_id: str,
        source: str,
        turn_id: str | None = None,
        step_id: str | None = None,
        task_id: str | None = None,
        tool_call_id: str | None = None,
        correlation_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> SessionEvent:
        if not event_type.strip() or not source.strip():
            raise ValueError("event_type and source are required")
        safe_payload = dict(payload or {})
        _validate_payload(safe_payload)
        encoded = json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"))
        event_id = new_id("event")
        timestamp = _utc_now()
        monotonic_ns = time.monotonic_ns()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = None
                if idempotency_key:
                    existing = connection.execute(
                        "SELECT * FROM events WHERE session_id=? AND idempotency_key=?",
                        (session_id, idempotency_key),
                    ).fetchone()
                if existing is not None:
                    connection.execute("COMMIT")
                    return self._row_to_event(existing)
                row = connection.execute(
                    "SELECT next_sequence FROM sessions WHERE session_id=?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown session: {session_id}")
                sequence = int(row["next_sequence"])
                connection.execute(
                    "UPDATE sessions SET next_sequence=? WHERE session_id=?",
                    (sequence + 1, session_id),
                )
                connection.execute(
                    """
                    INSERT INTO events(
                        event_id, schema_version, session_id, sequence, event_type,
                        source, timestamp, monotonic_ns, turn_id, step_id, task_id,
                        tool_call_id, correlation_id, idempotency_key, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        SCHEMA_VERSION,
                        session_id,
                        sequence,
                        event_type,
                        source,
                        timestamp,
                        monotonic_ns,
                        turn_id,
                        step_id,
                        task_id,
                        tool_call_id,
                        correlation_id,
                        idempotency_key,
                        encoded,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return SessionEvent(
            event_id,
            SCHEMA_VERSION,
            session_id,
            sequence,
            event_type,
            source,
            timestamp,
            monotonic_ns,
            turn_id,
            step_id,
            task_id,
            tool_call_id,
            correlation_id,
            safe_payload,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> SessionEvent:
        return SessionEvent(
            event_id=str(row["event_id"]),
            schema_version=int(row["schema_version"]),
            session_id=str(row["session_id"]),
            sequence=int(row["sequence"]),
            event_type=str(row["event_type"]),
            source=str(row["source"]),
            timestamp=str(row["timestamp"]),
            monotonic_ns=int(row["monotonic_ns"]),
            turn_id=row["turn_id"],
            step_id=row["step_id"],
            task_id=row["task_id"],
            tool_call_id=row["tool_call_id"],
            correlation_id=row["correlation_id"],
            payload=json.loads(str(row["payload_json"])),
        )

    def events(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 1_000,
    ) -> list[SessionEvent]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM events
                WHERE session_id=? AND sequence>?
                ORDER BY sequence ASC LIMIT ?
                """,
                (session_id, max(0, after_sequence), max(1, min(limit, 10_000))),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def session_ids(self) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id FROM sessions ORDER BY created_at"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def tool_call_state(self, session_id: str, tool_call_id: str) -> str | None:
        """Check durable replay evidence using events_tool, without decoding payloads.

        Deliberately independent of task/turn: reusing a call ID anywhere in the
        same session must retain the existing no-replay guarantee.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_type FROM events WHERE tool_call_id=? AND session_id=? "
                "AND event_type IN ('tool/started', 'tool/result')",
                (tool_call_id, session_id),
            ).fetchall()
        kinds = {row["event_type"] for row in rows}
        if "tool/result" in kinds:
            return "finished"
        if "tool/started" in kinds:
            return "side_effect_unknown"
        return None

    def latest_session(self, metadata: Mapping[str, Any]) -> str | None:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id, metadata_json FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        for row in rows:
            previous = json.loads(row["metadata_json"])
            if all(previous.get(key) == value for key, value in metadata.items()):
                return str(row["session_id"])
        return None

    def iter_task_events(self, session_id: str, task_id: str):
        """Project a long task without scanning every historical session event."""
        cursor = 0
        while True:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT * FROM events WHERE session_id=? AND task_id=? AND sequence>? ORDER BY sequence LIMIT 1000",
                    (session_id, task_id, cursor),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._row_to_event(row)
            cursor = int(rows[-1]["sequence"])

    def iter_events(self, session_id: str, *, after_sequence: int = 0):
        """Read the complete stream in bounded pages, including long sessions."""
        while True:
            batch = self.events(session_id, after_sequence=after_sequence, limit=1_000)
            if not batch:
                return
            yield from batch
            after_sequence = batch[-1].sequence

    def context_events(self, session_id: str):
        """Start at the newest context checkpoint or reset, never the oldest page."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT sequence FROM events WHERE session_id=? "
                "AND event_type IN ('model/context', 'session/context_reset') "
                "ORDER BY sequence DESC LIMIT 1", (session_id,),
            ).fetchone()
        yield from self.iter_events(
            session_id, after_sequence=max(0, int(row["sequence"]) - 1) if row else 0
        )

    def mark_interrupted_turns(self, session_id: str) -> list[str]:
        """Close dangling turns after restart without replaying any tool."""

        stream = self.iter_events(session_id)
        opened: dict[str, SessionEvent] = {}
        closed: set[str] = set()
        for event in stream:
            if event.turn_id is None:
                continue
            if event.event_type == "turn/started":
                opened[event.turn_id] = event
            elif event.event_type in {"turn/completed", "turn/incomplete", "turn/interrupted"}:
                closed.add(event.turn_id)
        interrupted = sorted(set(opened) - closed)
        for turn_id in interrupted:
            started = opened[turn_id]
            self.emit(
                "turn/interrupted",
                session_id=session_id,
                turn_id=turn_id,
                task_id=started.task_id,
                source="session-recovery",
                payload={
                    "completed": False,
                    "task_status": "interrupted",
                    "automatic_tool_replay": False,
                    "reconciliation_required": True,
                },
                idempotency_key=f"recover:{turn_id}",
            )
        return interrupted


class CompositeEventSink:
    """Best-effort legacy UI projection after the durable write succeeds."""

    def __init__(self, primary: LuxiSessionStore, legacy: Any | None = None) -> None:
        self.primary = primary
        self.legacy = legacy

    def emit(self, event_type: str, **kwargs: Any) -> SessionEvent:
        event = self.primary.emit(event_type, **kwargs)
        if self.legacy is not None:
            payload = dict(kwargs.get("payload") or {})
            try:
                self.legacy.append(
                    kwargs.get("source", "runtime"),
                    event_type,
                    event_type,
                    str(payload.get("message", "")),
                    data={"session_event": event.__dict__},
                )
            except Exception:
                # UI availability never controls the durable source of truth.
                pass
        return event


def project_model_history(events: Iterable[SessionEvent]) -> list[Mapping[str, Any]]:
    history: list[Mapping[str, Any]] = []
    for event in events:
        if event.event_type == "session/context_reset":
            history = []
        elif event.event_type == "model/context":
            messages = event.payload.get("messages", [])
            if isinstance(messages, list):
                # Every request records the complete model-visible context. Use
                # it as a checkpoint instead of concatenating checkpoints and
                # exponentially duplicating history.
                history = [item for item in messages if isinstance(item, Mapping)]
        elif event.event_type == "model/replied":
            if event.payload.get("tool_argument_errors"):
                # Invalid calls were never executable assistant/tool exchanges.
                history.append({"role": "user", "content": str(event.payload.get("retry_feedback", ""))})
                continue
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": str(event.payload.get("content", "")),
            }
            tool_calls = event.payload.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": str(call.get("call_id", "")),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name", "")),
                            "arguments": json.dumps(
                                call.get("arguments", {}), ensure_ascii=False
                            ),
                        },
                    }
                    for call in tool_calls
                    if isinstance(call, Mapping)
                ]
            history.append(assistant)
        elif event.event_type == "tool/result":
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": event.tool_call_id,
                    "content": json.dumps(event.payload, ensure_ascii=False),
                }
            )
    return history


class SessionContextProvider:
    def __init__(
        self, store: LuxiSessionStore, *, max_events: int = 1_000,
        max_history_turns: int = 8, max_history_chars: int = 24_000,
    ) -> None:
        self.store = store
        self.max_events = max_events
        self.max_history_turns = max(0, max_history_turns)
        self.max_history_chars = max(0, max_history_chars)

    def history(self, session_id: str) -> list[Mapping[str, Any]]:
        messages = project_model_history(self.store.context_events(session_id))
        # A crash may leave a model tool_call without its result. Record unknown
        # outcome for context only; never infer success or replay physical work.
        repaired = []
        pending = set()
        def close_pending():
            for call_id in sorted(pending):
                repaired.append({"role": "tool", "tool_call_id": call_id,
                                 "content": '{"completed":false,"task_status":"interrupted","automatic_tool_replay":false}'})
            pending.clear()
        for message in messages:
            if message.get("role") != "tool":
                close_pending()
            if message.get("role") == "assistant":
                pending.update(str(call["id"]) for call in message.get("tool_calls", []))
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    continue
                pending.remove(call_id)
            repaired.append(message)
        close_pending()
        messages = repaired
        # Evict whole turns so an assistant tool_call and its result stay paired.
        # This is a deterministic character budget, not a tokenizer estimate.
        turns: list[list[Mapping[str, Any]]] = []
        for message in messages:
            if message.get("role") == "user":
                turns.append([])
            if turns:
                turns[-1].append(message)
        selected: list[list[Mapping[str, Any]]] = []
        size = 0
        for turn in reversed(turns):
            cost = len(json.dumps(turn, ensure_ascii=False))
            if len(selected) >= self.max_history_turns or size + cost > self.max_history_chars:
                break
            selected.append(turn)
            size += cost
        return [message for turn in reversed(selected) for message in turn]

    def build_context(
        self, *, session_id: str, turn_id: str, instruction: str
    ) -> list[Mapping[str, Any]]:
        history = self.history(session_id)
        history.append({"role": "user", "content": instruction})
        return history
