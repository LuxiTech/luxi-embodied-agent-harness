"""Legacy dashboard timeline projected from the durable SessionStore."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import threading
from uuid import uuid4
from typing import Any

from harness.runtime.session_store import LuxiSessionStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventStore:
    """Bounded UI projection with optional durable SessionStore dual-write.

    New orchestration code writes directly to ``LuxiSessionStore``. This
    compatibility projection lets legacy bridges migrate without making the UI
    deque a second source of truth.
    """

    def __init__(self, max_events: int = 1_500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self._durable_cursor = 0
        self._projection_id = uuid4().hex
        self._next_id = 1
        self._session_store: LuxiSessionStore | None = None
        self._session_id: str | None = None
        self._thread_context = threading.local()

    def set_thread_context(self, **correlation: str | None) -> None:
        self._thread_context.correlation = {
            key: value for key, value in correlation.items() if value is not None
        }

    def clear_thread_context(self) -> None:
        self._thread_context.correlation = {}

    def attach_session_store(
        self,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
        resume_latest: bool = False,
    ) -> str:
        store = LuxiSessionStore(path)
        metadata = metadata or {"projection": "operator-ui"}
        session_id = store.latest_session(metadata) if resume_latest else None
        if session_id is None:
            session_id = store.create_session(metadata=metadata)
        with self._lock:
            self._session_store = store
            self._session_id = session_id
        return session_id

    @property
    def session_store(self) -> LuxiSessionStore | None:
        with self._lock:
            return self._session_store

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def append(
        self,
        source: str,
        kind: str,
        title: str,
        message: str = "",
        *,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        correlation = dict(getattr(self._thread_context, "correlation", {}) or {})
        event_data = {**correlation, **(data or {})}
        with self._lock:
            event = {
                "id": self._next_id,
                "timestamp": _utc_now(),
                "source": source,
                "kind": kind,
                "title": title,
                "message": message,
                "level": level,
                "data": event_data,
            }
            self._next_id += 1
            self._events.append(event)
            store = self._session_store
            session_id = self._session_id
        if store is not None and session_id is not None:
            payload = {
                "legacy_id": event["id"],
                "title": title,
                "message": message,
                "level": level,
                "data": event_data,
            }
            try:
                store.emit(
                    f"legacy/{source}/{kind}",
                    session_id=session_id,
                    source="legacy-ui-projection",
                    turn_id=event_data.get("turn_id"),
                    step_id=event_data.get("step_id"),
                    task_id=event_data.get("task_id"),
                    tool_call_id=event_data.get("tool_call_id")
                    or event_data.get("call_id"),
                    payload=payload,
                    idempotency_key=f"ui:{self._projection_id}:{event['id']}",
                )
            except (OSError, ValueError, TypeError):
                # Legacy UI remains available during dual-write. Unified
                # physical execution fails closed if its authoritative write
                # cannot be committed before tool start.
                pass
        return event

    def _project_durable(self):
        """UI is a bounded projection of shared Loop facts, never a second log."""
        with self._lock:
            if self._session_store is None or self._session_id is None:
                return
            kinds = {"turn/started": ("agent", "instruction", "用户任务"),
                     "model/replied": ("agent", "model", "Harness model step"),
                     "tool/started": ("tool", "call", "工具调用"),
                     "tool/result": ("tool", "result", "工具结果"),
                     "tool/denied": ("tool", "result", "工具拒绝"),
                     "turn/completed": ("agent", "response", "任务结果"),
                     "turn/incomplete": ("agent", "response", "任务未完成"),
                     "turn/interrupted": ("agent", "response", "任务中断")}
            try:
                stream = self._session_store.iter_events(self._session_id, after_sequence=self._durable_cursor)
                for item in stream:
                    self._durable_cursor = item.sequence
                    if item.event_type not in kinds:
                        continue
                    source, kind, title = kinds[item.event_type]
                    data = dict(item.payload)
                    data.update(turn_id=item.turn_id, task_id=item.task_id,
                                step_id=item.step_id, tool_call_id=item.tool_call_id,
                                tool=data.get("capability_id"), loop="harness")
                    self._events.append({"id": self._next_id, "timestamp":item.timestamp,
                        "source":source,"kind":kind,"title":title,
                        "message":str(data.get("error") or data.get("instruction") or data.get("response") or data.get("content") or data.get("status") or data.get("task_status") or ""),
                        "level":"danger" if data.get("error") else "info","data":data})
                    self._next_id += 1
            except (OSError, ValueError):
                pass

    def since(self, cursor: int, limit: int = 300) -> list[dict[str, Any]]:
        self._project_durable()
        with self._lock:
            matches = [event.copy() for event in self._events if event["id"] > cursor]
        return matches[-limit:]

    @property
    def latest_id(self) -> int:
        self._project_durable()
        with self._lock:
            return self._events[-1]["id"] if self._events else 0
