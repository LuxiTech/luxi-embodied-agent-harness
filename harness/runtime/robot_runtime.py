"""Robot-neutral local RuntimeHost shared by G1 and Go2 deployments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import threading
import time
from typing import Any, Callable, Mapping

from .contracts import CancellationToken, new_id
from .ports import MotionPort, RobotWorldAdapter, RuntimeCommand


def _session_events(store: Any, session_id: str):
    after_sequence = 0
    while True:
        batch = store.events(
            session_id,
            after_sequence=after_sequence,
            limit=10_000,
        )
        if not batch:
            return
        yield from batch
        after_sequence = batch[-1].sequence
        if len(batch) < 10_000:
            return


def _health_fingerprint(health: Mapping[str, Any]) -> str:
    """Compare health state without high-frequency freshness telemetry."""

    stable = dict(health)
    stable.pop("observation_age_s", None)
    return json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


class RuntimeState(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class RuntimeStatus:
    robot_id: str
    robot_type: str
    backend: str
    boot_epoch: str
    runtime_revision: str
    state: RuntimeState
    active_task_id: str | None
    health: Mapping[str, Any]


class LuxiRobotRuntimeHost:
    """Lightweight lifecycle, fencing and cancellation owner for one robot."""

    def __init__(
        self,
        *,
        robot_id: str,
        adapter: RobotWorldAdapter,
        events: Any,
        runtime_revision: str = "runtime-v1",
        watchdog_interval_s: float = 0.10,
        watchdog_projection_interval_s: float = 5.0,
        watchdog_enabled: bool = True,
        emergency_stop: Callable[[str], Any] | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.adapter = adapter
        self.events = events
        self.runtime_revision = runtime_revision
        self.boot_epoch = new_id("boot")
        self.watchdog_interval_s = max(0.02, float(watchdog_interval_s))
        self.watchdog_projection_interval_s = max(
            1.0,
            float(watchdog_projection_interval_s),
        )
        self.watchdog_enabled = bool(watchdog_enabled)
        self._emergency_stop = emergency_stop
        self._state = RuntimeState.CREATED
        self._active_task_id: str | None = None
        self._active_cancel: CancellationToken | None = None
        self._active_deadline_monotonic: float | None = None
        self._fault_latched_reason: str | None = None
        self._seen_calls: set[tuple[str, str]] = set()
        self._lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._last_health_fingerprint: str | None = None
        self._last_status_emitted_monotonic = 0.0
        self._event_sink_error: str | None = None
        self._recover_previous_lifecycle()
        self._emit_status("runtime/created")

    def _store_context(self) -> tuple[Any, str] | None:
        try:
            store = getattr(self.events, "session_store", None)
            session_id = getattr(self.events, "session_id", None)
        except Exception as exc:  # Event logging may not own lifecycle safety.
            self._event_sink_error = str(exc)[:500]
            return None
        if store is None or not isinstance(session_id, str) or not session_id:
            return None
        return store, session_id

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        context = self._store_context()
        if context is None:
            return
        store, session_id = context
        try:
            store.emit(
                event_type,
                session_id=session_id,
                source="luxi-robot-runtime-host",
                payload={
                    "robot_id": self.robot_id,
                    "robot_type": self.adapter.robot_type,
                    "backend": self.adapter.backend,
                    "boot_epoch": self.boot_epoch,
                    "runtime_revision": self.runtime_revision,
                    **dict(payload),
                },
            )
            self._event_sink_error = None
        except Exception as exc:  # Keep robot-local lifecycle independent of storage.
            self._event_sink_error = str(exc)[:500]

    def _recover_previous_lifecycle(self) -> None:
        """Record an interrupted prior attachment without reusing its epoch."""

        context = self._store_context()
        if context is None:
            return
        store, current_session_id = context
        latest: Any | None = None
        try:
            for session_id in store.session_ids():
                for event in _session_events(store, session_id):
                    if event.event_type != "runtime/status_projected":
                        continue
                    payload = event.payload
                    if (
                        payload.get("robot_id") == self.robot_id
                        and payload.get("backend") == self.adapter.backend
                    ):
                        latest = event
        except Exception as exc:
            self._event_sink_error = str(exc)[:500]
            return
        if latest is None:
            return
        previous = latest.payload
        previous_state = str(previous.get("state", ""))
        if previous_state not in {
            RuntimeState.STARTING.value,
            RuntimeState.READY.value,
            RuntimeState.BUSY.value,
            RuntimeState.STOPPING.value,
        }:
            return
        self._emit(
            "runtime/recovered",
            {
                "state": RuntimeState.CREATED.value,
                "recovery_status": "interrupted",
                "previous_state": previous_state,
                "previous_boot_epoch": previous.get("boot_epoch"),
                "previous_active_task_id": previous.get("active_task_id"),
                "automatic_replay": False,
            },
        )

    def _health(self) -> Mapping[str, Any]:
        try:
            return dict(self.adapter.ports.lifecycle_health.health(self.robot_id))
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    def _emit_status(
        self,
        reason: str,
        *,
        health: Mapping[str, Any] | None = None,
    ) -> None:
        current_health = dict(health if health is not None else self._health())
        with self._lock:
            payload = {
                "state": self._state.value,
                "active_task_id": self._active_task_id,
                "health": current_health,
                "reason": reason,
                "watchdog_running": self.watchdog_running,
                "watchdog_interval_s": self.watchdog_interval_s,
                "watchdog_projection_interval_s": (
                    self.watchdog_projection_interval_s
                ),
                "fault_latched_reason": self._fault_latched_reason,
            }
        self._emit("runtime/status_projected", payload)
        self._last_health_fingerprint = _health_fingerprint(current_health)
        self._last_status_emitted_monotonic = time.monotonic()

    @property
    def watchdog_running(self) -> bool:
        thread = self._watchdog_thread
        return bool(thread is not None and thread.is_alive())

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            try:
                self.reconcile()
            except Exception as exc:
                # An unexpected health-check failure is itself unsafe. Cancel
                # local ownership without depending on UI or event storage.
                with self._lock:
                    if self._active_cancel is not None:
                        self._active_cancel.cancel()
                    if self._state in {
                        RuntimeState.STARTING,
                        RuntimeState.READY,
                        RuntimeState.BUSY,
                        RuntimeState.STOPPING,
                    }:
                        self._state = RuntimeState.FAULT
                        self._fault_latched_reason = "watchdog_error"
                self._emit_status(
                    "watchdog_error",
                    health={"ready": False, "error": str(exc)[:500]},
                )
                self._request_emergency_stop("runtime_watchdog_error")

    @property
    def emergency_stop_bound(self) -> bool:
        return self._emergency_stop is not None

    def bind_emergency_stop(self, callback: Callable[[str], Any]) -> None:
        """Bind the one robot-local stop coordinator before runtime work."""

        with self._lock:
            if self._emergency_stop is not None and self._emergency_stop is not callback:
                raise RuntimeError("RuntimeHost emergency stop already has an owner")
            self._emergency_stop = callback

    def _request_emergency_stop(self, reason: str) -> Any:
        callback = self._emergency_stop
        if callback is None:
            return None
        try:
            return callback(reason)
        except Exception as exc:
            self._emit(
                "runtime/emergency_stop_failed",
                {"reason": reason, "error": str(exc)[:500]},
            )
            return None

    def confirm_restart_stationary(self) -> None:
        """Recover safety admission only through a fresh physical stop barrier."""
        if self.reconcile().state is not RuntimeState.READY:
            raise RuntimeError("Runtime is not ready for restart stop verification")
        evidence = self._request_emergency_stop("runtime_restart_verification")
        # Isaac's stop service returns a legacy result envelope; MuJoCo's
        # bound kernel returns SafetyEvidence directly.
        details = evidence.get("safety_evidence", {}) if isinstance(evidence, Mapping) else {
            name: getattr(evidence, name, None) for name in (
                "stop_command_completed", "stationary_confirmed",
                "stop_command_completed_at", "stationary_confirmed_at", "details",
            )
        }
        if not isinstance(details, Mapping):
            details = {}
        confirmed = (details.get("stop_command_completed") is True
                     and details.get("stationary_confirmed") is True)
        self._emit("runtime/restart_stop_verified", {
            "boot_epoch": self.boot_epoch,
            "stationary_confirmed": confirmed,
            "stop_command_completed": details.get("stop_command_completed") is True,
            "stop_command_completed_at": details.get("stop_command_completed_at"),
            "stationary_confirmed_at": details.get("stationary_confirmed_at"),
            "details": details.get("details") or {},
        })
        if not confirmed:
            raise RuntimeError("新仿真停稳验证失败，安全故障未解除")
        if self.reconcile().state is not RuntimeState.READY:
            raise RuntimeError("Runtime health changed during restart stop verification")

    def _start_watchdog(self) -> None:
        if not self.watchdog_enabled:
            return
        with self._lock:
            if self.watchdog_running:
                return
            self._watchdog_stop.clear()
            thread = threading.Thread(
                target=self._watchdog_loop,
                name=f"luxi-runtime-watchdog-{self.robot_id}",
                daemon=True,
            )
            self._watchdog_thread = thread
            thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.watchdog_interval_s * 4.0))
        self._watchdog_thread = None

    def start(self) -> None:
        with self._lock:
            if self._state not in {RuntimeState.CREATED, RuntimeState.STOPPED}:
                raise RuntimeError(f"cannot start runtime from {self._state.value}")
            if self._state is RuntimeState.STOPPED:
                self.boot_epoch = new_id("boot")
                self._seen_calls.clear()
            self._active_deadline_monotonic = None
            self._fault_latched_reason = None
            self._state = RuntimeState.STARTING
            motion = self.adapter.ports.motion
            bind_epoch = getattr(motion, "bind_boot_epoch", None)
            if callable(bind_epoch):
                bind_epoch(self.boot_epoch)
        try:
            self.adapter.ports.lifecycle_health.start(self.robot_id, self.boot_epoch)
            health = self._health()
        except Exception:
            with self._lock:
                self._state = RuntimeState.FAULT
            self._emit_status("lifecycle_start_failed")
            raise
        with self._lock:
            self._state = (
                RuntimeState.READY
                if health.get("ready") is True
                else RuntimeState.STARTING
            )
        self._start_watchdog()
        self._emit_status("lifecycle_started", health=health)

    def bind_motion_port(
        self,
        port: MotionPort,
        *,
        capability_ids: frozenset[str],
    ) -> None:
        """Bind one delegating port before work starts; never replace an owner."""

        if not capability_ids:
            raise ValueError("motion port requires at least one capability")
        with self._lock:
            if self._active_task_id is not None or self._state not in {
                RuntimeState.CREATED,
                RuntimeState.STARTING,
                RuntimeState.READY,
            }:
                raise RuntimeError(
                    f"cannot bind motion port from {self._state.value}"
                )
            existing = self.adapter.ports.motion
            if existing is not None and existing is not port:
                raise RuntimeError("RuntimeHost motion port already has an owner")
            if existing is port:
                self.adapter.capabilities = frozenset(
                    set(self.adapter.capabilities) | set(capability_ids)
                )
            else:
                self.adapter.ports = replace(self.adapter.ports, motion=port)
                self.adapter.capabilities = frozenset(
                    set(self.adapter.capabilities) | set(capability_ids)
                )
            observation = self.adapter.ports.observation
            mark_bound = getattr(observation, "set_motion_command_port_bound", None)
            if callable(mark_bound):
                mark_bound(True)
            bind_epoch = getattr(port, "bind_boot_epoch", None)
            if callable(bind_epoch):
                bind_epoch(self.boot_epoch)
        self._emit(
            "runtime/port_bound",
            {
                "port": "MotionPort",
                "capability_ids": sorted(capability_ids),
                "publisher_implementation": "existing-safe-command-gateway",
            },
        )

    def bind_navigation_port(
        self,
        port: Any,
        *,
        capability_ids: frozenset[str],
    ) -> None:
        """Bind one reviewed navigation owner without replacing MotionPort."""

        if not capability_ids:
            raise ValueError("navigation port requires at least one capability")
        with self._lock:
            if self._active_task_id is not None or self._state not in {
                RuntimeState.CREATED,
                RuntimeState.STARTING,
                RuntimeState.READY,
            }:
                raise RuntimeError(
                    f"cannot bind navigation port from {self._state.value}"
                )
            existing = self.adapter.ports.navigation
            if existing is not None and existing is not port:
                raise RuntimeError("RuntimeHost navigation port already has an owner")
            if existing is not port:
                self.adapter.ports = replace(self.adapter.ports, navigation=port)
            self.adapter.capabilities = frozenset(
                set(self.adapter.capabilities) | set(capability_ids)
            )
            bind_epoch = getattr(port, "bind_boot_epoch", None)
            if callable(bind_epoch):
                bind_epoch(self.boot_epoch)
        self._emit(
            "runtime/port_bound",
            {
                "port": "NavigationPort",
                "capability_ids": sorted(capability_ids),
                "publisher_implementation": "existing-native-navigation-skill",
            },
        )

    def reconcile(self) -> RuntimeStatus:
        """Refresh readiness from the adapter without taking command ownership."""

        health = self._health()
        changed = False
        reason = "health_reconciled"
        with self._lock:
            if self._state is RuntimeState.STARTING and health.get("ready") is True:
                self._state = RuntimeState.READY
                changed = True
            elif (
                self._state is RuntimeState.STARTING
                and health.get("startup_failed") is True
            ):
                self._state = RuntimeState.FAULT
                self._fault_latched_reason = "startup_failed"
                changed = True
            elif self._state in {
                RuntimeState.READY,
                RuntimeState.BUSY,
                RuntimeState.STOPPING,
            } and health.get("ready") is not True:
                if self._active_cancel is not None:
                    self._active_cancel.cancel()
                self._state = RuntimeState.FAULT
                self._fault_latched_reason = "health_faulted"
                changed = True
            elif (
                self._state is RuntimeState.BUSY
                and self._active_deadline_monotonic is not None
                and self._active_deadline_monotonic <= time.monotonic()
            ):
                if self._active_cancel is not None:
                    self._active_cancel.cancel()
                self._state = RuntimeState.STOPPING
                reason = "command_deadline_expired"
                changed = True
            state = self._state
            active_task_id = self._active_task_id
        fingerprint = _health_fingerprint(health)
        projection_due = (
            time.monotonic() - self._last_status_emitted_monotonic
            >= self.watchdog_projection_interval_s
        )
        if changed or fingerprint != self._last_health_fingerprint or projection_due:
            self._emit_status(
                (
                    "health_faulted"
                    if state is RuntimeState.FAULT
                    else reason
                    if changed or fingerprint != self._last_health_fingerprint
                    else "watchdog_heartbeat"
                ),
                health=health,
            )
        if changed and state in {RuntimeState.STOPPING, RuntimeState.FAULT}:
            self._request_emergency_stop(
                "runtime_deadline_expired"
                if reason == "command_deadline_expired"
                else "runtime_health_fault"
            )
        return RuntimeStatus(
            self.robot_id,
            self.adapter.robot_type,
            self.adapter.backend,
            self.boot_epoch,
            self.runtime_revision,
            state,
            active_task_id,
            health,
        )

    def accept(self, command: RuntimeCommand) -> CancellationToken:
        with self._lock:
            if self._state is not RuntimeState.READY:
                raise RuntimeError(f"runtime is not ready: {self._state.value}")
            if command.robot_id != self.robot_id or command.boot_epoch != self.boot_epoch:
                raise PermissionError("robot identity or boot epoch mismatch")
            if command.deadline_monotonic <= time.monotonic():
                raise TimeoutError("runtime command deadline expired")
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("runtime command replay rejected")
            self._seen_calls.add(key)
            self._active_task_id = command.task_id
            self._active_cancel = CancellationToken()
            self._active_deadline_monotonic = command.deadline_monotonic
            self._state = RuntimeState.BUSY
            token = self._active_cancel
        self._emit_status("command_accepted")
        return token

    def refresh(self, command: RuntimeCommand) -> CancellationToken:
        """Refresh one active owner's deadline with a newly fenced command.

        Operator dead-man commands use one Runtime task while each keyboard
        edge receives a distinct Tool Call id.  Refreshing never changes the
        owner or creates a second cancellation token.
        """

        with self._lock:
            if self._state is not RuntimeState.BUSY:
                raise RuntimeError(f"runtime is not busy: {self._state.value}")
            if command.task_id != self._active_task_id:
                raise PermissionError("task does not own this runtime")
            if command.robot_id != self.robot_id or command.boot_epoch != self.boot_epoch:
                raise PermissionError("robot identity or boot epoch mismatch")
            if command.deadline_monotonic <= time.monotonic():
                raise TimeoutError("runtime command deadline expired")
            key = (command.boot_epoch, command.tool_call_id)
            if key in self._seen_calls:
                raise PermissionError("runtime command replay rejected")
            token = self._active_cancel
            if token is None or token.cancelled:
                raise RuntimeError("runtime command cancellation is no longer active")
            self._seen_calls.add(key)
            self._active_deadline_monotonic = command.deadline_monotonic
        self._emit_status("command_refreshed")
        return token

    def finish(self, task_id: str, *, stationary_confirmed: bool) -> None:
        with self._lock:
            if task_id != self._active_task_id:
                raise PermissionError("task does not own this runtime")
            self._active_task_id = None
            self._active_cancel = None
            self._active_deadline_monotonic = None
            if self._state is RuntimeState.FAULT or self._fault_latched_reason:
                self._state = RuntimeState.FAULT
            elif stationary_confirmed:
                self._state = RuntimeState.READY
            else:
                self._state = RuntimeState.FAULT
                self._fault_latched_reason = "stationarity_unconfirmed"
        self._emit_status("command_finished")

    def cancel(self, reason: str) -> Any:
        should_stop = False
        with self._lock:
            token = self._active_cancel
            if token is not None:
                token.cancel()
            if self._state is RuntimeState.BUSY:
                self._state = RuntimeState.STOPPING
                should_stop = True
        self._emit_status(reason)
        if should_stop:
            return self._request_emergency_stop(f"runtime_cancel:{reason}")
        return None

    def stop(self) -> None:
        self._stop_watchdog()
        with self._lock:
            was_busy = self._state is RuntimeState.BUSY
            was_created = self._state is RuntimeState.CREATED
        self.cancel("runtime_stop")
        if not was_busy and not was_created:
            self._request_emergency_stop("runtime_stop")
        self.adapter.ports.lifecycle_health.stop(self.robot_id, self.boot_epoch)
        with self._lock:
            self._state = RuntimeState.STOPPED
            self._active_task_id = None
            self._active_cancel = None
            self._active_deadline_monotonic = None
        self._emit_status("lifecycle_stopped")

    def status(self) -> RuntimeStatus:
        with self._lock:
            state = self._state
            active_task_id = self._active_task_id
        health = self._health()
        return RuntimeStatus(
            self.robot_id,
            self.adapter.robot_type,
            self.adapter.backend,
            self.boot_epoch,
            self.runtime_revision,
            state,
            active_task_id,
            health,
        )

    @property
    def event_sink_error(self) -> str | None:
        return self._event_sink_error


class RuntimeStatusProjection:
    """Build the operator runtime view only from durable RuntimeHost events."""

    def __init__(self, events: Any, *, robot_id: str) -> None:
        self.events = events
        self.robot_id = robot_id

    def snapshot(self) -> Mapping[str, Any]:
        try:
            store = getattr(self.events, "session_store", None)
            session_id = getattr(self.events, "session_id", None)
        except Exception as exc:
            return {
                "available": False,
                "robot_id": self.robot_id,
                "error": str(exc)[:500],
            }
        if store is None or not isinstance(session_id, str) or not session_id:
            return {"available": False, "robot_id": self.robot_id}
        latest: Any | None = None
        recovery: Any | None = None
        try:
            for event in _session_events(store, session_id):
                if event.payload.get("robot_id") != self.robot_id:
                    continue
                if event.event_type == "runtime/recovered":
                    recovery = event
                elif event.event_type == "runtime/status_projected":
                    latest = event
        except Exception as exc:
            return {
                "available": False,
                "robot_id": self.robot_id,
                "error": str(exc)[:500],
            }
        if latest is None:
            return {"available": False, "robot_id": self.robot_id}
        payload = dict(latest.payload)
        payload["available"] = True
        payload["event_sequence"] = latest.sequence
        payload["projected_from"] = "LuxiSessionStore"
        if recovery is not None:
            payload["recovery"] = {
                key: recovery.payload.get(key)
                for key in (
                    "recovery_status",
                    "previous_state",
                    "previous_boot_epoch",
                    "previous_active_task_id",
                    "automatic_replay",
                )
            }
        return payload
