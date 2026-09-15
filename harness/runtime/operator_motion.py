"""Runtime-fenced operator motion without creating another robot publisher."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from .contracts import SafetyRequest, SideEffect, new_id
from .ports import RuntimeCommand
from .robot_runtime import RuntimeState


class IsaacOperatorMotionGateway:
    """Own Isaac keyboard pulses through RuntimeHost and SafetyKernel.

    The gateway is an operator control-plane boundary, not a model Tool.  It
    reuses the already-bound Isaac MotionPort and the common stop barrier.
    Each non-zero edge is fenced by boot epoch and a unique call id, expires
    after 350 ms, and is never replayed.
    """

    CAPABILITY_ID = "operator_manual_velocity"
    WATCHDOG_GRACE_S = 0.25

    def __init__(
        self,
        *,
        runtime_host: Any,
        motion_port: Any,
        safety: Any,
        safety_observation: Callable[[str], Mapping[str, Any]],
        stop_callback: Callable[[str], Mapping[str, Any]],
        events: Any | None = None,
        safety_robot_id: str = "g1-isaac",
        clock: Callable[[], float] = time.monotonic,
        watchdog_interval_s: float = 0.02,
    ) -> None:
        self.runtime_host = runtime_host
        self.motion_port = motion_port
        self.safety = safety
        self.safety_observation = safety_observation
        self.stop_callback = stop_callback
        self.events = events
        self.safety_robot_id = safety_robot_id
        self.clock = clock
        self.watchdog_interval_s = max(0.01, float(watchdog_interval_s))
        self._enabled_boot_epoch: str | None = None
        self._active_task_id: str | None = None
        self._active_deadline_monotonic: float | None = None
        self._last_call_id: str | None = None
        self._last_sequence: int | None = None
        self._last_stop: Mapping[str, Any] | None = None
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="luxi-isaac-operator-deadman",
            daemon=True,
        )
        self._watchdog.start()

    @staticmethod
    def _stationary(result: Any) -> bool:
        evidence = result.get("safety_evidence") if isinstance(result, Mapping) else None
        return bool(
            isinstance(evidence, Mapping)
            and evidence.get("stop_command_completed") is True
            and evidence.get("stationary_confirmed") is True
        )

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        try:
            store = getattr(self.events, "session_store", None)
            session_id = getattr(self.events, "session_id", None)
            if store is not None and isinstance(session_id, str) and session_id:
                store.emit(
                    event_type,
                    session_id=session_id,
                    source="isaac-operator-motion-gateway",
                    task_id=self._active_task_id,
                    tool_call_id=self._last_call_id,
                    payload={
                        "robot_id": self.runtime_host.robot_id,
                        "boot_epoch": self._enabled_boot_epoch,
                        "capability_id": self.CAPABILITY_ID,
                        **dict(payload),
                    },
                )
        except Exception:
            # Operator dead-man and stop behavior never depends on logging.
            pass

    def enable(self) -> tuple[bool, str]:
        status = self.runtime_host.reconcile()
        if status.state is not RuntimeState.READY:
            return False, f"RuntimeHost 当前为 {status.state.value}，键盘不能抢占控制权"
        result = self.stop_callback("manual_control_enable")
        if not self._stationary(result):
            return False, "键盘接管前未确认机器人静止"
        # Recheck after the blocking stop barrier to close an Agent race.
        status = self.runtime_host.reconcile()
        if status.state is not RuntimeState.READY:
            return False, f"RuntimeHost 当前为 {status.state.value}，键盘不能抢占控制权"
        with self._lock:
            self._enabled_boot_epoch = status.boot_epoch
            self._last_stop = dict(result)
        self._emit("operator_motion/enabled", {"stationary_confirmed": True})
        return True, ""

    def owns_runtime_control(self) -> bool:
        with self._lock:
            task_id = self._active_task_id
        status = self.runtime_host.status()
        return bool(task_id and status.active_task_id == task_id)

    def command_velocity(
        self,
        linear: tuple[float, float, float],
        angular: tuple[float, float, float],
        *,
        duration_s: float,
    ) -> int:
        moving = any(abs(float(value)) > 1e-9 for value in (*linear, *angular))
        if not moving:
            result = self.stop("operator_key_release")
            if not self._stationary(result):
                raise RuntimeError("operator stop was not physically confirmed")
            return int(self._last_sequence or 0)

        now = self.clock()
        with self._lock:
            boot_epoch = self._enabled_boot_epoch
            if boot_epoch is None:
                raise PermissionError("operator motion gateway is disabled")
            if boot_epoch != self.runtime_host.boot_epoch:
                raise PermissionError("retired operator boot epoch rejected")
            task_id = self._active_task_id or new_id("operator-task")
            call_id = new_id("operator-call")
            command_deadline = now + float(duration_s)
            runtime_deadline = command_deadline + self.WATCHDOG_GRACE_S
            command = RuntimeCommand(
                robot_id=self.runtime_host.robot_id,
                boot_epoch=boot_epoch,
                task_id=task_id,
                tool_call_id=call_id,
                deadline_monotonic=runtime_deadline,
                cancel_token_id=new_id("operator-cancel"),
                payload={
                    "capability_id": self.CAPABILITY_ID,
                    "linear": tuple(float(value) for value in linear),
                    "angular": tuple(float(value) for value in angular),
                    "duration_s": float(duration_s),
                    "retry": "never",
                },
            )
            first_command = self._active_task_id is None
            if first_command:
                decision = self.safety.admit(
                    SafetyRequest(
                        robot_id=self.safety_robot_id,
                        capability_id=self.CAPABILITY_ID,
                        side_effect=SideEffect.PHYSICAL,
                        deadline_monotonic=command_deadline,
                        observation=dict(
                            self.safety_observation(self.safety_robot_id)
                        ),
                    )
                )
                if not decision.admitted:
                    raise PermissionError(
                        f"SafetyKernel denied operator motion: {decision.reason}"
                    )
                self.runtime_host.accept(command)
                self._active_task_id = task_id
                self.safety.mark_active(self.safety_robot_id)
            else:
                self.runtime_host.refresh(command)
            self._last_call_id = call_id
            self._active_deadline_monotonic = command_deadline

        try:
            evidence = self.motion_port.command_operator_velocity(command)
            if not evidence.accepted:
                raise RuntimeError(evidence.status)
        except Exception:
            self.stop("operator_command_failed")
            raise
        sequence = evidence.evidence.get("command_sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            self.stop("operator_result_invalid")
            raise RuntimeError("operator MotionPort returned no command sequence")
        with self._lock:
            self._last_sequence = sequence
        self._emit(
            "operator_motion/commanded",
            {
                "command_sequence": sequence,
                "deadline_monotonic": command_deadline,
                "retry": "never",
            },
        )
        return sequence

    def stop(self, reason: str) -> Mapping[str, Any]:
        with self._lock:
            task_id = self._active_task_id
            self._active_task_id = None
            self._active_deadline_monotonic = None
        result: Any = None
        if task_id is not None:
            result = self.runtime_host.cancel(reason)
        if not isinstance(result, Mapping):
            result = self.stop_callback(reason)
        stationary = self._stationary(result)
        if task_id is not None:
            try:
                self.runtime_host.finish(
                    task_id,
                    stationary_confirmed=stationary,
                )
            except PermissionError:
                # A concurrent RuntimeHost lifecycle transition may already
                # have retired the task; the stop evidence remains authoritative.
                pass
        with self._lock:
            self._last_stop = dict(result)
        self._emit(
            "operator_motion/stopped",
            {
                "reason": reason,
                "stop_command_completed": bool(
                    isinstance(result.get("safety_evidence"), Mapping)
                    and result["safety_evidence"].get("stop_command_completed")
                ),
                "stationary_confirmed": stationary,
            },
        )
        return result

    def disable(self, reason: str = "manual_control_disable") -> Mapping[str, Any]:
        with self._lock:
            was_enabled = self._enabled_boot_epoch is not None
            active = self._active_task_id is not None
            self._enabled_boot_epoch = None
            previous = self._last_stop
        if not was_enabled and not active and previous is not None:
            return previous
        return self.stop(reason)

    def _watchdog_loop(self) -> None:
        while not self._closed.wait(self.watchdog_interval_s):
            with self._lock:
                deadline = self._active_deadline_monotonic
            if deadline is not None and deadline <= self.clock():
                self.stop("operator_command_expired")

    def close(self) -> None:
        self.disable("operator_gateway_close")
        self._closed.set()
        if self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=max(1.0, self.watchdog_interval_s * 4.0))

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled_boot_epoch is not None,
                "boot_epoch": self._enabled_boot_epoch,
                "active_task_id": self._active_task_id,
                "deadline_monotonic": self._active_deadline_monotonic,
                "last_call_id": self._last_call_id,
                "last_sequence": self._last_sequence,
                "publisher_implementation": (
                    "existing-isaac-filesystem-command-writer"
                ),
                "retry": "never",
            }
