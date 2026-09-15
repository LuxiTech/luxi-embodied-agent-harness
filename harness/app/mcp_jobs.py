"""UI-owned supervision for one long-running robot MCP task."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable
import uuid

from harness.control.long_task_control import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    LongTaskControlChannel,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_OUTPUT_CHARS = 64_000
ALLOWED_TOOLS = frozenset({"fetch_object"})


class MCPJobManager:
    """Own one long motion call and expose evidence-backed lifecycle state."""

    def __init__(
        self,
        root: Path,
        *,
        command_builder: Callable[[str, dict[str, Any], str], list[str]],
        channel: LongTaskControlChannel | None = None,
        events: Any | None = None,
        enabled: bool = True,
        idle_provider: Callable[[], dict[str, Any]] | None = None,
        stop_callback: Callable[[], None] | None = None,
        cooperative_cancel_seconds: float = 8.0,
        idle_timeout_seconds: float = 30.0,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._command_builder = command_builder
        self._channel = channel or LongTaskControlChannel()
        self._events = events
        self._enabled = bool(enabled)
        self._idle_provider = idle_provider
        self._stop_callback = stop_callback
        self._cooperative_cancel_seconds = max(
            0.1, float(cooperative_cancel_seconds)
        )
        self._idle_timeout_seconds = max(0.1, float(idle_timeout_seconds))
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._result_ready: dict[str, threading.Event] = {}

    @staticmethod
    def _validate_arguments(tool: str, arguments: Any) -> dict[str, Any]:
        if tool != "fetch_object":
            raise ValueError("only fetch_object is a UI-managed long task")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        allowed = {"object_id", "pickup_pose", "destination", "hand"}
        if set(arguments) - allowed:
            raise ValueError("fetch_object received unknown arguments")
        normalized = {
            "object_id": arguments.get("object_id"),
            "pickup_pose": arguments.get("pickup_pose"),
            "destination": arguments.get("destination", "start"),
            "hand": arguments.get("hand", "right"),
        }
        expected = {
            "object_id": {"water_bottle"},
            "pickup_pose": {"kitchen"},
            "destination": {"start", "living_room"},
            "hand": {"right"},
        }
        for name, accepted in expected.items():
            if normalized[name] not in accepted:
                raise ValueError(
                    f"unsupported fetch_object {name}: {normalized[name]!r}"
                )
        return normalized

    def _append_event(
        self,
        kind: str,
        title: str,
        detail: str,
        *,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._events is not None:
            self._events.append(
                "tool",
                kind,
                title,
                detail,
                level=level,
                data=data or {},
            )

    def _active_control(self) -> dict[str, Any] | None:
        payload = self._channel.read()
        if payload is not None and payload.get("state") in ACTIVE_STATES:
            return payload
        return None

    def start(
        self,
        tool: str,
        arguments: Any,
        *,
        source: str = "api",
    ) -> dict[str, Any]:
        if not self._enabled:
            raise RuntimeError("this backend does not support UI-managed object tasks")
        tool = str(tool)
        if tool not in ALLOWED_TOOLS:
            raise ValueError("unsupported UI-managed MCP tool")
        normalized = self._validate_arguments(tool, arguments)
        with self._lock:
            if any(job["state"] not in TERMINAL_STATES for job in self._jobs.values()):
                raise RuntimeError("another long robot task is already active")
            active_control = self._active_control()
            if active_control is not None:
                raise RuntimeError("another long robot task is already active")
            job_id = str(uuid.uuid4())
            now = time.time()
            job = {
                "job_id": job_id,
                "tool": tool,
                "arguments": normalized,
                "source": str(source),
                "state": "starting",
                "stage": "starting",
                "created_at": now,
                "updated_at": now,
                "output": "",
                "last_structured_result": None,
                "cancel_reason": None,
                "_transport_result": None,
            }
            self._jobs[job_id] = job
            self._result_ready[job_id] = threading.Event()
            self._channel.begin(
                job_id,
                tool=tool,
                owner="ui",
                arguments=normalized,
            )
            thread = threading.Thread(
                target=self._run,
                args=(job_id,),
                name=f"mcp-job-{job_id[:8]}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
        self._append_event(
            "job",
            "Long robot task started",
            f"{tool} · {job_id[:8]}",
            data={"job_id": job_id, "tool": tool, "arguments": normalized},
        )
        return self.status(job_id)

    @staticmethod
    def _parse_output(output: str, returncode: int) -> dict[str, Any]:
        clean = ANSI_RE.sub("", output).strip()
        parsed: Any = None
        candidates = [clean]
        if "\n" in clean:
            candidates.extend(reversed([line.strip() for line in clean.splitlines()]))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                    break
                except (ValueError, SyntaxError):
                    continue
        tool_error = returncode == 0 and clean.casefold().startswith("error")
        result: dict[str, Any] = {
            "ok": returncode == 0 and not tool_error,
            "exit_code": int(returncode),
        }
        if isinstance(parsed, (dict, list)):
            result["result"] = parsed
        elif tool_error:
            result["error"] = clean[-MAX_OUTPUT_CHARS:]
        elif clean:
            result["output"] = clean[-MAX_OUTPUT_CHARS:]
        return result

    @staticmethod
    def _structured_payload(transport: dict[str, Any]) -> dict[str, Any] | None:
        payload = transport.get("result")
        if isinstance(payload, dict):
            nested = payload.get("result")
            if isinstance(nested, dict) and "task_status" in nested:
                return nested
            return payload
        return None

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            tool = str(job["tool"])
            arguments = dict(job["arguments"])
        output_path = self.root / f"{job_id}.log"
        returncode = 1
        output = ""
        supervisor_error = ""
        try:
            command = self._command_builder(tool, arguments, job_id)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.root,
                start_new_session=True,
                close_fds=True,
            )
            with self._lock:
                self._processes[job_id] = process
                job = self._jobs[job_id]
                cancelled_before_spawn = bool(job.get("cancel_reason"))
                job.update(
                    started_at=time.time(),
                    updated_at=time.time(),
                    supervisor_pid=process.pid,
                )
                if not cancelled_before_spawn:
                    job.update(state="running", stage="starting")
            if cancelled_before_spawn:
                self._terminate_after_grace(process)
            output, _ = process.communicate()
            returncode = int(process.returncode or 0)
        except Exception as error:  # noqa: BLE001 - terminal supervisor evidence
            supervisor_error = f"{type(error).__name__}: {error}"[:2_000]
            output = f"mcp job supervisor error: {supervisor_error}\n"
        try:
            output_path.write_text(output[-MAX_OUTPUT_CHARS:], encoding="utf-8")
        except OSError:
            pass
        transport = self._parse_output(output, returncode)
        if supervisor_error:
            transport.update(ok=False, error=supervisor_error)
        payload = self._structured_payload(transport)
        with self._lock:
            self._processes.pop(job_id, None)
            job = self._jobs[job_id]
            cancelled = bool(job.get("cancel_reason"))
            execution_state = (
                "cancelled"
                if cancelled or (payload or {}).get("task_status") == "cancelled"
                else "completed"
                if (
                    transport.get("ok") is True
                    and isinstance(payload, dict)
                    and payload.get("tool_ok") is True
                    and payload.get("completed") is True
                    and payload.get("task_status") == "object_fetched"
                )
                else "failed"
            )
            job.update(
                state="settling",
                stage="waiting_for_idle",
                execution_state=execution_state,
                exit_code=returncode,
                output=ANSI_RE.sub("", output)[-MAX_OUTPUT_CHARS:],
                last_structured_result=payload,
                _transport_result=transport,
                execution_completed_at=time.time(),
                updated_at=time.time(),
            )
            ready = self._result_ready[job_id]
            ready.set()

        idle_evidence = self._wait_for_idle()
        with self._lock:
            job = self._jobs[job_id]
            final_state = str(job["execution_state"])
            if not idle_evidence.get("idle"):
                final_state = "failed"
                job["idle_error"] = "agent, navigation, and arm did not all become idle"
            job.update(
                state=final_state,
                stage=final_state,
                idle_evidence=idle_evidence,
                completed_at=time.time(),
                updated_at=time.time(),
            )
        self._channel.update(
            job_id,
            state=final_state,
            stage=final_state,
            result=payload,
            cancel_reason=job.get("cancel_reason"),
        )
        self._append_event(
            "job",
            "Long robot task finished",
            f"{tool} · {final_state}",
            level="info" if final_state == "completed" else "warning",
            data=self._summary_status(job_id),
        )

    def _idle_status_once(self) -> dict[str, Any]:
        if self._idle_provider is None:
            return {"idle": True, "agent_idle": True, "navigation_idle": True, "arm_idle": True}
        try:
            return dict(self._idle_provider())
        except Exception as error:  # noqa: BLE001 - evidence must fail closed
            return {"idle": False, "error": str(error)[:500]}

    def _wait_for_idle(self) -> dict[str, Any]:
        deadline = time.monotonic() + self._idle_timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self._idle_status_once()
            if last.get("idle") is True:
                return last
            time.sleep(0.1)
        return {**last, "idle": False, "timed_out": True}

    def run_and_wait(
        self,
        tool: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        started = self.start(tool, arguments, source="qwen")
        job_id = str(started["job_id"])
        ready = self._result_ready[job_id]
        if not ready.wait(max(0.1, float(timeout))):
            self.cancel(job_id, reason="tool_timeout")
            return {
                "ok": False,
                "error": f"机器人工具在 {float(timeout):.1f}s 后超时",
                "job_id": job_id,
            }
        with self._lock:
            result = dict(self._jobs[job_id].get("_transport_result") or {})
        result["job_id"] = job_id
        return result

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in job.items()
            if not key.startswith("_") and key != "arguments"
        }

    def _summary_status(self, job_id: str) -> dict[str, Any]:
        result = self.status(job_id)
        result.pop("output", None)
        return result

    def status(self, job_id: str) -> dict[str, Any]:
        try:
            normalized = str(uuid.UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError) as error:
            raise KeyError("invalid job id") from error
        with self._lock:
            job = self._jobs.get(normalized)
            if job is not None:
                result = self._public(dict(job))
            else:
                result = None
        control = self._channel.read()
        if result is None:
            if control is None or control.get("job_id") != normalized:
                raise KeyError("job not found")
            return dict(control)
        if control is not None and control.get("job_id") == normalized:
            if control.get("stage") and result.get("state") in ACTIVE_STATES:
                result["stage"] = control["stage"]
            if isinstance(control.get("last_structured_result"), dict):
                result["last_structured_result"] = control["last_structured_result"]
            if control.get("cancel_reason"):
                result["cancel_reason"] = control["cancel_reason"]
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
            active = next(
                (job for job in reversed(jobs) if job["state"] not in TERMINAL_STATES),
                None,
            )
            latest = jobs[-1] if jobs else None
        control = self._channel.read()
        if active is not None:
            return {
                "supported": self._enabled,
                "active": True,
                "job": self._summary_status(str(active["job_id"])),
            }
        if control is not None and control.get("state") in ACTIVE_STATES:
            return {
                "supported": self._enabled,
                "active": True,
                "job": dict(control),
            }
        if latest is not None:
            return {
                "supported": self._enabled,
                "active": False,
                "job": self._summary_status(str(latest["job_id"])),
            }
        if control is not None:
            idle_evidence = self._idle_status_once()
            if not idle_evidence.get("idle"):
                return {
                    "supported": self._enabled,
                    "active": True,
                    "job": {
                        **control,
                        "execution_state": control.get("state"),
                        "state": "settling",
                        "stage": "waiting_for_idle",
                        "idle_evidence": idle_evidence,
                    },
                }
            return {
                "supported": self._enabled,
                "active": False,
                "job": {**control, "idle_evidence": idle_evidence},
            }
        return {"supported": self._enabled, "active": False, "job": None}

    def cancel(self, job_id: str, *, reason: str = "operator_cancelled") -> dict[str, Any]:
        status = self.status(job_id)
        normalized = str(status["job_id"])
        if status.get("state") in TERMINAL_STATES:
            return status
        with self._lock:
            job = self._jobs.get(normalized)
            process = self._processes.get(normalized)
            if job is not None:
                job.update(
                    state="cancelling",
                    stage="stopping",
                    cancel_reason=str(reason)[:500],
                    cancel_requested_at=time.time(),
                    updated_at=time.time(),
                )
        self._channel.cancel(normalized, reason)
        if self._stop_callback is not None:
            try:
                self._stop_callback()
            except Exception:  # noqa: BLE001 - cooperative cancel still proceeds
                pass
        if process is not None and process.poll() is None:
            threading.Thread(
                target=self._terminate_after_grace,
                args=(process,),
                name=f"mcp-job-cancel-{normalized[:8]}",
                daemon=True,
            ).start()
        self._append_event(
            "job",
            "Long robot task cancellation requested",
            f"{normalized[:8]} · {reason}",
            level="warning",
            data={"job_id": normalized, "cancel_reason": reason},
        )
        return self.status(normalized)

    def _terminate_after_grace(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=self._cooperative_cancel_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=1.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def cancel_active(self, reason: str) -> bool:
        snapshot = self.snapshot()
        job = snapshot.get("job")
        if not snapshot.get("active") or not isinstance(job, dict):
            return False
        self.cancel(str(job["job_id"]), reason=reason)
        return True

    def close(self) -> None:
        self.cancel_active("ui_shutdown")
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=self._cooperative_cancel_seconds + 2.0)
