"""Extracted shared implementation; independent of Agent entry points."""

from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any
from harness.control.terminal_cancellation import TerminalCancellationChannel

from harness.skills.tool_results import (
    MAX_TOOL_OUTPUT_CHARS,
    _parse_structured_cli_output,
    _redact,
)


class RobotMcpClient:
    """Cancellable MCP transport used by robot skills."""

    def _mcp_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        command = [str(self.project_root / "scripts/dimos.sh"), "mcp", "call", tool_name]
        if arguments:
            command.extend(["--json-args", json.dumps(arguments, ensure_ascii=False)])
        broker = getattr(self, "native_broker", None)
        request = getattr(self._trace_local, "tool_request", None)
        if broker is not None and tool_name in broker.capability_ids:
            if request is None:
                raise PermissionError("Native motion requires an admitted Harness request")
            with broker.invocation(request, tool_name, arguments or {}, self._trace_local.cancel):
                return self._command_runner(command, timeout)
        return self._command_runner(command, timeout)

    @staticmethod
    def _run_subprocess(command: list[str], timeout: float) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"机器人工具在 {timeout:.1f}s 后超时"}
        except OSError as exc:
            return {"ok": False, "error": _redact(str(exc))[:1_000]}
        output = (completed.stdout + completed.stderr).strip()[-MAX_TOOL_OUTPUT_CHARS:]
        parsed = _parse_structured_cli_output(output)
        redacted_output = _redact(output)
        tool_error = completed.returncode == 0 and redacted_output.lower().startswith("error")
        result: dict[str, Any] = {
            "ok": completed.returncode == 0 and not tool_error,
            "exit_code": completed.returncode,
        }
        if parsed is not None:
            result["result"] = parsed
        elif tool_error:
            result["error"] = redacted_output
        elif output:
            result["output"] = redacted_output
        return result

    def _run_cancelable_subprocess(
        self, command: list[str], timeout: float
    ) -> dict[str, Any]:
        """Run one local tool with live cancellation and bounded process cleanup."""

        started = time.monotonic()
        launch_command = [
            sys.executable,
            "-m",
            "harness.control.parent_death_exec",
            "--",
            *command,
        ]
        try:
            process = subprocess.Popen(
                launch_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return {"ok": False, "error": _redact(str(exc))[:1_000]}
        reason = ""
        while process.poll() is None:
            pipeline_cancelled = False
            probe = self._physical_cancel_probe
            if probe is not None:
                try:
                    pipeline_cancelled = bool(probe())
                except Exception:
                    # A broken cancellation channel is fail-safe for physical work.
                    pipeline_cancelled = True
            if self._cancel.is_set() or pipeline_cancelled:
                reason = "cancelled"
                break
            if time.monotonic() - started >= timeout:
                reason = "timeout"
                break
            time.sleep(0.025)
        if reason:
            if self._physical_cancel_probe is not None or self._cancel.is_set():
                # Killing an MCP client does not cancel the already-running
                # server-side terminal Skill.  Publish a separate cancellation
                # edge before cleanup so that process observes and stops.
                try:
                    TerminalCancellationChannel().signal(reason)
                except OSError:
                    # The independent emergency-zero path below remains the
                    # fail-safe fallback if the IPC filesystem is unavailable.
                    pass
            try:
                # The old refresher must lose ownership before zero is sent;
                # a graceful delay would let it overwrite the stop command.
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                process.wait(timeout=0.5)
            # A killed velocity refresher may not execute its Python finally.
            # Use the non-cancellable primitive for a fresh, repeated zero.
            if len(command) >= 2 and command[1] == "move":
                self._publish_emergency_zero(command[0])
            return {
                "ok": False,
                "cancelled": reason == "cancelled",
                "error": (
                    "机器人工具已取消并发送零速命令"
                    if reason == "cancelled"
                    else f"机器人工具在 {timeout:.1f}s 后超时并发送零速命令"
                ),
            }
        stdout, stderr = process.communicate()
        output = (stdout + stderr).strip()[-MAX_TOOL_OUTPUT_CHARS:]
        parsed = _parse_structured_cli_output(output)
        redacted_output = _redact(output)
        tool_error = process.returncode == 0 and redacted_output.lower().startswith("error")
        result: dict[str, Any] = {
            "ok": process.returncode == 0 and not tool_error,
            "exit_code": process.returncode,
        }
        if parsed is not None:
            result["result"] = parsed
        elif tool_error:
            result["error"] = redacted_output
        elif output:
            result["output"] = redacted_output
        return result

    def _publish_emergency_zero(self, dimos_script: str) -> bool:
        """Publish zero through the backend's configured local command port."""

        try:
            if self.backend == "isaac-g1":
                from harness.robots.g1.isaac.isaac_protocol import IsaacCommandWriter, IsaacRuntimePaths

                IsaacCommandWriter(IsaacRuntimePaths.configured()).stop()
                return True
            # The launcher owns the dedicated LCM URL. A UI or acceptance
            # process may not have that environment in its Python transport;
            # using it directly could report a zero no robot consumed.
            result = self._run_subprocess(
                [
                    dimos_script,
                    "move",
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--yaw",
                    "0",
                    "--duration",
                    "0.6",
                ],
                6.0,
            )
            return result.get("ok") is True
        except Exception:
            return False
