"""Acceptance driver for the production Harness composition, without a second pipeline."""
import json
import tempfile
import threading
import time
from pathlib import Path
from harness.runtime.composition import create_agent_runtime_service
from harness.runtime.contracts import CancellationToken, ToolRequest
from harness.runtime.safety_kernel import LuxiSafetyKernel

class HarnessAcceptanceDriver:
    def __init__(self, skills, events, *, backend, enabled_tools, gateway=None, runtime_host=None, runtime_fenced_tools=(), policy=None, timeout_overrides=None):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root/'config').mkdir()
        (root/'config/physical-cutover.json').write_text(json.dumps({'schema_version':1,'backends':{backend:{'status':'accepted','accepted_tools':list(enabled_tools),'default_enabled':list(enabled_tools),'runtime_fenced_tools':list(runtime_fenced_tools)}}}))
        skills.events = events
        skills.blind_mode = False
        skills._trace_local = threading.local()
        if not hasattr(skills, 'set_physical_cancel_probe'):
            skills.set_physical_cancel_probe = lambda probe: setattr(skills, '_physical_cancel_probe', probe)
        from harness.runtime.command_gateway import MonitorStopGateway
        gateway = gateway or MonitorStopGateway(skills)
        self.safety = LuxiSafetyKernel(gateway, policy=policy)
        self.service = create_agent_runtime_service(events, object(), project_root=root, backend=backend, skills=skills, runtime_host=runtime_host, safety=self.safety, command_gateway=gateway, enabled_tools=enabled_tools, model_provider=object())
        self.enabled_tools = frozenset(enabled_tools)
        self._scope = self.service.scope
        self._resolver = self.service.loop.capabilities
        self._runtime_fence = self.service.loop.tools.execution_fence
        self._timeout_overrides = dict(timeout_overrides or {})
        self.host = runtime_host
        self._runtime_host = runtime_host
    def execute(self, name, arguments, turn_id, task_id, step_id, call_id, cancelled):
        scope = self._scope
        snapshot = self._resolver.snapshot(scope)
        cancel = CancellationToken()
        if cancelled: cancel.cancel()
        request = ToolRequest(session_id=scope.session_id, turn_id=turn_id, task_id=task_id, step_id=step_id, tool_call_id=call_id, agent_id=scope.agent_id, capability_id=name, arguments=arguments, snapshot_revision=snapshot.snapshot_revision, robot_id=next(iter(scope.robot_ids)), boot_epoch=self.host.boot_epoch if self.host else None, deadline_monotonic=time.monotonic()+self._timeout_overrides.get(name, 10))
        result = self.service.loop.tools.execute(request, scope=scope, snapshot=snapshot, cancel=cancel)
        value = dict(result.raw or result.payload)
        value.update(ok=result.tool_ok, completed=result.completed, task_status=result.status, safety_evidence=dict(result.evidence), error=result.error)
        return value
    def reconcile_unfinished(self):
        return self.service.recover_interrupted()
