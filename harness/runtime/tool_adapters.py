"""Small adapters over existing skill implementations, with explicit cancellation."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from .contracts import CancellationToken


@dataclass
class FunctionToolAdapter:
    handler: Callable

    def execute(self, request, cancel):
        cancel.raise_if_cancelled()
        return self.handler(dict(request.arguments))


class DeadlineCancellation(CancellationToken):
    def __init__(self, parent, deadline, extra_cancel=lambda: False):
        super().__init__()
        self.parent = parent
        self.deadline = deadline
        self.extra_cancel = extra_cancel

    def cancel(self):
        super().cancel()
        self.parent.cancel()

    @property
    def cancelled(self):
        return (super().cancelled or self.parent.cancelled or self.extra_cancel()
                or (self.deadline is not None and time.monotonic() >= self.deadline))


class SkillToolAdapter:
    def __init__(self, skills, delegate=None):
        self.skills = skills
        self.delegate = delegate

    def execute(self, request, cancel):
        bounded = DeadlineCancellation(cancel, request.deadline_monotonic, self.skills._cancel.is_set)
        bounded.raise_if_cancelled()
        self.skills.set_physical_cancel_probe(lambda: bounded.cancelled)
        self.skills._trace_local.tool_request = request
        self.skills._trace_local.cancel = bounded
        self.skills._trace_local.turn_id = request.turn_id
        self.skills._trace_local.task_id = request.task_id
        set_context = getattr(self.skills.events, 'set_thread_context', None)
        if callable(set_context):
            set_context(turn_id=request.turn_id, task_id=request.task_id, step_id=request.step_id,
                        tool_call_id=request.tool_call_id, loop='harness')
        try:
            if self.delegate is not None:
                result = self.delegate.execute(request, cancel)
            else:
                result = self.skills._dispatch_tool(request.capability_id, dict(request.arguments))
            bounded.raise_if_cancelled()
            return result
        finally:
            self.skills._trace_local.tool_request = None
            self.skills._trace_local.cancel = None
            self.skills.set_physical_cancel_probe(None)
            clear_context = getattr(self.skills.events, 'clear_thread_context', None)
            if callable(clear_context):
                clear_context()


def safety_observation(observe):
    from .command_gateway import _iso_timestamp
    snapshot = observe()
    pose = snapshot.get('pose') or {}
    stamp = pose.get('timestamp')
    sampled = _iso_timestamp(snapshot.get('sampled_at'))
    fresh = (isinstance(stamp, (float, int)) and not isinstance(stamp, bool)
             and 0 <= time.time() - stamp <= 1.5
             and sampled is not None and 0 <= time.time() - sampled <= 1.5)
    risk = (snapshot.get('metrics') or {}).get('risk', 'unknown')
    return {'timestamp_monotonic': time.monotonic() if fresh else 0.0,
            'risk': risk if risk in {'normal', 'clear', 'warning', 'critical'} else 'fault'}
