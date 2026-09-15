"""Correlate an already-admitted Harness tool with its native DimOS execution.

This transport adapter does not create a Loop, Scope, registry or ToolPipeline.
An invocation is single-use and exists only while its admitted MCP call is active.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time
from typing import Any, Mapping

from .isaac_protocol import BACKEND_NAME, SCHEMA_VERSION, atomic_write_json, read_json


class IsaacToolBroker:
    def __init__(self, runtime_host, capability_ids, *, parse_result):
        self.runtime_host = runtime_host
        self.capability_ids = frozenset(capability_ids)
        self.parse_result = parse_result
        self._lock = threading.RLock()
        self._invocation = None

    def prepare_owner(self):
        state = self.runtime_host.status().state
        return getattr(state, 'value', state) in {'READY', 'BUSY'}

    @contextmanager
    def invocation(self, request, capability_id, arguments, cancel):
        if capability_id not in self.capability_ids:
            yield
            return
        if request.capability_id != capability_id:
            raise PermissionError('Native capability differs from admitted Harness tool')
        with self._lock:
            if self._invocation is not None:
                raise PermissionError('Another native tool invocation is active')
            self._invocation = {'request': request, 'capability_id': capability_id,
                                'arguments': dict(arguments), 'cancel': cancel,
                                'claimed': False, 'opened_at': time.time()}
        try:
            yield
        finally:
            with self._lock:
                self._invocation = None

    def cancel_active(self, reason):
        with self._lock:
            invocation = self._invocation
        if invocation is not None:
            invocation['cancel'].cancel()

    def execute_request(self, native: Mapping[str, Any], channel):
        with self._lock:
            invocation = self._invocation
            if invocation is None or invocation['claimed']:
                return self._denied('No unclaimed Harness admission exists')
            request = invocation['request']
            cancel = invocation['cancel']
            supplied = native.get('arguments') or {}
            if (native.get('action') != invocation['capability_id']
                    or native.get('requested_at', 0) < invocation['opened_at']
                    or any(supplied.get(k) != v for k, v in invocation['arguments'].items())
                    or request.boot_epoch != self.runtime_host.boot_epoch
                    or request.deadline_monotonic is None
                    or time.monotonic() >= request.deadline_monotonic
                    or cancel.cancelled):
                return self._denied('Native request does not match active Harness admission')
            invocation['claimed'] = True
        token = str(native['token'])
        atomic_write_json(channel.grant_path, {
            'schema_version': SCHEMA_VERSION, 'backend': BACKEND_NAME,
            'action': native['action'], 'token': token, 'admitted': True,
            'boot_epoch': request.boot_epoch, 'tool_call_id': request.tool_call_id,
            'granted_at': time.time(),
        })
        cancel_written = False
        while time.monotonic() < request.deadline_monotonic + 10.0:
            if (cancel.cancelled or time.monotonic() >= request.deadline_monotonic) and not cancel_written:
                atomic_write_json(channel.cancel_path, {
                    'schema_version': SCHEMA_VERSION, 'backend': BACKEND_NAME,
                    'token': token, 'reason': 'harness_cancelled', 'cancelled_at': time.time(),
                })
                cancel_written = True
            result = read_json(channel.result_path)
            if result and result.get('token') == token:
                return self.parse_result(str(result.get('raw_result', '')), str(native['action']))
            time.sleep(.02)
        atomic_write_json(channel.cancel_path, {
            'schema_version': SCHEMA_VERSION, 'backend': BACKEND_NAME,
            'token': token, 'reason': 'native_result_missing', 'cancelled_at': time.time(),
        })
        return {'ok': False, 'completed': False, 'task_status': 'side_effect_unknown',
                'automatic_tool_replay': False, 'error': 'Native tool result missing after admission'}

    @staticmethod
    def _denied(error):
        return {'ok': False, 'completed': False, 'task_status': 'tool_denied', 'error': error}
