"""MuJoCo composition adapter over existing navigation and entity services.

Navigation/observation callbacks must belong to the robot-local runtime. This
module never plans paths, writes simulator poses or discovers other sessions.
"""
import math
import time
import threading

from .contracts import CancelledError, CancellationToken


class MujocoCompositionBackend:
    backend = "mujoco"
    interaction_model = "sim_attachment"

    def __init__(self, *, observe, navigate, entity_port, entity_id="water_bottle", locate=None):
        self._observe = observe
        self._navigate = navigate
        self._locate = locate
        self.scene_id_provider = lambda: None
        self.location_catalog_path = None
        from pathlib import Path
        self.entity_catalog_path = Path(__file__).resolve().parents[2] / "config/composed/entities.json"
        self.entity_port = entity_port
        self.entity_id = entity_id
        self.supported_entities = (entity_id,)
        self._released_entity = None
        self.world_revision = lambda: ""
        self.references = {}
        self.release_supported = False
        self.place_supported = callable(getattr(entity_port, "place_attachment", None))
        self._placed_surface = None
        self.emergency_stop = lambda reason: None

    def observe(self):
        # Callback supplies fresh sensor odometry; entity coordinates are never
        # substituted for robot localization. Only attachment metadata is exposed.
        observation = dict(self._observe())
        state = self.entity_port.entity_state(self.entity_id, timeout=1.0)
        stamp = state.get("evidence_timestamp")
        unavailable = state.get("task_status") == "entity_unavailable"
        if ((not state.get("operation_ok") and not unavailable) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or not 0 <= time.time()-stamp <= 1.0):
            return {}
        observation["world_revision"] = self.world_revision()
        observation["entity_available"] = not unavailable
        observation["attached_entity"] = self.entity_id if state.get("attached") is True else None
        observation["released_entity"] = self._released_entity
        if self._placed_surface is not None:
            placement = state.get("details", {}).get("placement")
            if isinstance(placement, dict) and placement.get("surface") == self._placed_surface:
                observation["placement"] = placement
        # Release verification requires an independent backend/sensor callback
        # supplying object_xy/object_stationary, never assumes release implies settling.
        return observation

    def reference_catalog(self):
        from harness.skills.composed_tasks import _fresh
        observation = self.observe()
        if not _fresh(observation):
            raise RuntimeError("缺少新鲜位置，不能准备组合任务")
        catalog = {'references': {}, 'visual_regions': {}}
        if self.location_catalog_path is not None:
            from .composed_locations import location_catalog
            catalog = location_catalog(self.location_catalog_path, self.scene_id_provider())
        for key, pose in self.references.items():
            if key in catalog['references'] and tuple(pose) != catalog['references'][key]:
                raise ValueError(f'配置位置 {key} 与已标注目录冲突')
        import json
        descriptions = json.loads(self.entity_catalog_path.read_text())
        operations = ['attach']
        if self._locate is not None:
            operations.append('locate')
        if self.place_supported:
            operations.append('place')
        if self.release_supported:
            operations.append('release')
        entities = {entity: {**descriptions.get(entity, {'name': entity, 'aliases': []}),
                             'operations': list(operations)} for entity in self.supported_entities}
        metadata = dict(catalog.get('reference_metadata', {}))
        for key in self.references:
            metadata.setdefault(key, {'kind': 'pose', 'aliases': [], 'description': '操作者提供的导航位姿。'})
        metadata['start'] = {'kind': 'pose', 'aliases': ['起点'], 'description': '本轮新鲜起始机器人位姿。'}
        return {**catalog, "entity_catalog": entities, "reference_metadata": metadata, "references": {**catalog['references'], **self.references, "start": tuple(observation["pose"])},
                "world_revision": observation["world_revision"], "supported_entities": self.supported_entities}

    @staticmethod
    def _timeout(cancel, deadline):
        cancel.raise_if_cancelled()
        remaining = deadline - time.monotonic() if deadline is not None else 0
        if remaining <= 0:
            raise CancelledError("composition deadline expired")
        return min(2.0, remaining)

    # 将单次导航交给后端；携物时另起本地监控，附着丢失即取消并停车，不等待模型决策。
    def navigate(self, pose, cancel, deadline):
        self._timeout(cancel, deadline)
        # A local monitor cancels the same command path; it never plans or emits motion.
        from .command_gateway import CombinedCancellationToken
        before = self.observe()
        from harness.skills.composed_tasks import _fresh
        if not _fresh(before):
            return {"operation_ok": False, "task_status": "observation_unavailable"}
        held = before.get("attached_entity")
        if held is None:
            return self._navigate(pose, cancel, deadline)
        stopped, lost = threading.Event(), threading.Event()
        local_cancel = CancellationToken()
        combined = CombinedCancellationToken(cancel, local_cancel)
        def watch():
            while not stopped.wait(.1):
                try:
                    state = self.entity_port.entity_state(held, timeout=.3)
                    stamp = state.get("evidence_timestamp")
                    valid = (state.get("operation_ok") is True and state.get("attached") is True
                             and isinstance(stamp, (float, int)) and not isinstance(stamp, bool)
                             and 0 <= time.time()-stamp <= 1.0)
                except Exception:
                    valid = False
                if not valid:
                    lost.set()
                    local_cancel.cancel()
                    self.emergency_stop("composition_attachment_lost")
                    return
        watcher = threading.Thread(target=watch, name="composition-attachment-monitor", daemon=True)
        watcher.start()
        try:
            try:
                value = self._navigate(pose, combined, deadline)
            except CancelledError:
                if not lost.is_set():
                    raise
                value = {}
        finally:
            stopped.set()
            watcher.join(timeout=7)
        if lost.is_set() or watcher.is_alive():
            return {"operation_ok": False, "task_status": "side_effect_unknown"}
        return value

    # 通过实体控制端口在指定取物位姿执行仿真附着，并转换后端回执；不是接触抓取。
    def attach(self, entity_id, pickup, cancel, deadline):
        if entity_id != self.entity_id:
            return {"operation_ok": False, "task_status": "invalid_input"}
        result = self.entity_port.attach_at_pose(entity_id, pickup,
                                               timeout=self._timeout(cancel, deadline), cancel=cancel)
        if result.get("operation_ok"):
            self._released_entity = None
            self._placed_surface = None
        return self._operation(result)

    def locate(self, entity_id, cancel, deadline):
        self._timeout(cancel, deadline)
        if self._locate is None or entity_id != self.entity_id:
            return {'operation_ok': False, 'task_status': 'tool_denied'}
        return self._locate(entity_id, cancel, deadline)

    def place(self, entity_id, surface, cancel, deadline):
        from harness.robots.composed_placement import placement_matches
        from .composed_locations import location_catalog
        self._timeout(cancel, deadline)
        if not self.place_supported or entity_id != self.entity_id or self.location_catalog_path is None:
            return {"operation_ok": False, "task_status": "tool_denied"}
        catalog = location_catalog(self.location_catalog_path, self.scene_id_provider())
        if surface not in catalog.get('placement_surfaces', {}).values():
            return {"operation_ok": False, "task_status": "tool_denied", "error": "placement annotation changed"}
        # Physical owner independently checks stationarity, pose, support and occupied volume.
        raw = self.entity_port.place_attachment(entity_id, surface,
                    timeout=self._timeout(cancel, deadline), cancel=cancel)
        if not raw.get('operation_ok'):
            return self._operation(raw)
        self._placed_surface = surface
        self._released_entity = entity_id
        until = min(deadline, time.monotonic()+3.0)
        while time.monotonic() < until:
            cancel.raise_if_cancelled()
            observation = self.observe()
            if (observation.get('attached_entity') is None
                    and placement_matches(observation.get('placement'), surface, entity_id)):
                return {'operation_ok':True, 'interaction_model':'sim_placement'}
            time.sleep(.03)
        return {'operation_ok':False, 'task_status':'side_effect_unknown',
                'error':'placement applied but fresh stable support evidence unavailable'}

    def release(self, entity_id, cancel, deadline):
        if entity_id != self.entity_id:
            return {"operation_ok": False, "task_status": "invalid_input"}
        result = self.entity_port.release_attachment(entity_id,
                                                    timeout=self._timeout(cancel, deadline), cancel=cancel)
        if result.get("operation_ok"):
            self._released_entity = entity_id
        return self._operation(result)

    @staticmethod
    def _operation(result):
        if result.get("task_status") == "entity_backend_timeout":
            return {**result, "task_status": "side_effect_unknown", "operation_ok": False}
        return result


class MultiplexExecutionFence:
    """Route admission within the same Pipeline; never execute a nested call."""
    def __init__(self, *fences):
        self.fences = tuple(f for f in fences if f is not None)

    def _fence(self, request):
        matches = [f for f in self.fences if f.applies(request)]
        if len(matches) != 1:
            raise PermissionError("command must have exactly one runtime fence")
        return matches[0]

    def applies(self, request):
        return any(f.applies(request) for f in self.fences)

    def admit(self, request, cancel):
        return self._fence(request).admit(request, cancel)

    def finish(self, request, result):
        self._fence(request).finish(request, result)


def mujoco_candidate_backend(skills):
    """Explicit acceptance assembly only; never selected by production env alone."""
    from harness.robots.g1.mujoco.entity_manipulation import EntityControlChannel
    from harness.control.terminal_cancellation import TerminalCancellationChannel
    from harness.skills.tool_results import _structured_mcp_payload
    last_stamp = [0.0]

    def observe():
        requested_at = time.time()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            pose = skills.monitor.probe.pose()
            if pose is not None and pose.timestamp > max(last_stamp[0], requested_at) and 0 <= time.time()-pose.timestamp <= 1.0:
                last_stamp[0] = pose.timestamp
                return {"pose": [pose.x, pose.y, pose.yaw],
                        "timestamp_monotonic": time.monotonic()-(time.time()-pose.timestamp)}
            time.sleep(.02)
        return {}

    def rpc(name, arguments, cancel, deadline):
        cancel.raise_if_cancelled()
        remaining = min(115.0, deadline-time.monotonic())
        if remaining <= 0:
            return {"operation_ok": False, "task_status": "tool_timeout"}
        # Sensor readiness may precede MCP registration. Check this endpoint's actual catalog.
        import json
        import urllib.request
        registered = False
        ready_deadline = min(deadline, time.monotonic()+5)
        while time.monotonic() < ready_deadline:
            cancel.raise_if_cancelled()
            try:
                query = urllib.request.Request("http://127.0.0.1:9990/mcp", data=json.dumps({
                    "jsonrpc": "2.0", "id": "composition-readiness", "method": "tools/list", "params": {}}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(query, timeout=.5) as response:
                    catalog = json.load(response)
                registered = any(tool.get("name") == name for tool in catalog.get("result", {}).get("tools", []))
            except (OSError, ValueError):
                registered = False
            if registered:
                break
            time.sleep(.1)
        if not registered:
            return {"operation_ok": False, "task_status": "observation_unavailable", "reason": f"组合 RPC {name} 尚未注册"}
        remaining = min(115.0, deadline-time.monotonic())
        if remaining <= 0:
            return {"operation_ok": False, "task_status": "tool_timeout"}
        revision = TerminalCancellationChannel().revision() or ""
        from harness.integrations.mcp.tool_client import RobotMcpClient
        client = RobotMcpClient()
        client.project_root, client.backend = skills.project_root, "mujoco"
        client._trace_local = threading.local()
        client._cancel = threading.Event()
        client._physical_cancel_probe = lambda: cancel.cancelled or time.monotonic() >= deadline
        client._command_runner = client._run_cancelable_subprocess
        try:
            transport = client._mcp_call(name, {**arguments,
                "expires_at": time.time()+remaining, "cancel_revision": revision,
            }, timeout=remaining)
            result = _structured_mcp_payload(transport)
            return result or {"operation_ok": False, "task_status": "side_effect_unknown"}
        finally:
            client._physical_cancel_probe = None

    def navigate(pose, cancel, deadline):
        return rpc('navigate_composed_pose', {'x': pose[0], 'y': pose[1],
            'yaw': pose[2] if pose[2] is not None else 0.0, 'require_heading': pose[2] is not None}, cancel, deadline)

    def locate(entity_id, cancel, deadline):
        return rpc('locate_composed_object', {'entity_id': entity_id}, cancel, deadline)

    return MujocoCompositionBackend(observe=observe, navigate=navigate, locate=locate, entity_port=EntityControlChannel())
