"""Single-Go2 terminal skills over the existing robot-local file protocol."""
from __future__ import annotations

import os
import math
import threading
import time
from pathlib import Path

from .go2_protocol import Go2CommandWriter, Go2RuntimePaths, read_fresh_go2_state

GO2_AGENT_CONTRACT = '''你是 LuxiAgent Harness 的单 Go2 机器人代理，只控制 go2-01。
只使用提供的工具。巡检使用 inspect_warehouse，寻物使用 object_search，跟随使用 follow_person。
不能执行双机分工、第二台机器人或跨机转派。历史观测不代表当前状态。
工具 completed=false 时必须如实报告未完成；工具失败后不得重复可能已产生运动的终端调用。
'''


def _tool(name, description, properties):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': list(properties), 'additionalProperties': False}}}


GO2_TOOLS = [
    _tool('observe_environment', '读取当前机器人的新鲜观测', {}),
    _tool('get_dimos_status', '读取单 Go2 运行状态', {}),
    _tool('stop_robot', '停止单 Go2 并验证静止', {}),
    _tool('inspect_warehouse', '巡检仓库指定区域', {'region': {'type': 'string', 'enum': ['north', 'south']}}),
    _tool('object_search', '寻找蓝色球、红色方块或水瓶', {'target': {'type': 'string', 'enum': ['blue_ball', 'red_cube', 'bottle']}}),
    _tool('follow_person', '跟随当前可见的人指定时长', {'duration_s': {'type': 'number', 'minimum': 1, 'maximum': 60}}),
]


class Go2RobotTools:
    def __init__(self, events, monitor, project_root, *, paths=None):
        self.events = events
        self.monitor = monitor
        self.project_root = Path(project_root)
        self.paths = paths or Go2RuntimePaths.configured()
        self.writer = Go2CommandWriter(self.paths)
        self.backend = 'mujoco-go2'
        self.blind_mode = False
        self._tools = GO2_TOOLS
        self._cancel = threading.Event()
        self._trace_local = threading.local()
        self._physical_cancel_probe = None
        self._observed_this_turn = False
        self._visual_branch_failed = False

    def set_physical_cancel_probe(self, probe):
        self._physical_cancel_probe = probe

    def _observation(self):
        return self.monitor.agent_snapshot()

    def _stop_robot(self, arguments):
        self.writer.request_stop('harness_stop')
        return {'ok': True, 'completed': False, 'task_status': 'stop_requested'}

    def _dispatch_tool(self, name, arguments):
        if name == 'observe_environment':
            return {'ok': True, 'observation': self._observation()}
        if name == 'get_dimos_status':
            state = read_fresh_go2_state(self.paths)
            return {'ok': state is not None, 'state': state or {'available': False}}
        if name == 'stop_robot':
            return self._stop_robot(arguments)
        if name == 'inspect_warehouse':
            region = {'north': '北区', 'south': '南区'}[arguments['region']]
            instruction = f'巡检仓库{region}'
        elif name == 'object_search':
            target = {'blue_ball': '蓝色球', 'red_cube': '红色方块', 'bottle': '水瓶'}[arguments['target']]
            instruction = f'寻找{target}'
        elif name == 'follow_person':
            instruction = f"跟随这个人 {float(arguments['duration_s']):g} 秒"
        else:
            return {'ok': False, 'completed': False, 'task_status': 'tool_denied', 'error': 'Unsupported Go2 skill'}
        state = read_fresh_go2_state(self.paths)
        if state is None or state.get('robot_id', 'go2-01') != 'go2-01' or state.get('busy'):
            return {'ok': False, 'completed': False, 'task_status': 'runtime_unavailable', 'error': 'Single Go2 is unavailable or busy'}
        if state.get('healthy') is not True:
            return {'ok': False, 'completed': False, 'task_status': 'risk_blocked', 'error': 'Go2 physical safety hold requires reset'}
        sequence = self.writer.submit(instruction, via_ros=os.environ.get("LUXI_GO2_EXTERNAL_ROS_AGENT", "0").lower() in {"1", "true", "yes"})
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            if self._cancel.is_set() or (self._physical_cancel_probe and self._physical_cancel_probe()):
                self.writer.request_stop('harness_cancelled')
                return {'ok': False, 'completed': False, 'task_status': 'cancelled'}
            state = read_fresh_go2_state(self.paths)
            if state is None:
                self.writer.request_stop('observation_unavailable')
                return {'ok': False, 'completed': False, 'task_status': 'observation_unavailable'}
            if state.get('healthy') is not True:
                self.writer.request_stop('physical_safety_hold')
                return {'ok': False, 'completed': False, 'task_status': 'risk_blocked'}
            if state.get('last_command_sequence') == sequence and not state.get('busy'):
                result = state.get('last_result')
                if not isinstance(result, dict):
                    return {'ok': False, 'completed': False, 'task_status': 'invalid_tool_result'}
                status = result.get('task_status')
                finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                metric = finite(result.get('final_distance_m')) and finite(result.get('final_bearing_deg'))
                verified = (status == 'arrived_verified' and metric and result['final_distance_m'] <= 1.08 and abs(result['final_bearing_deg']) <= 12
                            if name == 'object_search' else
                            status == 'follow_verified' and metric and abs(result['final_distance_m']-1.2) <= .35 and abs(result['final_bearing_deg']) <= 30 and finite(result.get('tracking_duration_s')) and result['tracking_duration_s'] >= arguments['duration_s'] and finite(result.get('frame_coverage')) and result['frame_coverage'] >= .90
                            if name == 'follow_person' else
                            status == 'inspection_route_verified' and isinstance(result.get('route_waypoints_completed'), int) and result['route_waypoints_completed'] > 0)
                completed = result.get('completed') is True and result.get('stationary_confirmed') is True and verified
                return {'ok': True, **result, 'completed': completed,
                        'task_status': status if completed or result.get('completed') is not True else 'verification_failed',
                        'evidence': {'verification_observed': completed and name == 'object_search',
                                     'follow_verified': completed and name == 'follow_person',
                                     'inspection_route_verified': completed and name == 'inspect_warehouse'}}
            time.sleep(.05)
        self.writer.request_stop('tool_timeout')
        return {'ok': False, 'completed': False, 'task_status': 'tool_timeout'}
