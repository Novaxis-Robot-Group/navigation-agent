"""通过控制台接入 MuJoCo 感知、控制权和运动，不发起模型会话。"""

import asyncio
import math
from collections import deque
import time

from navigation_agent.contracts.agent_backend import ToolError, make_observation
from navigation_agent.contracts.robot import MotionSample, RobotStatus
from navigation_agent.contracts.transport import ControlChannel
from navigation_agent.robots.mujoco import recovery as recover


class MujocoRobot:
    # 现有 0.8 rad/s、1.5 秒开环转向的标称角度，保持原行为。
    look_angle_deg = 70

    def __init__(self, channel: ControlChannel):
        self.channel = channel
        self.channel.on_snapshot = self.on_snapshot
        self.telemetry = None
        self.sample_sink = None
        self.motion_action = None
        self.preserve_history = False
        self.navigation_heading = None
        self.breadcrumbs = deque(maxlen=600)

    @property
    def state(self):
        return self.channel.state

    @state.setter
    def state(self, value):
        self.channel.state = value

    @property
    def error(self):
        return self.channel.error

    async def __aenter__(self):
        await self.channel.__aenter__()
        try:
            await self.hold()
        except BaseException:
            await self.channel.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.channel.__aexit__(exc_type, exc, traceback)

    def require_manual(self):
        if self.error:
            raise self.error
        if not self.state.get('control', {}).get('manual'):
            raise ToolError('Robot control was interrupted')

    async def hold(self):
        if self.preserve_history and self.state.get('control', {}).get('manual'):
            await self.channel.send({'type': 'twist', 'linear': 0., 'angular': 0.})
            return
        await self.channel.command('acquire_control')
        if not self.state['control'].get('manual'):
            raise ToolError('Robot control was not acquired')

    def status(self):
        control = self.state.get('control', {})
        return RobotStatus(
            manual=bool(control.get('manual')),
            automatic=bool(control.get('auto')),
            locked=bool(control.get('locked')),
            blocked=bool(self.state.get('simulation', {}).get('backend', {}).get('blocked')),
            error=self.state.get('mpc', {}).get('error', ''),
        )

    def begin_motion(self, action):
        self.navigation_heading = None
        self.motion_action = action
        if action == 'navigate':
            self.remember_pose()

    def end_motion(self):
        self.motion_action = None
        self.navigation_heading = None

    async def _snapshot(self):
        await self.channel.wait_snapshot()

    async def wait_update(self):
        await self.channel.wait_snapshot()

    def on_snapshot(self):
        self.navigation_heading = self._navigation_heading()
        if self.motion_action == 'navigate':
            self.remember_pose()
        if self.motion_action:
            simulation = self.state.get('simulation', {})
            sample = self.channel.diagnostics(self.motion_action)
            if self.sample_sink:
                self.sample_sink(MotionSample(
                    monotonic_s=sample['monotonic_s'],
                    pose=dict(sample['pose']),
                    blocked=bool(simulation.get('backend', {}).get('blocked')),
                ))
            if self.telemetry:
                self.telemetry(sample)

    def _navigation_heading(self):
        """只读避障后的路径方向；使用同一状态中的位姿，独立于相机采图时间。"""
        if self.motion_action != 'navigate' or not self.state.get('control', {}).get('auto'):
            return None
        mpc = self.state.get('mpc', {})
        route = mpc.get('reference_world', [])
        simulation = self.state.get('simulation', {})
        pose = simulation.get('pose', {})
        if len(route) < 2 or not all(k in pose for k in ('x', 'y', 'yaw')):
            return None
        # 路径起点是控制器求解时的位置，第一段方向允许正常转弯。
        start = route[0]
        point = next((p for p in route[1:] if math.hypot(p[0] - start[0], p[1] - start[1]) > .05), None)
        if point is None:
            return None
        heading = math.atan2(point[1] - start[1], point[0] - start[0])
        error = math.atan2(math.sin(heading - pose['yaw']), math.cos(heading - pose['yaw']))
        return {'stamp_ns': time.time_ns(), 'path_heading_deg': math.degrees(heading),
                'heading_error_deg': math.degrees(error),
                'linear_speed_m_s': simulation.get('velocity', {}).get('linear', 0.)}

    def remember_pose(self):
        return recover.remember_pose(self)

    def recovery_path(self):
        return recover.recovery_path(self)

    async def recover(self):
        return await recover.recover(self)

    def require_control(self):
        self.require_manual()

    async def snapshot_observation(self):
        """导航中的只读观察不申请控制权，也不触发停车。"""
        jpeg, payload = await self.channel.snapshot_observation()
        observation = make_observation(
            pose=payload['pose'], blocked=self.status().blocked,
            camera_stamp_ns=payload['camera_stamp_ns'],
        )
        if self.navigation_heading is not None:
            observation['navigation_heading'] = dict(self.navigation_heading)
        return jpeg, observation

    async def observe(self):
        self.require_manual()
        await self.hold()

        # 停车后等一帧新图，避免使用移动中的旧画面
        previous = self.state['simulation']['camera'].get('stamp_ns')
        async with asyncio.timeout(5):
            while True:
                await self._snapshot()
                camera = self.state['simulation']['camera']
                if camera.get('ready') and camera.get('stamp_ns') != previous:
                    break
        if not self.state['control'].get('manual'):
            raise ToolError('Robot control was interrupted')
        return await self.snapshot_observation()

    async def look(self, direction, *, angle_deg=None):
        self.require_manual()
        if direction not in ('left', 'right'):
            raise ToolError('Invalid look direction')
        await self.hold()
        self.begin_motion('look')

        # 短时转动机器人，结束后停车再决策
        duration = 1.5
        if angle_deg is not None:
            if not math.isfinite(angle_deg) or not 0 < angle_deg <= 90:
                raise ToolError('Invalid turn angle')
            duration = math.radians(angle_deg) / .8
        deadline = time.monotonic() + duration

        # 保持短时转向，控制权丢失时立即退出
        while time.monotonic() < deadline:
            if not self.state['control'].get('manual'):
                raise ToolError('Robot control was interrupted')
            await self.channel.send(
                {
                    'type': 'twist',
                    'linear': 0.0,
                    'angular': 0.8 if direction == 'left' else -0.8,
                }
            )
            await self._snapshot()
            await asyncio.sleep(0.05)
        await self.hold()  # 模型决策前先停车。
        return {'reason': 'turned', 'direction': direction}

