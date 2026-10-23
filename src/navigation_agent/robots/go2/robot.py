"""GO2 感知与动作；使用本体位姿、控制权和独立停车保障。"""

import asyncio
import math
import time

from navigation_agent.contracts.agent_backend import ToolError, make_observation
from navigation_agent.contracts.robot import MotionSample, RobotStatus
from navigation_agent.contracts.vln import SessionStatus
from navigation_agent.contracts.runtime import ProtectionSettings


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stationary(odom, settings=None):
    settings = settings or ProtectionSettings()
    vectors = [odom.get('twist_linear'), odom.get('twist_angular')]
    return all(isinstance(v, list) and len(v) == 3 and all(finite(x) for x in v) for v in vectors) and (
        math.hypot(*vectors[0][:2]) <= settings.stationary_linear and abs(vectors[1][2]) <= settings.stationary_angular
    )


class Go2Robot:
    def __init__(self, channel, *, look_angle_deg=70, look_tolerance_deg=10,
                 look_angular_min=.2, look_angular_max=.4, look_timeout_s=8, look_stall_s=2.5,
                 look_gain=1.5, look_progress_deg=1, protection=None):
        self.protection = protection or ProtectionSettings()
        self.look_gain, self.look_progress_deg = look_gain, look_progress_deg
        if not finite(look_angle_deg) or not 30 <= look_angle_deg <= 90:
            raise ValueError('GO2 look_angle_deg must be between 30 and 90')
        self.look_angle_deg = float(look_angle_deg)
        if not finite(look_tolerance_deg) or not 0 < look_tolerance_deg <= 10:
            raise ValueError('GO2 look_tolerance_deg must be greater than 0 and at most 10')
        self.look_tolerance_deg = float(look_tolerance_deg)
        if (not all(finite(v) and v > 0 for v in (look_angular_min, look_angular_max, look_timeout_s, look_stall_s))
            or look_angular_min > look_angular_max or look_stall_s > look_timeout_s):
            raise ValueError('Invalid GO2 turn speed or timeout')
        self.look_angular_min, self.look_angular_max = look_angular_min, look_angular_max
        self.look_timeout_s, self.look_stall_s = look_timeout_s, look_stall_s
        self.channel = channel
        self.telemetry = self.sample_sink = None
        self.motion_action = None
        self.mode = 'initial'
        self.channel.on_snapshot = self.on_snapshot

    async def __aenter__(self):
        await self.channel.__aenter__()
        try:
            await self.channel.wait_for(lambda: self.channel.odom_stamp is not None)
            await self.hold()
            return self
        except BaseException:
            await self.channel.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *args):
        await self.channel.__aexit__(*args)

    def require_ready(self):
        self.channel.check()
        robot = self.channel.state.get('robot', {})
        if not robot.get('connected') or robot.get('mode') != 'WALK':
            raise ToolError('GO2 is disconnected or not in WALK mode')
        odom = self.channel.state.get('odom', {})
        pose = odom.get('pose', {})
        if (time.monotonic() - self.channel.odom_received > self.protection.odom_max_age_s or not odom.get('pose_valid')
            or not all(finite(pose.get(axis)) for axis in ('x', 'y', 'yaw'))):
            raise ToolError('GO2 odometry is missing, stale or invalid')
        if self.channel.locked():
            raise ToolError('GO2 control belongs to another connection')
        return odom

    def require_control(self):
        self.require_ready()
        error = self.channel.robot_error()
        if error:
            raise ToolError(error)
        if not self.channel.control.get('owner') or self.channel.state.get('control_source') != 'manual':
            raise ToolError('GO2 manual control was interrupted')

    def status(self):
        self.require_ready()
        control, state = self.channel.control, self.channel.state
        error = self.channel.robot_error() or state.get('mpc', {}).get('error', '')
        if not error and state.get('vln', {}).get('state') == 'ERROR':
            error = 'GO2 navigation model reported an error'
        return RobotStatus(
            manual=bool(control.get('owner')) and state.get('control_source') == 'manual',
            automatic=bool(control.get('auto_owner')) and state.get('control_source') == 'auto',
            locked=self.channel.locked(), blocked=False, error=error,
        )

    def begin_motion(self, action):
        self.motion_action = action

    def end_motion(self):
        self.motion_action = None

    def on_snapshot(self):
        if not self.motion_action:
            return
        odom = self.channel.state.get('odom', {})
        pose = odom.get('pose', {}) if odom.get('pose_valid') else {}
        if self.sample_sink:
            self.sample_sink(MotionSample(time.monotonic(), dict(pose), False))
        if self.telemetry:
            self.telemetry({
                'time_ns': time.time_ns(), 'monotonic_s': time.monotonic(),
                'action': self.motion_action, 'pose': dict(pose),
                'velocity': {'linear': odom.get('twist_linear', [None])[0],
                             'angular': odom.get('twist_angular', [None, None, None])[2]},
                'vln': dict(self.channel.state.get('vln', {})),
                'mpc': dict(self.channel.state.get('mpc', {})),
                'backend': {'obstacle_feedback_available': False},
            })

    async def wait_update(self):
        await asyncio.sleep(0.05)
        self.require_ready()

    async def release(self, *, preserve_history=False):
        self.require_ready()
        self.mode = 'transition'
        # 控制权增量可能先于释放状态到达；始终等待 SDK 释放回执，不能仅信任缓存快照。
        parameters = ({'preserve_history': True, 'generation': self.channel.navigation_generation}
                      if preserve_history else {})
        await self.channel.command('agent_release', **parameters)
        await self.channel.wait_for(lambda: not self.channel.state['agent']['release_pending']
                                    and self.channel.state.get('control_source') == 'disabled'
                                    and not self.channel.owns_control())
        self.require_ready()

    async def settle(self):
        stable_since = None
        try:
            async with asyncio.timeout(self.protection.settle_timeout_s):
                while True:
                    self.require_control()
                    odom = self.channel.state['odom']
                    if stationary(odom, self.protection):
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= self.protection.settle_s:
                            return odom['stamp_ns']
                    else:
                        stable_since = None
                    await asyncio.sleep(0.05)
        except TimeoutError:
            raise ToolError('GO2 did not become stationary') from None

    async def hold(self):
        self.require_ready()
        if self.channel.control.get('owner'):
            await self.channel.send({'type': 'twist', 'x': 0.0, 'y': 0.0, 'z': 0.0})
        else:
            result = self.channel.navigation_result or {}
            if self.mode == 'manual' or (self.mode == 'auto' and not self.channel.control.get('auto_owner')
                                         and result.get('reason') not in ('model_stop', 'navigation_error')):
                raise ToolError('GO2 control was interrupted; refusing to reacquire')
            if self.mode != 'initial':
                await self.release()
            await self.channel.command('acquire_control')
            await self.channel.wait_for(lambda: self.channel.control.get('owner')
                                        and self.channel.state.get('control_source') == 'manual')
        self.mode = 'manual'
        return await self.settle()

    def _observation_metadata(self, data):
        """停车和行进采图共用校验，保留本体原始采图时间与匹配位姿。"""
        sequence, stamp = data.get('sequence'), data.get('camera_stamp_ns')
        if type(sequence) is not int or type(stamp) is not int or stamp <= 0:
            raise ToolError('GO2 observation timestamps are invalid')
        observation = make_observation(pose=data.get('pose'), blocked=False, camera_stamp_ns=stamp)
        if (data.get('pose_valid') is not True or not observation['pose_valid']
            or not self.channel.odom_frame or data.get('odom_frame_id') != self.channel.odom_frame
            or not finite(data.get('camera_age_ms'))
            or not 0 <= data['camera_age_ms'] <= self.protection.camera_max_age_ms
            or not finite(data.get('odom_match_ms'))
            or not 0 <= data['odom_match_ms'] <= self.protection.odom_match_ms):
            raise ToolError('GO2 observation metadata is invalid')
        return observation

    async def snapshot_observation(self):
        """只读取同一帧的图片与位姿，不停车、不申请或切换控制权。"""
        self.channel.check()
        result = await self.channel.observation()
        if result is None:
            raise ToolError('GO2 live observation is temporarily unavailable')
        jpeg, data = result
        observation = self._observation_metadata(data)
        self.channel.check()
        return jpeg, observation

    async def observe(self):
        self.require_control()
        stopped_stamp = await self.hold()
        after = self.channel.state.get('camera', {}).get('sequence', -1)
        try:
            async with asyncio.timeout(self.protection.observation_timeout_s):
                while True:
                    self.require_control()
                    result = await self.channel.observation(after)
                    if result is not None:
                        jpeg, data = result
                        sequence, stamp = data.get('sequence'), data.get('camera_stamp_ns')
                        if type(sequence) is not int or type(stamp) is not int or stamp <= 0 or sequence <= after:
                            raise ToolError('GO2 observation timestamps are invalid')
                        after = max(after, sequence)
                        if stamp >= stopped_stamp:
                            observation = self._observation_metadata(data)
                            self.require_control()
                            if not stationary(data.get('odom', {}), self.protection) or not stationary(self.channel.state['odom'], self.protection):
                                raise ToolError('GO2 moved during observation')
                            return jpeg, observation
                    await asyncio.sleep(0.05)
        except TimeoutError:
            raise ToolError('GO2 did not provide a fresh observation after stopping') from None

    async def look(self, direction, *, angle_deg=None):
        self.require_control()
        if direction not in ('left', 'right'):
            raise ToolError('Invalid look direction')
        await self.hold()
        self.begin_motion('look')
        previous = self.channel.state['odom']['pose']['yaw']
        turned = 0.0
        angle = self.look_angle_deg if angle_deg is None else angle_deg
        if not finite(angle) or not 0 < angle <= 90:
            raise ToolError('Invalid turn angle')
        target = math.radians(angle) * (1 if direction == 'left' else -1)
        deadline = time.monotonic() + self.look_timeout_s
        progress_at = time.monotonic()
        progress_angle = 0.0
        try:
            while time.monotonic() < deadline:
                self.require_control()
                yaw = self.channel.state['odom']['pose']['yaw']
                turned += math.atan2(math.sin(yaw - previous), math.cos(yaw - previous))
                previous = yaw
                remaining = target - turned
                if abs(remaining) <= math.radians(self.look_tolerance_deg):
                    return {'reason': 'turned', 'direction': direction}
                # 实机两次在 64–66° 停滞；末段比例控制降到约 0.1 rad/s 后
                # 没有继续转动。保持有限的最低转速，仍用真实 yaw 判定完成。
                if abs(turned - progress_angle) >= math.radians(self.look_progress_deg):
                    progress_at, progress_angle = time.monotonic(), turned
                elif time.monotonic() - progress_at > self.look_stall_s:
                    raise ToolError(f'GO2 turn stalled: turned {math.degrees(turned):.1f} deg, '
                                    f'target {math.degrees(target):.1f} deg')
                angular = math.copysign(min(self.look_angular_max, max(self.look_angular_min, self.look_gain * abs(remaining))), remaining)
                await self.channel.send({'type': 'twist', 'x': 0.0, 'y': 0.0, 'z': angular})
                await asyncio.sleep(0.1)
            raise ToolError(f'GO2 turn timed out: turned {math.degrees(turned):.1f} deg, '
                            f'target {math.degrees(target):.1f} deg')
        finally:
            if self.channel.control.get('owner') and not self.channel.error:
                await self.hold()

    async def recover(self):
        await self.hold()
        return {'reason': 'recovery_unavailable', 'detail': 'GO2 reverse-path perception is not configured'}


class Go2NavigationSession:
    def __init__(self, robot, model_url):
        self.robot = robot
        self.channel = robot.channel
        self.model_url = model_url
        self.preserve_history = False
        self.history_paused = False

    async def configure_history(self, enabled):
        capabilities = self.channel.state.get('agent', {}).get('capabilities', {})
        if enabled and capabilities.get('navigation_history_continuation') is not True:
            raise ToolError('Go2 服务不支持历史延续，请同步更新并重启 Web 与 VLN 节点')
        self.preserve_history = enabled

    async def pause(self):
        generation = self.channel.navigation_generation
        await self.channel.command('pause_vln', generation=generation)
        await self.channel.wait_for(lambda: not self.channel.state['agent']['release_pending']
                                    and self.channel.state.get('control_source') == 'disabled'
                                    and not self.channel.owns_control())
        self.history_paused = True
        await self.channel.command('acquire_control')
        await self.channel.wait_for(lambda: self.channel.control.get('owner')
                                    and self.channel.state.get('control_source') == 'manual')
        self.robot.mode = 'manual'

    async def finish(self):
        generation = self.channel.navigation_generation
        if generation is not None:
            await self.channel.command('finish_vln', generation=generation)
        self.history_paused = False

    @property
    def history_reused(self):
        return self.channel.state.get('agent', {}).get('history_reused', False)

    async def start(self, instruction):
        self.robot.require_control()
        await self.robot.release(preserve_history=self.preserve_history and self.history_paused)
        await self.channel.command('set_server_url', server_url=self.model_url)
        self.channel.begin_navigation()
        self.robot.mode = 'auto'
        parameters = ({'preserve_history': True, 'generation': self.channel.navigation_generation}
                      if self.preserve_history else {})
        await self.channel.command('set_vln', enabled=True, mode='objnav', instruction=instruction, **parameters)
        self.history_paused = False
        await self.channel.wait_for(lambda: (
            self.channel.state.get('agent', {}).get('navigation_generation') == self.channel.navigation_generation
            and (self.status().stopped or self.status().error or (
                self.channel.control.get('auto_owner') and self.channel.state.get('control_source') == 'auto'
            ))
        ))
        self.robot.require_ready()

    def status(self):
        result = self.channel.navigation_result or {}
        reason = result.get('reason')
        error = (self.channel.navigation_robot_error or self.channel.navigation_mpc_error
                 or self.channel.navigation_model_error)
        if reason in ('navigation_error', 'lease_expired', 'disconnected'):
            error = error or reason
        retryable = (
            bool(self.channel.navigation_model_error)
            and not self.channel.navigation_robot_error
            and not self.channel.navigation_mpc_error
            and not self.channel.navigation_received_result
            and not self.channel.navigation_motion_started
            and reason in (None, 'navigation_error')
        )
        return SessionStatus(
            stopped=reason == 'model_stop', error=error, startup_retryable=retryable,
        )
