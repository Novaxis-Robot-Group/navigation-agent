"""组合机器人与导航会话，实现 Agent 的动作契约。"""

import asyncio
import math
import time

from navigation_agent.contracts.agent_backend import ToolError
from navigation_agent.contracts.robot import Robot
from navigation_agent.contracts.vln import NavigationSession
from navigation_agent.navigation.progress import MotionProgress


class NavigationBackend:
    def __init__(
        self, robot: Robot, session: NavigationSession, *, max_attempts=1,
        time_limit_enabled=True, inactivity_timeout_s=15.0,
    ):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError('max_attempts 必须为 1 到 5 的整数')
        if type(time_limit_enabled) is not bool:
            raise ValueError('time_limit_enabled 必须为布尔值')
        if (isinstance(inactivity_timeout_s, bool)
                or not isinstance(inactivity_timeout_s, (int, float))
                or not math.isfinite(inactivity_timeout_s) or inactivity_timeout_s <= 0):
            raise ValueError('inactivity_timeout_s 必须为有限正数')
        self.max_attempts = max_attempts
        self.time_limit_enabled = time_limit_enabled
        self.inactivity_timeout_s = inactivity_timeout_s
        self.robot = robot
        self.session = session
        self.progress = MotionProgress()
        self.blocked_timeout_s = 2
        self.robot.sample_sink = self.progress.record
        self.active_navigation_id = None
        self.preserve_history = False
        self._navigation_sequence = 0
        self._navigation_interrupt = None

    async def configure_history(self, enabled):
        configure = getattr(self.session, 'configure_history', None)
        if enabled and configure is None:
            raise ToolError('导航会话未提供历史延续能力')
        if configure is not None:
            await configure(enabled)
        self.preserve_history = enabled
        self.robot.preserve_history = enabled

    async def _finish_history(self):
        if self.preserve_history:
            await self.session.finish()

    async def __aenter__(self):
        await self.robot.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.robot.__aexit__(exc_type, exc, traceback)

    @property
    def look_angle_deg(self):
        return self.robot.look_angle_deg

    @property
    def telemetry(self):
        return self.robot.telemetry

    @telemetry.setter
    def telemetry(self, callback):
        self.robot.telemetry = callback

    def require_control(self):
        self.robot.require_control()

    async def hold(self):
        await self.robot.hold()

    @property
    def supports_streaming_observation(self):
        return callable(getattr(self.robot, 'snapshot_observation', None))

    async def snapshot_observation(self):
        return await self.robot.snapshot_observation()

    def request_navigation_interrupt(self, identity, reason):
        if identity is None or identity != self.active_navigation_id:
            return False
        if self._navigation_interrupt is not None:
            return False
        self._navigation_interrupt = reason
        return True

    async def observe(self):
        return await self.robot.observe()

    def begin_motion(self, action):
        self.progress.reset()
        self.robot.begin_motion(action)

    async def look(self, direction):
        await self._finish_history()
        self.progress.reset()
        try:
            result = await self.robot.look(direction)
            return {**result, 'motion': self.progress.summary()}
        finally:
            self.robot.end_motion()

    async def orient(self, heading_deg):
        """按当前位置的实际朝向分段转回；只转向，导航须由下一帧重新确认。"""
        if (isinstance(heading_deg, bool) or not isinstance(heading_deg, (int, float))
                or not math.isfinite(heading_deg)):
            raise ToolError('Invalid orientation heading')
        step_angle = self.look_angle_deg
        if (isinstance(step_angle, bool) or not isinstance(step_angle, (int, float))
                or not math.isfinite(step_angle) or not 0 < step_angle <= 90):
            raise ToolError('Invalid orientation step angle')
        self.require_control()
        await self._finish_history()
        self.begin_motion('orient')
        remaining_turns = None
        try:
            while True:
                _, observation = await self.observe()
                yaw = observation.get('yaw')
                if not observation.get('pose_valid') or yaw is None:
                    raise ToolError('Orientation requires a valid current pose')
                delta = (heading_deg - math.degrees(yaw) + 180) % 360 - 180
                if abs(delta) <= 10:
                    return {'reason': 'oriented', 'heading_deg': math.degrees(yaw),
                            'motion': self.progress.summary()}
                if remaining_turns is None:
                    # 按初始角差和配置转角分段，另留两次纠偏；最后一段后仍先检查新朝向。
                    remaining_turns = math.ceil(abs(delta) / step_angle) + 2
                if remaining_turns == 0:
                    raise ToolError('Orientation did not converge')
                direction = 'left' if delta > 0 else 'right'
                result = await self.robot.look(direction, angle_deg=min(abs(delta), step_angle))
                remaining_turns -= 1
                if result['reason'] != 'turned':
                    raise ToolError('Could not turn back to the selected view')
        finally:
            self.robot.end_motion()

    async def recover(self):
        self.progress.reset()
        invalidate_history = True
        try:
            result = await self.robot.recover()
            invalidate_history = result['reason'] != 'recovery_unavailable'
            if invalidate_history:
                result['motion'] = self.progress.summary()
            return result
        finally:
            try:
                # 后端未执行恢复动作时保留暂停历史；异常或实际恢复后作废。
                if invalidate_history:
                    await self._finish_history()
            finally:
                self.robot.end_motion()

    async def navigate(self, instruction, seconds):
        self.require_control()
        if not instruction.strip() or len(instruction) > 500 or not 0 < seconds <= 60:
            raise ToolError('Invalid navigation arguments')
        await self.hold()
        self.begin_motion('navigate')
        deadline = time.monotonic() + seconds if self.time_limit_enabled else None
        attempts = []
        history_retained = False
        self._navigation_interrupt = None
        try:
            for attempt in range(1, self.max_attempts + 1):
                self.require_control()
                if attempt > 1 and not self.session.status().startup_retryable:
                    result.pop('startup_failed', None)
                    break
                try:
                    # 有时限时所有尝试共用预算；连续模式仍接受外层任务取消。
                    async with asyncio.timeout_at(deadline) as navigation_timeout:
                        result = await self._navigate_attempt(instruction, deadline)
                except TimeoutError:
                    if navigation_timeout.expired():
                        result = {'reason': 'timeout'}
                    else:
                        result = {
                            'reason': 'navigation_error',
                            'detail': 'Navigation service response timed out',
                        }
                attempts.append({'attempt': attempt, **result})
                # 本次尝试已结束，停车和重试期间不再接受它的观察中断。
                self.active_navigation_id = None
                self._navigation_interrupt = None
                if self.preserve_history:
                    if result.get('reason') in ('agent_interrupt', 'blocked', 'no_progress', 'path_repeated'):
                        await self.session.pause()
                        history_retained = True
                    else:
                        await self._finish_history()
                await self.hold()

                # 停车期间到达的结果同样阻止重试，不能只检查报错那一刻。
                if result.get('startup_failed') and not self.session.status().startup_retryable:
                    result.pop('startup_failed')
                if not result.get('startup_failed') or attempt == self.max_attempts:
                    break
                if deadline is not None and deadline - time.monotonic() <= 1:
                    break
                await asyncio.sleep(1)
                if deadline is not None and time.monotonic() >= deadline:
                    break
            return {
                **result,
                'instruction': instruction,
                'motion': self.progress.summary(),
                'arrival_verified': False,
                'time_limit_s': seconds if self.time_limit_enabled else None,
                'attempts': attempts,
                'history_retained': history_retained,
                'history_reused': getattr(self.session, 'history_reused', False),
            }
        finally:
            self.active_navigation_id = None
            self._navigation_interrupt = None
            try:
                if not history_retained:
                    await self._finish_history()
                # 正常返回、超时和内部取消均只结束本连接拥有的运动。
                status = self.robot.status()
                if status.automatic and not status.locked:
                    await self.hold()
            except ToolError:
                # 连接失效时不能重新取得控制；退出连接负责释放本连接的运动。
                pass
            finally:
                self.robot.end_motion()

    async def _navigate_attempt(self, instruction, deadline):
        await self.session.start(instruction)
        self._navigation_sequence += 1
        self.active_navigation_id = self._navigation_sequence
        blocked_since = None

        # 保留停止原因的优先级；模型停止不代表找到目标。
        while deadline is None or time.monotonic() < deadline:
            robot = self.robot.status()
            model = self.session.status()
            if robot.locked:
                raise ToolError('Robot control was interrupted')
            if model.error or robot.error:
                result = {'reason': 'navigation_error', 'detail': model.error or robot.error}
                if model.startup_retryable and not robot.blocked:
                    result['startup_failed'] = True
                return result
            if model.stopped:
                return {'reason': 'model_stop'}
            if not robot.automatic:
                raise ToolError('Navigation was interrupted by another controller')
            if self.progress.stalled():
                return {'reason': 'no_progress'}
            # 连续导航不能因局部轨迹已走完、模型不出 STOP 而永久静止等待。
            if not self.time_limit_enabled and self.progress.inactive(self.inactivity_timeout_s):
                return {'reason': 'no_progress', 'detail': 'No translation or rotation progress'}
            if not self.time_limit_enabled and self.progress.path_repeated:
                return {'reason': 'path_repeated', 'detail': 'Returned to an earlier point after a navigation loop'}
            now = time.monotonic()
            if robot.blocked:
                blocked_since = blocked_since or now
                if now - blocked_since >= self.blocked_timeout_s:
                    return {'reason': 'blocked'}
            else:
                blocked_since = None
            if self._navigation_interrupt is not None:
                return {'reason': 'agent_interrupt', 'detail': self._navigation_interrupt}
            await self.robot.wait_update()
        return {'reason': 'timeout'}
