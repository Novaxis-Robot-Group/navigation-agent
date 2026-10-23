"""用独立机器人与会话替换整个底层，验证 Agent 后端无需设备判断。"""
from tests.config_support import agent_settings

import asyncio
from dataclasses import replace
import unittest

from navigation_agent.contracts.agent_backend import ToolError, make_observation
from navigation_agent.contracts.robot import MotionSample, RobotStatus
from navigation_agent.contracts.vln import SessionStatus
from navigation_agent.navigation.backend import NavigationBackend


class Robot:
    def __init__(self):
        self.telemetry = None
        self.sample_sink = None
        self.current = RobotStatus(True, False, False, False)
        self.closed = False
        self.moving = False
        self.waiting = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True
        self.moving = False

    def require_control(self):
        if self.current.locked:
            raise ToolError('Control lost')

    def status(self):
        return self.current

    def begin_motion(self, action):
        self.moving = True
        self.sample_sink(MotionSample(0, {'x': 0, 'y': 0, 'yaw': 0}, False))

    def end_motion(self):
        pass

    async def hold(self):
        self.require_control()
        self.moving = False

    async def observe(self):
        await self.hold()
        return b'\xff\xd8image', make_observation(
            pose={'x': 0, 'y': 0, 'yaw': 0}, blocked=False, camera_stamp_ns=1
        )

    async def look(self, direction):
        self.begin_motion('look')
        await self.hold()
        return {'reason': 'turned', 'direction': direction}

    async def recover(self):
        return {'reason': 'recovery_unavailable'}

    async def wait_update(self):
        await self.waiting.wait()


class Session:
    def __init__(self, stopped=True):
        self.stopped = stopped
        self.instructions = []

    async def start(self, instruction):
        self.instructions.append(instruction)

    def status(self):
        return SessionStatus(self.stopped, '')


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_recovery_retains_history_but_actual_recovery_invalidates_it(self):
        from unittest.mock import AsyncMock
        robot, session = Robot(), Session()
        session.finish = AsyncMock()
        backend = NavigationBackend(robot, session)
        backend.preserve_history = True
        await backend.recover()
        session.finish.assert_not_awaited()
        robot.recover = AsyncMock(return_value={'reason': 'recovered'})
        await backend.recover()
        session.finish.assert_awaited_once()

    async def test_swappable_robot_and_model_session(self):
        robot, session = Robot(), Session()
        async with NavigationBackend(robot, session) as backend:
            image, observation = await backend.observe()
            self.assertTrue(observation['pose_valid'])
            self.assertTrue(image.startswith(b'\xff\xd8'))
            self.assertEqual((await backend.look('left'))['reason'], 'turned')
            result = await backend.navigate('Go to the doorway.', seconds=agent_settings().execution.navigation_seconds)
            self.assertEqual(result['reason'], 'model_stop')
            self.assertFalse(result['arrival_verified'])
            self.assertFalse(robot.moving)
            self.assertEqual((await backend.recover())['reason'], 'recovery_unavailable')
        self.assertTrue(robot.closed)
        self.assertEqual(session.instructions, ['Go to the doorway.'])

    async def test_cancel_closes_owned_robot_and_stops_motion(self):
        robot, session = Robot(), Session(stopped=False)
        robot.current = replace(robot.current, automatic=True)

        async def navigate():
            async with NavigationBackend(robot, session) as backend:
                await backend.navigate('Go forward.', seconds=agent_settings().execution.navigation_seconds)

        task = asyncio.create_task(navigate())
        await asyncio.sleep(0)
        self.assertTrue(robot.moving)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(robot.closed)
        self.assertFalse(robot.moving)

    async def test_continuous_navigation_exits_after_sustained_inactivity(self):
        robot, session = Robot(), Session(stopped=False)
        robot.current = replace(robot.current, automatic=True)
        elapsed = 0

        async def update():
            nonlocal elapsed
            elapsed += 1
            self.assertLessEqual(elapsed, 15)
            robot.sample_sink(MotionSample(elapsed, {'x': 0, 'y': 0, 'yaw': 0}, False))

        robot.wait_update = update
        backend = NavigationBackend(robot, session, time_limit_enabled=False)
        result = await backend.navigate('Go forward.', seconds=0.001)
        self.assertEqual(result['reason'], 'no_progress')
        self.assertEqual(elapsed, 15)
        self.assertFalse(robot.moving)

    async def test_continuous_navigation_returns_control_after_a_closed_path(self):
        robot, session = Robot(), Session(stopped=False)
        robot.current = replace(robot.current, automatic=True)
        points = iter([(2, 0), (2, 2), (0, 2), (0, 0)])
        elapsed = 0

        async def update():
            nonlocal elapsed
            elapsed += 1
            x, y = next(points)
            robot.sample_sink(MotionSample(elapsed, {'x': x, 'y': y, 'yaw': 0}, False))

        robot.wait_update = update
        backend = NavigationBackend(robot, session, time_limit_enabled=False)
        result = await backend.navigate('Go to the doorway.', seconds=15.)
        self.assertEqual(result['reason'], 'path_repeated')
        self.assertFalse(robot.moving)
        self.assertEqual(result['motion']['path_length_m'], 8.)

    async def test_slow_translation_or_turning_can_continue_until_model_stop(self):
        for motion in ('translation', 'turning'):
            with self.subTest(motion=motion):
                robot, session = Robot(), Session(stopped=False)
                robot.current = replace(robot.current, automatic=True)
                elapsed = 0

                async def update():
                    nonlocal elapsed
                    elapsed += 1
                    pose = {'x': 0, 'y': 0, 'yaw': 0}
                    pose['x' if motion == 'translation' else 'yaw'] = elapsed * 0.02
                    robot.sample_sink(MotionSample(elapsed, pose, False))
                    if elapsed == 40:
                        session.stopped = True

                robot.wait_update = update
                backend = NavigationBackend(robot, session, time_limit_enabled=False)
                result = await backend.navigate('Go forward.', seconds=0.001)
                self.assertEqual(result['reason'], 'model_stop')
                self.assertEqual(elapsed, 40)
                self.assertFalse(robot.moving)

    async def test_outer_task_budget_still_cancels_continuous_navigation_and_stops(self):
        robot, session = Robot(), Session(stopped=False)
        robot.current = replace(robot.current, automatic=True)
        backend = NavigationBackend(robot, session, time_limit_enabled=False)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.03):
                await backend.navigate('Go forward.', seconds=15)
        self.assertFalse(robot.moving)
        self.assertIsNone(backend.active_navigation_id)

    async def test_connection_timeout_is_not_reported_as_navigation_budget(self):
        for enabled in (True, False):
            with self.subTest(time_limit_enabled=enabled):
                robot, session = Robot(), Session(stopped=False)
                robot.current = replace(robot.current, automatic=True)

                async def update():
                    raise TimeoutError('no control snapshots')

                robot.wait_update = update
                backend = NavigationBackend(robot, session, time_limit_enabled=enabled)
                result = await backend.navigate('Go forward.', seconds=15)
                self.assertEqual(result['reason'], 'navigation_error')
                self.assertIn('service response', result['detail'])
                self.assertFalse(robot.moving)

    async def test_continuous_mode_keeps_bounded_startup_retries(self):
        robot = Robot()
        robot.current = replace(robot.current, automatic=True)

        class RetrySession(Session):
            def status(self):
                if len(self.instructions) == 1:
                    return SessionStatus(False, 'startup failed', startup_retryable=True)
                return SessionStatus(True, '')

        session = RetrySession()
        backend = NavigationBackend(robot, session, max_attempts=2, time_limit_enabled=False)
        result = await backend.navigate('Go forward.', seconds=0.001)
        self.assertEqual(result['reason'], 'model_stop')
        self.assertEqual(len(result['attempts']), 2)
        self.assertFalse(robot.moving)

    async def test_report_from_finished_attempt_cannot_interrupt_startup_retry(self):
        robot = Robot()
        robot.current = replace(robot.current, automatic=True)

        class RetrySession(Session):
            def status(self):
                if len(self.instructions) == 1:
                    return SessionStatus(False, 'startup failed', startup_retryable=True)
                return SessionStatus(self.stopped, '')

        session = RetrySession(stopped=False)
        backend = NavigationBackend(robot, session, max_attempts=2)
        late_requests = []
        original_hold = robot.hold

        async def hold():
            if len(session.instructions) == 1:
                late_requests.append(backend.request_navigation_interrupt(1, 'target_seen'))
            await original_hold()

        async def update():
            session.stopped = True

        robot.hold, robot.wait_update = hold, update
        async with backend:
            result = await backend.navigate('Go forward.', seconds=3)
        self.assertEqual(len(result['attempts']), 2)
        self.assertEqual(result['reason'], 'model_stop')
        self.assertEqual(late_requests, [False])

    async def test_lost_control_prevents_model_start(self):
        robot, session = Robot(), Session()
        robot.current = replace(robot.current, locked=True)
        async with NavigationBackend(robot, session) as backend:
            with self.assertRaises(ToolError):
                await backend.navigate('Go forward.', seconds=agent_settings().execution.navigation_seconds)
        self.assertEqual(session.instructions, [])

    async def test_orient_uses_shortest_turn_and_never_starts_vln(self):
        import math
        robot, session = Robot(), Session()
        yaw = 170.
        turns = []
        async def observe():
            return b'\xff\xd8image', make_observation(
                pose={'x': 0, 'y': 0, 'yaw': math.radians(yaw)}, blocked=False, camera_stamp_ns=1)
        async def look(direction, *, angle_deg=None):
            nonlocal yaw
            turns.append((direction, angle_deg))
            yaw += angle_deg if direction == 'left' else -angle_deg
            return {'reason': 'turned'}
        robot.observe, robot.look = observe, look
        robot.look_angle_deg = 75
        async with NavigationBackend(robot, session) as backend:
            result = await backend.orient(-170)
        self.assertEqual(result['reason'], 'oriented')
        self.assertEqual(turns, [('left', 20.)])
        self.assertEqual(session.instructions, [])

    async def test_orient_splits_large_turn_and_refuses_missing_pose(self):
        import math
        robot, session = Robot(), Session()
        yaw = 0.
        turns = []
        async def observe():
            return b'\xff\xd8image', make_observation(
                pose={'x': 0, 'y': 0, 'yaw': math.radians(yaw)}, blocked=False, camera_stamp_ns=1)
        async def look(direction, *, angle_deg=None):
            nonlocal yaw
            turns.append(angle_deg)
            yaw += angle_deg if direction == 'left' else -angle_deg
            return {'reason': 'turned'}
        robot.observe, robot.look, robot.look_angle_deg = observe, look, 75
        async with NavigationBackend(robot, session) as backend:
            self.assertEqual((await backend.orient(170))['reason'], 'oriented')
        self.assertEqual(turns, [75, 75, 20.])
        async def unknown():
            return b'\xff\xd8image', make_observation(pose=None, blocked=False, camera_stamp_ns=1)
        robot.observe = unknown
        async with NavigationBackend(robot, session) as backend:
            with self.assertRaisesRegex(ToolError, 'valid current pose'):
                await backend.orient(0)
        self.assertEqual(session.instructions, [])

    async def test_orient_respects_small_step_angles_and_checks_after_last_turn(self):
        import math
        for step_angle in (30, 45, 60, 75):
            with self.subTest(step_angle=step_angle):
                robot, session = Robot(), Session()
                yaw = 0.
                turns = []
                async def observe():
                    return b'image', make_observation(
                        pose={'x': 0, 'y': 0, 'yaw': math.radians(yaw)},
                        blocked=False, camera_stamp_ns=1)
                async def look(direction, *, angle_deg=None):
                    nonlocal yaw
                    turns.append(angle_deg)
                    yaw += angle_deg if direction == 'left' else -angle_deg
                    return {'reason': 'turned'}
                robot.observe, robot.look, robot.look_angle_deg = observe, look, step_angle
                async with NavigationBackend(robot, session) as backend:
                    result = await backend.orient(170)
                self.assertEqual(result['reason'], 'oriented')
                self.assertAlmostEqual(result['heading_deg'], 170)
                self.assertTrue(all(angle <= step_angle for angle in turns))
                self.assertEqual(session.instructions, [])

    async def test_orient_stops_when_reported_turns_do_not_change_heading(self):
        robot, session = Robot(), Session()
        robot.look_angle_deg = 30
        turns = []
        async def look(direction, *, angle_deg=None):
            turns.append(angle_deg)
            return {'reason': 'turned'}
        robot.look = look
        async with NavigationBackend(robot, session) as backend:
            with self.assertRaisesRegex(ToolError, 'did not converge'):
                await backend.orient(170)
        self.assertGreater(len(turns), 0)
        self.assertLessEqual(len(turns), 10)
        self.assertEqual(session.instructions, [])


class NavigationConfigurationTests(unittest.TestCase):
    def test_deployment_default_and_explicit_limits_reach_backend(self):
        from navigation_agent.app.deployments.factory import create_backend
        from navigation_agent.app.deployments.settings import load_backend, PROJECT_ROOT

        mujoco = load_backend(PROJECT_ROOT / 'configs/mujoco-lightnav.toml')
        backend = create_backend(mujoco)
        self.assertFalse(backend.time_limit_enabled)
        self.assertEqual(backend.inactivity_timeout_s, 15)
        navigation = mujoco.navigation.model_copy(update={
            'time_limit_enabled': True, 'inactivity_timeout_s': 9.0,
        })
        backend = create_backend(mujoco.model_copy(update={'navigation': navigation}))
        self.assertTrue(backend.time_limit_enabled)
        self.assertEqual(backend.inactivity_timeout_s, 9)
        go2 = create_backend(load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml'))
        self.assertFalse(go2.time_limit_enabled)

    def test_omitted_configuration_keeps_limits_and_invalid_values_fail(self):
        from navigation_agent.app.deployments.settings import NavigationSettings

        self.assertTrue(NavigationSettings().time_limit_enabled)
        for value in (0, -1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NavigationSettings(inactivity_timeout_s=value)
        with self.assertRaises(ValueError):
            NavigationSettings(time_limit_enabled='false')


class InactivityTests(unittest.TestCase):
    def test_inactivity_uses_recent_window_and_wraps_heading(self):
        import math
        from navigation_agent.navigation.progress import MotionProgress

        progress = MotionProgress()
        for i in range(31):
            # 先移动，后静止；角度跨越正负 pi 的微小噪声不算真实转向。
            progress.record(MotionSample(i, {
                'x': min(i * 0.1, 1.0), 'y': 0,
                'yaw': math.pi - 0.001 if i % 2 else -math.pi + 0.001,
            }, False))
            if i == 15:
                self.assertFalse(progress.inactive(15))
        self.assertTrue(progress.inactive(15))
        progress.reset()
        self.assertFalse(progress.inactive(15))

    def test_short_observation_and_missing_pose_are_not_evidence_of_inactivity(self):
        from navigation_agent.navigation.progress import MotionProgress

        progress = MotionProgress()
        for i in range(15):
            progress.record(MotionSample(i, {'x': 0, 'y': 0, 'yaw': 0}, False))
        self.assertFalse(progress.inactive(15))
        progress.record(MotionSample(15, {}, False))
        self.assertFalse(progress.inactive(15))


class PathReturnTests(unittest.TestCase):
    def test_forward_progress_detour_and_in_place_turn_do_not_count_as_a_loop(self):
        from navigation_agent.navigation.progress import MotionProgress
        paths = [
            [(0, 0), (2, 0), (4, 0), (6, 0)],
            [(0, 0), (0, 2), (2, 2), (2, 0)],
            [(0, 0)] * 10,
        ]
        for positions in paths:
            progress = MotionProgress()
            for step, (x, y) in enumerate(positions):
                progress.record(MotionSample(step, {'x': x, 'y': y, 'yaw': step}, False))
            self.assertFalse(progress.path_repeated)
        progress.record(MotionSample(12, {'x': 0, 'y': 0, 'yaw': 0}, False))
        self.assertFalse(progress.path_repeated)

    def test_loop_state_resets_and_missing_positions_break_the_evidence(self):
        from navigation_agent.navigation.progress import MotionProgress
        progress = MotionProgress()
        for step, (x, y) in enumerate([(0, 0), (2, 0), (2, 2), (0, 2), (.1, .1)]):
            progress.record(MotionSample(step, {'x': x, 'y': y, 'yaw': 0}, False))
        self.assertTrue(progress.path_repeated)
        progress.reset()
        self.assertFalse(progress.path_repeated)
        self.assertEqual(progress.path_points, [])
        for step, pose in enumerate([{'x': 0, 'y': 0}, {'x': 2, 'y': 2}, {}, {'x': 0, 'y': 0}]):
            progress.record(MotionSample(step, pose, False))
        self.assertFalse(progress.path_repeated)
