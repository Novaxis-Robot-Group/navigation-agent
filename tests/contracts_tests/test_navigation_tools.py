from tests.config_support import agent_settings
import asyncio
import base64
import time
from contextlib import suppress
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestServer

from navigation_agent.agent.tools import ToolError
from tests.backend_support import backend_class

SimTools = backend_class()


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = 'idle'
        self.stop = False
        self.fail = False
        self.busy = False
        self.commands = []
        self.navigation_result = 'model_stop'
        self.closed = asyncio.Event()
        app = web.Application()

        async def websocket(request):
            socket = web.WebSocketResponse(heartbeat=1)
            await socket.prepare(request)
            counter = 0

            async def snapshot():
                nonlocal counter
                counter += 1
                await socket.send_json(
                    {
                        'type': 'runtime',
                        'data': {
                            'control': {
                                'manual': self.mode == 'manual',
                                'auto': self.mode == 'auto',
                                'locked': self.busy,
                            },
                            'simulation': {
                                'camera': {'ready': True, 'stamp_ns': counter},
                                'pose': {'yaw': 0},
                                'backend': {'blocked': self.navigation_result == 'blocked'},
                            },
                            'vln': {'stop': self.stop, 'error': 'error' if self.fail else ''},
                            'mpc': {'error': ''},
                        },
                    }
                )

            async def updates():
                while True:
                    await asyncio.sleep(0.05)
                    await snapshot()

            await snapshot()
            task = asyncio.create_task(updates())
            try:
                async for message in socket:
                    data = message.json()
                    self.commands.append(data)
                    kind = data['type']
                    if self.busy:
                        await socket.send_json({'type': 'command_result', 'ok': False})
                        continue
                    if kind == 'acquire_control':
                        self.mode, self.stop, self.fail = 'manual', False, False
                    elif kind == 'set_vln':
                        self.mode = 'auto'
                    if kind != 'twist':
                        await socket.send_json({'type': 'command_result', 'ok': True})
                    await snapshot()
                    if kind == 'set_vln':
                        await asyncio.sleep(0.1)
                        if self.navigation_result == 'model_stop':
                            self.stop, self.mode = True, 'idle'
                        elif self.navigation_result == 'error':
                            self.fail = True
                        await snapshot()
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                if not self.busy:
                    self.mode = 'idle'
                self.closed.set()
            return socket

        async def camera(request):
            return web.Response(body=b'\xff\xd8camera', content_type='image/jpeg')

        async def observation(request):
            return web.json_response({
                'jpeg_base64': base64.b64encode(b'\xff\xd8camera').decode(),
                'pose': {'yaw': 0}, 'camera_stamp_ns': time.time_ns(),
            })

        app.router.add_get('/api/observation', observation)
        app.router.add_get('/ws', websocket)
        app.router.add_get('/api/camera.jpg', camera)
        self.server = TestServer(app)
        await self.server.start_server()

    async def asyncTearDown(self):
        await self.server.close()

    def client(self):
        return SimTools(str(self.server.make_url('/')), 'ws://127.0.0.1:8050')

    async def test_navigation_completion_and_socket_cleanup(self):
        async with self.client() as tools:
            samples = []
            tools.telemetry = samples.append
            image, _ = await tools.observe()
            self.assertTrue(image.startswith(b'\xff\xd8'))
            result = await tools.navigate('穿过门口', seconds=1)
            self.assertEqual(result['reason'], 'model_stop')
            self.assertEqual(self.mode, 'manual')
            self.assertGreater(len(samples), 0)
            self.assertTrue(any(s['vln']['stop'] for s in samples))
            self.assertIsNone(result['motion']['displacement_m'])
            self.assertFalse(result['arrival_verified'])
        await asyncio.wait_for(self.closed.wait(), 1)
        self.assertEqual(self.mode, 'idle')
        self.assertNotIn('stop', [item['type'] for item in self.commands])

    async def test_passive_frame_does_not_send_commands(self):
        async with self.client() as tools:
            before = list(self.commands)
            mode = self.mode
            image, observation = await tools.snapshot_observation()
            self.assertTrue(image.startswith(b'\xff\xd8'))
            self.assertGreater(observation['camera_stamp_ns'], 0)
            self.assertEqual(self.commands, before)
            self.assertEqual(self.mode, mode)

    async def test_only_current_navigation_can_be_interrupted(self):
        self.navigation_result = 'running'
        async with self.client() as tools:
            task = asyncio.create_task(tools.navigate('forward', seconds=1))
            while tools.active_navigation_id is None:
                await asyncio.sleep(.01)
            identity = tools.active_navigation_id
            self.assertFalse(tools.request_navigation_interrupt(identity + 1, 'stale'))
            self.assertTrue(tools.request_navigation_interrupt(identity, 'target_seen'))
            result = await task
            self.assertEqual(result['reason'], 'agent_interrupt')
            self.assertEqual(self.mode, 'manual')
            self.assertFalse(tools.request_navigation_interrupt(identity, 'late'))

    async def test_timeout_stops_owned_navigation(self):
        self.navigation_result = 'running'
        async with self.client() as tools:
            tools.time_limit_enabled = True
            result = await tools.navigate('向前走', seconds=0.2)
            self.assertEqual(result['reason'], 'timeout')
            self.assertEqual(self.mode, 'manual')

    async def test_continuous_navigation_waits_for_stop_beyond_requested_seconds(self):
        # 使用实际 MuJoCo 装配和控制连接；模拟服务稍后发出 STOP。
        async with self.client() as tools:
            self.assertFalse(tools.time_limit_enabled)
            async with asyncio.timeout(2):
                result = await tools.navigate('Go forward.', seconds=0.03)
            self.assertEqual(result['reason'], 'model_stop')
            self.assertIsNone(result['time_limit_s'])
            self.assertFalse(result['arrival_verified'])
            self.assertEqual(self.mode, 'manual')

    async def test_model_error_stops_navigation(self):
        self.navigation_result = 'error'
        async with self.client() as tools:
            self.assertEqual(
                (await tools.navigate('向前走', seconds=1))['reason'], 'navigation_error'
            )
            self.assertEqual(self.mode, 'manual')

    async def test_blocked_navigation_stops(self):
        self.navigation_result = 'blocked'
        async with self.client() as tools:
            self.assertEqual((await tools.navigate('向前走', seconds=4))['reason'], 'blocked')

    async def test_busy_robot_is_not_stopped(self):
        self.busy = True
        with self.assertRaises(ToolError):
            async with self.client():
                self.fail('Should not acquire another client control')
        self.assertEqual([item['type'] for item in self.commands], ['acquire_control'])

    async def test_cloud_wait_keeps_connection_live_and_detects_manual_stop(self):
        async with self.client() as tools:
            await asyncio.sleep(3.5)
            await tools.observe()
            self.mode = 'idle'
            await asyncio.sleep(0.1)
            with self.assertRaises(ToolError):
                await tools.look('left')

    async def test_cancel_during_navigation_releases_control(self):
        self.navigation_result = 'running'

        async def navigate():
            async with self.client() as tools:
                await tools.navigate('向前走', seconds=agent_settings().execution.navigation_seconds)

        task = asyncio.create_task(navigate())
        for _ in range(100):
            if self.mode == 'auto':
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.closed.wait(), 1)
        self.assertEqual(self.mode, 'idle')


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        client = SimTools('http://unused', 'ws://unused').robot
        client.state = {
            'control': {'manual': True},
            'simulation': {
                'pose': {'x': 0.6, 'y': 0, 'yaw': 0},
                'backend': {'collision_guard': True, 'blocked': False},
            },
        }
        client.breadcrumbs.extend((i * 0.02, 0) for i in range(31))
        client.hold = AsyncMock()
        client.channel.send = AsyncMock()
        return client

    async def test_missing_guard_or_discontinuous_path_does_not_move(self):
        for condition in ('guard', 'gap', 'empty'):
            client = self.client()
            if condition == 'guard':
                client.state['simulation']['backend']['collision_guard'] = False
            elif condition == 'gap':
                client.state['simulation']['pose']['x'] = 2
            else:
                client.breadcrumbs.clear()
            self.assertEqual((await client.recover())['reason'], 'recovery_unavailable')
            client.channel.send.assert_not_awaited()

    async def test_new_collision_stops_recovery(self):
        client = self.client()

        async def snapshot():
            client.state['simulation']['backend']['blocked'] = True

        client._snapshot = snapshot
        self.assertEqual((await client.recover())['reason'], 'recovery_blocked')
        self.assertEqual(client.channel.send.await_count, 1)
        self.assertEqual(client.hold.await_count, 2)
        self.assertIsNone(client.motion_action)

    async def test_cancellation_stops_recovery(self):
        client = self.client()
        client._snapshot = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await client.recover()
        self.assertEqual(client.hold.await_count, 2)
        self.assertIsNone(client.motion_action)

    def test_intermittent_collision_is_stalled_but_translation_is_progress(self):
        from navigation_agent.contracts.robot import MotionSample
        from navigation_agent.navigation.progress import MotionProgress
        progress = MotionProgress()
        progress.samples = [
            MotionSample(i * 0.1, {'x': 0, 'y': 0, 'yaw': i * 0.2}, i % 3 == 0)
            for i in range(32)
        ]
        self.assertTrue(progress.stalled())
        progress.samples[-1].pose['x'] = 0.2
        self.assertFalse(progress.stalled())


if __name__ == '__main__':
    unittest.main()


class ProgressSettingsTests(unittest.TestCase):
    def test_path_samples_include_short_final_leg_without_changing_motion_checks(self):
        from navigation_agent.contracts.robot import MotionSample
        from navigation_agent.navigation.progress import MotionProgress
        progress = MotionProgress()
        for step, x in enumerate([0., .05, .1, .22, .3]):
            progress.record(MotionSample(step, {'x': x, 'y': 0., 'yaw': 0.}, False))
        expected = [[0., 0.], [.22, 0.], [.3, 0.]]
        self.assertEqual(progress.summary()['sampled_positions'], expected)
        self.assertEqual(progress.summary()['sampled_positions'], expected)
        self.assertEqual([point['position'] for point in progress.path_points], [(0., 0.), (.22, 0.)])
        self.assertFalse(progress.path_repeated)
        progress.record(MotionSample(6, {'yaw': 0.}, False))
        self.assertNotIn('sampled_positions', progress.summary())
        progress.record(MotionSample(7, {'x': 2., 'y': 0., 'yaw': 0.}, False))
        self.assertEqual(progress.summary()['sampled_positions'], [[2., 0.]])
        progress.reset()
        self.assertNotIn('sampled_positions', progress.summary())

    def test_stall_window_changes_stop_decision(self):
        from navigation_agent.navigation.progress import MotionProgress
        from navigation_agent.contracts.robot import MotionSample
        from navigation_agent.contracts.runtime import ProtectionSettings
        progress=MotionProgress()
        progress.samples=[MotionSample(i*.1,{'x':0,'y':0,'yaw':0},True) for i in range(22)]
        self.assertFalse(progress.stalled())
        progress.settings=ProtectionSettings(stall_window_s=2.)
        self.assertTrue(progress.stalled())


class NavigationHeadingTests(unittest.TestCase):
    def make_robot(self):
        from types import SimpleNamespace
        from navigation_agent.robots.mujoco.robot import MujocoRobot
        channel = SimpleNamespace(state={
            'control': {'auto': True},
            'simulation': {'pose': {'x': 0., 'y': 0., 'yaw': 0.},
                           'velocity': {'linear': .2}},
            'mpc': {'reference_world': [[0., 0.], [0., 0.], [0., 1.]]},
        })
        robot = MujocoRobot(channel)
        robot.begin_motion('navigate')
        return robot

    def test_heading_uses_current_route_and_pose(self):
        import math
        robot = self.make_robot()
        self.assertAlmostEqual(robot._navigation_heading()['heading_error_deg'], 90.)
        robot.state['simulation']['pose']['yaw'] = math.pi / 2
        self.assertAlmostEqual(robot._navigation_heading()['heading_error_deg'], 0.)

    def test_missing_route_and_control_loss_disable_feedback(self):
        robot = self.make_robot()
        robot.state['control']['auto'] = False
        self.assertIsNone(robot._navigation_heading())
        robot.state['control']['auto'] = True
        robot.state['mpc']['reference_world'] = []
        self.assertIsNone(robot._navigation_heading())

    def test_motion_end_clears_previous_feedback(self):
        robot = self.make_robot()
        robot.navigation_heading = robot._navigation_heading()
        robot.end_motion()
        self.assertIsNone(robot.navigation_heading)
        self.assertIsNone(robot._navigation_heading())
