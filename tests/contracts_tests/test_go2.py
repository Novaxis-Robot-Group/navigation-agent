"""真实 HTTP/WS 连接配替代本体，验证 GO2 动作和失败收尾。"""

import asyncio
import base64
from contextlib import suppress
import math
import os
import time
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import load_backend, PROJECT_ROOT
from navigation_agent.app.go2.connection import Go2Connection, REQUIRED_CAPABILITIES
from navigation_agent.contracts.agent_backend import ToolError


class Body:
    def __init__(self):
        self.socket = None
        self.owner = self.auto = self.other = False
        self.source = 'disabled'
        self.generation = self.episode = 0
        self.terminal = None
        self.commands = []
        self.angular = 0.
        self.angular_deadband = 0.
        self.motion_blocked = False
        self.yaw = math.pi - .2
        self.stamp = time.time_ns()
        self.sequence = 0
        self.auto_started = 0.
        self.model_stop = True
        self.startup_failures = 0
        self.result_before_error = False
        self.command_before_error = False
        self.mpc_failure = False
        self.odom_updates = True
        self.heartbeats = True
        self.mpc_ack = True
        self.lease_s = 3.
        self.old_observation = False
        self.drop_on_start = False
        self.runtime_every = 1
        self.released = asyncio.Event()
        self.release_pending = False
        self.release_delay = 0.
        self.release_runtime_before_ack = True

    def odom(self):
        return {'stamp_ns': self.stamp, 'frame_id': 'odom', 'pose_valid': True,
                'pose': {'x': 1., 'y': 2., 'z': .3, 'yaw': self.yaw},
                'twist_linear': [0., 0., 0.], 'twist_angular': [0., 0., self.angular]}

    def snapshot(self):
        return {
            'agent': {'protocol_version': 1, 'lease_s': self.lease_s,
                      'capabilities': {name: True for name in REQUIRED_CAPABILITIES},
                      'release_pending': self.release_pending, 'navigation_generation': self.generation, 'terminal': self.terminal},
            'robot': {'connected': True, 'mode': 'WALK'}, 'odom': self.odom(),
            'camera': {'sequence': self.sequence, 'stamp_ns': self.stamp},
            'vln': {'state': 'RUNNING' if self.auto else 'IDLE', 'stop': None,
                    'episode': self.episode, 'server_url': 'ws://127.0.0.1:8052'},
            'mpc': {'error': ''}, 'control_source': self.source,
        }

    async def send(self, value):
        if self.socket and not self.socket.closed:
            await self.socket.send_json(value)

    async def control(self):
        await self.send({'type': 'control_state', 'owner': self.owner,
                         'auto_owner': self.auto, 'locked': self.owner or self.auto or self.other})
        await self.send({'type': 'control_source', 'source': self.source})

    async def ack(self, command, **fields):
        await self.send({'type': 'command_result', 'command': command, 'ok': True, **fields})

    async def tick(self):
        while True:
            await asyncio.sleep(.02)
            self.yaw = math.atan2(math.sin(self.yaw + self.angular * .02), math.cos(self.yaw + self.angular * .02))
            if self.odom_updates:
                self.stamp = time.time_ns()
            self.sequence += 1
            if self.auto and self.generation <= self.startup_failures and time.monotonic() - self.auto_started > .1:
                if self.result_before_error:
                    await self.send({'type': 'vln_status', 'data': {
                        'episode': self.episode, 'last_sequence': 1, 'stop': False}})
                if self.command_before_error:
                    await self.send({'type': 'sent_command', 'data': {'linear': .1, 'angular': 0}})
                if self.mpc_failure:
                    await self.send({'type': 'mpc_status', 'data': {'error': 'controller failed'}})
                else:
                    await self.send({'type': 'vln_status', 'data': {
                        'state': 'ERROR', 'message': 'connection reset'}})
                self.terminal = {'generation': self.generation, 'reason': 'navigation_error'}
                self.auto = False
                self.source = 'disabled'
                await self.control()
                # 同一轮清空实时状态，验证适配层仍保留有效结果和错误。
                await self.send({'type': 'vln_status', 'data': {
                    'state': 'IDLE', 'last_sequence': 0, 'stop': None}})
                await self.send({'type': 'mpc_status', 'data': {'error': ''}})
                await self.send({'type': 'sent_command', 'data': {'linear': 0, 'angular': 0}})
                await self.send({'type': 'runtime', 'data': self.snapshot()})
            if self.auto and self.model_stop and time.monotonic() - self.auto_started > .2:
                await self.send({'type': 'vln_status', 'data': {'stop': True, 'episode': self.episode}})
                self.terminal = {'generation': self.generation, 'reason': 'model_stop'}
                self.auto = False
                self.source = 'disabled'
                self.release_pending = True
                await self.control()
                if self.release_runtime_before_ack:
                    await self.send({'type': 'runtime', 'data': self.snapshot()})
                await self.send({'type': 'vln_status', 'data': {'stop': None, 'state': 'IDLE'}})
                await asyncio.sleep(self.release_delay)
                self.release_pending = False
                await self.ack('agent_release')
            if self.sequence % self.runtime_every == 0:
                await self.send({'type': 'runtime', 'data': self.snapshot()})

    async def websocket(self, request):
        self.socket = web.WebSocketResponse()
        await self.socket.prepare(request)
        await self.send({'type': 'snapshot', 'data': self.snapshot()})
        await self.control()
        task = asyncio.create_task(self.tick())
        try:
            async for message in self.socket:
                payload = message.json()
                kind = payload['type']
                self.commands.append(payload)
                if kind == 'agent_heartbeat':
                    if self.heartbeats:
                        await self.send({'type': 'agent_heartbeat', 'lease_s': self.lease_s})
                elif kind == 'acquire_control':
                    if self.release_pending:
                        await self.send({'type': 'command_result', 'ok': False,
                                         'message': 'waiting for robot control release'})
                        continue
                    if self.other:
                        await self.send({'type': 'command_result', 'ok': False, 'message': 'busy'})
                        continue
                    self.owner, self.source = True, 'manual'
                    await self.control()
                    await self.ack('manual_control', enabled=True)
                elif kind == 'agent_release':
                    if self.release_pending:
                        continue
                    self.owner = self.auto = False
                    self.angular, self.source = 0., 'disabled'
                    await self.control()
                    await self.ack('agent_release')
                    self.released.set()
                elif kind == 'set_server_url':
                    await self.ack('server_url')
                elif kind == 'set_vln':
                    assert payload['mode'] == 'objnav'
                    assert not self.owner
                    self.auto, self.source = True, 'auto'
                    self.generation += 1
                    self.episode += 1
                    self.terminal = None
                    self.auto_started = time.monotonic()
                    await self.control()
                    await self.ack('vln', enabled=True)
                    # 重复与无关回执不能代替 MPC 启用回执。
                    await self.ack('vln', enabled=True)
                    await self.ack('mpc', enabled=False)
                    if self.drop_on_start:
                        await self.socket.close()
                    elif self.mpc_ack:
                        await self.ack('mpc', enabled=True)
                elif kind == 'twist':
                    assert self.owner and payload['x'] == payload['y'] == 0
                    self.angular = (payload['z'] if abs(payload['z']) >= self.angular_deadband
                                    and not self.motion_blocked else 0.)
        finally:
            self.owner = self.auto = False
            self.angular, self.source = 0., 'disabled'
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return self.socket

    async def observation(self, request):
        after = int(request.query.get('after_sequence', -1))
        if self.sequence <= after:
            return web.Response(status=409)
        return web.json_response({
            'protocol_version': 1, 'sequence': self.sequence,
            'camera_stamp_ns': self.stamp, 'pose': self.odom()['pose'], 'pose_valid': True,
            'camera_age_ms': 2000 if self.old_observation else 0,
            'odom_frame_id': 'odom', 'odom_match_ms': 0, 'odom': self.odom(),
            'jpeg_base64': base64.b64encode(b'\xff\xd8test').decode(),
        })

    def app(self):
        app = web.Application()
        app.router.add_get('/agent/ws', self.websocket)
        app.router.add_get('/api/observation', self.observation)
        return app


class Go2Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.body = Body()
        self.server = TestServer(self.body.app())
        await self.server.start_server()
        self.env = patch.dict(os.environ, {'VLN_AGENT_CONFIG': 'configs/go2-lightnav.toml'}, clear=True)
        self.env.start()

    async def asyncTearDown(self):
        await self.server.close()
        self.env.stop()

    def backend(self):
        return create_backend(load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml'), base_url=str(self.server.make_url('/')))

    async def test_observation_navigation_stop_and_recovery(self):
        async with self.backend() as backend:
            image, observation = await backend.observe()
            self.assertTrue(image.startswith(b'\xff\xd8'))
            self.assertTrue(observation['pose_valid'])
            result = await backend.navigate('Go to the chair.', seconds=2)
            self.assertEqual(result['reason'], 'model_stop')
            self.assertFalse(result['arrival_verified'])
            self.assertTrue(self.body.owner)
            self.assertEqual((await backend.recover())['reason'], 'recovery_unavailable')
            result = await backend.navigate('Go to the table.', seconds=2)
            self.assertEqual(result['reason'], 'model_stop')
        self.assertEqual(self.body.source, 'disabled')
        self.assertFalse(self.body.auto)

    async def test_consecutive_navigation_waits_for_pending_sdk_release(self):
        self.body.release_delay = .15
        async with self.backend() as backend:
            for instruction in ('Go to the sofa.', 'Go to the table.'):
                result = await backend.navigate(instruction, seconds=2)
                self.assertEqual(result['reason'], 'model_stop')
                self.assertFalse(self.body.release_pending)
                self.assertTrue(self.body.owner)
                self.assertIsNone(backend.robot.channel.error)

    async def test_model_stop_waits_for_release_ack_with_stale_runtime_snapshot(self):
        # 控制权增量先到、完整快照后到时，不能把旧的 pending=false 当作释放成功。
        self.body.release_runtime_before_ack = False
        self.body.release_delay = .15
        async with self.backend() as backend:
            for instruction in ('Go to the sofa.', 'Go to the table.'):
                result = await backend.navigate(instruction, seconds=2)
                self.assertEqual(result['reason'], 'model_stop')
                self.assertFalse(result['arrival_verified'])
                self.assertFalse(self.body.release_pending)
                self.assertTrue(self.body.owner)
                self.assertIsNone(backend.robot.channel.error)
                self.assertTrue((await backend.observe())[1]['pose_valid'])

    async def test_failed_model_stop_release_is_not_reacquired(self):
        self.body.release_runtime_before_ack = False
        self.body.release_delay = .15
        acknowledge = self.body.ack

        async def fail_release(command, **fields):
            if command == 'agent_release' and self.body.terminal:
                self.body.release_pending = True
                await self.body.send({'type': 'command_result', 'command': command,
                                      'ok': False, 'message': 'SDK release failed'})
            else:
                await acknowledge(command, **fields)

        self.body.ack = fail_release
        async with self.backend() as backend:
            start = len(self.body.commands)
            with self.assertRaisesRegex(ToolError, 'SDK release failed'):
                await backend.navigate('Go to the sofa.', seconds=2)
            self.assertFalse(any(command['type'] == 'acquire_control'
                                 for command in self.body.commands[start:]))

    async def test_missing_model_stop_release_ack_is_not_reacquired(self):
        self.body.release_runtime_before_ack = False
        self.body.release_delay = .15
        acknowledge = self.body.ack

        async def omit_release(command, **fields):
            if command != 'agent_release' or not self.body.terminal:
                await acknowledge(command, **fields)

        self.body.ack = omit_release
        async with self.backend() as backend:
            backend.robot.channel.command_timeout = .3
            start = len(self.body.commands)
            with self.assertRaisesRegex(ToolError, 'Timed out waiting for GO2 state'):
                await backend.navigate('Go to the sofa.', seconds=2)
            self.assertFalse(any(command['type'] == 'acquire_control'
                                 for command in self.body.commands[start:]))
            self.assertIsNotNone(backend.robot.channel.error)

    async def test_live_observation_during_navigation_never_changes_control(self):
        self.body.model_stop = False
        async with self.backend() as backend:
            action = asyncio.create_task(backend.navigate('Go forward.', seconds=3))
            try:
                await backend.robot.channel.wait_for(lambda: self.body.auto)
                before = len(self.body.commands)
                jpeg, observation = await backend.snapshot_observation()
                self.assertTrue(jpeg.startswith(b'\xff\xd8'))
                self.assertTrue(observation['pose_valid'])
                self.assertEqual(observation['pose']['x'], 1.)
                self.assertTrue(self.body.auto)
                self.assertFalse(action.done())
                commands = self.body.commands[before:]
                self.assertTrue(all(item['type'] == 'agent_heartbeat' for item in commands))
                self.body.old_observation = True
                with self.assertRaisesRegex(ToolError, 'metadata'):
                    await backend.snapshot_observation()
            finally:
                self.body.old_observation = False
                action.cancel()
                await asyncio.gather(action, return_exceptions=True)

    async def test_hardware_errors_reach_agent_status_and_control_checks(self):
        async with self.backend() as backend:
            for diagnostic in ({'sdk_error': 'Move failed'}, {'safety_error': 'avoidance failed'},
                               {'level': 2, 'message': 'SDK unavailable'}):
                backend.robot.channel.state['robot'] = {'connected': True, 'mode': 'WALK', **diagnostic}
                self.assertTrue(backend.robot.status().error)
                with self.assertRaises(ToolError):
                    backend.require_control()
            backend.robot.channel.state['robot'] = {'connected': True, 'mode': 'WALK'}

    def test_transient_hardware_error_is_latched_for_the_navigation_attempt(self):
        from types import SimpleNamespace
        from navigation_agent.robots.go2.robot import Go2NavigationSession
        channel = Go2Connection('http://127.0.0.1:8088')
        channel.state = {'agent': {'navigation_generation': 0}}
        session = Go2NavigationSession(SimpleNamespace(channel=channel), '')
        channel.begin_navigation()
        channel._accept({'type': 'robot_diagnostics', 'data': {'sdk_error': 'Move failed'}})
        channel._accept({'type': 'robot_diagnostics', 'data': {'sdk_error': '', 'level': 0}})
        channel._accept({'type': 'vln_status', 'data': {'state': 'ERROR'}})
        self.assertEqual(channel.robot_error(), '')
        self.assertEqual(session.status().error, 'Move failed')
        self.assertFalse(session.status().startup_retryable)
        channel.begin_navigation()
        self.assertEqual(channel.navigation_robot_error, '')

    async def test_old_client_is_rejected_before_acquiring_control(self):
        snapshot = self.body.snapshot
        def old_snapshot():
            value = snapshot()
            del value['agent']['capabilities']
            return value
        self.body.snapshot = old_snapshot
        with self.assertRaisesRegex(ToolError, '需要更新'):
            async with self.backend():
                self.fail('old client must not be accepted')
        self.assertFalse(self.body.commands)

    async def test_real_adapter_target_watch_stops_and_can_navigate_again(self):
        from types import SimpleNamespace
        from navigation_agent.agent.tools import NavigationTools
        from navigation_agent.agent.tools.target_watch import TargetWatch
        from navigation_agent.contracts.detection import Capabilities, Detection, FrameResponse
        from tests.config_support import agent_settings

        frames, events = [], []
        async def capabilities():
            return Capabilities(ready=True, model='fake', vocabulary='open')
        async def open_session(labels):
            self.assertEqual(labels, ['sofa'])
            return 'test-session'
        async def detect(session, jpeg, stamp):
            frames.append((jpeg, stamp))
            return FrameResponse(request_id='test', camera_stamp_ns=stamp, width=480, height=270,
                                 inference_ms=1., detections=[Detection(label='sofa', confidence=.9,
                                 bbox=(.1, .1, .9, .9), continuity_id='sofa-1')])
        async def close_session(session):
            pass
        client = SimpleNamespace(capabilities=capabilities, open=open_session,
                                 detect=detect, close=close_session)
        settings = agent_settings().target_watch.model_copy(update={
            'server_url': 'http://fake', 'frame_interval_s': .02})
        self.body.model_stop = False
        async with self.backend() as backend:
            tools = NavigationTools(backend)
            watch = TargetWatch(settings, client, tools,
                                event=lambda kind, **data: events.append((kind, data)),
                                capture=lambda *args: {'image': 'images/001.jpg', 'step': 1})
            try:
                await watch.configure(True, ['sofa'])
                result = await watch.execute(lambda: tools.navigate('Go forward.', seconds=2),
                                             1, search_navigation=True)
                self.assertEqual(result['detail'], 'target_detected')
                self.assertTrue(frames)
                self.assertTrue(any(kind == 'target_detection_hit' for kind, _ in events))
                self.assertTrue(self.body.owner)
                self.assertFalse(self.body.auto)
                self.assertTrue((await backend.observe())[1]['pose_valid'])
                await watch.configure(False, [])
                self.body.model_stop = True
                self.assertEqual((await backend.navigate('Go to the table.', seconds=2))['reason'], 'model_stop')
            finally:
                await watch.pause()

    async def test_real_adapter_enforces_explicit_displacement_budget(self):
        from navigation_agent.agent.tools import NavigationTools
        self.body.model_stop = False
        async with self.backend() as backend:
            result = await NavigationTools(backend).navigate(
                'Move forward.', seconds=2, max_displacement_m=.5, origin=[0., 2.])
            self.assertEqual(result['detail'], 'local_observation_distance')
            self.assertTrue(self.body.owner)
            self.assertFalse(self.body.auto)

    async def test_turn_crosses_pi_and_stops(self):
        start = self.body.yaw
        async with self.backend() as backend:
            self.assertEqual(backend.look_angle_deg, 70)
            result = await backend.look('left')
            self.assertEqual(result['reason'], 'turned')
            turned = math.atan2(math.sin(self.body.yaw-start), math.cos(self.body.yaw-start))
            self.assertAlmostEqual(math.degrees(turned), 70, delta=10)
            self.assertEqual(self.body.angular, 0)

    async def test_turn_finishes_with_body_low_speed_deadband(self):
        # 原比例控制会在剩余约 7° 时停滞，必须继续依据反馈完成转向。
        self.body.angular_deadband = .18
        for direction, sign in (('right', -1), ('left', 1)):
            with self.subTest(direction=direction):
                start = self.body.yaw
                async with self.backend() as backend:
                    result = await backend.look(direction)
                    self.assertEqual(result['reason'], 'turned')
                    turned = math.atan2(math.sin(self.body.yaw-start), math.cos(self.body.yaw-start))
                    self.assertAlmostEqual(math.degrees(turned), sign*70, delta=10)
                    self.assertEqual(self.body.angular, 0)
                self.assertEqual(self.body.source, 'disabled')
        speeds = [abs(c['z']) for c in self.body.commands if c['type'] == 'twist']
        self.assertLessEqual(max(speeds), .4)

    async def test_turn_stall_fails_and_releases_instead_of_claiming_success(self):
        self.body.motion_blocked = True
        with self.assertRaisesRegex(ToolError, 'turn stalled: turned 0.0 deg'):
            async with self.backend() as backend:
                await backend.look('right')
        self.assertEqual(self.body.angular, 0)
        self.assertEqual(self.body.source, 'disabled')
        self.assertFalse(self.body.owner)

    async def test_turn_accepts_ten_degree_tolerance_but_rejects_outside(self):
        self.body.motion_blocked = True
        for angle, succeeds in ((55.8, True), (69.9, True), (49.0, False), (71.0, False)):
            with self.subTest(angle=angle):
                async with self.backend() as backend:
                    backend.robot.look_angle_deg = 60
                    start = self.body.yaw
                    command_start = len(self.body.commands)
                    task = asyncio.create_task(backend.look('right'))
                    try:
                        async with asyncio.timeout(2):
                            while not any(c['type'] == 'twist' and c['z'] != 0
                                          for c in self.body.commands[command_start:]):
                                await asyncio.sleep(.02)
                        self.body.yaw = start - math.radians(angle)
                        if succeeds:
                            self.assertEqual((await task)['reason'], 'turned')
                        else:
                            with self.assertRaisesRegex(ToolError, 'turn stalled'):
                                await task
                        self.assertEqual(self.body.angular, 0)
                    finally:
                        if not task.done():
                            task.cancel()
                            with suppress(asyncio.CancelledError):
                                await task

    async def test_cancel_navigation_releases_owned_control(self):
        self.body.model_stop = False
        async def run():
            async with self.backend() as backend:
                await backend.navigate('Go forward.', seconds=10)
        task = asyncio.create_task(run())
        async with asyncio.timeout(3):
            while not self.body.auto:
                await asyncio.sleep(.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.body.auto)
        self.assertEqual(self.body.source, 'disabled')
        self.assertTrue(self.body.released.is_set())

    async def test_cancel_turn_stops(self):
        async def run():
            async with self.backend() as backend:
                await backend.look('right')
        task = asyncio.create_task(run())
        async with asyncio.timeout(3):
            while not self.body.angular:
                await asyncio.sleep(.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.body.angular, 0)

    async def test_stale_odom_and_old_observation_rejected(self):
        async with self.backend() as backend:
            self.body.old_observation = True
            with self.assertRaisesRegex(ToolError, 'metadata'):
                await backend.observe()
            self.body.odom_updates = False
            await asyncio.sleep(.9)
            with self.assertRaisesRegex(ToolError, 'odometry'):
                backend.require_control()

    async def test_manual_takeover_is_not_reacquired_or_stopped(self):
        async with self.backend() as backend:
            self.body.owner = False
            self.body.other = True
            await self.body.control()
            await asyncio.sleep(.05)
            with self.assertRaises(ToolError):
                await backend.hold()
            count = len(self.body.commands)
        self.assertFalse(any(c['type'] in ('agent_release', 'acquire_control', 'stop') for c in self.body.commands[count:]))

    async def test_partial_start_ack_never_completes_and_connection_is_discarded(self):
        self.body.model_stop = False
        self.body.mpc_ack = False
        backend = self.backend()
        backend.robot.channel.command_timeout = .3
        async with backend:
            with self.assertRaises(ToolError):
                await backend.navigate('Go forward.', seconds=2)
            self.assertIsNotNone(backend.robot.channel.error)
        self.assertFalse(self.body.auto)

    async def test_disconnected_start_does_not_leave_navigation_running(self):
        self.body.drop_on_start = True
        async with self.backend() as backend:
            with self.assertRaises(ToolError):
                await backend.navigate('Go forward.', seconds=2)
        self.assertFalse(self.body.auto)

    async def test_missing_heartbeat_ack_closes_connection(self):
        self.body.lease_s = .6
        self.body.heartbeats = False
        channel = Go2Connection(str(self.server.make_url('/')))
        async with channel:
            await asyncio.sleep(.8)
            self.assertIsNotNone(channel.error)
        self.assertFalse(self.body.auto)

    async def test_other_owner_prevents_any_control_command(self):
        self.body.other = True
        with self.assertRaises(ToolError):
            async with self.backend():
                pass
        self.assertFalse(any(c['type'] != 'agent_heartbeat' for c in self.body.commands))

    def test_only_current_generation_terminal_is_accepted(self):
        channel = Go2Connection('http://127.0.0.1:8088')
        channel.state = {'agent': {'navigation_generation': 4}}
        channel.begin_navigation()
        channel._accept({'type': 'runtime', 'data': {'agent': {
            'navigation_generation': 4, 'terminal': {'generation': 4, 'reason': 'model_stop'}}}})
        self.assertIsNone(channel.navigation_result)
        channel._accept({'type': 'runtime', 'data': {'agent': {
            'navigation_generation': 5, 'terminal': {'generation': 5, 'reason': 'model_stop'}}}})
        self.assertEqual(channel.navigation_result['reason'], 'model_stop')

    async def test_fast_model_stop_before_full_snapshot_does_not_reuse_generation(self):
        self.body.runtime_every = 25
        async with self.backend() as backend:
            for _ in range(2):
                result = await backend.navigate('Go to the chair.', seconds=2)
                self.assertEqual(result['reason'], 'model_stop')
            self.assertEqual(backend.robot.channel.navigation_generation, 2)

    async def test_navigation_budget_stops_before_return(self):
        self.body.model_stop = False
        async with self.backend() as backend:
            backend.time_limit_enabled = True
            result = await backend.navigate('Go to the chair.', seconds=.15)
            self.assertEqual(result['reason'], 'timeout')
            self.assertFalse(self.body.auto)
            self.assertTrue(self.body.owner)

    async def test_heartbeat_continues_during_cloud_idle(self):
        async with self.backend():
            await asyncio.sleep(1.1)
            self.assertTrue(self.body.owner)
            self.assertGreaterEqual(sum(c['type'] == 'agent_heartbeat' for c in self.body.commands), 2)

    def test_same_stamp_snapshots_do_not_refresh_odometry_age(self):
        channel = Go2Connection('http://127.0.0.1:8088')
        channel._accept({'type': 'odom', 'data': self.body.odom()})
        channel.odom_received -= 10
        before = channel.odom_received
        channel._accept({'type': 'runtime', 'data': self.body.snapshot()})
        self.assertEqual(channel.odom_received, before)

    async def test_startup_service_error_retries_then_succeeds(self):
        self.body.startup_failures = 2
        async with self.backend() as backend:
            self.assertEqual(backend.max_attempts, 3)
            result = await backend.navigate('Go to the chair.', seconds=8)
        self.assertEqual(result['reason'], 'model_stop')
        self.assertNotIn('startup_failed', result)
        self.assertEqual(self.body.generation, 3)
        self.assertEqual([a['reason'] for a in result['attempts']],
                         ['navigation_error', 'navigation_error', 'model_stop'])
        self.assertEqual(result['attempts'][0]['detail'], 'connection reset')
        self.assertEqual(self.body.source, 'disabled')

    async def test_startup_service_error_stops_at_max_attempts(self):
        self.body.startup_failures = 10
        async with self.backend() as backend:
            result = await backend.navigate('Go to the chair.', seconds=8)
            self.assertTrue(self.body.owner)
        self.assertEqual(self.body.generation, 3)
        self.assertTrue(result['startup_failed'])
        self.assertEqual(len(result['attempts']), 3)

    async def test_result_motion_or_mpc_error_prevents_retry(self):
        for field in ('result_before_error', 'command_before_error', 'mpc_failure'):
            with self.subTest(field=field):
                self.body.generation = self.body.episode = 0
                self.body.terminal = None
                self.body.startup_failures = 10
                setattr(self.body, field, True)
                async with self.backend() as backend:
                    result = await backend.navigate('Go to the chair.', seconds=5)
                self.assertEqual(self.body.generation, 1)
                self.assertEqual(result['reason'], 'navigation_error')
                self.assertFalse(result.get('startup_failed', False))
                setattr(self.body, field, False)

    async def test_startup_retry_shares_navigation_deadline(self):
        self.body.startup_failures = 10
        async with self.backend() as backend:
            backend.time_limit_enabled = True
            result = await backend.navigate('Go to the chair.', seconds=.6)
        self.assertEqual(self.body.generation, 1)
        self.assertTrue(result['startup_failed'])

    async def test_cancel_during_retry_delay_does_not_start_again(self):
        self.body.startup_failures = 10
        async def run():
            async with self.backend() as backend:
                await backend.navigate('Go to the chair.', seconds=10)
        task = asyncio.create_task(run())
        try:
            async with asyncio.timeout(3):
                while not (self.body.generation == 1 and self.body.owner):
                    await asyncio.sleep(.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.body.generation, 1)
            self.assertEqual(self.body.source, 'disabled')
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    def test_incremental_result_and_error_survive_idle_until_next_attempt(self):
        from navigation_agent.robots.go2.robot import Go2NavigationSession
        from types import SimpleNamespace
        channel = Go2Connection('http://127.0.0.1:8088')
        channel.state = {'agent': {'navigation_generation': 0}}
        session = Go2NavigationSession(SimpleNamespace(channel=channel), '')
        channel.begin_navigation()
        channel._accept({'type': 'vln_status', 'data': {
            'episode': 1, 'last_sequence': 1, 'stop': True}})
        channel._accept({'type': 'vln_status', 'data': {'state': 'ERROR'}})
        channel._accept({'type': 'vln_status', 'data': {
            'state': 'IDLE', 'last_sequence': 0, 'stop': None}})
        self.assertTrue(channel.navigation_received_result)
        self.assertFalse(session.status().startup_retryable)
        channel.begin_navigation()
        self.assertEqual(channel.navigation_generation, 2)
        channel._accept({'type': 'vln_status', 'data': {'state': 'ERROR'}})
        self.assertTrue(session.status().startup_retryable)
        channel._accept({'type': 'runtime', 'data': {'agent': {
            'navigation_generation': 2,
            'terminal': {'generation': 2, 'reason': 'lease_expired'}}}})
        self.assertFalse(session.status().startup_retryable)

    async def test_orient_turns_to_historical_heading_without_starting_vln(self):
        target = (math.degrees(self.body.yaw) + 130 + 180) % 360 - 180
        async with self.backend() as backend:
            result = await backend.orient(target)
            actual = math.degrees(self.body.yaw)
            self.assertLessEqual(abs((actual - target + 180) % 360 - 180), 10)
            self.assertEqual(result['reason'], 'oriented')
            self.assertEqual(self.body.angular, 0)
        self.assertFalse(any(command['type'] == 'set_vln' for command in self.body.commands))


class HistoryGenerationTests(unittest.TestCase):
    def test_same_episode_stop_and_old_ack_are_isolated(self):
        channel = Go2Connection('http://127.0.0.1:8088')
        channel.state = {'agent': {'navigation_generation': 1}}
        channel.last_episode = 4
        channel.begin_navigation()
        channel._accept({'type': 'vln_status', 'data': {
            'episode': 4, 'generation': 1, 'last_sequence': 2, 'stop': True}})
        self.assertIsNone(channel.navigation_result)
        channel._accept({'type': 'vln_status', 'data': {
            'episode': 4, 'generation': 2, 'last_sequence': 3, 'stop': True}})
        self.assertEqual(channel.navigation_result['reason'], 'model_stop')
        received = set()
        channel.pending = {'vln_history'}, received, None
        channel.pending_history = (2, 'resume')
        channel._accept({'type': 'command_result', 'command': 'vln_history',
                         'generation': 1, 'operation': 'resume', 'ok': True})
        self.assertEqual(received, set())
        channel._accept({'type': 'command_result', 'command': 'vln_history',
                         'generation': 2, 'operation': 'pause', 'ok': True})
        self.assertEqual(received, set())
        channel._accept({'type': 'command_result', 'command': 'vln_history',
                         'generation': 2, 'operation': 'resume', 'ok': True})
        self.assertEqual(received, {'vln_history'})
