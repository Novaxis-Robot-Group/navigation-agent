"""只读连接检查不发送模型输入或机器人命令。"""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import base64
import time
from pathlib import Path
import tempfile
import unittest
from aiohttp import web
from aiohttp.test_utils import TestServer

from navigation_agent.app.deployments.check import check
from navigation_agent.app.deployments.settings import load_settings, load_backend, PROJECT_ROOT
from navigation_agent.app.go2.connection import REQUIRED_CAPABILITIES


class ConnectionCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_transport_and_detector_without_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(env_file=Path(directory) / '.env', environ={})
            requests, messages = [], []

            async def health(request):
                requests.append(request.path)
                return web.json_response({'ok': True, 'agent': {'capabilities': ['navigation_history_continuation']}})

            async def model(request):
                requests.append(request.path)
                socket = web.WebSocketResponse()
                await socket.prepare(request)
                async for message in socket:
                    messages.append(message)
                return socket

            detector_payload = {}

            async def detector(request):
                requests.append(request.path)
                return web.json_response(detector_payload)

            app = web.Application()
            app.router.add_get('/api/health', health)
            app.router.add_get('/model', model)
            app.router.add_get('/capabilities', detector)
            async with TestServer(app) as server:
                url = str(server.make_url('')).rstrip('/')
                backend = settings.backend.model_copy(update={
                    'robot': settings.backend.robot.model_copy(update={'base_url': url}),
                    'vln': settings.backend.vln.model_copy(update={'server_url': url.replace('http:', 'ws:') + '/model'}),
                })
                cases = (
                    ('disabled', '', {}, None),
                    ('ready', url, {'ready': True, 'model': 'fake', 'vocabulary': 'open'}, True),
                    ('loading', url, {'ready': False, 'model': 'fake', 'vocabulary': 'open'}, False),
                    ('invalid', url, {'status': 'ok'}, False),
                    ('missing', url + '/missing', {}, False),
                )
                for name, endpoint, payload, expected in cases:
                    with self.subTest(name=name):
                        requests.clear()
                        detector_payload = payload
                        watch = settings.agent.target_watch.model_copy(update={'server_url': endpoint})
                        agent = settings.agent.model_copy(update={'target_watch': watch})
                        output = StringIO()
                        with redirect_stdout(output):
                            status = await check(replace(settings, backend=backend, agent=agent))
                        report = json.loads(output.getvalue())
                        self.assertEqual(status, 1 if expected is False else 0)
                        self.assertEqual(report.get('target_detector'), expected)
                        paths = ['/api/health', '/model']
                        if endpoint == url:
                            paths.append('/capabilities')
                        self.assertEqual(requests, paths)
            self.assertEqual(messages, [])


class Go2ConnectionCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_observation_and_detector_are_read_only(self):
        from navigation_agent.app.go2.check import check as check_go2

        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(env_file=Path(directory) / '.env', environ={})
            requests = []
            state, observation, detector_payload = {}, {}, {}
            observation_status = 200

            async def health(request):
                requests.append(request.path)
                return web.json_response(state)

            async def observe(request):
                requests.append(request.path)
                return web.json_response(observation, status=observation_status)

            async def detector(request):
                requests.append(request.path)
                return web.json_response(detector_payload)

            app = web.Application()
            app.router.add_get('/api/health', health)
            app.router.add_get('/api/observation', observe)
            app.router.add_get('/capabilities', detector)
            async with TestServer(app) as server:
                url = str(server.make_url('')).rstrip('/')
                go2 = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
                backend = go2.model_copy(update={
                    'robot': go2.robot.model_copy(update={'base_url': url}),
                })
                cases = (
                    ('ready', 1, True, 200, True, True, 0),
                    ('detector_disabled', 1, True, 200, False, False, 0),
                    ('old_client', None, True, 200, True, True, 1),
                    ('missing_capability', 1, True, 200, True, True, 1),
                    ('missing_history', 1, True, 200, True, True, 1),
                    ('stale_observation', 1, True, 503, True, True, 1),
                    ('invalid_pose', 1, False, 200, True, True, 1),
                    ('detector_loading', 1, True, 200, True, False, 1),
                )
                for name, protocol, pose_valid, code, enabled, ready, expected in cases:
                    with self.subTest(name=name):
                        requests.clear()
                        state = {'agent': {'protocol_version': protocol, 'capabilities': {
                            name: True for name in (*REQUIRED_CAPABILITIES, 'navigation_history_continuation')}}, 'odom': {'frame_id': 'odom'}}
                        if name == 'missing_history':
                            state['agent']['capabilities'].pop('navigation_history_continuation')
                        if name == 'missing_capability':
                            state['agent']['capabilities'] = {}
                        observation = {
                            'protocol_version': 1, 'sequence': 1, 'camera_stamp_ns': time.time_ns(),
                            'pose_valid': pose_valid, 'pose': {'x': 0., 'y': 0., 'yaw': 0.},
                            'odom_frame_id': 'odom', 'camera_age_ms': 0., 'odom_match_ms': 0.,
                            'jpeg_base64': base64.b64encode(b'\xff\xd8test').decode(),
                        }
                        observation_status = code
                        detector_payload = {'ready': ready, 'model': 'fake', 'vocabulary': 'open'}
                        watch = settings.agent.target_watch.model_copy(update={'server_url': url if enabled else ''})
                        agent = settings.agent.model_copy(update={'target_watch': watch})
                        output = StringIO()
                        with redirect_stdout(output):
                            status = await check_go2(replace(settings, backend=backend, agent=agent))
                        report = json.loads(output.getvalue())
                        self.assertEqual(status, expected)
                        self.assertEqual(report.get('target_detector'), ready if enabled else None)
                        expected_paths = ['/api/health']
                        if protocol == 1 and name not in ('missing_capability', 'missing_history'):
                            expected_paths.append('/api/observation')
                        if enabled:
                            expected_paths.append('/capabilities')
                        self.assertEqual(requests, expected_paths)
                        if protocol is None:
                            self.assertIn('Agent 协议版本 1', report['error'])
