import asyncio
import base64
import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


from navigation_agent.app.mujoco.console import ConsoleServer
from navigation_agent.app.navigation_server import NavigationServer
from navigation_agent.navigation.runtime import NavigationRuntime
from navigation_agent.vln.lightnav.client import LightNavClient
from vln_mujoco.vln_client import VlnClient


class DiagnosticTests(unittest.TestCase):
    def test_result_capture_metadata_and_world_reference(self):
        result = SimpleNamespace(
            sequence=7, stamp_ns=42, waypoints=((1, 0, 0),), stop=False, latency_ms=300
        )

        paths = []

        def initialize(server, **kwargs):
            server.simulation = SimpleNamespace(frame=lambda: None)
            tracker = SimpleNamespace(
                reference=np.array([[3, 3, 0]]), prediction=np.empty((0, 3))
            )
            executor = SimpleNamespace(
                frame=lambda: None,
                accept_path=lambda points, pose: paths.append((points, pose)) or (0, 0),
                poll=lambda: None,
                tracker=tracker,
            )
            server.executor = executor
            server.vln = LightNavClient(None, client=SimpleNamespace(take_results=lambda: [result]))
            server.navigation = NavigationRuntime(server.vln, executor)
            server.navigation.capture_poses[42] = (2, 3, 0)

        with patch.object(NavigationServer, '__init__', initialize), patch(
            'navigation_agent.app.mujoco.console.create_simulation_components', return_value=(None, None, {})
        ), patch('navigation_agent.app.mujoco.console.create_vln_client'):
            server = ConsoleServer(agent_url='http://127.0.0.1:8090', model=object())
        server.navigation.step(False)
        self.assertEqual(paths, [(result.waypoints, (2, 3, 0))])
        with patch.object(NavigationServer, 'snapshot', return_value={'vln': {}, 'mpc': {}}):
            state = server.snapshot(None)
        self.assertEqual(state['vln']['diagnostic']['capture_pose'], (2, 3, 0))
        self.assertEqual(state['vln']['diagnostic']['capture_stamp_ns'], 42)
        self.assertEqual(state['mpc']['reference_world'], [[3, 3, 0]])

    def test_console_keeps_official_client_and_pending_frame(self):
        # 控制台必须保留官方客户端，繁忙时不得替换已排队的图像。
        with patch('threading.Thread.start'):
            client = VlnClient(logging.getLogger('test'), 'ws://localhost:8050')

        def initialize(server, **kwargs):
            server.vln = LightNavClient(None, client=client)

        with patch.object(NavigationServer, '__init__', initialize), patch(
            'navigation_agent.app.mujoco.console.create_simulation_components', return_value=(None, None, {})
        ), patch('navigation_agent.app.mujoco.console.create_vln_client'):
            server = ConsoleServer(agent_url='http://127.0.0.1:8090', model=object())
        self.assertIs(server.vln.client, client)
        client.start('Go forward.')
        client._connected = True
        self.assertTrue(server.vln.offer_frame(10, b'first'))
        self.assertFalse(server.vln.offer_frame(20, b'newer'))
        self.assertEqual(client._take_frame(client._session), (10, b'first'))
        client.stop()


class ShutdownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.console = object.__new__(ConsoleServer)
        self.console.update_task = None
        self.console.vln = Mock()
        self.console.executor = Mock()
        self.console.simulation = Mock()
        self.console.sockets = set()
        self.console.recorder = SimpleNamespace(
            active=SimpleNamespace(metadata={'task_id': 'a' * 32}), stop=AsyncMock()
        )
        self.console.agent_http = SimpleNamespace(close=AsyncMock())

    async def test_motion_stops_before_slow_recording_finalization(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def finish_recording(*args, **kwargs):
            entered.set()
            await release.wait()

        self.console.recorder.stop.side_effect = finish_recording
        shutdown = asyncio.create_task(self.console.on_shutdown(None))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            self.assertFalse(shutdown.done())
            self.console.vln.stop.assert_called_once_with()
            self.console.executor.close.assert_called_once_with()
            self.console.simulation.set_velocity.assert_called_once_with(0.0, 0.0)
            self.console.simulation.stop.assert_called_once_with()
        finally:
            release.set()
            await shutdown
        self.console.recorder.stop.assert_awaited_once_with('a' * 32, status='interrupted')
        self.console.agent_http.close.assert_awaited_once_with()

    async def test_recording_failure_still_stops_motion_and_closes_http(self):
        self.console.recorder.stop.side_effect = OSError('recording write failed')
        with self.assertRaisesRegex(OSError, 'recording write failed'):
            await self.console.on_shutdown(None)
        self.console.simulation.set_velocity.assert_called_once_with(0.0, 0.0)
        self.console.simulation.stop.assert_called_once_with()
        self.console.agent_http.close.assert_awaited_once_with()

    async def test_navigation_cleanup_failure_still_finalizes_recording_and_http(self):
        with patch.object(NavigationServer, 'on_shutdown', side_effect=RuntimeError('shutdown failed')):
            with self.assertRaisesRegex(RuntimeError, 'shutdown failed'):
                await self.console.on_shutdown(None)
        self.console.recorder.stop.assert_awaited_once_with('a' * 32, status='interrupted')
        self.console.agent_http.close.assert_awaited_once_with()


class LightweightConsole(ConsoleServer):
    def __init__(self, agent_url):
        self.agent_url = agent_url
        self.agent_http = None

    async def on_startup(self, app):
        self.agent_http = aiohttp.ClientSession(trust_env=False)

    async def on_shutdown(self, app):
        await self.agent_http.close()


class ConsoleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []

        async def agent(request):
            self.received.append((request.method, request.path, request.query_string))
            if request.path.endswith('.html'):
                return web.Response(text='<a href="memory_visualize.html">记忆</a>', content_type='text/html')
            if request.path.endswith('.jpg'):
                return web.Response(body=b'\xff\xd8image', content_type='image/jpeg')
            if request.method == 'POST':
                return web.json_response({'status': 'running', **await request.json()}, status=202)
            return web.json_response({'tasks': []})

        app = web.Application()
        app.router.add_route('*', '/{path:.*}', agent)
        self.agent = TestServer(app)
        await self.agent.start_server()
        self.console = LightweightConsole(str(self.agent.make_url('/')).rstrip('/'))
        self.client = TestClient(TestServer(self.console.app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.agent.close()

    async def test_atomic_frame_uses_capture_pose_without_motion(self):
        frame = SimpleNamespace(jpeg=b'jpeg', stamp_ns=123, pose=(1., 2., .3))
        simulation = Mock()
        simulation.frame.return_value = frame
        self.console.simulation = simulation
        response = await self.console.observation(None)
        payload = json.loads(response.body)
        self.assertEqual(base64.b64decode(payload['jpeg_base64']), frame.jpeg)
        self.assertEqual(payload['camera_stamp_ns'], frame.stamp_ns)
        self.assertEqual(payload['pose'], {'x': 1., 'y': 2., 'yaw': .3})
        self.assertEqual(simulation.mock_calls, [unittest.mock.call.frame()])

    async def test_console_serves_local_assets(self):
        response = await self.client.get('/')
        self.assertIn('探索工作台', await response.text())
        self.assertEqual((await self.client.get('/console-static/app.js')).status, 200)

    async def test_same_origin_task_proxy_and_image(self):
        response = await self.client.post('/api/agent/tasks', json={'goal': '找到沙发'})
        self.assertEqual(response.status, 202)
        self.assertEqual((await response.json())['goal'], '找到沙发')
        await self.client.get('/api/agent/tasks?limit=10')
        self.assertEqual(self.received[-1][2], 'limit=10')
        response = await self.client.get('/api/agent/tasks/' + 'a' * 32 + '/images/001.jpg')
        self.assertEqual(response.content_type, 'image/jpeg')

    async def test_report_proxy_preserves_relative_resources(self):
        base = '/api/agent/tasks/' + 'a' * 32 + '/logs/'
        response = await self.client.get(base + 'index.html')
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, 'text/html')
        self.assertIn('href="memory_visualize.html"', await response.text())
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        resources = [
            'memory_visualize.html', 'images/001.jpg', 'images/1001.jpg',
            'events.jsonl', 'result.json', 'navigation.jsonl', 'execution.json',
            'memory.sqlite3', 'prompts/' + 'b' * 64 + '.md',
        ]
        for resource in resources:
            with self.subTest(resource=resource):
                response = await self.client.get(base + resource)
                self.assertEqual(response.status, 200)
                self.assertEqual(self.received[-1][1], base.removeprefix('/api/agent') + resource)

    async def test_report_proxy_rejects_writes_and_unrelated_files(self):
        base = '/api/agent/tasks/' + 'a' * 32 + '/logs/'
        for resource in ['.env', 'other.html', 'script.js', 'nested/result.json',
                         '%2e%2e%2fresult.json', 'images/%2e%2e%2fresult.json']:
            with self.subTest(resource=resource):
                self.assertEqual((await self.client.get(base + resource)).status, 404)
        self.assertEqual((await self.client.post(base + 'index.html', json={})).status, 405)
        self.assertEqual(self.received, [])

    async def test_agent_settings_proxy_is_read_only(self):
        response = await self.client.get('/api/agent/settings/agent')
        self.assertEqual(response.status, 200)
        self.assertEqual(self.received[-1][1], '/settings/agent')
        self.assertEqual((await self.client.post('/api/agent/settings/agent', json={})).status, 405)
        self.assertEqual((await self.client.get('/api/agent/settings/motion')).status, 404)
        self.assertEqual(len(self.received), 1)

    async def test_rejects_other_origins_and_unknown_proxy_targets(self):
        response = await self.client.post(
            '/api/agent/tasks',
            json={'goal': 'test'},
            headers={'Origin': 'https://unrelated.example'},
        )
        self.assertEqual(response.status, 403)
        self.assertEqual((await self.client.get('/api/agent/.env')).status, 404)
        self.assertEqual((await self.client.post('/api/agent/health')).status, 405)
        self.assertEqual(self.received, [])

    async def test_agent_offline_is_actionable(self):
        await self.agent.close()
        response = await self.client.get('/api/agent/tasks')
        self.assertEqual(response.status, 503)
        self.assertIn('启动 Agent 服务', (await response.json())['detail'])


if __name__ == '__main__':
    unittest.main()


class ReplayConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_uses_backend_camera_and_optional_model_override(self):
        from tests import replay_navigation
        from navigation_agent.app.deployments.settings import load_backend

        settings = load_backend()
        settings = settings.model_copy(update={
            'simulation': settings.simulation.model_copy(update={'camera_height_m': 0.65}),
            'vln': settings.vln.model_copy(update={'server_url': 'ws://configured-model:8053'}),
        })
        for override in (None, 'ws://override-model:8054'):
            with self.subTest(override=override):
                args = SimpleNamespace(
                    backend_config='custom-mujoco.toml', model_url=override, pose=(1., 2., 0.),
                )
                robot = SimpleNamespace(
                    _base_qpos=0, data=SimpleNamespace(qpos=np.zeros(8)), model=object(),
                    _collision=lambda: False,
                )
                socket = Mock()
                socket.__enter__ = Mock(return_value=Mock(connect_ex=Mock(return_value=1)))
                socket.__exit__ = Mock(return_value=False)
                # 在创建服务前停止，验证配置传递，不连接模型或启动仿真。
                with patch.object(replay_navigation, 'load_backend', return_value=settings) as load, patch(
                    'tests.replay_navigation.socket.socket', return_value=socket,
                ), patch.object(replay_navigation, 'SafeTurtleBot', return_value=robot) as create_robot, patch(
                    'tests.replay_navigation.mujoco.mj_forward',
                ), patch.object(replay_navigation, 'Simulation'), patch.object(
                    replay_navigation, 'ConsoleServer', side_effect=RuntimeError('stop-before-service'),
                ) as create_console:
                    with self.assertRaisesRegex(RuntimeError, 'stop-before-service'):
                        await replay_navigation.replay(args)
                load.assert_called_once_with('custom-mujoco.toml')
                create_robot.assert_called_once_with(camera_height_m=0.65)
                self.assertEqual(
                    create_console.call_args.kwargs['vln_settings'].server_url,
                    override or 'ws://configured-model:8053',
                )
