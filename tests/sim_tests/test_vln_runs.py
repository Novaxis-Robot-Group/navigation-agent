"""纯 VLN 留档的离线生命周期与资源回归，不连接模型或机器人。"""

import asyncio
from datetime import datetime, timezone
import json
import io
import shutil
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from navigation_agent.app.deployments.settings import RecordingSettings
from navigation_agent.app.mujoco.console import ConsoleServer
from navigation_agent.app.mujoco.vln_runs import VlnRuns
from navigation_agent.contracts.vln import Prediction
from navigation_agent.navigation.runtime import NavigationRuntime


class FakeModel:
    def __init__(self):
        self.state = 'IDLE'
        self.instruction = ''
        self.error = ''
        self.results = []
        self.starts = []

    def start(self, instruction):
        self.starts.append(instruction)
        self.instruction = instruction
        self.state = 'RUNNING'

    def stop(self):
        self.state = 'IDLE'

    def offer_frame(self, stamp, jpeg):
        return False

    def take_results(self):
        results, self.results = self.results, []
        return results

    def snapshot(self):
        return SimpleNamespace(state=self.state, error=self.error, connected=True,
                               server_url='', instruction=self.instruction, sequence=0)


class OfflineConsole(ConsoleServer):
    async def broadcast(self):
        pass

    def snapshot(self, socket):
        return {}

    async def on_startup(self, app):
        self.agent_http = aiohttp.ClientSession(trust_env=False)

    async def on_shutdown(self, app):
        self.stop_vln('shutdown')
        await self.vln_runs.close()
        await self.agent_http.close()


def console_at(root):
    frame = SimpleNamespace(stamp_ns=time.time_ns(), pose=(1., 2., .3), jpeg=b'jpeg')
    robot = SimpleNamespace(frame=lambda: frame, robot_name='offline',
                            set_velocity=Mock(), reset=Mock())
    executor = SimpleNamespace(frame=lambda: None, control_period_s=.1,
                               reset=Mock(), drive=Mock(), accept_path=Mock(return_value=(.1, 0)),
                               poll=lambda: None, submit=Mock(), fail=Mock(), diagnostics=lambda: {})
    with patch('navigation_agent.app.mujoco.console.create_simulation_components',
               return_value=(robot, executor, {'id': 'offline-scene'})):
        return OfflineConsole(agent_url='http://127.0.0.1:1', model=FakeModel(),
                              recording=RecordingSettings(directory=Path(root)))


class VlnRunTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.console = console_at(self.temp.name)
        # 控制权使用集合成员检查，需要可哈希的连接。
        self.socket = Mock(closed=False, send_json=AsyncMock())

    async def asyncTearDown(self):
        self.console.stop_vln('shutdown')
        await self.console.vln_runs.close()
        self.temp.cleanup()

    async def start(self, **extra):
        await self.console.handle_message(self.socket, {
            'type': 'set_vln', 'enabled': True, 'record_run': True,
            'instruction': 'Go to the chair.', 'goal': '走到椅子旁', **extra,
        })
        return self.console.vln_runs.active

    async def finish(self, reason='manual_stop'):
        self.console.stop_vln(reason)
        await self.console.vln_runs.close()

    async def test_model_stop_is_not_success_and_all_predictions_are_saved(self):
        record = await self.start()
        run_id = record['id']
        self.console.navigation.capture_poses[10] = (1, 2, .3)
        self.console.vln.results = [
            Prediction(1, 10, ((1, 0, 0),), False, 12),
            Prediction(2, 20, (), True, 13),
        ]
        stopped = self.console.navigation.step(True)
        self.console.navigation_tick(stopped)
        await self.console.vln_runs.close()
        result = self.console.vln_runs.read(run_id)
        self.assertEqual(result['end_reason'], 'model_stop')
        self.assertEqual(result['status'], 'stopped')
        self.assertIsNone(result['success'])
        self.assertEqual(result['prediction_count'], 2)
        self.assertEqual(result['recording']['status'], 'disabled')
        directory = self.console.vln_runs.directory(run_id)
        predictions = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        self.assertEqual([row['sequence'] for row in predictions], [1, 2])
        self.assertEqual([row['execution'] for row in predictions], ['submitted', 'model_stop'])
        self.assertTrue((directory / result['final_image']).is_file())
        timeline = self.console.vln_runs.timeline(run_id, {'status': 'disabled'})
        self.assertEqual(timeline['task']['mode'], 'vln')
        self.assertEqual([node['label'] for node in timeline['nodes'] if node['key']],
                         ['开始直接导航', '收到首次预测', '导航结束'])
        self.assertTrue(timeline['images'])
        self.assertTrue(timeline['trajectory'])

    async def test_agent_navigation_is_not_duplicated_and_can_use_agent_recorder(self):
        self.console.recorder.active = SimpleNamespace(metadata={'task_id': 'a' * 32})
        await self.console.handle_message(self.socket, {
            'type': 'set_vln', 'enabled': True, 'instruction': 'Go forward.',
        })
        self.assertEqual(self.console.vln.starts, ['Go forward.'])
        self.assertIsNone(self.console.vln_runs.active)
        self.assertFalse(self.console.vln_runs.root.exists())
        self.console.recorder.active = None

    async def test_manual_end_reset_and_takeover_keep_distinct_reasons(self):
        for kind, reason in [('set_vln', 'manual_stop'), ('reset', 'reset'), ('acquire_control', 'manual_takeover')]:
            record = await self.start()
            await self.console.handle_message(self.socket, {'type': kind, 'enabled': False})
            await self.console.vln_runs.close()
            self.assertEqual(self.console.vln_runs.read(record['id'])['end_reason'], reason)
            self.console.manual_owner = None

    async def test_timeout_and_error_stop_without_claiming_success(self):
        for reason in ('timeout', 'error'):
            record = await self.start(max_seconds=1)
            if reason == 'timeout':
                self.console.vln_runs.deadline = time.monotonic() - 1
            else:
                self.console.vln.error = 'offline model error'
            self.console.navigation_tick(False)
            await self.console.vln_runs.close()
            self.assertEqual(self.console.vln_runs.read(record['id'])['end_reason'], reason)
            self.assertIsNone(self.console.auto_owner)
            self.console.vln.error = ''

    async def test_invalid_request_does_not_start_or_create_records(self):
        for extra in ({'instruction': ''}, {'max_seconds': -1}, {'max_seconds': float('nan')},
                      {'max_seconds': True}, {'max_seconds': '600'}):
            await self.start(**extra)
            self.assertIsNone(self.console.vln_runs.active)
        self.assertEqual(self.console.vln.starts, [])
        self.assertFalse(self.console.vln_runs.root.exists())

    async def test_stop_is_not_blocked_by_recording_start_and_prevents_late_motion(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_start(record):
            entered.set()
            await release.wait()
            return {'status': 'disabled'}
        self.console.recorder.start = slow_start
        task = asyncio.create_task(self.start())
        await entered.wait()
        await asyncio.wait_for(self.console.handle_message(self.socket, {'type': 'stop'}), .2)
        self.assertEqual(self.console.vln.starts, [])
        await asyncio.sleep(.15)
        self.assertIsNotNone(self.console.vln_runs.active)
        release.set()
        await task
        await self.console.vln_runs.close()
        self.assertEqual(self.console.vln.starts, [])

    async def test_second_start_is_rejected_during_recording_finalization(self):
        record = await self.start()
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_stop(run_id):
            entered.set()
            await release.wait()
            return {'status': 'ready', 'views': []}
        self.console.recorder.stop = slow_stop
        self.console.stop_vln()
        await entered.wait()
        await self.start()
        self.assertEqual(len(self.console.vln.starts), 1)
        release.set()
        await self.console.vln_runs.close()
        self.assertEqual(self.console.vln_runs.read(record['id'])['status'], 'stopped')

    async def test_crash_recovery_does_not_require_agent_service(self):
        record = await self.start()
        await self.console.vln_runs.save_pending(record)
        recovered = VlnRuns(self.console)
        result = recovered.read(record['id'])
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['end_reason'], 'service_interrupted')
        self.assertIsNone(result['success'])

    async def test_logging_failure_does_not_change_prediction_execution(self):
        record = await self.start()
        self.console.navigation.on_prediction = Mock(side_effect=OSError('disk unavailable'))
        self.console.navigation.capture_poses[10] = (1, 2, 0)
        self.console.vln.results = [Prediction(1, 10, ((1, 0, 0),), False, 12)]
        with self.assertLogs('navigation_agent.navigation.runtime', level='ERROR'):
            self.console.navigation.step(True)
        self.console.executor.accept_path.assert_called_once_with(((1, 0, 0),), (1, 2, 0))
        self.assertEqual(record['status'], 'running')

    async def test_missing_encoder_does_not_prevent_direct_navigation(self):
        self.console.recorder.start = AsyncMock(side_effect=OSError('encoder missing'))
        record = await self.start()
        self.assertEqual(self.console.vln.starts, ['Go to the chair.'])
        await self.finish()
        self.assertEqual(self.console.vln_runs.read(record['id'])['recording']['status'], 'failed')


class VlnRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.console = console_at(self.temp.name)
        self.client = TestClient(TestServer(self.console.app()))
        await self.client.start_server()
        record = await self.console.vln_runs.start('Go to <chair>.', '<目标>', None)
        self.run_id = record['id']
        self.console.vln_runs.finish('manual_stop')
        await self.console.vln_runs.close()

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_history_report_and_replay_work_with_agent_offline(self):
        response = await self.client.get('/api/vln/runs')
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['tasks'][0]['id'], self.run_id)
        response = await self.client.get('/api/replay/' + self.run_id)
        data = await response.json()
        self.assertEqual(data['recording']['status'], 'disabled')
        self.assertEqual(data['task']['mode'], 'vln')
        self.assertEqual(len([node for node in data['nodes'] if node['key']]), 2)
        base = '/api/vln/runs/' + self.run_id
        response = await self.client.get(base + '/logs/index.html')
        html = await response.text()
        self.assertIn('&lt;目标&gt;', html)
        self.assertNotIn('<目标>', html)
        self.assertIn('任务成功：未评定', html)
        response = await self.client.get(base + '/events?after=0')
        self.assertGreater((await response.json())['next'], 0)
        self.assertEqual((await self.client.get(base + '/logs/predictions.jsonl')).status, 200)
        self.assertEqual((await self.client.get(base + '/logs/images/000001.jpg')).status, 200)

    @unittest.skipUnless(shutil.which('gst-launch-1.0'), '需要本地 GStreamer 编码器')
    async def test_recorded_vln_video_and_timeline_share_run_id(self):
        image = io.BytesIO()
        Image.new('RGB', (160, 120), '#e6e0d6').save(image, 'JPEG')
        frame = SimpleNamespace(stamp_ns=time.time_ns(), pose=(1., 2., .3),
                                jpeg=image.getvalue())
        self.console.simulation.frame = lambda: frame
        self.console.simulation.third_person_frame = lambda: frame
        self.console.recorder.settings = RecordingSettings(
            enabled=True, directory=Path(self.temp.name), fps=10)
        record = await self.console.vln_runs.start('Go to the chair.', '离线录像验证', None)
        await asyncio.sleep(.25)
        self.console.vln_runs.finish('model_stop')
        await self.console.vln_runs.close()
        response = await self.client.get('/api/replay/' + record['id'])
        data = await response.json()
        self.assertEqual(data['recording']['status'], 'ready')
        self.assertEqual(data['recording']['views'], ['first', 'third'])
        self.assertGreater(len(data['frames']), 0)
        self.assertTrue(data['trajectory'])
        for view in ('first', 'third'):
            response = await self.client.get('/api/recordings/' + record['id'] + '/' + view + '.mp4')
            self.assertEqual(response.status, 200)
            self.assertIn(b'ftyp', (await response.read())[:32])

    async def test_websocket_disconnect_records_interruption(self):
        socket = await self.client.ws_connect('/ws')
        await socket.receive_json()
        await socket.send_json({'type': 'set_vln', 'enabled': True, 'record_run': True,
                                'instruction': 'Go to the chair.'})
        reply = await socket.receive_json()
        self.assertTrue(reply['ok'])
        run_id = self.console.vln_runs.active['id']
        await socket.close()
        # 让服务端处理断开；等待日志收尾不触发任何外部服务。
        for _ in range(50):
            if self.console.vln_runs.active is None:
                break
            await asyncio.sleep(.01)
        result = self.console.vln_runs.read(run_id)
        self.assertEqual(result['end_reason'], 'connection_closed')
        self.assertEqual(result['status'], 'interrupted')
        self.assertIsNone(self.console.auto_owner)

    async def test_routes_reject_writes_traversal_and_invalid_cursor(self):
        base = '/api/vln/runs/' + self.run_id
        for file in ('.env', '../result.json', '%2e%2e%2fresult.json', 'images/../../result.json'):
            self.assertEqual((await self.client.get(base + '/logs/' + file)).status, 404)
        self.assertEqual((await self.client.post('/api/vln/runs')).status, 405)
        self.assertEqual((await self.client.get(base + '/events?after=bad')).status, 400)
        self.assertEqual((await self.client.get('/api/vln/runs/' + 'b' * 32 + '/logs/result.json')).status, 404)
