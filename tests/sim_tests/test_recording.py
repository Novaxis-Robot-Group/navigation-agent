"""验证录像时间映射、独立编码与回放资源边界。"""

import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from navigation_agent.app.deployments.settings import RecordingSettings
from navigation_agent.app.mujoco.recording import Recorder
from navigation_agent.app.recording import TaskRecording
from navigation_agent.app.mujoco.replay import ReplayService
from navigation_agent.app.replay import build_timeline, event_subtitles
from navigation_agent.contracts.robot import CameraFrame
from tests.sim_tests.test_console import LightweightConsole

TASK_ID = 'a' * 32


def task_record():
    return {'id': TASK_ID, 'status': 'running', 'goal': '录像测试',
            'started_at': datetime.now(timezone.utc).isoformat(), 'max_seconds': 5}


class TimelineTests(unittest.TestCase):
    def test_capture_decision_and_action_use_distinct_times(self):
        task = {'id': TASK_ID, 'started_at': '2026-01-01T00:00:00+00:00',
                'ended_at': '2026-01-01T00:00:20+00:00'}
        start = datetime.fromisoformat(task['started_at']).timestamp()
        events = [
            {'type': 'observation', 'step': 1, 'time': '2026-01-01T00:00:11+00:00',
             'image': 'images/001.jpg', 'observation': {'camera_stamp_ns': (start + 10) * 1e9}},
            {'type': 'decision', 'step': 1, 'time': '2026-01-01T00:00:14+00:00',
             'decision': {'action': 'navigate', 'reason': '查看门口'}},
            {'type': 'action_input', 'step': 1, 'time': '2026-01-01T00:00:15+00:00', 'action': 'navigate'},
        ]
        samples = [{'time_ns': (start + 15) * 1e9, 'pose': {'x': 1, 'y': 2, 'yaw': .3}}]
        frames = [{'stamp_ns': (start + 10) * 1e9, 'video_s': 0, 'pose': [0, 0, 0]},
                  {'stamp_ns': (start + 14) * 1e9, 'video_s': .1, 'pose': [1, 1, 0]}]
        data = build_timeline(task, events, samples, {'status': 'ready'}, frames)
        self.assertEqual([n['t'] for n in data['nodes']], [10, 14, 15])
        self.assertEqual(data['nodes'][1]['observation_t'], 10)
        self.assertEqual([n['key'] for n in data['nodes']], [False, True, False])
        self.assertEqual(data['frames'][1], {'t': 14, 'v': .1})
        self.assertEqual(data['trajectory'][-1]['pose'], [1, 2, .3])
        self.assertEqual(data['duration'], 20)

    def test_subtitles_follow_video_clock_after_missing_frames(self):
        data = {'frames': [{'t': 10, 'v': 0}, {'t': 14, 'v': .1}, {'t': 14.1, 'v': .2}],
                'recording': {'video_duration_s': .3},
                'nodes': [{'t': 10, 'type': 'decision', 'label': '决策', 'description': '<观察>'},
                          {'t': 14, 'type': 'action_input', 'label': '动作', 'description': '前进'}]}
        subtitles = event_subtitles(data)
        self.assertIn('00:00:00.000 --> 00:00:00.100', subtitles)
        self.assertIn('00:00:00.100 --> 00:00:00.300', subtitles)
        self.assertIn('&lt;观察&gt;', subtitles)
        self.assertNotIn('00:00:14.', subtitles)

    def test_old_logs_without_video_keep_images_and_reject_unsafe_paths(self):
        task = {'id': TASK_ID, 'started_at': '2026-01-01T00:00:00+00:00'}
        events = [{'type': 'observation', 'time': '2026-01-01T00:00:01+00:00', 'image': '../secret.jpg'},
                  {'type': 'observation', 'time': '2026-01-01T00:00:02+00:00', 'image': 'images/001.jpg'}]
        data = build_timeline(task, events, [], {'status': 'missing'}, [])
        self.assertEqual(data['images'], [{'t': 2, 'image': 'images/001.jpg'}])
        self.assertEqual(data['frames'], [])


class RecorderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_default_disabled_does_not_create_encoder_or_files(self):
        from unittest.mock import patch
        recorder = Recorder(None, RecordingSettings(directory=self.root))
        with patch('navigation_agent.app.mujoco.recording.TaskRecording') as encoder:
            result = await recorder.start(task_record())
        self.assertEqual(result, {'status': 'disabled'})
        encoder.assert_not_called()
        self.assertIsNone(recorder.sampler)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_missing_encoder_is_visible_without_raising_into_navigation(self):
        from unittest.mock import patch
        with patch('navigation_agent.app.recording.shutil.which', return_value=None):
            recording = TaskRecording(self.root / TASK_ID, task_record(), 10)
            await asyncio.to_thread(recording.thread.join, 3)
            result = await recording.close()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('GStreamer', result['error'])
        self.assertFalse((recording.directory / 'first.mp4').exists())

    async def test_stale_recording_is_reported_as_interrupted(self):
        path = self.root / TASK_ID
        path.mkdir()
        (path / 'recording.json').write_text(json.dumps({'status': 'recording', 'views': ['first', 'third']}))
        recorder = Recorder(None, RecordingSettings(directory=self.root))
        result = recorder.read(TASK_ID)
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['views'], [])

    @unittest.skipUnless(shutil.which('gst-launch-1.0'), '需要 GStreamer 验证真实 MP4 编码')
    async def test_dual_video_finalization_and_frame_timestamps(self):
        buffer = io.BytesIO()
        Image.new('RGB', (320, 240), '#796347').save(buffer, 'JPEG')
        def frame():
            stamp = int(time.time() * 10) * 100_000_000
            return CameraFrame(stamp_ns=stamp, jpeg=buffer.getvalue(), pose=(1., 2., .3))
        simulation = SimpleNamespace(frame=frame, third_person_frame=frame)
        recorder = Recorder(simulation, RecordingSettings(enabled=True, directory=self.root, fps=10))
        task = task_record()
        result = await recorder.start(task)
        self.assertEqual(result['status'], 'recording')
        await asyncio.sleep(.7)
        result = await recorder.stop(TASK_ID)
        self.assertEqual(result['status'], 'ready', result)
        self.assertGreaterEqual(result['frames'], 3)
        frames = [json.loads(line) for line in (self.root / TASK_ID / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(frames), result['frames'])
        self.assertEqual(frames[1]['video_s'], .1)
        self.assertLess(frames[0]['stamp_ns'], frames[1]['stamp_ns'])
        for view in ('first', 'third'):
            video = self.root / TASK_ID / (view + '.mp4')
            self.assertGreater(video.stat().st_size, 1000)
            payload = video.read_bytes()
            sample_table = payload.index(b'stsz')
            encoded_frames = int.from_bytes(payload[sample_table + 12:sample_table + 16], 'big')
            self.assertEqual(encoded_frames, result['frames'])
            # 解复用真实产物，确认不是仅创建了扩展名为 MP4 的文件。
            decode = await asyncio.to_thread(subprocess.run,
                ['gst-launch-1.0', '-q', 'filesrc', 'location=' + str(video), '!',
                 'qtdemux', '!', 'h264parse', '!', 'fakesink'], capture_output=True, timeout=10)
            self.assertEqual(decode.returncode, 0, decode.stderr.decode())
        self.assertIsNone(recorder.active)
        self.assertEqual((await recorder.stop(TASK_ID))['status'], 'ready')


    @unittest.skipUnless(shutil.which('gst-launch-1.0'), '需要 GStreamer 验证不限时任务录像')
    async def test_unlimited_task_records_and_finalizes_normally(self):
        from unittest.mock import patch
        task = task_record()
        task['max_seconds'] = None
        with patch(__name__ + '.task_record', return_value=task):
            await self.test_dual_video_finalization_and_frame_timestamps()


class ReplayRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        task = task_record()
        task['status'] = 'completed'
        async def agent(request):
            if request.path.endswith('/logs/events.jsonl'):
                return web.Response(text=json.dumps({'type': 'started', 'time': task['started_at']}))
            if request.path.endswith('/logs/navigation.jsonl'):
                raise web.HTTPNotFound()
            return web.json_response(task)
        app = web.Application()
        app.router.add_get('/{path:.*}', agent)
        self.agent = TestServer(app)
        await self.agent.start_server()
        console = LightweightConsole(str(self.agent.make_url('/')).rstrip('/'))
        console.recorder = Recorder(None, RecordingSettings(directory=self.root))
        console.replay = ReplayService(console, console.recorder)
        self.client = TestClient(TestServer(console.app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.agent.close()
        self.temp.cleanup()

    async def test_old_task_page_and_data(self):
        response = await self.client.get('/replay/' + TASK_ID)
        self.assertEqual(response.status, 200)
        self.assertIn('关键节点', await response.text())
        response = await self.client.get('/api/replay/' + TASK_ID)
        data = await response.json()
        self.assertEqual(data['recording']['status'], 'missing')
        self.assertEqual(data['nodes'][0]['type'], 'started')
        self.assertEqual((await self.client.post('/api/recordings/' + TASK_ID + '/start', json={})).status, 409)

    async def test_video_range_and_path_boundaries(self):
        directory = self.root / TASK_ID
        directory.mkdir()
        (directory / 'first.mp4').write_bytes(b'0123456789')
        (directory / 'recording.json').write_text(json.dumps({'status': 'ready', 'views': ['first', 'third'],
                                                               'video_duration_s': .1}))
        (directory / 'frames.jsonl').write_text(json.dumps({'stamp_ns': time.time_ns(), 'video_s': 0}) + '\n')
        base = '/api/recordings/' + TASK_ID
        response = await self.client.get(base + '/first.mp4', headers={'Range': 'bytes=2-5'})
        self.assertEqual(response.status, 206)
        self.assertEqual(await response.read(), b'2345')
        self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
        response = await self.client.get('/api/replay/' + TASK_ID + '/subtitles.vtt')
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, 'text/vtt')
        self.assertTrue((await response.text()).startswith('WEBVTT'))
        self.assertIn('attachment', response.headers['Content-Disposition'])
        self.assertEqual((await self.client.get(base + '/recording.json')).status, 404)
        self.assertEqual((await self.client.get('/api/recordings/../first.mp4')).status, 404)
        self.assertEqual((await self.client.post(base + '/stop', headers={'Origin': 'https://example.com'})).status, 403)
