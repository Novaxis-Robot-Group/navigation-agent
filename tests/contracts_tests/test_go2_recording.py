"""Go2 录像只读采图；回放与 MuJoCo 共用时间轴和页面。"""

import asyncio
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from navigation_agent.app.go2.recording import record_task
from navigation_agent.app.go2.replay import ReplayService
from navigation_agent.app.recording import TaskRecording
from navigation_agent.app.replay import WEB_DIR

TASK_ID = 'b' * 32


class Go2RecordingTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(shutil.which('gst-launch-1.0'), '需要 GStreamer 验证真实 MP4')
    async def test_first_person_keeps_original_frames_and_finishes_after_runner_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jpeg = subprocess.check_output([
                'gst-launch-1.0', '-q', 'videotestsrc', 'num-buffers=1', '!',
                'video/x-raw,width=320,height=240', '!', 'jpegenc', '!', 'fdsink', 'fd=1',
            ], timeout=10)
            stamps, requests = [], []
            async def observation(request):
                requests.append((request.method, request.path))
                stamp = time.time_ns()
                stamps.append(stamp)
                return web.json_response({
                    'sequence': len(stamps), 'camera_stamp_ns': stamp,
                    'pose_valid': True, 'pose': {'x': 1., 'y': 2., 'yaw': .3},
                    'jpeg_base64': base64.b64encode(jpeg).decode(),
                })
            app = web.Application()
            app.router.add_get('/api/observation', observation)
            async with TestServer(app) as server:
                runtime = SimpleNamespace(
                    backend=SimpleNamespace(
                        recording=SimpleNamespace(directory=root, fps=10),
                        robot=SimpleNamespace(base_url=str(server.make_url('')).rstrip('/')),
                    ),
                    agent=SimpleNamespace(storage=None),
                )
                task = {'id': TASK_ID, 'started_at': datetime.now(timezone.utc).isoformat()}
                async def runner(record, *, settings):
                    self.assertEqual(settings, 'runner-settings')
                    await asyncio.sleep(.8)
                    raise RuntimeError('任务异常')
                with patch('navigation_agent.app.go2.recording.LogStore') as store:
                    with self.assertRaisesRegex(RuntimeError, '任务异常'):
                        await record_task(runner, task, runtime_settings=runtime, settings='runner-settings')
                    store.return_value.save_result.assert_called_once_with(task)
            self.assertEqual(task['recording']['status'], 'ready', task['recording'])
            self.assertEqual(task['recording']['views'], ['first'])
            self.assertGreaterEqual(task['recording']['frames'], 3)
            self.assertGreater((root / TASK_ID / 'first.mp4').stat().st_size, 1000)
            self.assertFalse((root / TASK_ID / 'third.mp4').exists())
            frames = [json.loads(row) for row in (root / TASK_ID / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual([row['stamp_ns'] for row in frames], stamps)
            self.assertTrue(all(row['pose'] == [1., 2., .3] for row in frames))
            self.assertEqual(set(requests), {('GET', '/api/observation')})


class Go2ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_without_video_and_first_view_path_boundaries(self):
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                'id': TASK_ID, 'goal': '找到垃圾桶', 'status': 'completed',
                'started_at': '2026-01-01T00:00:00+00:00',
                'ended_at': '2026-01-01T00:00:03+00:00',
            }
            (root / 'events.jsonl').write_text(json.dumps({
                'type': 'observation', 'time': '2026-01-01T00:00:01+00:00',
                'image': 'images/001.jpg',
            }))
            def read_record(task_id):
                if task_id != TASK_ID:
                    raise HTTPException(404)
                return task
            api = SimpleNamespace(
                read_record=read_record,
                log_store=lambda: SimpleNamespace(run_directory=lambda record: root),
                current_settings=lambda: SimpleNamespace(
                    backend=SimpleNamespace(recording=SimpleNamespace(directory=root / 'recordings')),
                ),
            )
            service = ReplayService(api)
            data = await service.data(TASK_ID)
            self.assertEqual(data['images'], [{'t': 1., 'image': 'images/001.jpg'}])
            self.assertEqual(data['recording']['status'], 'missing')
            self.assertEqual(data['log_base'], '/tasks/' + TASK_ID + '/logs/')
            self.assertEqual(Path((await service.page(TASK_ID)).path), WEB_DIR / 'replay.html')
            path = root / 'recordings' / TASK_ID
            path.mkdir(parents=True)
            (path / 'recording.json').write_text(json.dumps({'status': 'ready', 'views': ['first']}))
            (path / 'first.mp4').write_bytes(b'video')
            self.assertEqual(Path((await service.video(TASK_ID, 'first')).path), path / 'first.mp4')
            for view in ('third', '../first'):
                with self.assertRaises(HTTPException):
                    await service.video(TASK_ID, view)
            with self.assertRaises(HTTPException):
                await service.data('../secret')
