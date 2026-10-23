"""无权重 ASGI 测试：会话隔离、关闭与协议错误，不额外安装测试客户端。"""

import json
import time
import unittest
from unittest.mock import patch

from navigation_agent.app.detector.server import create_app, detect_with_deadline
from navigation_agent.contracts.detection import Detection, FrameResponse


class FakeEngine:
    model_name = 'offline-fake'

    def new_state(self, confidence):
        return {}

    def detect(self, session, request):
        return FrameResponse(
            request_id=request.request_id, camera_stamp_ns=request.camera_stamp_ns,
            width=480, height=270, inference_ms=1,
            detections=[Detection(label=session['labels'][0], confidence=.9,
                                  bbox=(.1,.2,.8,.9), continuity_id='temporary')],
        )


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = create_app(FakeEngine)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()

    async def asyncTearDown(self):
        await self.lifespan.__aexit__(None, None, None)

    async def request(self, method, path, payload=None):
        messages = []
        body = json.dumps(payload).encode() if payload is not None else b''
        scope = {
            'type':'http', 'asgi':{'version':'3.0'}, 'http_version':'1.1',
            'method':method, 'scheme':'http', 'path':path, 'raw_path':path.encode(),
            'query_string':b'', 'root_path':'', 'server':('test',80), 'client':('test',1),
            'headers':[(b'content-type',b'application/json')],
        }
        async def receive():
            return {'type':'http.request', 'body':body, 'more_body':False}
        async def send(message):
            messages.append(message)
        await self.app(scope, receive, send)
        status = next(item['status'] for item in messages if item['type'] == 'http.response.start')
        output = b''.join(item.get('body',b'') for item in messages if item['type'] == 'http.response.body')
        return status, json.loads(output)

    async def test_sessions_keep_vocabulary_separate_and_close_is_idempotent(self):
        self.assertTrue((await self.request('GET','/capabilities'))[1]['ready'])
        first = (await self.request('POST','/sessions', {'labels':['sofa']}))[1]['session_id']
        second = (await self.request('POST','/sessions', {'labels':['ball']}))[1]['session_id']
        for session, label in ((first,'sofa'), (second,'ball'), (first,'sofa')):
            status, data = await self.request('POST','/sessions/' + session + '/frames', {
                'request_id':'frame-1', 'camera_stamp_ns':time.time_ns(), 'jpeg_base64':'test',
            })
            self.assertEqual(status, 200)
            self.assertEqual(data['detections'][0]['label'], label)
        self.assertEqual((await self.request('DELETE','/sessions/' + first))[0], 200)
        self.assertEqual((await self.request('DELETE','/sessions/' + first))[0], 200)
        self.assertEqual((await self.request('POST','/sessions/' + first + '/frames', {
            'request_id':'frame-1','camera_stamp_ns':1,'jpeg_base64':'test',
        }))[0], 404)

    async def test_invalid_request_and_session_capacity(self):
        self.assertEqual((await self.request('POST','/sessions', {'labels':[]}))[0], 422)
        for _ in range(4):
            self.assertEqual((await self.request('POST','/sessions', {'labels':['sofa']}))[0], 200)
        self.assertEqual((await self.request('POST','/sessions', {'labels':['sofa']}))[0], 503)


class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_loads_once_across_requests(self):
        count = 0
        def factory():
            nonlocal count
            count += 1
            return FakeEngine()
        app = create_app(factory)
        async with app.router.lifespan_context(app):
            engine = app.state.engine
            for _ in range(3):
                engine.new_state(.25)
                self.assertIs(app.state.engine, engine)
        self.assertEqual(count, 1)

    async def test_hung_inference_requests_process_restart(self):
        class SlowEngine:
            def detect(self, session, request):
                time.sleep(.08)
        with patch('navigation_agent.app.detector.server.terminate_worker') as terminate:
            with self.assertRaises(TimeoutError):
                await detect_with_deadline(SlowEngine(), {}, None, .01)
            terminate.assert_called_once_with()

    async def test_normal_inference_keeps_process_running(self):
        class Engine:
            def detect(self, session, request):
                return {'result': request}
        with patch('navigation_agent.app.detector.server.terminate_worker') as terminate:
            self.assertEqual(await detect_with_deadline(Engine(), {}, 'image', 1), {'result': 'image'})
            terminate.assert_not_called()
