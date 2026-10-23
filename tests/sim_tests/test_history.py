"""使用本机假模型验证暂停和恢复；不启动仿真或真实导航。"""

import asyncio
import logging
import unittest

from aiohttp import web

from navigation_agent.vln.lightnav.transport import LatestFrameClient


class HistoryWireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.logins = 0
        self.resets = 0
        self.release = asyncio.Event()
        self.first = asyncio.Event()

        async def endpoint(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                payload = message.json()
                if payload['action'] == 'login':
                    self.logins += 1
                    await socket.send_json({'data': {'rc': 0, 'capture_timestamps': True}})
                elif payload['action'] == 'reset':
                    self.resets += 1
                else:
                    data = payload['data']
                    self.requests.append(data)
                    if len(self.requests) == 1:
                        self.first.set()
                        await self.release.wait()
                    await socket.send_json({'data': {
                        'rc': 0, 'seq': data['seq'], 'capture_stamp_ns': data['capture_stamp_ns'],
                        'history_elapsed_s': float(data['seq']), 'actions': [[.1, 0., 0.]],
                        'stop': False, 'pointing': {},
                    }})
            return socket
        app = web.Application()
        app.router.add_get('/', endpoint)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.client = LatestFrameClient(logging.getLogger(__name__), f'ws://127.0.0.1:{port}')

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.to_thread(self.client.close)
        await self.runner.cleanup()

    async def until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(.01)

    async def test_pause_drops_inflight_action_and_resumes_same_history(self):
        self.client.start('Go to the door.')
        await self.until(lambda: self.client.snapshot().connected)
        self.client.offer_frame(1, b'first')
        await self.first.wait()
        self.client.pause()
        self.assertFalse(self.client.offer_frame(2, b'paused'))
        self.assertTrue(self.client.resume('Go to the door.'))
        self.release.set()
        await asyncio.sleep(.05)
        self.assertEqual(self.client.take_results(), [])
        self.client.offer_frame(3, b'resumed')
        results = []
        def received():
            results.extend(self.client.take_results())
            return bool(results)
        await self.until(received)
        self.assertEqual(self.logins, 1)
        self.assertEqual(self.resets, 0)
        self.assertEqual([p['capture_stamp_ns'] for p in self.requests], [1, 3])
        self.assertEqual(results[0].stamp_ns, 3)
        self.assertEqual(results[0].sequence, 2)

    async def test_changed_instruction_and_finish_start_fresh_history(self):
        self.client.start('Go to the door.')
        await self.until(lambda: self.client.snapshot().connected)
        self.client.pause()
        self.assertFalse(self.client.resume('Go to the chair.'))
        await self.until(lambda: self.logins == 2 and self.client.snapshot().connected)
        self.assertGreaterEqual(self.resets, 1)
        self.client.stop()
        self.assertFalse(self.client.resume('Go to the chair.'))
        await self.until(lambda: self.logins == 3 and self.client.snapshot().connected)

    async def test_old_generation_cannot_deliver_stop_to_resumed_action(self):
        from types import SimpleNamespace
        self.client.start('Go to the door.')
        await self.until(lambda: self.client.snapshot().connected)
        session, token = self.client._session, self.client._action_generation
        self.client.pause()
        self.client.resume('Go to the door.')
        self.client._accept_result(session, token, SimpleNamespace(stop=True))
        self.assertEqual(self.client.take_results(), [])
        self.assertGreater(self.client._session, session)
        self.client.stop()
        stopped_session = self.client._session
        self.client._accept_result(session, token, SimpleNamespace(stop=True))
        self.assertEqual(self.client._session, stopped_session)
        self.assertEqual(self.client.snapshot().instruction, '')
