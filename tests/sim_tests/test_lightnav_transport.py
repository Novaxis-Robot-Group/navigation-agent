"""验证传输更新不会改写采图标识、动作或旧会话隔离。"""

import threading
import unittest
from navigation_agent.vln.lightnav.transport import LatestFrameClient


class LatestFrameTests(unittest.TestCase):
    def setUp(self):
        self.client = LatestFrameClient.__new__(LatestFrameClient)
        self.client._lock = threading.Lock()
        self.client._instruction = "Go to the door."
        self.client._connected = True
        self.client._session = 3
        self.client._paused = False
        self.client._action_generation = 1
        self.client._pending_frame = None

    def test_inflight_frame_is_immutable_and_pending_is_latest(self):
        self.assertTrue(self.client.offer_frame(10, b"first"))
        inflight = self.client._take_frame(3)
        self.client.offer_frame(20, b"old-pending")
        self.client.offer_frame(30, b"latest")
        self.assertFalse(self.client.offer_frame(20, b"out-of-order"))
        self.assertEqual(inflight, (10, b"first", 1))
        self.assertEqual(self.client._take_frame(3), (30, b"latest", 1))

    def test_inactive_and_stale_session_cannot_consume_frame(self):
        self.client.offer_frame(10, b"frame")
        self.assertIsNone(self.client._take_frame(2))
        self.assertEqual(self.client._take_frame(3), (10, b"frame", 1))
        self.client._instruction = ""
        self.assertFalse(self.client.offer_frame(20, b"frame"))


class TransportWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_image_timestamp_and_actions_cross_websocket(self):
        import asyncio
        import logging
        from aiohttp import web

        first_seen = asyncio.Event()
        release_first = asyncio.Event()
        second_seen = asyncio.Event()
        requests = []

        async def endpoint(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                value = message.json()
                action = value["action"]
                if action == "login":
                    self.assertTrue(value["data"]["capture_timestamps"])
                    await socket.send_json({"data": {"rc": 0, "capture_timestamps": True}})
                elif action == "next":
                    data = value["data"]
                    requests.append(data)
                    if len(requests) == 1:
                        first_seen.set()
                        await release_first.wait()
                    else:
                        second_seen.set()
                    await socket.send_json({"data": {
                        "rc": 0, "seq": data["seq"], "actions": [[.1, -.2, .3]],
                        "stop": False, "pointing": {},
                        "capture_stamp_ns": data["capture_stamp_ns"],
                        "history_elapsed_s": (data["capture_stamp_ns"] - 1_000_000_000) / 1e9,
                    }})
            return socket

        app = web.Application()
        app.router.add_get("/", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = LatestFrameClient(logging.getLogger(__name__), f"ws://127.0.0.1:{port}")
        try:
            client.start("Go to the door.")
            async with asyncio.timeout(5):
                while not client.snapshot().connected:
                    await asyncio.sleep(.01)
                client.offer_frame(1_000_000_000, b"first")
                await first_seen.wait()
                client.offer_frame(1_200_000_000, b"old")
                client.offer_frame(1_700_000_000, b"latest")
                release_first.set()
                await second_seen.wait()
                results = []
                while len(results) < 2:
                    results.extend(client.take_results())
                    await asyncio.sleep(.01)
            self.assertEqual([r["capture_stamp_ns"] for r in requests], [1_000_000_000, 1_700_000_000])
            self.assertEqual(requests[1]["image"], "bGF0ZXN0")
            self.assertEqual(results[1].waypoints, ((.1, -.2, .3),))
            self.assertFalse(results[1].stop)
            self.assertEqual(results[1].history_elapsed_s, .7)
        finally:
            release_first.set()
            await asyncio.to_thread(client.close)
            await runner.cleanup()
