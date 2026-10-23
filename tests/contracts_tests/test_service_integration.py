"""真实 HTTP/WS 链路配替代设备与模型，验证装配和控制权闭环。"""

import time
import unittest

from aiohttp.test_utils import TestServer

from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import load_backend
from navigation_agent.app.navigation_server import NavigationServer
from navigation_agent.contracts.robot import CameraFrame
from navigation_agent.contracts.vln import Prediction, VlnStatus
from tests.contracts_tests.test_runtime import Executor


class Device:
    robot_name = 'test-device'

    def __init__(self):
        self.command = (0.0, 0.0)
        self.stamp = 0
        self.started = False

    def frame(self):
        self.stamp = time.time_ns()
        return CameraFrame(self.stamp, b'\xff\xd8image', (1, 2, 0))

    third_person_frame = frame

    def snapshot(self):
        return {
            'sim_time': time.monotonic(),
            'pose': {'x': 1, 'y': 2, 'z': 0, 'yaw': 0},
            'velocity': {'linear': 0, 'angular': 0},
            'command': {'linear': self.command[0], 'angular': self.command[1]},
            'camera': {'ready': True, 'stamp_ns': self.stamp},
            'backend': {'blocked': False},
        }

    def set_velocity(self, linear, angular):
        self.command = (linear, angular)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def reset(self):
        self.command = (0, 0)


class DeviceExecutor(Executor):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.closed = False

    def frame(self):
        return self.device.frame()

    def drive(self, command):
        super().drive(command)
        self.device.set_velocity(*command)

    def diagnostics(self):
        return {'error': self.error, 'solve_ms': 0, 'parameters': {}}

    def close(self):
        self.closed = True


class AlternativeModel:
    def __init__(self):
        self.instruction = ''
        self.url = ''
        self.sequence = 0
        self.results = []
        self.closed = False

    def set_server_url(self, url):
        self.url = url
        return url

    def start(self, instruction):
        self.instruction = instruction
        self.sequence = 0

    def offer_frame(self, stamp_ns, jpeg):
        if not self.instruction:
            return False
        self.sequence += 1
        self.results.append(Prediction(
            self.sequence, stamp_ns, ((0.3, 0, 0),), self.sequence >= 2, 1
        ))
        return True

    def take_results(self):
        results, self.results = self.results, []
        return results

    def snapshot(self):
        return VlnStatus(
            'RUNNING' if self.instruction else 'IDLE', True,
            self.url, self.instruction, self.sequence, '',
        )

    def stop(self):
        self.instruction = ''
        self.results = []

    def close(self):
        self.closed = True


class ServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_server_runtime_and_replacement_implementations(self):
        device, model = Device(), AlternativeModel()
        executor = DeviceExecutor(device)
        service = NavigationServer(robot=device, executor=executor, model=model)
        server = TestServer(service.app())
        await server.start_server()
        try:
            async with create_backend(load_backend(),
                base_url=str(server.make_url('/')), model_url='ws://alternative-model'
            ) as backend:
                image, observation = await backend.observe()
                self.assertTrue(image.startswith(b'\xff\xd8'))
                self.assertEqual(observation['pose']['x'], 1)
                result = await backend.navigate('Go to the doorway.', seconds=2)
                self.assertEqual(result['reason'], 'model_stop')
                self.assertFalse(result['arrival_verified'])
                self.assertTrue(executor.paths)
                self.assertEqual(executor.paths[0][1], (1, 2, 0))
                self.assertEqual(device.command, (0, 0))
                self.assertIsNotNone(service.manual_owner)
        finally:
            await server.close()
        self.assertTrue(executor.closed)
        self.assertTrue(model.closed)
        self.assertFalse(device.started)


class HistoryModel(AlternativeModel):
    supports_history_continuation = True

    def __init__(self):
        super().__init__()
        self.paused = False
        self.starts = 0
        self.reuses = 0
        self.request_stop = False

    def start(self, instruction):
        super().start(instruction)
        self.starts += 1
        self.paused = False

    def pause(self):
        self.paused = True
        self.results.clear()

    def resume(self, instruction):
        reused = self.paused and self.instruction == instruction
        if reused:
            self.reuses += 1
            self.paused = False
        else:
            self.start(instruction)
        return reused

    def offer_frame(self, stamp_ns, jpeg):
        if not self.instruction or self.paused:
            return False
        self.sequence += 1
        self.results.append(Prediction(self.sequence, stamp_ns, ((.3, 0., 0.),), self.request_stop, 1.))
        return True


class HistoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_observe_resume_stop_and_disconnect(self):
        import asyncio
        device, model = Device(), HistoryModel()
        service = NavigationServer(robot=device, executor=DeviceExecutor(device), model=model)
        server = TestServer(service.app())
        await server.start_server()
        try:
            async with create_backend(load_backend(), base_url=str(server.make_url('/')),
                                      model_url='ws://alternative-model') as backend:
                await backend.configure_history(True)
                async def moving():
                    task = asyncio.create_task(backend.navigate('Go to the doorway.', seconds=2))
                    async with asyncio.timeout(5):
                        while backend.active_navigation_id is None:
                            await asyncio.sleep(.01)
                    return task
                first = await moving()
                identity = backend.active_navigation_id
                self.assertTrue(backend.request_navigation_interrupt(identity, 'target_detected'))
                self.assertFalse(backend.request_navigation_interrupt(identity, 'selective_no_motion'))
                result = await first
                self.assertTrue(result['history_retained'])
                self.assertEqual(result['detail'], 'target_detected')
                self.assertEqual(device.command, (0., 0.))
                await backend.observe()
                self.assertTrue(model.paused)
                self.assertEqual(model.starts, 1)
                second = await moving()
                self.assertNotEqual(backend.active_navigation_id, identity)
                self.assertFalse(backend.request_navigation_interrupt(identity, 'stale'))
                model.request_stop = True
                result = await second
                self.assertEqual(result['reason'], 'model_stop')
                self.assertTrue(result['history_reused'])
                self.assertFalse(result['history_retained'])
                self.assertEqual(model.starts, 1)
                self.assertEqual(model.reuses, 1)
                self.assertEqual(model.instruction, '')
                model.request_stop = False
                third = await moving()
                backend.request_navigation_interrupt(backend.active_navigation_id, 'local_observation_distance')
                await third
                self.assertEqual(model.starts, 2)
                self.assertTrue(model.paused)
            async with asyncio.timeout(5):
                while model.instruction:
                    await asyncio.sleep(.01)
            self.assertIsNone(service.paused_owner)
            self.assertEqual(device.command, (0., 0.))
        finally:
            await server.close()
