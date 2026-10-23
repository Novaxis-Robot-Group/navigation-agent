"""导航工具执行边界与清理；不启用后台规划模型。"""

import asyncio
import time
import unittest

from navigation_agent.agent.tools import NavigationTools
from navigation_agent.contracts.agent_backend import ToolError
from navigation_agent.agent.tools.navigation import navigation_registry
from navigation_agent.agent.tools.registry import ToolCall


class Backend:
    supports_streaming_observation = True
    active_navigation_id = None

    def __init__(self):
        self.interrupts = []
        self.stopped = asyncio.Event()
        self.position = 2.
        self.pose_valid = True
        self.stale = False
        self.cancelled = False
        self.samples = 0
        self.duration = .05

    async def navigate(self, instruction, *, seconds):
        self.active_navigation_id = 'navigation'
        try:
            await asyncio.wait_for(self.stopped.wait(), self.duration)
            return {'reason': 'agent_interrupt', 'detail': self.interrupts[-1][1]}
        except TimeoutError:
            return {'reason': 'model_stop'}
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.active_navigation_id = None

    async def snapshot_observation(self):
        self.samples += 1
        return b'jpeg', {
            'camera_stamp_ns': time.time_ns() - (5_000_000_000 if self.stale else 0),
            'pose_valid': self.pose_valid, 'pose': {'x': self.position, 'y': 0.},
        }

    def request_navigation_interrupt(self, identity, reason):
        if identity != self.active_navigation_id:
            return False
        self.interrupts.append((identity, reason))
        self.stopped.set()
        return True

    async def look(self, direction):
        return {}

    async def orient(self, heading_deg):
        return {}

    async def recover(self):
        return {}


class NavigationExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def run_navigation(self, backend, *, bounded=True, origin=None):
        arguments = {'instruction': 'Go to the doorway.', 'seconds': 5.}
        if bounded:
            arguments.update(max_displacement_m=1.5, origin=origin or [0., 0.])
        return await navigation_registry(NavigationTools(backend)).execute(
            ToolCall('navigate', arguments),
        )

    async def test_viewpoint_budget_stops_motion_without_claiming_arrival(self):
        backend = Backend()
        result = await self.run_navigation(backend)
        self.assertEqual(result['reason'], 'agent_interrupt')
        self.assertEqual(backend.interrupts, [('navigation', 'local_observation_distance')])
        self.assertNotIn('arrival_verified', result)

    async def test_search_and_approach_do_not_sample_or_interrupt(self):
        backend = Backend()
        result = await self.run_navigation(backend, bounded=False)
        self.assertEqual(result['reason'], 'model_stop')
        self.assertEqual(backend.samples, 0)
        self.assertEqual(backend.interrupts, [])

    async def test_default_viewpoint_continues_past_old_distance_limit(self):
        from navigation_agent.agent.schemas import Decision
        from navigation_agent.agent.skills.object_search.context import ObjectSearchContext
        from tests.config_support import agent_settings
        from tests.agent_tests.test_loop import navigation
        from tests.agent_tests.test_target_tracking import view

        skill = ObjectSearchContext(agent_settings(), 70)
        skill.observe(view(1))
        decision = Decision.model_validate(navigation(kind='viewpoint'))
        selected = skill.accept(decision)
        call, _ = skill.prepare_call(decision, selected, 15.)
        backend = Backend()
        backend.position = 3.
        result = await navigation_registry(NavigationTools(backend)).execute(call)
        self.assertEqual(result['reason'], 'model_stop')
        self.assertNotIn('max_displacement_m', call.arguments)
        self.assertEqual(backend.interrupts, [])
        self.assertEqual(backend.samples, 0)

    async def test_invalid_or_stale_pose_stops_bounded_motion(self):
        for options in ({'pose_valid': False}, {'stale': True},
                        {'position': float('nan')}):
            with self.subTest(options=options):
                backend = Backend()
                for name, value in options.items():
                    setattr(backend, name, value)
                result = await self.run_navigation(backend)
                self.assertEqual(result['detail'], 'observation_unavailable')
                self.assertEqual(backend.interrupts, [('navigation', 'observation_unavailable')])

    async def test_unsupported_streaming_refuses_bounded_navigation(self):
        backend = Backend()
        backend.supports_streaming_observation = False
        with self.assertRaisesRegex(ToolError, '位移限制'):
            await self.run_navigation(backend)
        self.assertEqual(backend.samples, 0)

    async def test_capture_timeout_stops_bounded_motion(self):
        backend = Backend()
        async def unavailable():
            raise TimeoutError('camera busy')
        backend.snapshot_observation = unavailable
        result = await self.run_navigation(backend)
        self.assertEqual(result['detail'], 'observation_unavailable')

    async def test_rotation_without_displacement_keeps_navigation(self):
        backend = Backend()
        backend.position = 0.
        self.assertEqual((await self.run_navigation(backend))['reason'], 'model_stop')
        self.assertFalse(backend.interrupts)

    async def test_cancel_and_tool_timeout_wait_for_motion_and_sampler_cleanup(self):
        for bounded in (True, False):
            backend = Backend()
            backend.position = 0.
            backend.duration = 10.
            with self.assertRaises(TimeoutError):
                async with asyncio.timeout(.025):
                    await self.run_navigation(backend, bounded=bounded)
            self.assertTrue(backend.cancelled)
            self.assertIsNone(backend.active_navigation_id)
            samples = backend.samples
            await asyncio.sleep(.02)
            self.assertEqual(backend.samples, samples)

    async def test_short_reacquisition_budget_cancels_continuous_backend(self):
        backend = Backend()
        backend.duration = 10.
        result = await navigation_registry(NavigationTools(backend)).execute(ToolCall(
            'navigate', {'instruction': 'Go to the doorway.', 'seconds': 5.},
            max_seconds=.02,
        ))
        self.assertEqual(result['reason'], 'timeout')
        self.assertTrue(backend.cancelled)
        self.assertIsNone(backend.active_navigation_id)
