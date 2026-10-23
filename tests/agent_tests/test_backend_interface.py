from navigation_agent.agent.skills.object_search.context import ObjectSearchContext
"""用不含仿真协议的后端，验证 Agent 只依赖动作契约。"""

import ast
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from navigation_agent.contracts.agent_backend import make_observation
from tests.support import clear_navigation_check
from navigation_agent.agent.core import loop
from navigation_agent.agent.schemas import Decision, TaskRequest
from navigation_agent.agent.tools import NavigationTools
from tests.config_support import test_runtime
from tests.support import patch_runs


class FakeBackend:
    look_angle_deg = 70
    def __init__(self):
        self.telemetry = None
        self.actions = []
        self.closed = False
        self.waiting = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def require_control(self):
        pass

    async def observe(self):
        self.actions.append('observe')
        if self.waiting is not None:
            await self.waiting.wait()
        return b'\xff\xd8test', make_observation(
            pose={'x': 0, 'y': 0, 'yaw': 0}, blocked=False, camera_stamp_ns=10
        )

    async def navigate(self, instruction, seconds=15):
        self.actions.append('navigate')
        return {'reason': 'blocked', 'motion': {'displacement_m': 0, 'blocked_samples': 2}}

    async def recover(self):
        self.actions.append('recover')
        return {'reason': 'recovered', 'displacement_m': 0.5}

    async def look(self, direction):
        self.actions.append('look')
        return {'reason': 'turned', 'direction': direction}

    async def hold(self):
        self.actions.append('hold')


class BackendInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_loop_accepts_backend_without_simulator_state(self):
        backend = FakeBackend()
        decisions = [
            (
                Decision(
                    action='navigate', navigation_check=clear_navigation_check(),
                    reason='接近已见目标', scene={'target_seen': 'yes'},
                    instruction='接近门口',
                    candidate_id='new_1',
                    candidates=[
                        {
                            'id': 'new_1',
                            'kind': 'target',
                            'description': '门口',
                            'evidence': '可见地面',
                            'bearing_deg': 0,
                        }
                    ],
                ),
                {},
            ),
            (Decision(action='finish', reason='找到', evidence='新画面中看到了目标',
                      target_review={'observation_step': 3, 'status': 'arrived', 'evidence': '当前图已确认目标'}), {}),
        ]
        with tempfile.TemporaryDirectory() as directory, patch_runs(Path(directory)):
            record = {'id': 'c' * 32, 'status': 'running'}
            (Path(directory) / record['id'] / 'images').mkdir(parents=True)
            with patch.object(loop, 'decide', AsyncMock(side_effect=decisions)) as decide:
                await loop.execute(
                    record, TaskRequest(goal='找目标', max_steps=3, max_seconds=300, navigation_seconds=15.), NavigationTools(backend),
                    settings=test_runtime(directory).agent, api_key='',
                    search=ObjectSearchContext(
                        test_runtime(directory).agent, backend.look_angle_deg)
                )
            self.assertEqual(decide.await_count, 2)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(backend.actions, ['observe', 'navigate', 'observe', 'recover', 'observe'])
        self.assertTrue(backend.closed)

    async def test_cancel_observation_closes_injected_backend(self):
        backend = FakeBackend()
        backend.waiting = asyncio.Event()

        async def observe():
            async with NavigationTools(backend) as tools:
                await tools.observe()

        task = asyncio.create_task(observe())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(backend.closed)

    def test_missing_or_invalid_pose_is_explicit(self):
        for pose in ({}, {'x': float('nan'), 'y': 0, 'yaw': 0}, {'x': 0, 'y': 0}):
            observation = make_observation(pose=pose, blocked=False, camera_stamp_ns=None)
            self.assertFalse(observation['pose_valid'])
        valid = make_observation(pose={'x': 1, 'y': 2, 'yaw': 0}, blocked=True, camera_stamp_ns=10)
        self.assertTrue(valid['pose_valid'])
        self.assertEqual(valid['pose'], {'x': 1, 'y': 2, 'yaw': 0})

    def test_agent_does_not_import_or_read_mujoco_protocol(self):
        import navigation_agent.agent as agent

        for path in Path(agent.__file__).parent.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(
                        node.module,
                        ('navigation_agent.robots.mujoco.robot', 'navigation_agent.robots.mujoco.recovery', 'navigation_agent.app.deployments.factory'),
                    )
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ('_command', '_snapshot', 'socket'))
