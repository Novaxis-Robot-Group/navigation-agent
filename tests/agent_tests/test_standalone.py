"""验证独立运行和工具边界，不需要设备或模型服务。"""

import asyncio
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError
from navigation_agent.agent.tools.registry import ToolCall, ToolRegistry
from navigation_agent.agent.tools.navigation import NavigateArguments
from navigation_agent.contracts.agent_backend import ToolError


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_arguments_do_not_execute_or_retry(self):
        execute = AsyncMock(return_value={'reason': 'model_stop'})
        registry = ToolRegistry()
        registry.register('navigate', NavigateArguments, execute)
        for parameters in ({'instruction': 'go', 'seconds': 0.0},
                           {'instruction': 'go', 'seconds': float('nan')},
                           {'instruction': 'go', 'seconds': 1.0, 'candidate_id': 'C1'}):
            with self.assertRaises(ValidationError):
                await registry.execute(ToolCall('navigate', parameters))
        execute.assert_not_awaited()
        with self.assertRaises(ToolError):
            await registry.execute(ToolCall('missing', {}))
        with self.assertRaises(ValueError):
            registry.register('navigate', NavigateArguments, execute)
        result = await registry.execute(ToolCall('navigate', {'instruction': 'go', 'seconds': 1.0}))
        self.assertEqual(result, {'reason': 'model_stop'})
        execute.assert_awaited_once_with(instruction='go', seconds=1.0)

    async def test_explicit_duration_cancels_continuous_motion_and_waits_for_cleanup(self):
        stopped = asyncio.Event()

        async def execute(**arguments):
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.set()

        registry = ToolRegistry()
        registry.register('navigate', NavigateArguments, execute)
        result = await registry.execute(ToolCall(
            'navigate', {'instruction': 'go', 'seconds': 3.0}, max_seconds=.01,
        ))
        self.assertEqual(result['reason'], 'timeout')
        self.assertTrue(stopped.is_set())
        self.assertEqual(result['time_limit_s'], .01)

    async def test_service_timeout_is_not_converted_to_an_explicit_duration_limit(self):
        registry = ToolRegistry()
        registry.register('navigate', NavigateArguments, AsyncMock(side_effect=TimeoutError('service')))
        with self.assertRaisesRegex(TimeoutError, 'service'):
            await registry.execute(ToolCall(
                'navigate', {'instruction': 'go', 'seconds': 3.0}, max_seconds=3.,
            ))

    async def test_motion_error_propagates_without_retry(self):
        execute = AsyncMock(side_effect=ToolError('lost control'))
        registry = ToolRegistry()
        registry.register('navigate', NavigateArguments, execute)
        with self.assertRaises(ToolError):
            await registry.execute(ToolCall('navigate', {'instruction': 'go', 'seconds': 1.0}))
        self.assertEqual(execute.await_count, 1)


class IndependentRunTests(unittest.TestCase):
    def test_demo_does_not_import_deployments_or_device_modules(self):
        source = Path(__file__).resolve().parents[2] / 'src'
        script = """
import importlib.abc, json, pathlib, sys, asyncio
sys.path.insert(0, sys.argv[1])
class RejectDevices(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('robots', 'navigation', 'vln', 'mujoco', 'rclpy', 'fastapi', 'uvicorn') or fullname.startswith('navigation_agent.app.deployments'):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, RejectDevices())
from navigation_agent.app.agent.settings import standalone_settings
from navigation_agent.app.agent.main import run_demo
record, directory = asyncio.run(run_demo(standalone_settings(output=sys.argv[2], api_key='')))
assert record['status'] == 'completed', record
assert (directory / 'index.html').is_file()
assert (directory / 'memory_visualize.html').is_file()
assert (directory / 'memory.sqlite3').is_file()
print(json.dumps({'status': record['status']}))
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-c', script, str(source), directory],
                                    cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
