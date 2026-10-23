"""从构建产物验证独立交付，不访问实际模型或本体。"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]


class PackageTests(unittest.TestCase):
    def test_wheel_resources_and_offline_entry_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix='agent-package-') as directory:
            temporary = Path(directory)
            build = subprocess.run([
                sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                '--wheel-dir', str(temporary), str(ROOT),
            ], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            wheel = next(temporary.glob('*.whl'))
            installed = temporary / 'installed'
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertTrue(any(name.endswith("/LICENSE") for name in names))
                self.assertTrue(any(name.endswith("/NOTICE") for name in names))
                self.assertIn('navigation_agent/agent/memory/trajectory.py', names)
                self.assertIn('navigation_agent/agent/tools/target_watch.py', names)
                self.assertIn('navigation_agent/contracts/detection.py', names)
                self.assertIn('navigation_agent/app/detector/server.py', names)
                metadata = archive.read(next(name for name in names if name.endswith('/METADATA'))).decode()
                for line in metadata.splitlines():
                    if line.startswith('Requires-Dist:') and any(word in line.lower() for word in ('torch', 'ultralytics', 'clip ', 'lap')):
                        self.assertIn('extra == "detector"', line)
                self.assertFalse(any(name.endswith(('.pt', '.onnx')) for name in names))
                for module in ('memory/topology.py', 'model/observation.py', 'model/completion.py'):
                    self.assertIn('navigation_agent/agent/' + module, names)
                self.assertIn('navigation_agent/agent/skills/scan/skill.py', names)
                for module in ('registry.py', 'search_navigation.py', 'approach_target.py',
                               'object_search/context.py'):
                    self.assertIn('navigation_agent/agent/skills/' + module, names)
                self.assertNotIn('navigation_agent/agent/core/monitor.py', names)
                self.assertNotIn('navigation_agent/agent/skills/object_search/skill.py', names)
                self.assertIn('navigation_agent/agent/skills/object_search/target.py', names)
                self.assertNotIn('navigation_agent/agent/skills/object_search/session.py', names)
                self.assertFalse(any(name.startswith(('navigation_agent/robots/', 'navigation_agent/navigation/', 'navigation_agent/vln/', 'navigation_agent/app/deployments/')) for name in names))
                self.assertEqual(
                    [name for name in names if name.startswith('navigation_agent/app/') and name.count('/') == 2],
                    ['navigation_agent/app/__init__.py'],
                )
                for relative in ('agent.toml', 'prompts/common.md', 'prompts/go2-office.md', 'prompts/scheduling.md'):
                    self.assertEqual(archive.read('navigation_agent/app/agent/resources/' + relative),
                                     (ROOT / 'configs' / relative).read_bytes())
                archive.extractall(installed)
            script = """
import asyncio, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from navigation_agent import agent
from navigation_agent.app.agent.main import run_demo
from navigation_agent.app.agent.settings import standalone_settings
assert pathlib.Path(agent.__file__).is_relative_to(sys.argv[1])
record, directory = asyncio.run(run_demo(standalone_settings(output=sys.argv[2], api_key='')))
assert record['status'] == 'completed', record
assert (directory / 'index.html').is_file()
assert (directory / 'memory.sqlite3').is_file()
print(json.dumps({'status':record['status']}))
"""
            run = subprocess.run([sys.executable, '-I', '-c', script, str(installed), str(temporary / 'runs')],
                                 cwd=temporary, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
