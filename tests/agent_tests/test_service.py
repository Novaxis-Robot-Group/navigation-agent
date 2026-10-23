"""用临时文件测试 HTTP 和配置加载，不使用用户 Key。"""

import json

import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


class ServiceTests(unittest.TestCase):
    def check_service(self, file_key, file_model, overrides, configured, reload=False, legacy_saved=False):
        with tempfile.TemporaryDirectory(prefix='agent config ') as directory:
            root = Path(directory)
            service = root / 'src' / 'navigation_agent' / 'app'
            shutil.copytree(
                Path(__file__).resolve().parents[2] / 'src',
                root / 'src',
                ignore=shutil.ignore_patterns('.venv', '.env', '__pycache__', 'tests'),
            )
            shutil.copytree(Path(__file__).resolve().parents[2] / 'configs', root / 'configs')
            configuration = (
                f'DASHSCOPE_API_KEY={file_key}\n'
            )
            (service / '.env').write_text(configuration, encoding='utf-8')
            (root / 'deploy/agent').mkdir(parents=True)
            devserver = Path(__file__).resolve().parents[2] / 'deploy/agent/devserver.py'
            shutil.copyfile(devserver, root / 'deploy/agent/devserver.py')
            (root / 'deploy' / '.env').write_text(configuration, encoding='utf-8')

            # 当前工作目录的配置不能覆盖服务自身的配置。
            (root / '.env').write_text('DASHSCOPE_API_KEY=wrong-cwd\nDASHSCOPE_MODEL=wrong\n')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(('DASHSCOPE_', 'ZHIPU_', 'AGENT_', 'NAV_', 'PYTHON_DOTENV_'))
            }
            env['PYTHONPATH'] = str(root / 'src')
            backend_name = 'go2' if legacy_saved else 'mujoco'
            profile = root / f'configs/{backend_name}-lightnav.toml'
            import tomllib
            backend = tomllib.loads(profile.read_text())
            service_port = backend['service']['port']
            profile.write_text(profile.read_text().replace(f'port = {service_port}', f'port = {port}', 1))
            if legacy_saved:
                from navigation_agent.app.go2.motion_settings import MotionSettings

                # 使用真实旧文件格式走完整启动路径，设备地址只指向不可用的测试端口。
                robot_url = 'http://127.0.0.1:1'
                profile.write_text(profile.read_text().replace(backend['robot']['base_url'], robot_url))
                saved_file = root / '.runtime/go2-motion.json'
                saved_file.parent.mkdir()
                values = MotionSettings().model_dump()
                values['exploration']['max_looks'] = 6
                values['agent'].update(max_steps=17, max_seconds=123)
                values['look']['navigation_seconds'] = 23.
                values['mpc']['q_x'] = 73.
                saved_file.write_text(json.dumps({
                    'schema_version': 3, 'robot_url': robot_url, 'values': values,
                }))
                original_saved = saved_file.read_bytes()
            agent_file = root / 'configs/agent.toml'
            # 固定测试用的密钥变量，不依赖公共模板当前选择的供应商。
            text = agent_file.read_text()
            llm = tomllib.loads(text)['llm']
            text = text.replace('model = "' + llm['model'] + '"', 'model = "' + file_model + '"')
            text = text.replace('api_key_env = "' + llm['api_key_env'] + '"', 'api_key_env = "DASHSCOPE_API_KEY"')
            agent_file.write_text(text)
            env.update(overrides)
            startupinfo = None
            creationflags = 0
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
                creationflags = subprocess.CREATE_NO_WINDOW
            with (root / 'service.log').open('w+') as log:
                process = subprocess.Popen(
                    [sys.executable, '-m', 'navigation_agent.app.main', '--backend-config', str(profile)]
                    + (['--reload'] if reload else []),
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    startupinfo=startupinfo,
                )
                try:
                    opener = build_opener(ProxyHandler({}))
                    deadline = time.monotonic() + 15
                    while True:
                        if process.poll() is not None:
                            log.seek(0)
                            self.fail('Service exited before becoming ready: ' + log.read())
                        try:
                            with opener.open(
                                f'http://127.0.0.1:{port}/health', timeout=1
                            ) as response:
                                raw = response.read().decode()
                            break
                        except (URLError, TimeoutError):
                            if time.monotonic() >= deadline:
                                self.fail('Service did not become ready')
                            time.sleep(0.1)
                    self.assertEqual(
                        json.loads(raw), {'status': 'ok', 'llm_configured': configured}
                    )
                    self.assertNotIn('test-secret', raw)
                    with opener.open(f'http://127.0.0.1:{port}/settings/agent') as response:
                        initial_settings = json.loads(response.read())
                    if legacy_saved:
                        self.assertEqual(initial_settings['task_defaults'], {'max_steps': 17, 'max_seconds': 123})
                        self.assertEqual(initial_settings['navigation_seconds'], 23.)
                        self.assertEqual(saved_file.read_bytes(), original_saved)
                    agent_file.write_text(agent_file.read_text().replace('navigation_seconds = 15.0', 'navigation_seconds = 9.0'))
                    profile.write_text(profile.read_text().replace('max_steps = 30', 'max_steps = 7'))
                    with opener.open(f'http://127.0.0.1:{port}/settings/agent') as response:
                        self.assertEqual(json.loads(response.read()), initial_settings)
                    self.assertNotIn('test-secret', json.dumps(initial_settings))
                    if reload:
                        # 只修改测试副本的响应，用它验证热重载。
                        time.sleep(0.5)
                        source = service / 'agent' / 'api.py'
                        source.write_text(
                            source.read_text(encoding='utf-8').replace(
                                "'status': 'ok'", "'status': 'reloaded'"
                            ),
                            encoding='utf-8',
                        )
                        deadline = time.monotonic() + 15
                        while time.monotonic() < deadline:
                            try:
                                with opener.open(
                                    f'http://127.0.0.1:{port}/health', timeout=1
                                ) as response:
                                    if json.loads(response.read())['status'] == 'reloaded':
                                        break
                            except (URLError, TimeoutError, ConnectionError):
                                pass
                            time.sleep(0.1)
                        else:
                            log.seek(0)
                            self.fail(
                                'Python edit did not reload the running service: ' + log.read()
                            )
                finally:
                    if os.name == 'nt' and process.poll() is None:
                        # Windows 虚拟环境可能创建子进程，清理时一起结束。
                        subprocess.run(
                            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    elif process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                log.seek(0)
                self.assertNotIn('test-secret', log.read())

    def test_blank_key_from_foreign_working_directory(self):
        self.check_service('', 'test-model', {}, False)

    def test_service_dotenv_is_loaded(self):
        self.check_service('test-secret', 'test-model', {}, True)

    def test_system_environment_overrides_dotenv(self):
        self.check_service('test-secret', 'test-model', {'DASHSCOPE_API_KEY': ''}, False)

    def test_go2_starts_with_legacy_saved_observation_limit(self):
        self.check_service('', 'test-model', {}, False, legacy_saved=True)

    @unittest.skipUnless(importlib.util.find_spec('watchfiles'), 'Requires agent/requirements.txt')
    def test_reload_loads_python_edits_without_manual_restart(self):
        self.check_service('', 'test-model', {}, False, reload=True)


if __name__ == '__main__':
    unittest.main()
