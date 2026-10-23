"""部署入口离线回归：环境隔离、参数透传和重复启动保护。"""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='deploy with spaces ')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = self.root / 'project checkout'
        for service in ('agent', 'sim', 'detector'):
            shutil.copytree(ROOT / 'deploy' / service, self.project / 'deploy' / service)
        shutil.copyfile(ROOT / 'deploy/common.sh', self.project / 'deploy/common.sh')
        (self.project / 'deploy/.env.example').write_text('DASHSCOPE_API_KEY=\n')
        self.bin = self.root / 'tools'
        self.bin.mkdir()
        self.base = self.root / 'conda install'
        hook = self.base / 'etc/profile.d/conda.sh'
        hook.parent.mkdir(parents=True)
        hook.write_text("""
conda() {
    [[ "$1" == activate ]] || return 9
    [[ "$MISSING_ENV" == 0 ]] || return 1
    export TEST_ENV="$2"
    export PATH="$TEST_BIN:$PATH"
}
""")
        conda = self.bin / 'conda'
        conda.write_text('#!/bin/bash\n[[ "$1" == info && "$2" == --base ]] || exit 7\nprintf "%s\\n" "$TEST_BASE"\n')
        conda.chmod(0o755)
        python = self.bin / 'python'
        python.write_text('#!' + sys.executable + '\n' + """
import json, os, sys
print(json.dumps(dict(args=sys.argv[1:], env=os.environ.get('TEST_ENV'),
                     pythonpath=os.environ.get('PYTHONPATH'), cwd=os.getcwd(),
                     gl=os.environ.get('MUJOCO_GL'), pid=os.getpid())))
""")
        python.chmod(0o755)
        self.env = dict(os.environ, CONDA_EXE=str(conda), TEST_BASE=str(self.base), MISSING_ENV='0',
                        TEST_BIN=str(self.bin), PATH=str(self.bin) + os.pathsep + os.environ['PATH'])

    def run_script(self, service, name='run', *args, **environment):
        return subprocess.run(
            ['bash', str(self.project / f'deploy/{service}/{name}.sh'), *args],
            cwd=self.root, env=dict(self.env, **environment),
            text=True, capture_output=True, timeout=15,
        )

    def test_shell_syntax(self):
        for path in (ROOT / 'deploy').rglob('*.sh'):
            with self.subTest(path=path):
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_selects_conda_and_preserves_relative_arguments(self):
        result = self.run_script('agent', 'run', '--backend-config', 'config files/backend.toml', '--reload')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads(result.stdout)
        self.assertEqual(call['env'], 'nav-agent')
        self.assertEqual(call['cwd'], str(self.root))
        self.assertEqual(call['pythonpath'], str(self.project / 'src'))
        self.assertEqual(call['args'][:2], ['-m', 'navigation_agent.app.main'])
        self.assertNotIn(str(self.project / 'configs/mujoco-lightnav.toml'), call['args'])
        self.assertEqual(call['args'][-3:], ['--backend-config', 'config files/backend.toml', '--reload'])

    def test_mujoco_session_uses_agent_environment(self):
        result = self.run_script('agent', 'run', '--session', 'mujoco')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads(result.stdout)
        self.assertEqual(call['env'], 'nav-agent')
        self.assertEqual(call['args'], [str(self.project / 'deploy/session.py'), 'mujoco'])

    def test_sim_environment_and_renderer_selection(self):
        for renderer in ('egl', 'osmesa'):
            result = self.run_script('sim', 'run', '--backend-config', 'sim config.toml', MUJOCO_GL=renderer)
            self.assertEqual(result.returncode, 0, result.stderr)
            call = json.loads(result.stdout)
            self.assertEqual(call['env'], 'nav-sim')
            self.assertEqual(call['gl'], renderer)
            self.assertEqual(call['args'][:2], ['-m', 'navigation_agent.app.mujoco.main'])
            self.assertEqual(call['args'][-2:], ['--backend-config', 'sim config.toml'])

    def test_detector_port_mapping_and_override(self):
        models = self.root / 'model files'
        (models / 'clip').mkdir(parents=True)
        (models / 'yolov8l-worldv2.pt').touch()
        (models / 'clip/ViT-B-32.pt').touch()
        docker = self.bin / 'docker'
        docker.write_text('#!' + sys.executable + '\n' + """
import json, sys
if sys.argv[1:3] == ['container', 'inspect']:
    raise SystemExit(1)
print(json.dumps(sys.argv[1:]))
""")
        docker.chmod(0o755)
        for configured, expected in (('', '18060'), ('18061', '18061')):
            with self.subTest(port=configured):
                result = self.run_script('detector', 'run', str(models),
                                         DETECTOR_PORT=configured)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = json.loads(result.stdout)
                self.assertEqual(arguments[arguments.index('-p') + 1],
                                 f'127.0.0.1:{expected}:8060')
                self.assertIn(f'type=bind,source={models},target=/models,readonly', arguments)
                self.assertEqual(arguments[arguments.index('--port') + 1], '8060')

    def test_missing_environment_does_not_install_or_start(self):
        for service in ('agent', 'sim'):
            result = self.run_script(service, MISSING_ENV='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f'deploy/{service}/setup.sh', result.stderr)
            self.assertEqual(result.stdout, '')

    def test_setup_preserves_secret_file_and_does_not_launch(self):
        secret = self.project / 'deploy/.env'
        secret.write_text('DASHSCOPE_API_KEY=existing-test-value\n')
        for service in ('agent', 'sim'):
            result = self.run_script(service, 'setup')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('navigation_agent.app.main', result.stdout)
            self.assertNotIn('navigation_agent.app.mujoco.main', result.stdout)
            self.assertNotIn('existing-test-value', result.stdout)
        self.assertEqual(secret.read_text(), 'DASHSCOPE_API_KEY=existing-test-value\n')

    def test_run_executes_python_without_extra_launcher_process(self):
        process = subprocess.Popen(
            ['bash', str(self.project / 'deploy/agent/run.sh'), '--check-config'],
            env=self.env, cwd=self.root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout)['pid'], process.pid)

    def test_reload_worker_is_importable_for_spawn(self):
        from deploy.agent.devserver import serve_worker
        self.assertIs(pickle.loads(pickle.dumps(serve_worker)), serve_worker)

    def test_obsolete_entrypoints_are_removed(self):
        for name in ('windows', 'orin', 'server', 'install.py', 'setup_support.py',
                     'requirements-agent.txt', 'requirements-config.txt', 'devserver.py'):
            self.assertFalse((ROOT / 'deploy' / name).exists(), name)
