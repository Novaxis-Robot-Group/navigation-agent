"""启动组合离线验证：配置保留、容器归属与当前会话清理。"""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_session", ROOT / "deploy/session.py")
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


class DeployFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="make checkout ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("common.sh", "manage.sh", "local.env.example", ".env.example"):
            destination = self.root / "deploy" / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / "deploy" / name, destination)
        for name in ("lightnav", "detector", "go2"):
            shutil.copytree(ROOT / "deploy" / name, self.root / "deploy" / name)
        shutil.copyfile(ROOT / "Makefile", self.root / "Makefile")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls.jsonl"
        docker = self.bin / "docker"
        docker.write_text("#!" + sys.executable + "\n" + """
import json, os, sys
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["container", "inspect"]:
    raise SystemExit(0 if os.environ.get("EXISTS") == "1" else 1)
if args[:1] == ["inspect"]:
    if ".State.Running" in " ".join(args):
        print(os.environ.get("RUNNING", "true"))
    else:
        print(os.environ.get("LABELS", ""))
elif args[:1] == ["run"]:
    print("test-container")
""")
        docker.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        CALLS=str(self.calls), REUSE_EXISTING="0", EXISTS="0")
        self.owner = subprocess.check_output(["id", "-un"], text=True).strip()
        self.env.pop("VLLM_BUILD_JOBS", None)
        self.env.pop("LIGHTNAV_SOURCE", None)
        self.env.pop("LIGHTNAV_WEIGHTS", None)
        self.env.pop("LIGHTNAV_PATCH", None)

    def run_script(self, name, *args, **env):
        return subprocess.run(["bash", str(self.root / "deploy" / name), *args],
                              cwd=self.root, env=dict(self.env, **env), text=True,
                              capture_output=True, timeout=10)

    def commands(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]


class MakeTests(DeployFixture):
    def test_configure_preserves_local_options_and_secrets(self):
        (self.root / "deploy/local.env").write_text('export NETWORK_INTERFACE="custom0"\n')
        (self.root / "deploy/.env").write_text("DASHSCOPE_API_KEY=test-existing\n")
        result = self.run_script("manage.sh", "configure")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom0", (self.root / "deploy/local.env").read_text())
        self.assertEqual((self.root / "deploy/.env").read_text(), "DASHSCOPE_API_KEY=test-existing\n")
        self.assertEqual(self.commands(), [])

    def test_existing_model_containers_do_not_require_build_files_or_downloads(self):
        for script, label in (("lightnav/run.sh", "lightnav"), ("detector/run.sh", "target-detector")):
            with self.subTest(script=script):
                result = self.run_script(script, EXISTS="1", LABELS=label + "|" + self.owner, RUNNING="false")
                self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.commands()
        self.assertEqual(len([args for args in calls if args[0] == "start"]), 2)
        self.assertFalse(any(args[0] in ("run", "build", "pull", "stop", "rm") for args in calls))

    def test_running_container_is_not_restarted(self):
        result = self.run_script("lightnav/run.sh", EXISTS="1", LABELS="lightnav|" + self.owner, RUNNING="true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(args[0] in ("start", "restart", "stop") for args in self.commands()))

    def test_unknown_container_requires_explicit_choice(self):
        result = self.run_script("lightnav/run.sh", EXISTS="1", LABELS="other|other-user")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REUSE_EXISTING", result.stderr)
        self.assertFalse(any(args[0] in ("start", "stop", "run") for args in self.commands()))

    def test_missing_or_wrong_labels_never_start_container(self):
        for labels in ("lightnav|", "lightnav|<no value>", "|" + self.owner,
                       "other|" + self.owner, "lightnav|other-user"):
            with self.subTest(labels=labels):
                result = self.run_script("lightnav/run.sh", EXISTS="1", LABELS=labels, RUNNING="false")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("REUSE_EXISTING", result.stderr)
        self.assertFalse(any(args[0] in ("start", "stop", "run", "rm") for args in self.commands()))

    def test_legacy_container_requires_explicit_reuse(self):
        result = self.run_script("lightnav/run.sh", EXISTS="1", LABELS="<no value>|<no value>",
                                 RUNNING="false", REUSE_EXISTING="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([args for args in self.commands() if args[0] == "start"]), 1)
        self.assertFalse(any(args[0] in ("run", "build", "pull", "stop", "rm") for args in self.commands()))

    def test_new_go2_container_requires_explicit_interface(self):
        result = self.run_script("go2/run.sh", NETWORK_INTERFACE="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NETWORK_INTERFACE", result.stderr)
        self.assertFalse(any(args[0] == "run" for args in self.commands()))

    def test_make_help_is_default_and_launch_targets_are_distinct(self):
        result = subprocess.run(["make"], cwd=self.root, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("make go2-client", result.stdout)
        self.assertEqual(self.commands(), [])
        for target in ("go2", "mujoco", "go2-client", "models"):
            result = subprocess.run(["make", "-n", target], cwd=self.root, env=self.env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "bash deploy/manage.sh " + target)

    def test_lightnav_missing_assets_never_trigger_download(self):
        result = self.run_script("lightnav/run.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("models-setup", result.stderr)
        self.assertFalse(any(args[0] in ("run", "build", "pull") for args in self.commands()))

    def test_lightnav_fresh_source_has_nested_mountpoints(self):
        source = self.root / ".runtime/lightnav/upstream"
        patched = self.root / ".runtime/lightnav/capture-time"
        for root in (source, patched):
            for name in ("inference/policies.py", "serving/ws_server.py", "serving/tracking_service.py",
                         "data_processor.py", "processing.py"):
                path = root / "src/lightnav" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        weights = source / "checkpoints/LightNav-0"
        weights.mkdir(parents=True)
        (weights / "config.json").write_text("{}")
        for name in ("prepare.py", "check_weights.py"):
            (self.root / "deploy/lightnav" / name).write_text("pass\n")
        result = self.run_script("lightnav/run.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((source / ".cache").is_dir())
        self.assertTrue((source / "output").is_dir())
        self.assertEqual(len([call for call in self.commands() if call[0] == "run"]), 1)

    def test_same_service_label_does_not_override_other_owner(self):
        result = self.run_script("lightnav/run.sh", EXISTS="1", LABELS="lightnav|other-user", RUNNING="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(args[0] == "start" for args in self.commands()))

    def test_make_session_interrupt_closes_only_its_sim_and_agent(self):
        self.check_make_session_exit(signal.SIGINT)

    def test_closed_terminal_closes_its_services(self):
        self.check_make_session_exit(signal.SIGHUP)

    def check_make_session_exit(self, exit_signal):
        shutil.copyfile(ROOT / "deploy/session.py", self.root / "deploy/session.py")
        (self.root / "configs").mkdir()
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            second.bind(("127.0.0.1", 0))
            agent_port, sim_port = first.getsockname()[1], second.getsockname()[1]
        (self.root / "configs/mujoco-lightnav.toml").write_text(
            '[service]\nhost="127.0.0.1"\nport=%d\n[simulation]\nhost="127.0.0.1"\nport=%d\n[vln]\nserver_url="ws://model-host:8053"\n'
            % (agent_port, sim_port))
        for name in ("agent", "sim"):
            path = self.root / "deploy" / name / "run.sh"
            path.parent.mkdir(exist_ok=True)
            code = 'import os,time; from pathlib import Path; Path("%s.pid").write_text(str(os.getpid())); time.sleep(30)' % name
            path.write_text('#!/bin/bash\nif [[ "$1" == --session ]]; then exec "$TEST_PYTHON" deploy/session.py "$2"; fi\n[[ " $* " != *" --check-config "* ]] || exit 0\n'
                            + 'exec "$TEST_PYTHON" -c ' + shlex.quote(code) + '\n')
        process = subprocess.Popen(["make", "mujoco"], cwd=self.root,
                                   env=dict(self.env, PYTHON=sys.executable, TEST_PYTHON=sys.executable, START_MODELS="0"),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not all((self.root / (name + ".pid")).exists() for name in ("agent", "sim")):
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail("前台会话未启动两个服务：" + str(process.poll()) + (process.communicate(timeout=2)[1] if process.poll() is not None else ""))
                time.sleep(0.05)
            children = [int((self.root / (name + ".pid")).read_text()) for name in ("agent", "sim")]
            os.killpg(process.pid, exit_signal)
            process.communicate(timeout=10)
            for pid in children:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.assertEqual(self.commands(), [])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=10)


class HostEnvironmentTests(DeployFixture):
    """使用假的 Conda 激活脚本，验证入口选择，不启动任何服务。"""

    def setUp(self):
        super().setUp()
        shutil.copytree(ROOT / "deploy/agent", self.root / "deploy/agent")
        (self.root / "deploy/session.py").write_text(
            'import os, sys; print(os.environ.get("CONDA_DEFAULT_ENV", "system"), sys.argv[1])\n')
        conda_base = self.root / "fake conda"
        profile = conda_base / "etc/profile.d/conda.sh"
        profile.parent.mkdir(parents=True)
        env_bin = conda_base / "envs/nav-agent/bin"
        env_bin.mkdir(parents=True)
        (env_bin / "python").symlink_to(sys.executable)
        profile.write_text('conda() { [[ "$1" == activate && "$2" == nav-agent ]] || return 1; '
                           'export CONDA_DEFAULT_ENV=nav-agent; export PATH=' + shlex.quote(str(env_bin)) + ':"$PATH"; }\n')
        conda = self.bin / "conda"
        conda.write_text('#!/bin/bash\n[[ "$1" == info && "$2" == --base ]] || exit 9\nprintf "%s\\n" '
                         + shlex.quote(str(conda_base)) + '\n')
        conda.chmod(0o755)
        self.env.update(CONDA_EXE=str(conda), PYTHON=str(self.bin / "python3"))
        self.env.pop("CONDA_DEFAULT_ENV", None)
        # 系统 Python 不可用于主机启动；激活后必须使用环境中的 python。
        for name in ("python", "python3"):
            path = self.bin / name
            path.write_text('#!/bin/bash\necho wrong-system-python >&2\nexit 97\n')
            path.chmod(0o755)

    def test_host_launchers_select_nav_agent_without_manual_activation(self):
        for target in ("go2", "mujoco"):
            with self.subTest(target=target):
                result = subprocess.run(["make", target], cwd=self.root, env=self.env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("nav-agent " + target, result.stdout)
        self.assertEqual(self.commands(), [])

    def test_robot_client_does_not_require_conda(self):
        self.env.pop("PYTHON")
        self.env["CONDA_EXE"] = str(self.root / "missing-conda")
        python = self.bin / "python3"
        python.write_text('#!/bin/bash\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        result = subprocess.run(["make", "go2-client"], cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("system go2-client", result.stdout)
        self.assertEqual(self.commands(), [])


class LightNavBuildTests(DeployFixture):
    """Docker 命令替身只记录参数，不构建镜像。"""

    def setUp(self):
        super().setUp()
        source = self.root / "source"
        source.mkdir()
        (source / "pyproject.toml").touch()
        self.env.update(LIGHTNAV_SOURCE=str(source), LIGHTNAV_PLATFORM="thor")

    def test_thor_default_and_local_override_reach_docker(self):
        result = self.run_script("lightnav/build.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VLLM_BUILD_JOBS=2", self.commands()[-1])
        (self.root / "deploy/local.env").write_text('export VLLM_BUILD_JOBS="${VLLM_BUILD_JOBS:-3}"\n')
        result = self.run_script("lightnav/build.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VLLM_BUILD_JOBS=3", self.commands()[-1])
        result = self.run_script("lightnav/build.sh", VLLM_BUILD_JOBS="4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VLLM_BUILD_JOBS=4", self.commands()[-1])

    def test_invalid_parallelism_fails_before_docker(self):
        for jobs in ("0", "-1", "1.5", "two", "2 3"):
            with self.subTest(jobs=jobs):
                result = self.run_script("lightnav/build.sh", VLLM_BUILD_JOBS=jobs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("VLLM_BUILD_JOBS", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_upstream_build_does_not_receive_thor_options(self):
        result = self.run_script("lightnav/build.sh", LIGHTNAV_PLATFORM="upstream", VLLM_BUILD_JOBS="4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("VLLM_BUILD_JOBS" in arg for arg in self.commands()[-1]))


class LightNavSetupTests(DeployFixture):
    """首次下载失败可重试，已有源码不会被安装器覆盖。"""

    def setUp(self):
        super().setUp()
        remote = self.root / "public upstream"
        remote.mkdir()
        (remote / "pyproject.toml").write_text('[project]\nname="test-lightnav"\nversion="0.1"\n')
        self.git = shutil.which("git")
        for args in (("init",), ("add", "pyproject.toml"),
                     ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")):
            subprocess.run([self.git, "-C", str(remote), *args], check=True, capture_output=True)
        self.revision = subprocess.check_output([self.git, "-C", str(remote), "rev-parse", "HEAD"], text=True).strip()
        manifest = self.root / "deploy/lightnav/sources.json"
        data = json.loads(manifest.read_text())
        data["revision"] = self.revision
        manifest.write_text(json.dumps(data))
        # 此测试仅覆盖安装事务，补丁和权重由实际公开产物验收验证。
        (self.root / "deploy/lightnav/prepare.py").write_text('print("fixture patch checked")\n')
        (self.root / "deploy/lightnav/check_weights.py").write_text('print("fixture weights checked")\n')
        wrapper = self.bin / "git"
        wrapper.write_text("#!" + sys.executable + "\n" + """
import os, sys
args = sys.argv[1:]
if "fetch" in args and os.environ.get("FAIL_FETCH") == "1":
    print("fixture download failed", file=sys.stderr)
    raise SystemExit(128)
if "remote" in args and "add" in args:
    args[-1] = os.environ["TEST_REMOTE"]
os.execv(os.environ["TEST_GIT"], [os.environ["TEST_GIT"], *args])
""")
        wrapper.chmod(0o755)
        self.env.update(TEST_REMOTE=str(remote), TEST_GIT=self.git)
        self.source = self.root / ".runtime/lightnav/upstream"

    def test_failed_fetch_leaves_no_partial_source_and_retry_succeeds(self):
        failed = self.run_script("lightnav/setup.sh", FAIL_FETCH="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(self.source.exists())
        self.assertEqual(list(self.source.parent.glob("upstream.download.*")), [])
        self.assertEqual(self.commands(), [])
        retry = self.run_script("lightnav/setup.sh", FAIL_FETCH="0")
        self.assertEqual(retry.returncode, 0, retry.stderr)
        revision = subprocess.check_output([self.git, "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(revision, self.revision)
        self.assertTrue((self.source / "pyproject.toml").is_file())
        self.assertEqual(list(self.source.parent.glob("upstream.download.*")), [])

    def test_existing_incomplete_source_is_preserved(self):
        self.source.mkdir(parents=True)
        marker = self.source / "keep.txt"
        marker.write_text("existing user data")
        result = self.run_script("lightnav/setup.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("源码目录不完整", result.stderr)
        self.assertEqual(marker.read_text(), "existing user data")
        self.assertFalse((self.source / ".git").exists())
        self.assertEqual(self.commands(), [])


class SessionTests(unittest.TestCase):
    def test_busy_port_rejected(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "已有服务"):
                session.require_free_port("127.0.0.1", listener.getsockname()[1])

    def test_close_only_terminates_own_children(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        owned = session.Session()
        try:
            child = owned.start([sys.executable, "-c", "import time; time.sleep(30)"])
            owned.close()
            self.assertIsNotNone(child.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait()
            owned.close()

    def test_child_failure_is_not_reported_as_success(self):
        owned = session.Session()
        process = owned.start([sys.executable, "-c", "raise SystemExit(7)"])
        process.wait()
        with self.assertRaisesRegex(RuntimeError, "状态 7"):
            owned.check()
        owned.close()

    def test_remote_model_endpoints_are_not_local(self):
        self.assertTrue(session.local_endpoint("ws://127.0.0.1:8053", 8053))
        self.assertFalse(session.local_endpoint("ws://model-host:8053", 8053))
        self.assertFalse(session.local_endpoint("", 18060))

    def test_go2_config_stays_go2_and_disabled_yolo_is_not_started(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            config = root / "configs/go2-lightnav.toml"
            original = '[service]\nhost="127.0.0.1"\nport=8090\n[vln]\nserver_url="ws://127.0.0.1:8053"\n[target_watch]\nserver_url=""\n'
            config.write_text(original)
            owned = mock.Mock()
            with mock.patch.object(session, "ROOT", root), mock.patch.dict(os.environ, {"GO2_CONFIG": str(config), "START_MODELS": "1"}), mock.patch.object(session, "require_free_port"), mock.patch.object(session.subprocess, "run") as run:
                session.server(owned, "go2")
            self.assertEqual(config.read_text(), original)
            calls = [call.args[0] for call in run.call_args_list]
            self.assertEqual(calls, [
                ["bash", "deploy/agent/run.sh", "--backend-config", str(config), "--check-config"],
                ["bash", "deploy/lightnav/run.sh"],
            ])
            owned.start.assert_called_once_with(["bash", "deploy/agent/run.sh", "--backend-config", str(config)])


class BootstrapTests(DeployFixture):
    """隔离所有安装命令，验证首次准备不会下载、启动或覆写配置。"""

    def setUp(self):
        super().setUp()
        (self.root / "deploy/sim").mkdir()
        shutil.copyfile(ROOT / "deploy/sim/bootstrap.sh", self.root / "deploy/sim/bootstrap.sh")
        self.steps = self.root / "steps"
        self.ready = self.root / "recording-ready"
        for name in ("agent", "sim", "lightnav"):
            path = self.root / "deploy" / name / "setup.sh"
            path.parent.mkdir(exist_ok=True)
            path.write_text('echo "' + name + '" >> "$STEPS"\n[[ "$FAIL_STEP" != "' + name + '" ]]')
        commands = {
            "conda": 'exit 0',
            "gst-launch-1.0": 'exit 0',
            "gst-inspect-1.0": '[[ -f "$RECORDING_READY" ]]',
            "id": 'echo 1000',
            "sudo": 'echo "sudo $*" >> "$STEPS"; exec "$@"',
            "apt-get": '[[ "$FAIL_APT" != 1 ]] || exit 8; [[ "$1" != install ]] || touch "$RECORDING_READY"',
        }
        for name, body in commands.items():
            path = self.bin / name
            path.write_text('#!/bin/bash\n' + body + '\n')
            path.chmod(0o755)
        self.env.update(CONDA_EXE=str(self.bin / "conda"), STEPS=str(self.steps),
                        RECORDING_READY=str(self.ready), START_MODELS="1", FAIL_STEP="", FAIL_APT="0")

    def test_full_prepare_installs_missing_plugins_then_environments_and_models(self):
        secret = self.root / "deploy/.env"
        secret.write_text('DASHSCOPE_API_KEY=keep-existing\n')
        result = subprocess.run(["make", "mujoco-setup"], cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        steps = self.steps.read_text().splitlines()
        self.assertEqual(steps[0], "sudo apt-get update")
        self.assertIn("gstreamer1.0-plugins-ugly", steps[1])
        self.assertEqual(steps[2:], ["agent", "sim", "lightnav"])
        self.assertEqual(secret.read_text(), 'DASHSCOPE_API_KEY=keep-existing\n')
        self.assertNotIn("keep-existing", result.stdout)
        self.assertIn("MuJoCo 准备完成", result.stdout)
        self.assertEqual(self.commands(), [["info"]])

    def test_external_models_skip_docker_and_model_preparation(self):
        self.ready.touch()
        result = self.run_script("sim/bootstrap.sh", START_MODELS="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.steps.read_text().splitlines(), ["agent", "sim"])
        self.assertEqual(self.commands(), [])

    def test_failed_system_install_does_not_continue(self):
        result = self.run_script("sim/bootstrap.sh", FAIL_APT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.steps.read_text().splitlines(), ["sudo apt-get update"])
        self.assertNotIn("准备完成", result.stdout)

    def test_failed_environment_does_not_prepare_models(self):
        self.ready.touch()
        result = self.run_script("sim/bootstrap.sh", FAIL_STEP="sim")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.steps.read_text().splitlines(), ["agent", "sim"])
        self.assertNotIn("准备完成", result.stdout)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.owned = mock.Mock()
        self.agent = ["bash", "deploy/agent/run.sh", "--backend-config", "custom.toml"]
        self.service = {"host": "0.0.0.0", "port": 8092}
        self.clock = 0
        self.monotonic = mock.patch.object(session.time, "monotonic", side_effect=lambda: self.clock)
        self.sleep = mock.patch.object(session.time, "sleep", side_effect=self.advance)
        self.monotonic.start()
        self.sleep.start()
        self.addCleanup(self.monotonic.stop)
        self.addCleanup(self.sleep.stop)
        self.opener_patch = mock.patch.object(session, "build_opener")
        self.opener = self.opener_patch.start().return_value
        self.addCleanup(self.opener_patch.stop)
        self.health({"status": "ok", "llm_configured": True})

    def advance(self, seconds):
        self.clock += seconds

    def health(self, value):
        self.opener.open.side_effect = lambda *args, **kwargs: io.StringIO(json.dumps(value))

    def test_retries_pending_services_then_reports_ready(self):
        failed = subprocess.CompletedProcess([], 1, '{"robot_service": true, "model_transport": false}')
        success = subprocess.CompletedProcess([], 0, '{"robot_service": true, "model_transport": true, "target_detector": true}')
        with mock.patch.object(session.subprocess, "run", side_effect=[failed, success]) as run:
            session.wait_for_mujoco(self.owned, self.agent, self.service, timeout=5)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0], self.agent + ["--check"])
        self.assertEqual(self.opener.open.call_args.args[0], "http://127.0.0.1:8092/health")
        self.owned.check.assert_called()

    def test_missing_key_fails_without_model_probe(self):
        self.health({"status": "ok", "llm_configured": False})
        with mock.patch.object(session.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "deploy/.env"):
                session.wait_for_mujoco(self.owned, self.agent, self.service)
        run.assert_not_called()

    def test_timeout_names_pending_detector(self):
        result = subprocess.CompletedProcess([], 1, '{"robot_service": true, "model_transport": true, "target_detector": false}')
        with mock.patch.object(session.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "target_detector"):
                session.wait_for_mujoco(self.owned, self.agent, self.service, timeout=3)

    def test_unreachable_agent_times_out_without_running_probe(self):
        self.opener.open.side_effect = OSError("not listening")
        with mock.patch.object(session.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Agent 服务"):
                session.wait_for_mujoco(self.owned, self.agent, self.service, timeout=3)
        run.assert_not_called()

    def test_invalid_probe_output_never_reports_ready(self):
        for output in ("not json", "[]", "{}"):
            with self.subTest(output=output):
                self.clock = 0
                result = subprocess.CompletedProcess([], 0, output)
                with mock.patch.object(session.subprocess, "run", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, "超时"):
                        session.wait_for_mujoco(self.owned, self.agent, self.service, timeout=3)

    def test_child_exit_stops_waiting_immediately(self):
        self.owned.check.side_effect = RuntimeError("子进程已退出")
        with self.assertRaisesRegex(RuntimeError, "子进程已退出"):
            session.wait_for_mujoco(self.owned, self.agent, self.service)
        self.opener.open.assert_not_called()

    def test_page_url_uses_configured_host_and_port(self):
        self.assertEqual(session.service_url({"host": "::", "port": 9000}), "http://[::1]:9000")
        self.assertEqual(session.service_url({"host": "localhost", "port": 9001}), "http://localhost:9001")
