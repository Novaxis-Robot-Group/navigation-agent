"""在一个前台终端管理本次启动的进程；退出不停止共享模型或其他会话。"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


class Session:
    def __init__(self):
        self.children = []

    def start(self, args):
        process = subprocess.Popen(args, cwd=ROOT, start_new_session=True)
        self.children.append(process)
        return process

    def check(self):
        for process in self.children:
            code = process.poll()
            if code is not None:
                raise RuntimeError("子进程已退出（状态 %s）：%s" % (code, process.args))

    def close(self):
        # 只终止本会话创建的进程组；从不扫描或匹配其他用户进程。
        for process in reversed(self.children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        for process in reversed(self.children):
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def require_free_port(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except ConnectionRefusedError:
        return
    except OSError as error:
        raise RuntimeError("不能确认 %s:%s 可用：%s" % (host, port, error))
    raise RuntimeError("%s:%s 已有服务；请使用已有页面，或先停止自己的旧服务。" % (host, port))


def local_endpoint(url, port):
    parsed = urlparse(url)
    return parsed.hostname in ("localhost", "127.0.0.1", "::1") and parsed.port == port


def service_url(service):
    host = service.get("host", "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    if ":" in host:
        host = "[" + host + "]"
    return "http://%s:%s" % (host, service["port"])


def wait_for_mujoco(session, agent, service, *, timeout=600):
    """只读等待 Agent、仿真和模型连接；不发任务、不调用模型推理。"""
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    pending = "Agent 服务"
    while time.monotonic() < deadline:
        session.check()
        try:
            with opener.open(service_url(service) + "/health", timeout=2) as response:
                health = json.load(response)
        except (OSError, ValueError):
            health = {}
        if isinstance(health, dict) and health.get("status") == "ok":
            if not health.get("llm_configured"):
                raise RuntimeError("Agent 缺少模型配置或密钥；请检查 deploy/.env 和 configs/agent.toml 后重新运行 make mujoco。")
            try:
                result = subprocess.run(
                    agent + ["--check"], capture_output=True, text=True,
                    timeout=min(35, max(0.1, deadline - time.monotonic())),
                )
                checks = json.loads(result.stdout)
                if not isinstance(checks, dict):
                    raise ValueError("连接检查未返回状态对象")
                ready = (
                    result.returncode == 0
                    and checks.get("robot_service") is True
                    and checks.get("model_transport") is True
                    and all(checks.values())
                )
                if ready:
                    session.check()
                    return
                pending = "、".join(name for name, value in checks.items() if not value) or "服务连接检查"
            except (subprocess.TimeoutExpired, ValueError):
                pending = "服务连接检查"
        else:
            pending = "Agent 服务"
        session.check()
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    raise RuntimeError("等待服务就绪超时（%s）；请查看上方服务日志，排查见 docs/zh/mujoco.md。" % pending)


def server(session, mode):
    try:
        import tomllib
    except ImportError:
        raise RuntimeError("主机启动器需要 Python 3.11+；可先 conda activate nav-agent。")
    config_path = Path(os.environ.get(mode.upper() + "_CONFIG", "configs/%s-lightnav.toml" % mode))
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    service = config["service"]
    require_free_port(service.get("host", "127.0.0.1"), service["port"])
    if mode == "mujoco":
        simulation = config["simulation"]
        require_free_port(simulation.get("host", "127.0.0.1"), simulation["port"])
    # 先验证配置，避免明显错误时启动模型容器。
    agent = ["bash", "deploy/agent/run.sh", "--backend-config", str(config_path)]
    try:
        subprocess.run(agent + ["--check-config"], check=True, stdout=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stderr)
        raise
    print("配置检查通过：%s" % config_path, flush=True)
    if os.environ.get("START_MODELS", "1") == "1":
        if local_endpoint(config["vln"]["server_url"], int(os.environ.get("LIGHTNAV_PORT", "8053"))):
            subprocess.run(["bash", "deploy/lightnav/run.sh"], check=True)
        detector_url = config.get("target_watch", {}).get("server_url")
        if detector_url is None:
            with (ROOT / "configs/agent.toml").open("rb") as stream:
                detector_url = tomllib.load(stream).get("target_watch", {}).get("server_url", "")
        if local_endpoint(detector_url, int(os.environ.get("DETECTOR_PORT", "18060"))):
            subprocess.run(["bash", "deploy/detector/run.sh",
                            os.environ.get("DETECTOR_MODELS", "runs/target-detector/models")], check=True)
    if mode == "mujoco":
        session.start(["bash", "deploy/sim/run.sh", "--backend-config", str(config_path)])
    session.start(agent)
    if mode == "mujoco":
        print("服务启动中，正在等待仿真和模型连接；首次加载模型可能需要数分钟。", flush=True)
        wait_for_mujoco(session, agent, service)
        print("MuJoCo 已就绪：" + service_url(simulation), flush=True)
        print("打开页面并输入目标，例如：找到棕色沙发。使用说明见 docs/zh/mujoco.md。", flush=True)
        print("取消任务并等待空闲后按 Ctrl+C；共享模型容器保持运行。", flush=True)
    else:
        print("服务启动中；连接结果请执行 make %s-check。Ctrl+C 结束本会话，模型容器保持运行。" % mode, flush=True)


def client(session):
    # 连接选项由用户配置；不保存密码、不修改 SSH 配置。
    host = os.environ.get("AGENT_HOST", "")
    if host:
        if host.startswith("-"):
            raise RuntimeError("AGENT_HOST 应为 user@host，不能是 SSH 选项。")
        require_free_port("127.0.0.1", int(os.environ.get("LIGHTNAV_PORT", "8053")))
        session.start([
            "ssh", "-N", "-T", "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-p", os.environ.get("AGENT_SSH_PORT", "22"),
            "-L", "127.0.0.1:%s:127.0.0.1:%s" % (
                os.environ.get("LIGHTNAV_PORT", "8053"), os.environ.get("LIGHTNAV_PORT", "8053")),
            "-R", "127.0.0.1:%s:127.0.0.1:8088" % os.environ.get("GO2_FORWARD_PORT", "18088"),
            host,
        ])
        time.sleep(1)
        session.check()
    else:
        print("未设置 AGENT_HOST：使用用户已有的直连网络或隧道。", flush=True)
    session.start(["bash", "deploy/go2/run.sh"])
    print("Go2 客户端启动中；取消任务并确认停稳后按 Ctrl+C 退出。", flush=True)


def main():
    session = Session()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        mode = sys.argv[1]
        if mode == "go2-client":
            client(session)
        else:
            server(session, mode)
        while True:
            session.check()
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        # 清理期间再次 Ctrl+C 不应留下子进程。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
