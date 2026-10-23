"""在日志位姿复测真实 LightNav 闭环；运行前停止日常仿真以节省内存。"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import mujoco
from aiohttp import web
from vln_mujoco.mpc import MpcTracker
from vln_mujoco.simulation import Simulation
from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.mujoco.console import ConsoleServer
from navigation_agent.app.deployments.settings import load_backend
from navigation_agent.robots.mujoco.mujoco_safe import SafeTurtleBot


async def replay(args):
    settings = load_backend(args.backend_config)
    if settings.robot.backend != 'mujoco' or settings.simulation is None:
        raise ValueError('复测需要 MuJoCo 后端配置')
    model_url = args.model_url or settings.vln.server_url
    # 完整场景占用较多提交内存，不与日常仿真实例同时加载。
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8088)) == 0:
            raise RuntimeError('请先停止自己启动的 8088 仿真；已配置的 LightNav 服务保持运行')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8098))

    robot = SafeTurtleBot(camera_height_m=settings.simulation.camera_height_m)
    x, y, yaw = args.pose
    offset = robot._base_qpos
    robot.data.qpos[offset:offset + 2] = [x, y]
    robot._yaw = yaw
    robot.data.qpos[offset + 3:offset + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(robot.model, robot.data)
    if robot._collision():
        raise ValueError('复现起点已与障碍重叠')
    simulation = Simulation(robot)
    server = ConsoleServer(
        simulation=simulation, agent_url=settings.simulation.agent_url,
        vln_settings=settings.vln.model_copy(update={'server_url': model_url}),
    )
    if args.official_tracker:
        server.mpc.close()
        server.mpc = MpcTracker()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    runner = web.AppRunner(server.app())
    try:
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 8098).start()
        async with asyncio.timeout(20):
            while simulation.frame() is None:
                await asyncio.sleep(0.1)
        (output / 'initial.jpg').write_bytes(simulation.frame().jpeg)
        async with create_backend(settings, base_url='http://127.0.0.1:8098', model_url=model_url) as backend:
            backend.telemetry = samples.append
            result = await backend.navigate(args.instruction, seconds=args.seconds)
        (output / 'final.jpg').write_bytes(simulation.frame().jpeg)
        report = {
            'instruction': args.instruction,
            'initial_pose': args.pose,
            'controller': 'official_mpc' if args.official_tracker else 'local_navigation',
            'result': result,
            'final_pose': robot.state().pose,
            'blocked_steps': robot.blocked_steps,
        }
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    finally:
        (output / 'telemetry.jsonl').write_text(
            '\n'.join(json.dumps(sample) for sample in samples), encoding='utf-8'
        )
        await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', nargs=3, type=float, required=True, metavar=('X', 'Y', 'YAW'))
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--backend-config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs/mujoco-lightnav.toml')
    parser.add_argument('--model-url', help='覆盖后端配置中的模型地址')
    parser.add_argument('--official-tracker', action='store_true')
    asyncio.run(replay(parser.parse_args()))
