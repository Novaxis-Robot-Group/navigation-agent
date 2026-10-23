"""仿真入口；与 Agent 使用同一份环境配置。"""

import argparse
import socket
import sys
from aiohttp import web
from navigation_agent.app.deployments.settings import DEFAULT_BACKEND_CONFIG, load_backend, launch_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-config', default=DEFAULT_BACKEND_CONFIG)
    args = parser.parse_args()
    try:
        launch_values()
        settings = load_backend(args.backend_config)
    except (ValueError, OSError, KeyError) as exc:
        if hasattr(exc, 'errors'):
            parser.error(str([(e['loc'], e['type']) for e in exc.errors(include_input=False)]))
        parser.error(str(exc))
    if settings.robot.backend != 'mujoco':
        parser.error('仿真入口需要 MuJoCo 环境配置')
    service = settings.simulation
    with socket.socket() as probe:
        if sys.platform != 'win32':
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((service.host, service.port))
    from navigation_agent.app.mujoco.console import ConsoleServer
    server = ConsoleServer(
        vln_settings=settings.vln, agent_url=service.agent_url, recording=service.recording,
        camera_height_m=service.camera_height_m,
    )
    web.run_app(server.app(), host=service.host, port=service.port)


if __name__ == '__main__':
    main()
