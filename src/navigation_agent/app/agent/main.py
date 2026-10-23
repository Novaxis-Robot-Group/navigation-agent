"""独立 Agent 入口；离线示例无需密钥或设备依赖。"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from uuid import uuid4

from navigation_agent.agent.core.runner import run_task
from navigation_agent.agent.schemas import TaskRequest
from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.app.agent.settings import DEFAULT_AGENT_CONFIG, StandaloneDeployment, standalone_settings
from navigation_agent.app.agent.demo import DemoBackend, DemoModel


async def run_demo(settings):
    store = LogStore(settings.agent.storage)
    request = TaskRequest(goal='运行离线接入示例', max_steps=2, max_seconds=30,
                          navigation_seconds=settings.agent.execution.navigation_seconds)
    record = {
        'id': uuid4().hex, 'goal': request.goal, 'status': 'running', 'phase': 'starting',
        'step': 0, 'max_steps': request.max_steps, 'max_seconds': request.max_seconds,
        'navigation_seconds': request.navigation_seconds,
        'configuration': settings.snapshot(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'model': 'offline-demo',
    }
    (store.run_directory(record) / 'images').mkdir(parents=True)
    store.event(record, 'started', goal=request.goal)
    await run_task(record, request, NavigationTools(DemoBackend()), settings=settings.agent,
                   api_key='', model=DemoModel(), store=store)
    return record, store.run_directory(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='运行固定响应的离线示例')
    parser.add_argument('--serve', action='store_true', help='启动示例任务 API，需要 service 依赖')
    parser.add_argument('--agent-config', default=DEFAULT_AGENT_CONFIG)
    parser.add_argument('--output', default='runs')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    if not args.demo:
        parser.error('此入口请显式指定 --demo；真实后端通过 StandaloneDeployment 注入')
    settings = standalone_settings(args.agent_config, output=args.output, api_key='')
    if settings.agent.prompt.scene is not None:
        parser.error('离线示例不加载部署场景规则，请使用通用 Agent 配置')
    if args.serve:
        import uvicorn
        from navigation_agent.app.agent import api
        api.configure(settings)
        api.install_deployment(StandaloneDeployment(backend_factory=DemoBackend, model_factory=DemoModel))
        uvicorn.run(api.app, host=args.host, port=args.port, access_log=False)
    else:
        record, directory = asyncio.run(run_demo(settings))
        print(json.dumps({'status': record['status'], 'directory': str(directory),
                          'report': str(directory / 'index.html')}, ensure_ascii=False))
        if record['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
