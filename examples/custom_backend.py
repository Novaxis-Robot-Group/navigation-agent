"""演示后端注入；默认使用固定图片与响应，不连接设备。"""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from navigation_agent.agent.core.runner import run_task
from navigation_agent.agent.schemas import TaskRequest
from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.tools import NavigationTools
from navigation_agent.app.agent.demo import DemoBackend, DemoModel
from navigation_agent.app.agent.settings import standalone_settings


async def main(backend):
    settings = standalone_settings(output='runs/custom-backend', api_key='')
    request = TaskRequest(goal='运行离线接入示例', max_steps=2, max_seconds=30,
                          navigation_seconds=settings.agent.execution.navigation_seconds)
    store = LogStore(settings.agent.storage)
    record = {
        'id': uuid4().hex, 'goal': request.goal, 'status': 'running', 'phase': 'starting',
        'step': 0, 'max_steps': request.max_steps, 'max_seconds': request.max_seconds,
        'navigation_seconds': request.navigation_seconds, 'configuration': settings.snapshot(),
        'started_at': datetime.now(timezone.utc).isoformat(), 'model': 'offline-demo',
    }
    (store.run_directory(record) / 'images').mkdir(parents=True)
    store.event(record, 'started', goal=request.goal)
    # 将 backend 换成自己的 RobotBackend 实现；真实任务还需配置决策模型。
    # 行进观察是可选能力；接入方式见 docs/zh/integration.md。
    await run_task(record, request, NavigationTools(backend), settings=settings.agent,
                   api_key='', model=DemoModel(), store=store)
    print(record['status'], store.run_directory(record))
    if record['status'] != 'completed':
        raise RuntimeError(record.get('summary', '示例未完成'))


if __name__ == '__main__':
    asyncio.run(main(DemoBackend()))
