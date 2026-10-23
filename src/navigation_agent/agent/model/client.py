"""模型请求与连接生命周期；可注入离线响应客户端。"""

import aiohttp
from navigation_agent.agent.schemas import AgentError


class ModelClient:
    def __init__(self, settings, api_key):
        self.settings = settings
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.request_timeout_s))
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def complete(self, body):
        body = body.copy()
        # 决策和辅助核验共用配置；显式请求参数用于受控实验覆盖。
        if self.settings.temperature is not None:
            body.setdefault('temperature', self.settings.temperature)
        # 两家服务的思考开关不同；仅转换请求，不改变决策内容。
        if self.settings.provider == 'zhipu':
            enabled = body.pop('enable_thinking', self.settings.enable_thinking)
            body['thinking'] = {'type': 'enabled' if enabled else 'disabled'}
        async with self.session.post(
            self.settings.base_url.rstrip('/') + '/chat/completions', json=body,
            headers={'Authorization': 'Bearer ' + self.api_key}, allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise AgentError(
                    'Cloud model HTTP %s; check model access, region and Key' % response.status)
            return await response.json()
