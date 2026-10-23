"""验证模型供应商请求参数，保持调用方数据不被修改。"""

import unittest
from aiohttp import web
from navigation_agent.agent.model.client import ModelClient
from tests.config_support import agent_settings


class ModelClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_thinking_fields(self):
        requests = []

        async def receive(request):
            requests.append(await request.json())
            return web.json_response({'choices': []})

        app = web.Application()
        app.router.add_post('/chat/completions', receive)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            for provider in ('dashscope', 'zhipu'):
                for enabled in (False, True):
                    settings = agent_settings().llm.model_copy(update={
                        'provider': provider, 'base_url': f'http://127.0.0.1:{port}',
                        'enable_thinking': enabled, 'temperature': 0.0,
                    })
                    body = {'model': 'test', 'messages': [], 'enable_thinking': enabled}
                    async with ModelClient(settings, 'test-key') as client:
                        await client.complete(body)
                    self.assertEqual(body['enable_thinking'], enabled)
                    self.assertNotIn('thinking', body)
                    self.assertNotIn('temperature', body)
                    payload = requests[-1]
                    self.assertEqual(payload['messages'], [])
                    self.assertEqual(payload['temperature'], 0.0)
                    if provider == 'zhipu':
                        self.assertNotIn('enable_thinking', payload)
                        self.assertEqual(payload['thinking'], {
                            'type': 'enabled' if enabled else 'disabled'})
                    else:
                        self.assertEqual(payload, {**body, 'temperature': 0.0})
        finally:
            await runner.cleanup()


    async def test_sampling_default_omission_and_explicit_override(self):
        requests = []

        async def receive(request):
            requests.append(await request.json())
            return web.json_response({'choices': []})

        app = web.Application()
        app.router.add_post('/chat/completions', receive)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            for temperature in (None, 0.0, 0.4):
                settings = agent_settings().llm.model_copy(update={
                    'base_url': f'http://127.0.0.1:{port}', 'temperature': temperature,
                })
                async with ModelClient(settings, 'test-key') as client:
                    for purpose in ('decision', 'region', 'route', 'completion'):
                        body = {'model': 'test', 'messages': [{'role': 'user', 'content': purpose}]}
                        await client.complete(body)
                        self.assertNotIn('temperature', body)
                        expected = body if temperature is None else {**body, 'temperature': temperature}
                        self.assertEqual(requests[-1], expected)
                    body = {'model': 'test', 'messages': [], 'temperature': 0.2}
                    await client.complete(body)
                    self.assertEqual(requests[-1], body)
        finally:
            await runner.cleanup()

    def test_sampling_config_rejects_invalid_values(self):
        from pydantic import ValidationError
        from navigation_agent.agent.config import LlmSettings
        data = agent_settings().llm.model_dump()
        self.assertIsNone(data['temperature'])
        for value in (-0.1, 2.0, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                LlmSettings.model_validate({**data, 'temperature': value})
        data.pop('temperature')
        self.assertIsNone(LlmSettings.model_validate(data).temperature)
