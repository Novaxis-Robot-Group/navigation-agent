"""只检查服务连通性，不获取控制权或提交导航任务。"""

import json
import aiohttp

from navigation_agent.contracts.detection import Capabilities


async def check(settings):
    results = {}
    if settings.agent.harness.preserve_same_instruction_history:
        results['navigation_history_continuation'] = False
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        try:
            async with session.get(settings.backend.robot.base_url + '/api/health') as response:
                response.raise_for_status()
                results['robot_service'] = True
                if settings.agent.harness.preserve_same_instruction_history:
                    state = await response.json()
                    capabilities = state.get('agent', {}).get('capabilities', [])
                    results['navigation_history_continuation'] = 'navigation_history_continuation' in capabilities
        except (aiohttp.ClientError, TimeoutError, ValueError):
            results['robot_service'] = False
        try:
            url = settings.backend.vln.server_url
            if url.startswith(('ws://', 'wss://')):
                async with session.ws_connect(url):
                    results['model_transport'] = True
            else:
                async with session.get(url) as response:
                    response.raise_for_status()
                    results['model_transport'] = True
        except (aiohttp.ClientError, TimeoutError):
            results['model_transport'] = False
        detector_url = settings.agent.target_watch.server_url
        if detector_url:
            results['target_detector'] = await check_detector(session, detector_url)
    print(json.dumps(results, ensure_ascii=False))
    return 0 if all(results.values()) else 1


async def check_detector(session, url):
    """读取检测服务能力，不创建检测会话。"""
    try:
        async with session.get(url.rstrip('/') + '/capabilities') as response:
            response.raise_for_status()
            capabilities = Capabilities.model_validate(await response.json())
            return capabilities.ready
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return False
