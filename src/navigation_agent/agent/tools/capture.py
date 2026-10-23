"""行进中只读采样；不依赖检测命中，也不触发停车或模型请求。"""

import asyncio
import time

from navigation_agent.contracts.agent_backend import ToolError


async def capture_during_navigation(execute, *, tools, capture, interval_s, diagnostic):
    if interval_s <= 0 or not tools.supports_streaming_observation:
        return await execute()
    action = asyncio.create_task(execute())

    async def sample():
        previous_stamp = None
        while not action.done():
            await asyncio.sleep(interval_s)
            identity = tools.navigation_id
            if identity is None or action.done():
                continue
            try:
                jpeg, observation = await asyncio.wait_for(tools.snapshot_observation(), 1.)
                stamp = observation.get('camera_stamp_ns')
                if (action.done() or identity != tools.navigation_id or stamp == previous_stamp
                        or stamp is None or not 0 <= (time.time_ns() - stamp) / 1e9 <= 1.5):
                    continue
                capture(jpeg, observation)
                previous_stamp = stamp
            except (TimeoutError, OSError, ValueError, KeyError, TypeError, ToolError) as exc:
                diagnostic(error=type(exc).__name__)

    sampler = asyncio.create_task(sample())
    try:
        return await action
    finally:
        sampler.cancel()
        await asyncio.gather(sampler, return_exceptions=True)
        if not action.done():
            action.cancel()
            await asyncio.gather(action, return_exceptions=True)
