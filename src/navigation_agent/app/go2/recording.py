"""Go2 任务第一视角录像；只读观测接口，不申请控制或改写图片。"""

import asyncio
import base64
from types import SimpleNamespace

import aiohttp

from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.app.recording import TaskRecording


async def sample_frames(recording, robot_url, fps):
    sequence = -1
    async with aiohttp.ClientSession(
        trust_env=False, timeout=aiohttp.ClientTimeout(total=2),
    ) as http:
        while not recording.error:
            try:
                url = robot_url.rstrip('/') + '/api/observation'
                async with http.get(url, params={'after_sequence': sequence}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('pose_valid') and data['sequence'] > sequence:
                            pose = data['pose']
                            frame = SimpleNamespace(
                                jpeg=base64.b64decode(data['jpeg_base64'], validate=True),
                                stamp_ns=data['camera_stamp_ns'],
                                pose=(pose['x'], pose['y'], pose['yaw']),
                            )
                            recording.offer(frame)
                            sequence = data['sequence']
            except (aiohttp.ClientError, TimeoutError, ValueError, KeyError):
                # 暂时缺帧只影响录像；不补造图片或采图时间。
                pass
            await asyncio.sleep(1 / fps)


async def record_task(runner, record, *args, runtime_settings, **kwargs):
    configuration = runtime_settings.backend.recording
    store = LogStore(runtime_settings.agent.storage)
    recording = sampler = None
    try:
        try:
            recording = TaskRecording(
                configuration.directory / record['id'], record, configuration.fps,
                views=('first',),
            )
            record['recording'] = dict(recording.metadata)
            sampler = asyncio.create_task(sample_frames(
                recording, runtime_settings.backend.robot.base_url, configuration.fps,
            ))
        except (OSError, ValueError) as exc:
            record['recording'] = {'status': 'failed', 'error': str(exc)}
        return await runner(record, *args, **kwargs)
    finally:
        if sampler:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        if recording:
            record['recording'] = await recording.close()
        # runner 可能已经把任务目录改名，始终通过记录中的实际目录保存。
        store.save_result(record)
