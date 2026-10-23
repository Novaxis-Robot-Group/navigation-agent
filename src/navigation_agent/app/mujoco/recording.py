"""MuJoCo 双视角采样，复用公共录像编码器。"""

import asyncio
import json
import time

from navigation_agent.app.recording import TaskRecording


class Recorder:
    def __init__(self, simulation, settings):
        self.simulation = simulation
        self.settings = settings
        self.active = None
        self.sampler = None
        self.lock = asyncio.Lock()

    async def start(self, task):
        async with self.lock:
            if not self.settings.enabled:
                return {'status': 'disabled'}
            if self.active:
                if self.active.metadata['task_id'] == task['id']:
                    return self.active.metadata
                raise ValueError('另一任务正在录像或收尾')
            directory = self.settings.directory / task['id']
            if directory.exists():
                return self.read(task['id'])
            self.active = TaskRecording(directory, task, self.settings.fps)
            # 不限时任务沿用录像自身的容量上限，不给导航增加任务截止时间。
            task_seconds = task.get('max_seconds')
            recording_seconds = 7230 if task_seconds is None else min(task_seconds + 30, 7230)
            self.sampler = asyncio.create_task(self._sample(self.active, recording_seconds))
            return self.active.metadata

    async def _sample(self, recording, seconds):
        deadline = time.monotonic() + seconds
        previous = None
        try:
            while time.monotonic() < deadline:
                first = self.simulation.frame()
                third = self.simulation.third_person_frame()
                if (first and third and first.stamp_ns == third.stamp_ns
                        and first.stamp_ns != previous):
                    previous = first.stamp_ns
                    recording.offer(first, third)
                if recording.error:
                    break
                await asyncio.sleep(1 / self.settings.fps)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            recording.error = '读取录像画面失败：' + type(exc).__name__
            await self.stop(recording.metadata['task_id'], status='interrupted')
        else:
            # Agent 异常退出时也有截止时间，避免永久录制。
            await self.stop(recording.metadata['task_id'], status='interrupted')

    async def stop(self, task_id, status='ready'):
        async with self.lock:
            if not self.active or self.active.metadata['task_id'] != task_id:
                return self.read(task_id)
            if self.sampler and self.sampler is not asyncio.current_task():
                self.sampler.cancel()
                await asyncio.gather(self.sampler, return_exceptions=True)
            result = await self.active.close(status)
            self.active = None
            self.sampler = None
            return result

    def read(self, task_id):
        path = self.settings.directory / task_id / 'recording.json'
        if not path.is_file():
            return {'status': 'missing', 'frames': 0}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'status': 'failed', 'error': '录像状态不可读取，请检查磁盘与文件完整性', 'views': []}
        if data['status'] == 'recording' and (not self.active or self.active.metadata['task_id'] != task_id):
            data.update(status='interrupted', error='录像服务在上次任务期间退出，视频尚未正常封装')
            data['views'] = []
        return data
