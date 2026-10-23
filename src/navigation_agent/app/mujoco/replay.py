"""MuJoCo 回放路由；时间轴与页面和 Go2 共用。"""

import asyncio
import json
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

from navigation_agent.app.replay import WEB_DIR, build_timeline, event_subtitles, rows

TASK_ID = r'[a-f0-9]{32}'


class ReplayService:
    def __init__(self, console, recorder):
        self.console = console
        self.recorder = recorder

    def install(self, app):
        app.router.add_static('/replay-static', WEB_DIR)
        app.router.add_get('/replay/{task_id:' + TASK_ID + '}', self.page)
        app.router.add_get('/api/replay/{task_id:' + TASK_ID + '}', self.data)
        app.router.add_get('/api/replay/{task_id:' + TASK_ID + '}/subtitles.vtt', self.subtitles)
        app.router.add_get('/api/recordings/{task_id:' + TASK_ID + '}/{view:first|third}.mp4', self.video)
        app.router.add_post('/api/recordings/{task_id:' + TASK_ID + '}/{action:start|stop}', self.control)

    async def agent_get(self, path, *, optional=False):
        try:
            async with self.console.agent_http.get(self.console.agent_url + path) as response:
                if response.status == 404:
                    if optional:
                        return ''
                    raise web.HTTPNotFound(text='任务不存在')
                if response.status != 200:
                    raise web.HTTPBadGateway(text='Agent 日志暂不可用')
                return await response.text()
        except (aiohttp.ClientError, TimeoutError):
            raise web.HTTPServiceUnavailable(text='Agent 服务未连接') from None

    async def page(self, request):
        return web.FileResponse(WEB_DIR / 'replay.html', headers={'Cache-Control': 'no-store'})

    async def control(self, request):
        origin = request.headers.get('Origin')
        if origin and urlparse(origin).netloc != request.host:
            raise web.HTTPForbidden(text='Invalid origin')
        task_id = request.match_info['task_id']
        if request.match_info['action'] == 'stop':
            data = await self.recorder.stop(task_id)
        else:
            task = json.loads(await self.agent_get('/tasks/' + task_id))
            if request.transport is None or request.transport.is_closing():
                raise web.HTTPConflict(text='录像启动请求已断开')
            if task['status'] != 'running':
                raise web.HTTPConflict(text='只能为运行中的任务启动录像')
            try:
                data = await self.recorder.start(task)
            except ValueError as exc:
                raise web.HTTPConflict(text=str(exc)) from None
        return web.json_response(data)

    async def video(self, request):
        task_id = request.match_info['task_id']
        info = self.recorder.read(task_id)
        if info['status'] not in ('ready', 'interrupted') or not info.get('views'):
            raise web.HTTPNotFound(text='录像尚未生成或不可用，请查看回放页的录像状态')
        root = self.recorder.settings.directory.resolve()
        path = root / task_id / (request.match_info['view'] + '.mp4')
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound()
        # FileResponse 原生处理 Range，拖动进度无需下载整段视频。
        return web.FileResponse(path, headers={'Cache-Control': 'no-store'})

    async def subtitles(self, request):
        response = await self.data(request)
        data = json.loads(response.body)
        if data['recording']['status'] not in ('ready', 'interrupted') or not data['recording'].get('views'):
            raise web.HTTPNotFound(text='录像尚未完成，暂不能导出字幕')
        return web.Response(text=event_subtitles(data), content_type='text/vtt', headers={
            'Content-Disposition': 'attachment; filename="' + request.match_info['task_id'] + '-events.vtt"',
            'Cache-Control': 'no-store',
        })

    async def data(self, request):
        task_id = request.match_info['task_id']
        runs = getattr(self.console, 'vln_runs', None)
        if runs is not None and runs.exists(task_id):
            recording = self.recorder.read(task_id)
            task = runs.read(task_id)
            if recording['status'] == 'missing':
                recording = task.get('recording', recording)
            data = await asyncio.to_thread(runs.timeline, task_id, recording)
            return web.json_response(data, headers={'Cache-Control': 'no-store'})
        base = '/tasks/' + task_id
        task, events, samples = await asyncio.gather(
            self.agent_get(base), self.agent_get(base + '/logs/events.jsonl', optional=True),
            self.agent_get(base + '/logs/navigation.jsonl', optional=True))
        recording = self.recorder.read(task_id)
        path = self.recorder.settings.directory / task_id / 'frames.jsonl'
        frames = await asyncio.to_thread(path.read_text, encoding='utf-8') if path.is_file() else ''
        data = await asyncio.to_thread(build_timeline, json.loads(task), rows(events), rows(samples), recording, rows(frames))
        data['log_base'] = '/api/agent/tasks/' + task_id + '/logs/'
        return web.json_response(data, headers={'Cache-Control': 'no-store'})
