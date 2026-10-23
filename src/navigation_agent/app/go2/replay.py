"""Go2 回放路由；复用 MuJoCo 的时间轴、字幕与前端。"""

import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from navigation_agent.app.replay import WEB_DIR, build_timeline, event_subtitles, rows


class ReplayService:
    def __init__(self, api):
        self.api = api

    def install(self, app):
        app.mount('/replay-static', StaticFiles(directory=WEB_DIR), name='replay-static')
        app.add_api_route('/replay/{task_id}', self.page, methods=['GET'])
        app.add_api_route('/api/replay/{task_id}', self.data, methods=['GET'])
        app.add_api_route('/api/replay/{task_id}/subtitles.vtt', self.subtitles, methods=['GET'])
        app.add_api_route('/api/recordings/{task_id}/{view}.mp4', self.video, methods=['GET'])

    def recording_directory(self, task_id):
        configuration = self.api.current_settings().backend.recording
        root = configuration.directory if configuration else self.api.current_settings().agent.storage.runs_dir / 'recordings'
        root = root.resolve()
        directory = (root / task_id).resolve()
        if directory.parent != root:
            raise HTTPException(404, '录像不存在')
        return directory

    def recording(self, task):
        path = self.recording_directory(task['id']) / 'recording.json'
        if not path.is_file():
            return {'status': 'missing', 'frames': 0, 'views': []}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['status'] == 'recording' and task['status'] != 'running':
                data.update(status='interrupted', views=[], error='录像尚未正常封装')
            return data
        except (OSError, ValueError, KeyError, TypeError):
            return {'status': 'failed', 'views': [], 'error': '录像状态不可读取'}

    async def page(self, task_id: str):
        self.api.read_record(task_id)
        return FileResponse(WEB_DIR / 'replay.html', headers={'Cache-Control': 'no-store'})

    def timeline(self, task_id):
        task = self.api.read_record(task_id)
        directory = self.api.log_store().run_directory(task)
        def read(path):
            return rows(path.read_text(encoding='utf-8')) if path.is_file() else []
        data = build_timeline(
            task, read(directory / 'events.jsonl'), read(directory / 'navigation.jsonl'),
            self.recording(task), read(self.recording_directory(task_id) / 'frames.jsonl'),
        )
        data['log_base'] = '/tasks/' + task_id + '/logs/'
        return data

    async def data(self, task_id: str):
        return await asyncio.to_thread(self.timeline, task_id)

    async def video(self, task_id: str, view: str):
        task = self.api.read_record(task_id)
        info = self.recording(task)
        if (view != 'first' or info['status'] not in ('ready', 'interrupted')
                or view not in info.get('views', [])):
            raise HTTPException(404, '录像尚未生成或视角不可用')
        directory = self.recording_directory(task_id)
        path = (directory / (view + '.mp4')).resolve()
        if path.parent != directory or not path.is_file():
            raise HTTPException(404, '录像不存在')
        return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'no-store'})

    async def subtitles(self, task_id: str):
        data = await self.data(task_id)
        if data['recording']['status'] not in ('ready', 'interrupted') or not data['recording'].get('views'):
            raise HTTPException(404, '录像尚未完成')
        return Response(event_subtitles(data), media_type='text/vtt', headers={
            'Content-Disposition': 'attachment; filename="' + task_id + '-events.vtt"',
            'Cache-Control': 'no-store',
        })
