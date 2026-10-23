"""纯 VLN 运行留档；记录只观察执行状态，不参与模型决策。"""

import asyncio
from datetime import datetime, timezone
from html import escape
import json
import logging
import re
import time
from uuid import uuid4

from aiohttp import web

from navigation_agent.app.recording import write_json
from navigation_agent.app.replay import build_timeline, rows

logger = logging.getLogger(__name__)
RUN_ID = r'[a-f0-9]{32}'
REASONS = {
    'model_stop': '模型发出停止信号（未评定任务成功）',
    'manual_stop': '手动结束导航',
    'manual_takeover': '手动接管',
    'reset': '重置场景',
    'connection_closed': '控制连接中断',
    'shutdown': '服务关闭',
    'timeout': '达到运行时间上限',
    'error': '导航异常',
    'start_failed': '导航启动失败',
}


def now():
    return datetime.now(timezone.utc).isoformat()


class VlnRuns:
    def __init__(self, console):
        self.console = console
        self.root = console.recorder.settings.directory / 'vln'
        self.active = None
        self.worker = None
        self.pending = []
        self.last_sample = 0
        self.last_image = 0
        self.image_count = 0

    def directory(self, run_id):
        if not re.fullmatch(RUN_ID, run_id):
            raise web.HTTPNotFound()
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise web.HTTPNotFound()
        return path

    def read(self, run_id):
        try:
            record = json.loads((self.directory(run_id) / 'result.json').read_text())
        except (OSError, ValueError):
            raise web.HTTPNotFound(text='VLN 记录不存在或不可读取') from None
        if record['status'] == 'running' and (not self.active or self.active['id'] != run_id):
            record.update(status='interrupted', end_reason='service_interrupted',
                          summary='上次服务退出，记录未正常收尾；任务结果未评定',
                          ended_at=record.get('updated_at', record['started_at']))
        return record

    def exists(self, run_id):
        return (self.directory(run_id) / 'result.json').is_file()

    async def start(self, instruction, goal, max_seconds):
        frame = self.console.simulation.frame()
        record = {
            'id': uuid4().hex, 'mode': 'vln', 'goal': goal or instruction,
            'instruction': instruction, 'status': 'running', 'started_at': now(),
            'updated_at': now(), 'ended_at': None, 'max_seconds': max_seconds,
            'model': 'LightNav', 'success': None, 'prediction_count': 0,
            'summary': '直接导航执行中，任务成功与否尚未评定',
            'scene': self.console.scene,
            'initial_pose': list(frame.pose) if frame else None,
            'configuration': {
                'vln_backend': type(self.console.vln).__name__,
                'executor': type(self.console.executor).__name__,
                'control_period_s': self.console.executor.control_period_s,
                'executor_parameters': self.console.executor.diagnostics().get('parameters', {}),
                'recording_enabled': self.console.recorder.settings.enabled,
                'recording_fps': self.console.recorder.settings.fps,
            },
        }
        directory = self.directory(record['id'])
        def prepare():
            (directory / 'images').mkdir(parents=True)
            write_json(directory / 'result.json', record)
            for name in ('events.jsonl', 'predictions.jsonl', 'navigation.jsonl'):
                (directory / name).touch()
        await asyncio.to_thread(prepare)
        self.active = record
        self.pending = []
        self.last_sample = self.last_image = 0
        self.image_count = 0
        self.deadline = time.monotonic() + max_seconds if max_seconds else None
        self.event('started', '开始直接导航', instruction)
        self.recording_ready = asyncio.Event()
        self.worker = asyncio.create_task(self.persist())
        try:
            record['recording'] = await self.console.recorder.start(record)
        except Exception as exc:
            record['recording'] = {'status': 'failed', 'error': str(exc)}
        finally:
            self.recording_ready.set()
        return record

    def event(self, kind, label, description):
        self.pending.append(('events.jsonl', {
            'type': kind, 'time': now(), 'label': label, 'summary': description, 'key': True,
        }))

    def prediction(self, diagnostic):
        record = self.active
        if not record or record['status'] != 'running':
            return
        record['prediction_count'] += 1
        # 高频输出单独保存，时间轴只展示首次输出和停止等关键事件。
        if len(self.pending) < 2048:
            self.pending.append(('predictions.jsonl', diagnostic.copy()))
        else:
            record['dropped_log_entries'] = record.get('dropped_log_entries', 0) + 1
        if record['prediction_count'] == 1:
            self.event('vln_prediction', '收到首次预测', '模型开始输出局部导航轨迹')
        if diagnostic.get('execution') == 'missing_pose':
            self.event('vln_error', '预测未执行', '缺少对应采图位姿')

    def capture(self, final=False):
        frame = self.console.simulation.frame()
        if frame is None:
            return
        if frame.stamp_ns != self.last_sample:
            self.last_sample = frame.stamp_ns
            self.pending.append(('navigation.jsonl', {
                'time_ns': frame.stamp_ns, 'pose': dict(zip(('x', 'y', 'yaw'), frame.pose)),
                'command': list(self.console.navigation.command),
            }))
        if final or frame.stamp_ns - self.last_image >= 1_000_000_000:
            self.last_image = frame.stamp_ns
            self.image_count += 1
            name = f'images/{self.image_count:06d}.jpg'
            self.pending.append((name, frame.jpeg))
            self.pending.append(('events.jsonl', {
                'type': 'observation', 'time': now(), 'image': name,
                'observation': {'camera_stamp_ns': frame.stamp_ns},
            }))
            if final:
                self.active['final_image'] = name
                self.active['final_pose'] = list(frame.pose)

    def finish(self, reason, detail=''):
        if not self.active or self.active['status'] != 'running':
            return
        try:
            self.capture(final=True)
        except Exception as exc:
            self.active['log_error'] = '最终画面读取失败：' + str(exc)
        status = 'stopped' if reason in ('model_stop', 'manual_stop', 'manual_takeover', 'reset') else 'interrupted'
        self.active.update(status=status, ended_at=now(), end_reason=reason,
                           summary=REASONS[reason] + ('：' + detail if detail else ''))
        self.event('finished', '导航结束', self.active['summary'])

    def tick(self, model_stopped):
        if not self.active or self.active['status'] != 'running':
            return
        if model_stopped:
            self.finish('model_stop')
            return
        status = self.console.vln.snapshot()
        error = status.error or self.console.executor.diagnostics().get('error')
        if error:
            self.console.stop_vln('error', str(error))
        elif self.deadline and time.monotonic() >= self.deadline:
            self.console.stop_vln('timeout')

    def flush(self, record, batch):
        directory = self.directory(record['id'])
        for name, value in batch:
            path = directory / name
            if isinstance(value, bytes):
                path.write_bytes(value)
            else:
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        write_json(directory / 'result.json', record)

    async def persist(self):
        record = self.active
        try:
            while record['status'] == 'running':
                try:
                    self.capture()
                except Exception as exc:
                    record['log_error'] = '运行采样失败：' + str(exc)
                await self.save_pending(record)
                await asyncio.sleep(.1)
            # 停车可发生在录像初始化期间，等初始化结束后再收尾。
            await self.recording_ready.wait()
            try:
                recording = await self.console.recorder.stop(record['id'])
                if recording['status'] != 'missing':
                    record['recording'] = recording
            except Exception as exc:
                record['recording'] = {'status': 'failed', 'error': str(exc)}
            await self.save_pending(record)
        finally:
            self.active = None

    async def save_pending(self, record):
        record['updated_at'] = now()
        batch, self.pending = self.pending, []
        try:
            await asyncio.to_thread(self.flush, record.copy(), batch)
        except Exception as exc:
            record['log_error'] = str(exc)
            logger.exception('VLN 日志写入失败，run=%s', record['id'])

    async def close(self):
        self.finish('shutdown')
        if self.worker:
            await self.worker

    def timeline(self, run_id, recording):
        task = self.read(run_id)
        directory = self.directory(run_id)
        def read(name):
            path = directory / name
            return rows(path.read_text()) if path.is_file() else []
        events = read('events.jsonl')
        if task.get('end_reason') == 'service_interrupted':
            events.append({'type': 'finished', 'time': task['ended_at'],
                           'label': '运行中断', 'summary': task['summary'], 'key': True})
        frames_path = self.console.recorder.settings.directory / run_id / 'frames.jsonl'
        frames = rows(frames_path.read_text()) if frames_path.is_file() else []
        data = build_timeline(task, events, read('navigation.jsonl'), recording, frames)
        data['log_base'] = '/api/vln/runs/' + run_id + '/logs/'
        return data

    def install(self, app):
        app.router.add_get('/api/vln/runs', self.history)
        app.router.add_get('/api/vln/runs/{run_id:' + RUN_ID + '}/events', self.events)
        app.router.add_get('/api/vln/runs/{run_id:' + RUN_ID + '}/logs/{file:.*}', self.resource)

    async def history(self, request):
        def records():
            result = []
            for path in self.root.glob('*/result.json'):
                try:
                    result.append(self.read(path.parent.name))
                except web.HTTPNotFound:
                    continue
            return sorted(result, key=lambda row: row['started_at'], reverse=True)[:100]
        return web.json_response({'tasks': await asyncio.to_thread(records)},
                                 headers={'Cache-Control': 'no-store'})

    async def events(self, request):
        self.read(request.match_info['run_id'])
        try:
            after = max(0, int(request.query.get('after', 0)))
        except ValueError:
            raise web.HTTPBadRequest(text='无效事件游标') from None
        path = self.directory(request.match_info['run_id']) / 'events.jsonl'
        entries = await asyncio.to_thread(path.read_text) if path.is_file() else ''
        events = rows(entries)[after:after + 200]
        return web.json_response({'events': events, 'next': after + len(events)},
                                 headers={'Cache-Control': 'no-store'})

    async def resource(self, request):
        run_id, name = request.match_info['run_id'], request.match_info['file']
        record = self.read(run_id)
        if name == 'index.html':
            data = await asyncio.to_thread(self.timeline, run_id, self.console.recorder.read(run_id))
            return web.Response(text=self.report(record, data), content_type='text/html',
                                headers={'Cache-Control': 'no-store'})
        if not re.fullmatch(r'(?:result|events|predictions|navigation)\.jsonl?|images/\d{6}\.jpg', name):
            raise web.HTTPNotFound()
        directory = self.directory(run_id)
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Cache-Control': 'no-store'})

    def report(self, record, data):
        e = lambda value: escape(str(value))
        events = ''.join('<li><b>' + e(node['label']) + '</b><p>' + e(node['description']) + '</p></li>'
                         for node in data['nodes'] if node['key'])
        final = ('<img style="max-width:100%" alt="最终观察" src="' + e(record['final_image']) + '">'
                 if record.get('final_image') else '')
        links = ''.join('<a href="' + name + '" download>' + label + '</a> '
                        for name, label in [('result.json', '运行信息'), ('events.jsonl', '事件日志'),
                                            ('predictions.jsonl', '模型预测'), ('navigation.jsonl', '实际轨迹')])
        return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>VLN 运行日志</title><link rel="stylesheet" href="/replay-static/replay.css">'
                '<main><header><a href="/">← 探索工作台</a><p>VLN · 运行日志</p>'
                '<h1>' + e(record['goal']) + '</h1><p>' + e(record['summary']) + '</p>'
                '<p>导航指令：' + e(record['instruction']) + '</p><p>开始：' + e(record['started_at']) +
                ' · 预测：' + e(record['prediction_count']) + ' 次</p><p>任务成功：未评定</p>'
                '<nav><a href="/replay/' + record['id'] + '">录像回放 ↗</a>' + links +
                '</nav></header><section class="panel"><h2>执行事件</h2><ol>' + events +
                '</ol></section><section class="panel"><h2>最终画面</h2>' + final +
                '</section></main></html>')
