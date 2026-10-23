"""通用任务 HTTP 接口；设备能力由部署入口注入。"""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import json
import re
from uuid import uuid4

import aiohttp
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from navigation_agent.app.agent.settings import StandaloneDeployment

from navigation_agent.agent.storage.logs import LogStore
from navigation_agent.agent.schemas import TaskRequest

from navigation_agent.agent.core.runner import run_task

runtime = None
deployment = StandaloneDeployment()


def configure(settings):
    global runtime
    runtime = settings


def current_settings():
    global runtime
    if runtime is None:
        runtime = deployment.load_settings()
    return runtime


def log_store():
    return LogStore(current_settings().agent.storage)


def resolve_task(request, settings=None):
    settings = settings or current_settings().agent
    values = request.model_dump()
    for name in ('max_steps', 'max_seconds'):
        if values[name] is None:
            values[name] = getattr(settings.task_defaults, name)
    if values['navigation_seconds'] is None:
        values['navigation_seconds'] = settings.execution.navigation_seconds
    limits = settings.task_limits
    if not limits.min_seconds <= values['max_seconds'] <= limits.max_seconds:
        raise HTTPException(422, '任务时间超出配置允许范围')
    if not limits.min_navigation_seconds <= values['navigation_seconds'] <= limits.max_navigation_seconds:
        raise HTTPException(422, '单段导航时间超出配置允许范围')
    return TaskRequest.model_validate(values)


records = {}
worker = None
configuration_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.settings = current_settings()
    # 启动只验证状态文件，绝不申请控制或应用参数。
    deployment.effective_settings(current_settings(), deployment.saved_motion(current_settings()))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2), trust_env=False) as session:
        app.state.monitor_http = session
        try:
            yield
        finally:
            if worker is not None and not worker.done():
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker


app = FastAPI(title='VLN Agent', lifespan=lifespan, redoc_url=None)


def install_deployment(value):
    global deployment
    deployment = value
    value.install(app)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'llm_configured': deployment.model_configured(current_settings()),
    }


@app.post('/tasks', status_code=202)
async def start_task(request: TaskRequest):
    async with configuration_lock:
        return await _start_task(request)


async def _start_task(request: TaskRequest):
    global worker
    if worker is not None and not worker.done():
        raise HTTPException(409, 'A task is already running; cancel it first')
    if not health()['llm_configured']:
        raise HTTPException(503, '请配置 Agent 云端模型密钥')

    settings = current_settings()
    settings, request, saved = await deployment.prepare_task(settings, request)

    request = resolve_task(request, settings.agent)

    # 装配失败时不创建运行中的任务记录。
    try:
        tools = deployment.create_tools(settings, motion=saved)
        validator = deployment.scene_validator(settings)
        model = deployment.create_model(settings)
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(503, 'Robot or navigation configuration is invalid') from exc

    # 创建独立任务和日志目录，再启动后台循环
    record = {
        'id': uuid4().hex,
        'goal': request.goal,
        'status': 'running',
        'phase': 'starting',
        'step': 0,
        'max_steps': request.max_steps,
        'max_seconds': request.max_seconds,
        'navigation_seconds': request.navigation_seconds,
        'navigation_time_limit_enabled': settings.agent.prompt.navigation_time_limit_enabled,
        'configuration': {**settings.snapshot(), 'saved_motion': saved.model_dump() if saved else None},
        'started_at': datetime.now(timezone.utc).isoformat(),
        'model': settings.agent.llm.model.strip(),
    }
    record['log_directory'] = log_store().log_time(record['started_at']) + '__running'
    if log_store().run_directory(record).exists():
        record['log_directory'] += '__' + record['id']
    (log_store().run_directory(record) / 'images').mkdir(parents=True)
    records[record['id']] = record
    log_store().event(
        record,
        'started',
        goal=request.goal,
        max_steps=request.max_steps,
        max_seconds=request.max_seconds,
        model=settings.agent.llm.model,
    )
    log_store().save_result(record)
    worker = asyncio.create_task(deployment.execute_task(run_task,
        record, request, tools, settings=settings.agent, api_key=settings.api_key,
        scene_validator=validator, model=model,
    ))
    return record.copy()


@app.get('/settings/agent')
def get_agent_settings():
    settings = current_settings()
    try:
        motion = deployment.saved_motion(current_settings())
        settings = deployment.effective_settings(settings, motion)
    except (OSError, ValueError):
        raise HTTPException(503, '运行设置损坏或不兼容，请先迁移') from None
    return {
        'task_defaults': settings.agent.task_defaults.model_dump(),
        'task_limits': settings.agent.task_limits.model_dump(),
        'navigation_seconds': settings.agent.execution.navigation_seconds,
        'navigation_time_limit_enabled': settings.agent.prompt.navigation_time_limit_enabled,
        'ui_task_defaults': deployment.ui_defaults(settings, motion).model_dump(),
        'sources': settings.snapshot()['sources'],
    }


def read_record(task_id):
    if not re.fullmatch(r'[a-f0-9]{32}', task_id):
        raise HTTPException(404, 'Task not found')
    if task_id in records:
        return records[task_id].copy()
    record = None

    # 先兼容旧 ID 目录，再查找时间戳目录
    for path in [current_settings().agent.storage.runs_dir / task_id / 'result.json', *current_settings().agent.storage.runs_dir.glob('*/result.json')]:
        try:
            candidate = json.loads(path.read_text(encoding='utf-8'))
            if candidate.get('id') == task_id:
                record = candidate
                record['log_directory'] = path.parent.name
                break
        except (OSError, ValueError):
            continue
    if record is None:
        raise HTTPException(404, 'Task not found')

    # 服务重启后只恢复历史展示，不恢复运动
    if record.get('status') == 'running':
        record.update(
            status='interrupted', phase='stopped', summary='Agent 服务已重启，任务未自动恢复'
        )
    return record


@app.get('/tasks')
async def list_tasks(limit: int = Query(default=30, ge=1, le=100)):
    paths = sorted(current_settings().agent.storage.runs_dir.glob('*/result.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for path in paths:
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
            result.append(read_record(stored['id']))
        except (HTTPException, OSError, ValueError, KeyError):
            continue
        if len(result) >= limit:
            break
    return {'tasks': result}


@app.get('/tasks/{task_id}')
async def get_task(task_id: str):
    return read_record(task_id)


@app.get('/tasks/{task_id}/logs/{filename:path}', include_in_schema=False)
async def task_log(task_id: str, filename: str):
    record = read_record(task_id)
    directory = log_store().run_directory(record).resolve()
    path = (directory / filename).resolve()

    # 报告及其相对资源仅能读取当前任务目录，禁止越界和其他文件类型。
    if (not path.is_relative_to(directory)
            or path.suffix.lower() not in {'.html', '.jpg', '.png', '.json', '.jsonl', '.sqlite3', '.md'}
            or not path.is_file()):
        raise HTTPException(404, 'Log file not found')
    return FileResponse(path, headers={'Cache-Control': 'no-store'})


@app.get('/tasks/{task_id}/events')
async def get_events(task_id: str, after: int = Query(default=0, ge=0)):
    record = read_record(task_id)
    path = log_store().run_directory(record) / 'events.jsonl'

    # 按游标读取一页日志，跳过不完整的行
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    entries = []
    for index, line in enumerate(lines[after : after + 200], start=after):
        try:
            entries.append({'index': index, **json.loads(line)})
        except ValueError:
            continue
    return {'events': entries, 'next': min(after + 200, len(lines))}


@app.get('/tasks/{task_id}/images/{filename}')
async def task_image(task_id: str, filename: str):
    record = read_record(task_id)
    if not re.fullmatch(r'\d{3,}\.jpg', filename):
        raise HTTPException(404, 'Image not found')
    path = log_store().run_directory(record) / 'images' / filename
    if not path.is_file():
        raise HTTPException(404, 'Image not found')
    return FileResponse(path, media_type='image/jpeg')


@app.post('/tasks/{task_id}/cancel')
async def cancel_task(task_id: str):
    record = await get_task(task_id)

    # 等待任务完成清理，再返回取消后的状态
    if record['status'] == 'running' and worker is not None:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        if records[task_id]['status'] == 'running':
            records[task_id].update(
                status='cancelled', phase='stopped', summary='Cancelled before start'
            )
            records[task_id]['ended_at'] = datetime.now(timezone.utc).isoformat()
            log_store().finish_log_directory(records[task_id])
            log_store().save_result(records[task_id])
    return read_record(task_id)
