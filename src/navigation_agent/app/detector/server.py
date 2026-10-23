"""检测 HTTP API；会话隔离词表与跟踪，串行使用常驻模型。"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from navigation_agent.contracts.detection import (
    Capabilities, FrameRequest, FrameResponse, SessionRequest, SessionResponse,
)


def terminate_worker():
    """推理线程无法安全取消，退出进程交给容器重启，避免带着挂死线程继续服务。"""
    logging.getLogger('uvicorn.error').critical('检测推理超过服务时限，退出进程等待自动重启')
    os._exit(1)


async def detect_with_deadline(engine, session, request, timeout_s):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(engine.detect, session, request), timeout=timeout_s)
    except TimeoutError:
        terminate_worker()
        raise


def create_app(engine_factory, *, inference_timeout_s=30.):
    if inference_timeout_s <= 0:
        raise ValueError('检测推理时限必须大于零')
    sessions = {}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await asyncio.to_thread(engine_factory)
        yield
        sessions.clear()

    app = FastAPI(title='Navigation target detector', lifespan=lifespan)

    @app.get('/capabilities', response_model=Capabilities)
    async def capabilities():
        return Capabilities(ready=True, model=app.state.engine.model_name, vocabulary='open')

    @app.post('/sessions', response_model=SessionResponse)
    async def create_session(request: SessionRequest):
        async with lock:
            now = time.monotonic()
            for identity in list(sessions):
                if now - sessions[identity]['used'] > 90:
                    del sessions[identity]
            if len(sessions) >= 4:
                raise HTTPException(503, '检测会话已满，请继续原有观察流程')
            identity = uuid4().hex
            sessions[identity] = {
                'labels': request.labels, 'confidence': request.confidence, 'used': now,
                'state': app.state.engine.new_state(request.confidence),
            }
            return SessionResponse(session_id=identity)

    @app.delete('/sessions/{identity}')
    async def close_session(identity: str):
        async with lock:
            sessions.pop(identity, None)
        return {'closed': True}

    @app.post('/sessions/{identity}/frames', response_model=FrameResponse)
    async def frame(identity: str, request: FrameRequest):
        # 只允许一个模型调用；取消 HTTP 请求也不能提前释放正在推理的模型。
        async with lock:
            session = sessions.get(identity)
            if session is None or time.monotonic() - session['used'] > 90:
                sessions.pop(identity, None)
                raise HTTPException(404, '检测会话不存在或已过期')
            session['used'] = time.monotonic()
            task = asyncio.create_task(detect_with_deadline(
                app.state.engine, session, request, inference_timeout_s))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise
            except (ValueError, IndexError) as exc:
                raise HTTPException(422, '图片或类别无效') from exc

    return app
