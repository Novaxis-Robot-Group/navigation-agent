"""只读监控：代理相机和状态，不建立控制连接。"""

from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response



router = APIRouter()
WEB_DIR = Path(__file__).resolve().parents[1] / 'web' / 'monitor'


@router.get('/', include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / 'index.html', headers={'Cache-Control': 'no-store'})


@router.get('/monitor/app.js', include_in_schema=False)
async def script():
    return FileResponse(WEB_DIR / 'app.js', media_type='text/javascript',
                        headers={'Cache-Control': 'no-store'})


async def robot_get(request, path):
    try:
        url = request.app.state.settings.backend.robot.base_url.rstrip('/')
        parsed = urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
            raise ValueError('Invalid robot URL')
    except (ValueError, OSError, KeyError):
        raise HTTPException(503, '本体地址配置无效') from None
    try:
        async with request.app.state.monitor_http.get(url + path, allow_redirects=False) as response:
            if response.status != 200:
                raise HTTPException(503, '本体画面或状态暂不可用')
            if path.endswith('.jpg'):
                data = await response.read()
                if response.content_type != 'image/jpeg' or not data.startswith(b'\xff\xd8'):
                    raise HTTPException(502, '本体返回了无效图像')
                return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})
            data = await response.json()
            if not isinstance(data, dict):
                raise ValueError('Invalid robot state')
            return data
    except (aiohttp.ClientError, TimeoutError, ValueError):
        raise HTTPException(503, '无法读取本体，请检查客户端和网络连接') from None


@router.get('/monitor/camera.jpg', include_in_schema=False)
async def camera(request: Request):
    return await robot_get(request, '/api/camera.jpg')


@router.get('/monitor/robot', include_in_schema=False)
async def robot(request: Request):
    return await robot_get(request, '/api/health')
