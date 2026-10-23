"""项目控制台与本地 Agent API 的同源代理。"""

import asyncio
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import aiohttp
from aiohttp import web
from navigation_agent.app.navigation_server import NavigationServer
from navigation_agent.app.deployments.settings import RecordingSettings
from navigation_agent.app.mujoco.recording import Recorder
from navigation_agent.app.mujoco.replay import ReplayService
from navigation_agent.app.mujoco.vln_runs import VlnRuns
from navigation_agent.app.deployments.factory import create_simulation_components, create_vln_client

WEB_DIR = Path(__file__).resolve().parents[1] / 'web' / 'mujoco'
# 报告使用相对链接，只开放任务报告及其图片、数据和 Prompt 资源。
REPORT_RESOURCE = (
    r'(?:index\.html|memory_visualize\.html|[a-z_]+\.(?:json|jsonl|sqlite3)'
    r'|images/(?:detection-)?\d+\.(?:jpg|png)|prompts/[a-f0-9]{64}\.md)'
)
ALLOWED = re.compile(
    r'(?:settings/agent|health|tasks|tasks/[a-f0-9]{32}'
    r'(?:/cancel|/events|/images/\d{3,}\.jpg|/logs/' + REPORT_RESOURCE + r')?)'
)


class ConsoleServer(NavigationServer):
    def __init__(self, *, agent_url, simulation=None, vln_settings=None, model=None, recording=None, camera_height_m=0.198):
        simulation, executor, scene = create_simulation_components(simulation, camera_height_m=camera_height_m)
        if model is None:
            if vln_settings is None:
                raise ValueError('未提供 VLN 配置或客户端')
            model = create_vln_client(vln_settings)
        super().__init__(robot=simulation, executor=executor, model=model, scene=scene)
        self.agent_url = agent_url.rstrip('/')
        parsed = urlparse(self.agent_url)
        if (
            parsed.scheme != 'http'
            or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('simulation.agent_url must be a local loopback HTTP address')
        self.agent_http = None
        self.recorder = Recorder(simulation, recording or RecordingSettings())
        self.replay = ReplayService(self, self.recorder)
        self.vln_runs = VlnRuns(self)
        self.command_lock = asyncio.Lock()
        self.control_epoch = 0
        if hasattr(self, 'navigation'):
            self.navigation.on_prediction = self.vln_runs.prediction

    @property
    def mpc(self):
        return self.executor.tracker

    @mpc.setter
    def mpc(self, tracker):
        self.executor.tracker = tracker

    def snapshot(self, socket):
        state = super().snapshot(socket)
        state['vln']['diagnostic'] = self.navigation.diagnostic
        state['mpc']['reference_world'] = self.mpc.reference.tolist()
        state['mpc']['prediction_world'] = self.mpc.prediction.tolist()
        state['mpc']['local_navigation'] = getattr(self.mpc, 'local_status', 'disabled')
        if hasattr(self.mpc, 'local_status'):
            state['mpc']['controller'] = 'local_navigation'
            state['mpc']['parameters'] = {
                'linear_max_m_s': 0.35, 'angular_max_rad_s': 1.0,
                'local_radius_m': 2.0, 'extra_clearance_m': 0.035,
            }
        if hasattr(self, 'vln_runs'):
            state['vln_run'] = self.vln_runs.active
        return state

    async def handle_message(self, socket, payload):
        start = payload.get('type') == 'set_vln' and bool(payload.get('enabled'))
        if not start:
            # 停车和接管不能等待日志目录或编码器准备。
            await super().handle_message(socket, payload)
            return
        async with self.command_lock:
            if self.vln_runs.active or (payload.get('record_run') is True and self.recorder.active):
                await self.result(socket, False, '已有运行正在记录或录像收尾，请稍后再试')
                return
            if payload.get('record_run') is not True:
                await super().handle_message(socket, payload)
                return
            if self.auto_owner is not None or self.manual_owner not in {None, socket}:
                await self.result(socket, False, 'another page owns robot control')
                return
            instruction = str(payload.get('instruction', '')).strip()
            goal = str(payload.get('goal', '')).strip()
            seconds = payload.get('max_seconds')
            valid_seconds = seconds is None or (
                isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                and math.isfinite(seconds) and 1 <= seconds <= 7200
            )
            if not instruction or len(instruction) > 500 or len(goal) > 1000 or not valid_seconds:
                await self.result(socket, False, '请填写英文指令；时限留空或填写 1–7200 秒')
                return
            epoch = self.control_epoch
            try:
                await self.vln_runs.start(instruction, goal, seconds)
            except Exception:
                await self.result(socket, False, '无法创建 VLN 日志，请检查存储目录')
                return
            if epoch != self.control_epoch or socket.closed:
                self.vln_runs.finish('connection_closed' if socket.closed else 'manual_stop')
                return
            try:
                await super().handle_message(socket, payload)
            finally:
                if self.auto_owner is not socket:
                    self.vln_runs.finish('start_failed')

    def stop_vln(self, reason='manual_stop', detail=''):
        # 先执行原有停车逻辑，录像封装在后台完成。
        super().stop_vln(reason)
        self.control_epoch = getattr(self, 'control_epoch', 0) + 1
        if hasattr(self, 'vln_runs'):
            self.vln_runs.finish(reason, detail)

    def navigation_tick(self, model_stopped):
        self.vln_runs.tick(model_stopped)

    async def on_startup(self, app):
        self.agent_http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=12), trust_env=False
        )
        await super().on_startup(app)

    async def on_shutdown(self, app):
        # 先停止导航并保存最后缓存帧，录像封装不能延迟停车。
        try:
            if hasattr(self, 'vln_runs'):
                self.vln_runs.finish('shutdown')
            await super().on_shutdown(app)
        finally:
            try:
                if hasattr(self, 'vln_runs'):
                    await self.vln_runs.close()
                if self.recorder.active:
                    await self.recorder.stop(
                        self.recorder.active.metadata['task_id'], status='interrupted'
                    )
            finally:
                if self.agent_http:
                    await self.agent_http.close()

    async def index(self, request):
        return web.FileResponse(WEB_DIR / 'index.html', headers={'Cache-Control': 'no-cache'})

    # 校验请求来源与路由后，转发给本地 Agent
    async def agent_proxy(self, request):
        path = request.match_info['path']
        if not ALLOWED.fullmatch(path):
            raise web.HTTPNotFound()
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and urlparse(origin).netloc != request.host:
                raise web.HTTPForbidden(text='Invalid origin')
            if path != 'tasks' and not path.endswith('/cancel'):
                raise web.HTTPMethodNotAllowed(request.method, ['GET'])
        elif path.endswith('/cancel'):
            raise web.HTTPMethodNotAllowed(request.method, ['POST'])
        try:
            body = await request.read()
            async with self.agent_http.request(
                request.method,
                self.agent_url + '/' + path,
                params=request.query,
                data=body or None,
                headers={'Content-Type': 'application/json'},
                allow_redirects=False,
            ) as response:
                return web.Response(
                    body=await response.read(),
                    status=response.status,
                    content_type=response.content_type,
                    headers={'Cache-Control': 'no-store'},
                )
        except (aiohttp.ClientError, TimeoutError):
            return web.json_response(
                {'detail': 'Agent 服务未连接，请按所选配置启动 Agent 服务'}, status=503
            )

    # 分别探测 Agent 和模型服务，不启动任务
    async def services(self, request):
        async def agent_status():
            try:
                async with self.agent_http.get(
                    self.agent_url + '/health', timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    data = await response.json()
                    return {
                        'online': response.status == 200,
                        'configured': bool(data.get('llm_configured')),
                    }
            except (aiohttp.ClientError, TimeoutError, ValueError):
                return {'online': False, 'configured': False}

        async def model_status():
            parsed = urlparse(self.vln.snapshot().server_url)
            if parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
                return {'online': False}
            try:
                async with asyncio.timeout(1):
                    async with self.agent_http.ws_connect(self.vln.snapshot().server_url):
                        pass
                return {'online': True}
            except (aiohttp.ClientError, OSError, TimeoutError, ValueError):
                return {'online': False}

        agent, model = await asyncio.gather(agent_status(), model_status())
        return web.json_response(
            {'agent': agent, 'vln': model, 'sim': {'online': True}},
            headers={'Cache-Control': 'no-store'},
        )

    def app(self):
        app = super().app()
        app.router.add_static('/console-static', WEB_DIR)
        if hasattr(self, 'replay'):
            self.replay.install(app)
        if hasattr(self, 'vln_runs'):
            self.vln_runs.install(app)
        app.router.add_get('/api/services', self.services)
        app.router.add_get('/api/agent/{path:.*}', self.agent_proxy)
        app.router.add_post('/api/agent/{path:.*}', self.agent_proxy)
        return app
