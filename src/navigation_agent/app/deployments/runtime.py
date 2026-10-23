"""现有部署的设备配置、Go2 参数恢复与监控装配。"""

import asyncio
from datetime import datetime, timezone
from dataclasses import replace
from navigation_agent.agent.storage.logs import LogStore
import aiohttp
from fastapi import HTTPException, Request
from navigation_agent.app.go2 import motion_settings
from navigation_agent.app.deployments.runtime_overrides import motion_defaults, apply_agent_settings
from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import from_launch_environment
from navigation_agent.app.go2.monitor import router as monitor_router
from navigation_agent.app.deployments.scene import scene_validator
from navigation_agent.agent.tools import NavigationTools


class Deployment:
    def __init__(self, api):
        self.api = api

    def load_settings(self):
        return from_launch_environment()

    def saved_motion(self, settings):
        robot = settings.backend.robot
        if robot.backend != 'go2':
            return None
        return motion_settings.load_saved(
            robot.motion_settings_file, robot.base_url,
            prompt_defaults=motion_defaults(settings).prompt,
        )

    def create_tools(self, settings, motion):
        return NavigationTools(create_backend(settings.backend, saved_motion=motion))

    def scene_validator(self, settings):
        return scene_validator(settings.agent.prompt.scene)

    def create_model(self, settings):
        return None

    def model_configured(self, settings):
        return bool(settings.api_key and settings.agent.llm.model.strip())

    def effective_settings(self, settings, motion):
        return replace(settings, agent=apply_agent_settings(settings.agent, motion))

    def ui_defaults(self, settings, motion):
        return settings.agent.task_defaults if motion else settings.backend.ui.task_defaults

    def install(self, app):
        app.include_router(monitor_router)
        if self.api.current_settings().backend.robot.backend == 'go2':
            from navigation_agent.app.go2.replay import ReplayService
            ReplayService(self.api).install(app)
        app.add_api_route('/settings/motion', self.get_motion_settings, methods=['GET'])
        app.add_api_route('/settings/motion', self.set_motion_settings, methods=['POST'])

    async def execute_task(self, runner, record, *args, **kwargs):
        settings = self.api.current_settings()
        if (settings.backend.robot.backend == 'go2'
                and settings.backend.recording is not None
                and settings.backend.recording.enabled):
            from navigation_agent.app.go2.recording import record_task
            return await record_task(runner, record, *args, runtime_settings=settings, **kwargs)
        if (settings.backend.robot.backend != 'mujoco'
                or not settings.backend.simulation.recording.enabled):
            return await runner(record, *args, **kwargs)
        store = LogStore(settings.agent.storage)
        url = settings.backend.robot.base_url.rstrip('/') + '/api/recordings/' + record['id']
        running = False

        async def recording(action):
            try:
                # 独立会话使封装视频的等待不占用监控连接；录像故障不能阻止导航。
                async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=3 if action == 'start' else 25)) as http:
                    async with http.post(url + '/' + action, json={}) as response:
                        if response.status != 200:
                            raise ValueError('录像服务返回 ' + str(response.status))
                        result = await response.json()
                        record['recording'] = result
            except (aiohttp.ClientError, OSError, ValueError, TimeoutError) as exc:
                record['recording'] = {'status': 'failed', 'error': str(exc)}

        try:
            await recording('start')
            store.save_result(record)
            running = True
            return await runner(record, *args, **kwargs)
        except asyncio.CancelledError:
            if not running:
                record.update(status='cancelled', phase='stopped', summary='Cancelled before start',
                              ended_at=datetime.now(timezone.utc).isoformat())
                store.finish_log_directory(record)
                store.event(record, 'finished', status=record['status'], summary=record['summary'])
            raise
        finally:
            await recording('stop')
            store.save_result(record)

    async def prepare_task(self, settings, request):
        saved = None
        if settings.backend.robot.backend == 'go2':
            try:
                saved = self.saved_motion(self.api.current_settings())
                settings = replace(settings, agent=apply_agent_settings(settings.agent, saved))
                request = self.api.resolve_task(request, settings.agent)
                if saved is not None:
                    applied = await motion_settings.apply(self.api.app.state.monitor_http, settings.backend.robot.base_url, saved)
                    saved = motion_settings.MotionSettings.model_validate(applied)
            except (ValueError, OSError, RuntimeError, aiohttp.ClientError, TimeoutError, KeyError) as exc:
                raise HTTPException(503, f'本体参数恢复失败，任务未启动：{exc}') from None

        # 只复制本次任务配置，执行器与模型使用同一模式，不写回 TOML 或保存值。
        enabled = request.navigation_time_limit_enabled
        if enabled is not None:
            navigation = settings.backend.navigation.model_copy(update={'time_limit_enabled': enabled})
            prompt = settings.agent.prompt.model_copy(update={'navigation_time_limit_enabled': enabled})
            settings = replace(
                settings,
                backend=settings.backend.model_copy(update={'navigation': navigation}),
                agent=settings.agent.model_copy(update={'prompt': prompt}),
            )

        return settings, request, saved

    async def get_motion_settings(self):
        profile = self.api.current_settings().backend.robot
        try:
            defaults = motion_defaults(self.api.current_settings())
        except ValueError:
            raise HTTPException(503, 'TOML 参数超出网页可编辑范围，请在配置文件中调整') from None
        try:
            saved = self.saved_motion(self.api.current_settings())
        except (OSError, ValueError):
            raise HTTPException(503, '运动配置损坏、未迁移或所属机器人不匹配，请检查配置中的状态文件') from None
        state, error = None, ''
        if profile.backend == 'go2':
            try:
                state = await motion_settings.robot_state(self.api.app.state.monitor_http, profile.base_url)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                error = '本体未连接，无法确认实际参数'
        else:
            error = '当前后端不是 GO2'
        supported = bool(state and state.get('agent', {}).get('motion_settings_version') == 1)
        if state and not supported:
            error = '本体需要更新并重启，才能在此应用参数'
        current = {'mpc': state.get('mpc_config', {}), 'body': state.get('robot', {})} if state else None
        if saved is not None and current:
            weights = {key: current['mpc'][key] for key in ('q_x', 'q_y', 'q_yaw', 'r_v', 'r_w')
                       if key not in saved.mpc.model_fields_set and key in current['mpc']}
            try:
                saved.mpc = motion_settings.MpcSettings.model_validate({**saved.mpc.model_dump(), **weights})
            except ValueError:
                error = error or '本体 MPC 权重无法读取，请核对后保存'
        if saved is None and current:
            try:
                defaults.mpc = motion_settings.MpcSettings(**{
                    k: current['mpc'][k] for k in motion_settings.MpcSettings.model_fields})
                defaults.body = motion_settings.BodySettings(**{
                    k: current['body'][k] for k in motion_settings.BodySettings.model_fields})
            except (ValueError, KeyError):
                error = error or '部分本体参数不可读取，缺失项显示默认值，请核对后保存'
        return {'values': (saved or defaults).model_dump(), 'schema': motion_settings.MotionSettings.model_json_schema(),
                'saved': saved is not None,
                'current': current, 'supported': supported, 'error': error,
                'task_running': self.api.worker is not None and not self.api.worker.done()}


    async def set_motion_settings(self, settings: motion_settings.MotionSettings, request: Request):
        origin = request.headers.get('origin')
        if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
            raise HTTPException(403, 'Invalid origin')
        async with self.api.configuration_lock:
            if self.api.worker is not None and not self.api.worker.done():
                raise HTTPException(409, '请先取消当前任务，再修改运动参数')
            profile = self.api.current_settings().backend.robot
            if profile.backend != 'go2':
                raise HTTPException(409, '当前后端不是 GO2')
            try:
                apply_agent_settings(self.api.current_settings().agent, settings)
                values = await motion_settings.apply(self.api.app.state.monitor_http, profile.base_url, settings, save=True, settings_file=profile.motion_settings_file)
            except (ValueError, OSError, RuntimeError, aiohttp.ClientError, TimeoutError, KeyError) as exc:
                raise HTTPException(503, str(exc) or '参数应用失败，请检查本体连接') from None
            return {'values': values, 'saved': True, 'applied': True}
