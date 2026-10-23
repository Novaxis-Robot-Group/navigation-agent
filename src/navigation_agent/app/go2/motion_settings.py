"""Persist operator settings and apply idle-only updates through the robot Web API."""

import asyncio
import json
import os
from pathlib import Path
import tempfile

import aiohttp
from pydantic import Field, model_validator
from navigation_agent.contracts.runtime import Values, AgentSettings, PromptSettings, ExplorationSettings, ProtectionSettings, MemorySettings




class LookSettings(Values):
    look_angle_deg: float = Field(70, ge=30, le=90, strict=True, title='转向目标（°）')
    look_tolerance_deg: float = Field(10, gt=0, le=10, strict=True, title='转向容差 ±（°）')
    look_angular_min: float = Field(.2, gt=0, le=1, strict=True, title='环视最低转速（rad/s）')
    look_angular_max: float = Field(.4, gt=0, le=1, strict=True, title='环视最高转速（rad/s）')
    look_timeout_s: float = Field(8, ge=2, le=30, strict=True, title='转向总时限（秒）')
    look_stall_s: float = Field(2.5, ge=.5, le=10, strict=True, title='转向停滞时限（秒）')
    look_gain: float = Field(1.5, ge=.1, le=5, title='转向比例增益')
    look_progress_deg: float = Field(1, ge=.1, le=5, title='转向进展最小角度（°）')
    navigation_seconds: float = Field(15, ge=1, le=60, strict=True, title='每段导航时限（秒）')

    @model_validator(mode='after')
    def consistent(self):
        if self.look_angular_min > self.look_angular_max:
            raise ValueError('环视最低转速不能大于最高转速')
        if self.look_stall_s > self.look_timeout_s:
            raise ValueError('停滞时限不能大于转向总时限')
        return self


class MpcSettings(Values):
    objnav_v_max: float = Field(.3, gt=0, le=1.5, strict=True, title='导航 X 速度上限（m/s）')
    track_v_max: float = Field(.3, gt=0, le=1.5, strict=True, title='跟随 X 速度上限（m/s）')
    w_max: float = Field(.6, gt=0, le=3, strict=True, title='导航角速度上限（rad/s）')
    v_output_scale: float = Field(1, ge=0, le=3, strict=True, title='线速度倍率')
    w_output_scale: float = Field(1, ge=0, le=3, strict=True, title='角速度倍率')
    a_max_v: float = Field(.5, gt=0, le=5, strict=True, title='线加速度上限（m/s²）')
    a_max_w: float = Field(1, gt=0, le=10, strict=True, title='角加速度上限（rad/s²）')

    q_x: float = Field(10, ge=0, le=1000, title='MPC X 误差权重')
    q_y: float = Field(10, ge=0, le=1000, title='MPC Y 误差权重')
    q_yaw: float = Field(1, ge=0, le=1000, title='MPC 朝向误差权重')
    r_v: float = Field(.1, gt=0, le=1000, title='MPC 线速度控制代价')
    r_w: float = Field(.1, gt=0, le=1000, title='MPC 角速度控制代价')

    @model_validator(mode='after')
    def tracking_weights(self):
        if not any((self.q_x, self.q_y, self.q_yaw)):
            raise ValueError('MPC 位置和朝向权重不能全部为零')
        return self


class BodySettings(Values):
    linear_limit: float = Field(.3, gt=0, le=1.5, strict=True, title='本体 X 速度上限（m/s）')
    angular_limit: float = Field(.6, gt=0, le=3, strict=True, title='本体角速度上限（rad/s）')


class MotionSettings(Values):
    look: LookSettings = Field(default_factory=LookSettings)
    mpc: MpcSettings = Field(default_factory=MpcSettings)
    body: BodySettings = Field(default_factory=BodySettings)

    agent: AgentSettings = Field(default_factory=AgentSettings)
    prompt: PromptSettings = Field(default_factory=PromptSettings)
    exploration: ExplorationSettings = Field(default_factory=ExplorationSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    protection: ProtectionSettings = Field(default_factory=ProtectionSettings)

    @model_validator(mode='after')
    def achievable(self):
        if self.look.look_angular_max > self.body.angular_limit:
            raise ValueError('本体角速度上限不能小于环视最高转速')
        return self


def load_saved(path, robot_url, *, prompt_defaults=None):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 3 or set(data) not in ({'schema_version', 'robot_url', 'values'}, {'schema_version', 'robot_url', 'values', 'preserve_mpc_weights'}):
        raise ValueError('运动状态文件版本过旧；请使用当前 TOML 配置并重新保存网页参数，旧文件需先由维护者迁移')
    if not isinstance(data['robot_url'], str) or data['robot_url'].rstrip('/') != robot_url.rstrip('/'):
        raise ValueError('运动状态所属机器人与当前目标不一致，拒绝自动恢复')
    values = data['values']
    if not isinstance(values, dict) or set(values) != set(MotionSettings.model_fields):
        raise ValueError('运动状态必须包含完整参数')
    # 旧默认句式改用当前后端指南；自定义旧模板需显式迁移，避免丢失用户要求。
    if isinstance(values.get('prompt'), dict) and set(values['prompt']) == {'instruction_template'}:
        if values['prompt']['instruction_template'] != 'Walk straight toward {target} {location} and stop.':
            raise ValueError('旧自定义 instruction_template 已停用；请将 prompt 改为 instruction_guide 写作规则后重试')
        values = {**values, 'prompt': (prompt_defaults or PromptSettings()).model_dump()}
    for name, field in MotionSettings.model_fields.items():
        kind = field.annotation
        if not isinstance(values[name], dict):
            raise ValueError(f'运动状态的 {name} 必须是参数对象')
        saved_fields = set(values[name])
        if name == 'exploration':
            # 旧观察次数已退出策略；仅忽略该字段，仍要求其余参数完整。
            saved_fields.discard('max_looks')
        if saved_fields != set(kind.model_fields):
            raise ValueError(f'运动状态的 {name} 缺少参数或包含未知参数，拒绝以默认值恢复')
    checked = MotionSettings.model_validate(values)
    preserve = data.get('preserve_mpc_weights', [])
    if (not isinstance(preserve, list) or any(not isinstance(k, str) or
        k not in ('q_x', 'q_y', 'q_yaw', 'r_v', 'r_w') for k in preserve)):
        raise ValueError('待保留 MPC 权重标记无效')
    checked.mpc.__pydantic_fields_set__.difference_update(preserve)
    return checked


def persist(settings, settings_file, robot_url):
    settings_file = Path(settings_file)
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=settings_file.parent,
                                         prefix='.go2-motion-', delete=False) as file:
            path = Path(file.name)
            file.write(json.dumps({'schema_version': 3, 'robot_url': robot_url.rstrip('/'),
                                   'values': settings.model_dump()}, ensure_ascii=False, indent=2) + '\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(path, settings_file)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


async def robot_state(session, base_url):
    async with session.get(base_url.rstrip('/') + '/api/health', allow_redirects=False) as response:
        response.raise_for_status()
        return await response.json()


def require_idle(state):
    if state.get('agent', {}).get('motion_settings_version') != 1:
        raise ValueError('本体客户端尚不支持参数设置，请更新并重启本体容器')
    if (state.get('control_source') != 'disabled' or state.get('agent', {}).get('release_pending')
        or state.get('vln', {}).get('enabled') or state.get('mpc', {}).get('enabled')):
        raise ValueError('请先取消任务并释放本体控制，再修改参数')


async def command(socket, kind, expected, config):
    await socket.send_json({'type': kind, 'config': config})
    async with asyncio.timeout(5):
        async for message in socket:
            if message.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError('参数连接已断开，结果未确认')
            data = message.json()
            if data.get('type') == 'command_result' and data.get('command') == expected:
                if data.get('ok') is not True:
                    raise ValueError(data.get('message') or '本体拒绝参数更新')
                return
    raise RuntimeError('未收到本体参数更新确认')


async def verify(session, base_url, mpc, body):
    async with asyncio.timeout(3):
        while True:
            state = await robot_state(session, base_url)
            require_idle(state)
            if all(state.get('mpc_config', {}).get(k) == v for k, v in mpc.items()) and all(
                state.get('robot', {}).get(k) == v for k, v in body.items()
            ):
                return
            await asyncio.sleep(.1)


async def apply(session, base_url, settings, *, save=False, settings_file=None):
    """No movement/control acquisition. Restore both nodes if either update or save fails."""
    if save and settings_file is None:
        raise ValueError('保存运动参数必须指定状态文件')
    state = await robot_state(session, base_url)
    require_idle(state)
    old_mpc = state.get('mpc_config', {})
    if old_mpc.get('available') is not True:
        raise ValueError('本体 MPC 参数尚不可读取')
    names = MpcSettings.model_fields
    old_mpc = {name: float(old_mpc[name]) for name in names}
    old_body = {name: float(state['robot'][name]) for name in BodySettings.model_fields}
    proposed_mpc = {**old_mpc, **{key: value for key, value in settings.mpc.model_dump().items()
        if key not in ('q_x', 'q_y', 'q_yaw', 'r_v', 'r_w') or key in settings.mpc.model_fields_set}}
    # Browser socket only: no lease, no acquire_control, no movement commands.
    async with session.ws_connect(base_url.rstrip('/') + '/ws', heartbeat=2) as socket:
        try:
            await command(socket, 'set_mpc_config', 'mpc_config', proposed_mpc)
            await command(socket, 'set_body_limits', 'body_limits', settings.body.model_dump())
            await verify(session, base_url, proposed_mpc, settings.body.model_dump())
            settings = settings.model_copy(update={
                "mpc": MpcSettings.model_validate(proposed_mpc)})
            if save:
                persist(settings, settings_file, base_url)
        except Exception as exc:
            # A timed-out WS acknowledgment may arrive late: rollback uses a fresh connection.
            await socket.close()
            try:
                async with session.ws_connect(base_url.rstrip('/') + '/ws', heartbeat=2) as rollback:
                    await command(rollback, 'set_mpc_config', 'mpc_config', old_mpc)
                    await command(rollback, 'set_body_limits', 'body_limits', old_body)
                    await verify(session, base_url, old_mpc, old_body)
            except Exception:
                raise RuntimeError('参数更新未完成且回退未确认，请重新读取本体实际参数；保存值未更新') from exc
            raise ValueError(f'参数更新失败，已恢复原参数：{exc}') from exc
    return settings.model_dump()
