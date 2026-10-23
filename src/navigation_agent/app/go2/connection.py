"""GO2 Agent Web 协议连接；合并增量状态，串行确认命令并维持租约。"""

import asyncio
import base64
import binascii
from contextlib import suppress
import math
import time
from urllib.parse import urlparse

import aiohttp

from navigation_agent.contracts.agent_backend import ToolError


REQUIRED_CAPABILITIES = ('streaming_observation', 'confirmed_control_release', 'navigation_session_guard')


def missing_capabilities(state):
    capabilities = state.get('agent', {}).get('capabilities') or {}
    return [name for name in REQUIRED_CAPABILITIES if capabilities.get(name) is not True]


class Go2Connection:
    def __init__(self, base_url, *, command_timeout=5.0):
        address = urlparse(base_url)
        if address.scheme not in ('http', 'https') or not address.hostname or address.username or address.password:
            raise ValueError('GO2 base URL must be an HTTP(S) address without credentials')
        if address.path not in ('', '/') or address.query or address.fragment:
            raise ValueError('GO2 base URL must not contain a path, query or fragment')
        self.base_url = base_url.rstrip('/')
        self.command_timeout = command_timeout
        self.state = {}
        self.control = {}
        self.error = None
        self.on_snapshot = lambda: None
        self.session = self.socket = self.reader = self.heartbeat = None
        self.changed = asyncio.Event()
        self.revision = 0
        self.pending = None
        self.pending_history = None
        self.command_lock = asyncio.Lock()
        self.last_heartbeat = 0.0
        self.odom_received = 0.0
        self.odom_stamp = None
        self.odom_frame = None
        self.navigation_generation = None
        self.navigation_result = None
        self.navigation_received_result = False
        self.navigation_motion_started = False
        self.navigation_robot_error = ''
        self.navigation_model_error = ''
        self.navigation_mpc_error = ''
        self.last_episode = 0
        self.previous_episode = 0
        self.closing = False

    def robot_error(self):
        diagnostics = self.state.get('robot', {})
        error = diagnostics.get('safety_error') or diagnostics.get('sdk_error') or ''
        if not error and diagnostics.get('level', 0) >= 2:
            error = diagnostics.get('message') or 'GO2 robot diagnostics reported an error'
        return error

    def check(self):
        if self.error:
            raise self.error

    def fail(self, message):
        self.error = self.error or ToolError(message)
        self.changed.set()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3), trust_env=False)
        try:
            self.socket = await self.session.ws_connect(self.base_url + '/agent/ws', heartbeat=2)
            self.reader = asyncio.create_task(self._read())
            await self.wait_for(lambda: bool(self.state) and bool(self.control))
            agent = self.state.get('agent', {})
            lease = agent.get('lease_s')
            if agent.get('protocol_version') != 1 or not isinstance(lease, (int, float)) or not math.isfinite(lease) or lease < 0.3:
                raise ToolError('GO2 Agent protocol version or lease is invalid')
            missing = missing_capabilities(self.state)
            if missing:
                raise ToolError('GO2 客户端需要更新，缺少能力：' + ', '.join(missing))
            self.last_heartbeat = time.monotonic()
            self.heartbeat = asyncio.create_task(self._heartbeat(min(1.0, lease / 3), lease))
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        # 即使动作被取消，仍尽力等待自己的释放回执；随后关闭连接让本体收尾。
        if not self.closing:
            try:
                if self.socket and not self.socket.closed and not self.error and self.owns_control():
                    await self.command('agent_release')
            except (Exception, asyncio.CancelledError):
                pass
        self.closing = True
        if self.heartbeat:
            self.heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self.heartbeat
        if self.socket:
            with suppress(Exception):
                await asyncio.wait_for(self.socket.close(), 2)
        if self.reader:
            self.reader.cancel()
            with suppress(asyncio.CancelledError):
                await self.reader
        if self.session:
            await self.session.close()

    def owns_control(self):
        return bool(self.control.get('owner') or self.control.get('auto_owner'))

    def locked(self):
        return bool(self.control.get('locked')) and not self.owns_control()

    async def _heartbeat(self, period, lease):
        try:
            while True:
                self.check()
                if time.monotonic() - self.last_heartbeat > lease * 0.8:
                    raise ToolError('GO2 heartbeat acknowledgement timed out')
                await self.send({'type': 'agent_heartbeat'})
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail('GO2 heartbeat failed')
            await self.socket.close()

    async def _read(self):
        try:
            async for message in self.socket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                self._accept(message.json())
                self.revision += 1
                self.changed.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail('GO2 control stream failed')
        finally:
            if not self.closing:
                self.fail('GO2 control connection closed')

    def _accept(self, data):
        kind = data.get('type')
        if kind in ('snapshot', 'runtime'):
            self.state.update(data['data'])
        elif kind == 'control_state':
            self.control = data
        elif kind == 'control_source':
            self.state['control_source'] = data['source']
        elif kind in ('vln_status', 'mpc_status', 'robot_diagnostics', 'odom', 'path', 'sent_command'):
            key = {'vln_status': 'vln', 'mpc_status': 'mpc', 'robot_diagnostics': 'robot'}.get(kind, kind)
            self.state.setdefault(key, {}).update(data['data'])
        elif kind == 'agent_heartbeat':
            self.last_heartbeat = time.monotonic()
        elif kind == 'command_result':
            if data.get('command') == 'vln_history':
                if self.pending_history != (data.get('generation'), data.get('operation')):
                    return
                self.state.setdefault('agent', {})['history_reused'] = data.get('history_reused', False)
            if not data.get('ok'):
                self.fail('GO2 rejected command: ' + str(data.get('message', 'unknown error'))[:200])
            elif self.pending:
                expected, received, enabled = self.pending
                command = data.get('command')
                if command in expected and (enabled is None or data.get('enabled') is enabled):
                    received.add(command)

        odom = self.state.get('odom', {})
        stamp = odom.get('stamp_ns')
        if stamp is not None and stamp != self.odom_stamp:
            frame = odom.get('frame_id')
            if self.odom_stamp is not None and (stamp < self.odom_stamp or frame != self.odom_frame):
                self.fail('GO2 odometry clock or coordinate frame changed')
            self.odom_stamp, self.odom_frame = stamp, frame
            self.odom_received = time.monotonic()
            self.on_snapshot()
        vln = self.state.get('vln', {})
        episode = vln.get('episode', 0)
        self.last_episode = max(self.last_episode, episode or 0)
        if self.navigation_generation is not None:
            self.navigation_robot_error = self.navigation_robot_error or self.robot_error()
            # 增量状态可能在下一次里程计前被 IDLE 清空，必须保留本次会话的结果和错误。
            current_model = (
                vln.get('generation') == self.navigation_generation if 'generation' in vln
                else (episode or 0) > self.previous_episode)
            if kind in ('vln_status', 'snapshot', 'runtime') and (current_model or 'generation' not in vln):
                if vln.get('stop') is True or (vln.get('last_sequence') or 0) > 0:
                    self.navigation_received_result = True
                if vln.get('state') == 'ERROR':
                    self.navigation_model_error = vln.get('error') or vln.get('message') or 'VLN ERROR'
            mpc = self.state.get('mpc', {})
            if mpc.get('error') or mpc.get('state') == 'ERROR':
                self.navigation_mpc_error = mpc.get('error') or 'MPC ERROR'
            command = self.state.get('sent_command', {})
            if any(abs(command.get(axis) or 0) > 0 for axis in ('linear', 'angular')):
                self.navigation_motion_started = True
            terminal = self.state.get('agent', {}).get('terminal')
            if terminal and terminal.get('generation') == self.navigation_generation:
                self.navigation_result = dict(terminal)
            # stop 的增量事件先于控制权释放；不能等下一次完整快照才识别模型停止。
            if kind == 'vln_status' and vln.get('stop') is True and current_model:
                self.navigation_result = {'generation': self.navigation_generation, 'reason': 'model_stop'}

    def begin_navigation(self):
        self.navigation_generation = max(
            self.state['agent']['navigation_generation'], self.navigation_generation or 0,
        ) + 1
        self.previous_episode = self.last_episode
        self.navigation_result = None
        self.navigation_received_result = False
        self.navigation_motion_started = False
        self.navigation_robot_error = ''
        self.navigation_model_error = ''
        self.navigation_mpc_error = ''

    async def wait_for(self, predicate, timeout=None):
        try:
            async with asyncio.timeout(timeout or self.command_timeout):
                while True:
                    self.check()
                    if predicate():
                        return
                    self.changed.clear()
                    await self.changed.wait()
        except TimeoutError:
            raise ToolError('Timed out waiting for GO2 state') from None

    async def wait_snapshot(self):
        revision = self.revision
        await self.wait_for(lambda: self.revision > revision)

    async def send(self, payload):
        self.check()
        try:
            await self.socket.send_json(payload)
        except (aiohttp.ClientError, ConnectionError, RuntimeError):
            self.fail('GO2 command connection failed')
            raise self.error from None

    async def command(self, kind, **parameters):
        expected = {
            'acquire_control': {'manual_control'}, 'agent_release': {'agent_release'},
            'set_server_url': {'server_url'}, 'set_vln': {'vln', 'mpc'},
            'pause_vln': {'vln_history', 'agent_release'}, 'finish_vln': {'vln_history'},
        }[kind]
        operation = {'pause_vln': 'pause', 'finish_vln': 'finish'}.get(kind)
        if kind == 'set_vln' and parameters.get('preserve_history'):
            expected = {'vln_history'}
            operation = 'resume'
        enabled = True if kind in ('acquire_control', 'set_vln') and not operation else None
        async with self.command_lock:
            received = set()
            self.pending = expected, received, enabled
            self.pending_history = (parameters.get('generation'), operation) if operation else None
            try:
                await self.send({'type': kind, **parameters})
                await self.wait_for(lambda: received == expected)
            except BaseException:
                # 回执没有请求 ID，超时/取消后废弃连接，避免迟到回执确认下一条命令。
                self.fail('GO2 command did not complete')
                if self.socket:
                    with suppress(Exception):
                        await asyncio.wait_for(self.socket.close(), 2)
                raise
            finally:
                self.pending = None
                self.pending_history = None

    async def observation(self, after_sequence=-1):
        self.check()
        try:
            async with self.session.get(self.base_url + '/api/observation', params={'after_sequence': after_sequence}) as response:
                if response.status in (409, 503):
                    return None
                if response.status != 200:
                    raise ToolError('GO2 observation endpoint failed')
                data = await response.json()
            jpeg = base64.b64decode(data['jpeg_base64'], validate=True)
        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError, binascii.Error):
            raise ToolError('Invalid GO2 observation response') from None
        if data.get('protocol_version') != 1 or not jpeg.startswith(b'\xff\xd8'):
            raise ToolError('Invalid GO2 observation image or protocol')
        return jpeg, data
