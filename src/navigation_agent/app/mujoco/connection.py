"""控制台通信与连接清理；设备和导航模块共享一个控制连接。"""

import asyncio
import base64
import time
from contextlib import suppress

import aiohttp

from navigation_agent.contracts.agent_backend import ToolError


class ConsoleConnection:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.socket = None
        self.state = {}
        self.revision = 0
        self.changed = asyncio.Event()
        self.reader = None
        self.pending = None
        self.pending_id = None
        self.command_sequence = 0
        self.command_lock = asyncio.Lock()
        self.error = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10), trust_env=False
        )
        try:
            # 连接仿真，收到初始状态后申请控制权
            self.socket = await self.session.ws_connect(self.base_url + '/ws', heartbeat=2)
            self.reader = asyncio.create_task(self._read_loop())
            await self._snapshot(after=0)
            if 'control' not in self.state:
                raise ToolError('Control service returned an invalid initial state')
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        # 断开连接时，仿真只停止本连接控制的运动。
        # 不发送全局停止，避免打断其他控制端。
        if self.socket is not None:
            await self.socket.close()
        if self.reader is not None:
            self.reader.cancel()
            with suppress(asyncio.CancelledError):
                await self.reader
        if self.session is not None:
            await self.session.close()

    async def _read_loop(self):
        try:
            async for message in self.socket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = message.json()

                # 状态消息唤醒观察者，命令回执唤醒等待中的调用
                if data.get('type') in ('snapshot', 'runtime'):
                    self.state = data['data']
                    self.on_snapshot()
                    self.revision += 1
                    self.changed.set()
                elif data.get('type') == 'command_result':
                    identified = 'command_request_id' in self.state.get('agent', {}).get('capabilities', [])
                    if identified and data.get('request_id') != self.pending_id:
                        continue
                    if self.pending is not None and not self.pending.done():
                        if data.get('ok'):
                            self.pending.set_result(self.revision)
                        else:
                            self.pending.set_exception(
                                ToolError(
                                    'Simulator rejected control command (possibly owned by another client)'
                                )
                            )
                    elif not data.get('ok'):
                        raise ToolError('Robot control was interrupted')
        except Exception:
            self.error = ToolError('Simulator control connection failed')
        finally:
            self.error = self.error or ToolError('Simulator control connection closed')
            self.changed.set()
            if self.pending is not None and not self.pending.done():
                self.pending.set_exception(self.error)

    async def _snapshot(self, after=None):
        # 等待指定版本之后的快照，避免读到旧状态
        after = self.revision if after is None else after
        async with asyncio.timeout(5):
            while self.revision <= after:
                if self.error:
                    raise self.error
                self.changed.clear()
                await self.changed.wait()
        if self.error:
            raise self.error
        return self.state

    async def _command(self, kind, **parameters):
        if self.error:
            raise self.error
        self.command_sequence += 1
        self.pending_id = self.command_sequence
        self.pending = asyncio.get_running_loop().create_future()
        try:
            await self.socket.send_json({'type': kind, 'request_id': self.pending_id, **parameters})

            # 命令确认后，再等待对应的新状态
            # 在当前任务内限制等待，避免回执与取消同时到达时吞掉取消。
            async with asyncio.timeout(5):
                after = await self.pending
            return await self._snapshot(after=after)
        except BaseException:
            # 旧服务没有回执编号；不完整命令结束后废弃连接。
            self.error = self.error or ToolError('Simulator command did not complete')
            with suppress(Exception):
                await asyncio.wait_for(self.socket.close(), 2)
            raise
        finally:
            self.pending = None
            self.pending_id = None

    async def command(self, kind, **parameters):
        async with self.command_lock:
            await self._command(kind, **parameters)

    async def send(self, payload):
        if self.error:
            raise self.error
        await self.socket.send_json(payload)

    async def wait_snapshot(self):
        await self._snapshot()

    async def camera(self):
        async with self.session.get(self.base_url + '/api/camera.jpg') as response:
            if response.status != 200:
                raise ToolError('Robot camera unavailable')
            return await response.read()

    async def snapshot_observation(self):
        async with self.session.get(self.base_url + '/api/observation') as response:
            if response.status != 200:
                raise ToolError('Robot observation unavailable')
            payload = await response.json()
        jpeg = base64.b64decode(payload['jpeg_base64'], validate=True)
        if not jpeg.startswith(b'\xff\xd8'):
            raise ToolError('Simulator returned an invalid camera image')
        return jpeg, payload

    def diagnostics(self, action):
        # 页面兼容字段只用于日志，不进入公共决策或运动契约。
        simulation = self.state.get('simulation', {})
        vln = self.state.get('vln', {})
        return {
            'time_ns': time.time_ns(),
            'monotonic_s': time.monotonic(),
            'action': action,
            'pose': simulation.get('pose', {}),
            'velocity': simulation.get('velocity', {}),
            'command': simulation.get('command', {}),
            'backend': simulation.get('backend', {}),
            'camera_stamp_ns': simulation.get('camera', {}).get('stamp_ns'),
            'vln': {
                key: vln.get(key)
                for key in ('sequence', 'waypoints', 'latency_ms', 'stop', 'error', 'diagnostic')
            },
            'mpc': self.state.get('mpc', {}),
        }

    def on_snapshot(self):
        """收到状态后的扩展入口；适配器本身不判断探索进展。"""
        pass
