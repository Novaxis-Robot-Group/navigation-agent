"""控制服务侧导航会话；沿用现有控制台命令，图像留在服务侧。"""

from navigation_agent.contracts.transport import ControlChannel
from navigation_agent.contracts.agent_backend import ToolError
from navigation_agent.contracts.vln import SessionStatus


class RemoteNavigationSession:
    def __init__(self, channel: ControlChannel, model_url: str):
        self.channel = channel
        self.model_url = model_url
        self.preserve_history = False
        self.generation = None

    async def configure_history(self, enabled):
        capabilities = self.channel.state.get('agent', {}).get('capabilities', [])
        if enabled and 'navigation_history_continuation' not in capabilities:
            raise ToolError('导航服务不支持历史延续，请更新并重启仿真服务')
        self.preserve_history = enabled

    async def pause(self):
        await self.channel.command('pause_vln', generation=self.generation)

    async def finish(self):
        if self.generation is not None:
            await self.channel.command('finish_vln', generation=self.generation)
            self.generation = None

    @property
    def history_reused(self):
        return self.channel.state.get('vln', {}).get('history_reused', False)

    async def start(self, instruction):
        await self.channel.command('set_server_url', server_url=self.model_url)
        await self.channel.command('resume_vln' if self.preserve_history else 'set_vln',
                                   enabled=True, instruction=instruction)
        self.generation = self.channel.state.get('vln', {}).get('generation')

    def status(self):
        state = self.channel.state.get('vln', {})
        return SessionStatus(stopped=state.get('stop') is True, error=state.get('error', ''))

