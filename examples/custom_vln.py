"""演示注册与参数装配，不是可执行真实导航的模型。"""

from pydantic import BaseModel, ConfigDict, Field
from navigation_agent.app.deployments.vln import VlnAdapter, register_vln
from navigation_agent.app.deployments.settings import VlnSettings
from navigation_agent.app.deployments.factory import create_vln_client
from navigation_agent.contracts.vln import VlnStatus


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    horizon: int = Field(default=4, gt=0)


class ExampleClient:
    def __init__(self, url, options):
        self.url, self.options, self.instruction = url, options, ''

    def set_server_url(self, value):
        self.url = value
        return value

    def start(self, instruction):
        raise RuntimeError('此示例没有真实模型；请实现服务协议')

    def offer_frame(self, stamp_ns, jpeg):
        return False

    def take_results(self):
        return []

    def snapshot(self):
        return VlnStatus('offline', False, self.url, self.instruction, 0, '')

    def stop(self):
        pass

    def close(self):
        pass


def register():
    # Agent 和控制服务应在读取配置之前调用同一个注册入口。
    register_vln('example', VlnAdapter(
        options=Options, create=ExampleClient,
        instruction_guide='使用简洁英文描述一个明确目标。可用 Approach 加目标描述。',
    ))


if __name__ == '__main__':
    register()
    settings = VlnSettings(backend='example', server_url='ws://localhost:9900', options={'horizon': 6})
    client = create_vln_client(settings)
    print(client.snapshot(), client.options)
