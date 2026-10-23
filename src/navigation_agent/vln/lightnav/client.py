"""LightNav 接口适配；保留官方预测数值及 STOP 信号。"""

from pydantic import BaseModel, ConfigDict

from navigation_agent.contracts.vln import Prediction, VlnStatus


class LightNavClient:
    def __init__(self, logger, server_url='', *, client=None):
        if client is None:
            # 只在运行模型客户端的服务进程加载官方依赖。
            from navigation_agent.vln.lightnav.transport import LatestFrameClient
            client = LatestFrameClient(logger, server_url)
        self.client = client

    def set_server_url(self, value):
        return self.client.set_server_url(value)

    def start(self, instruction):
        self.client.start(instruction)

    @property
    def supports_history_continuation(self):
        return callable(getattr(self.client, 'pause', None)) and callable(getattr(self.client, 'resume', None))

    def pause(self):
        self.client.pause()

    def resume(self, instruction):
        return self.client.resume(instruction)

    def offer_frame(self, stamp_ns, jpeg):
        return self.client.offer_frame(stamp_ns, jpeg)

    def take_results(self):
        return [
            Prediction(
                sequence=result.sequence,
                stamp_ns=result.stamp_ns,
                waypoints=result.waypoints,
                stop=result.stop,
                latency_ms=result.latency_ms,
                metadata={
                    name: getattr(result, name, None)
                    for name in ('visible', 'apos_state', 'opos_state', 'apos_px', 'opos_px', 'history_elapsed_s')
                },
            )
            for result in self.client.take_results()
        ]

    def snapshot(self):
        state = self.client.snapshot()
        return VlnStatus(
            state=state.state,
            connected=state.connected,
            server_url=state.server_url,
            instruction=state.instruction,
            sequence=state.sequence,
            error=state.error,
        )

    def stop(self):
        self.client.stop()

    def close(self):
        self.client.close()



# 未支持的参数必须明确拒绝。
class LightNavOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)


def create_lightnav(server_url: str, options: LightNavOptions):
    import logging
    return LightNavClient(logging.getLogger('vln_mujoco.vln'), server_url)
