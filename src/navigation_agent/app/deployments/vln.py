"""显式注册模型适配器；配置检查不导入模型或连接服务。"""

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel
from navigation_agent.contracts.runtime import PromptSettings
from navigation_agent.contracts.vln import VlnClient
from navigation_agent.vln.lightnav.client import LightNavOptions, create_lightnav
from navigation_agent.vln.lightnav.instructions import INSTRUCTION_GUIDE


@dataclass(frozen=True)
class VlnAdapter:
    options: type[BaseModel]
    create: Callable[[str, BaseModel], VlnClient]
    instruction_guide: str
    robot_backends: tuple[str, ...] | None = None
    url_schemes: tuple[str, ...] = ('ws', 'wss')


_ADAPTERS: dict[str, VlnAdapter] = {}


def register_vln(name: str, adapter: VlnAdapter):
    """应用启动前注册一次；禁止静默替换同名实现。"""
    if not name or name in _ADAPTERS:
        raise ValueError('VLN 名称为空或重复：' + name)
    PromptSettings(instruction_guide=adapter.instruction_guide)
    _ADAPTERS[name] = adapter


def get_vln(name: str) -> VlnAdapter:
    if name not in _ADAPTERS:
        raise ValueError('未注册的 VLN：' + name)
    return _ADAPTERS[name]


def validate_vln(name: str, server_url: str, options: dict[str, Any]):
    adapter = get_vln(name)
    parsed = urlsplit(server_url)
    parsed.port
    if (parsed.scheme not in adapter.url_schemes or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('VLN 服务地址与适配器协议不匹配，且不能包含凭据')
    return adapter.options.model_validate(options)


register_vln('lightnav', VlnAdapter(
    options=LightNavOptions,
    create=create_lightnav,
    instruction_guide=INSTRUCTION_GUIDE,
))
