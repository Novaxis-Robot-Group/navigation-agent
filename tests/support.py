"""任务测试的临时配置与视觉证据夹具。"""

from contextlib import ExitStack
from unittest.mock import patch


class patch_runs:
    def __init__(self, path):
        self.path, self.stack = path, ExitStack()

    def start(self):
        from navigation_agent.app.agent import api
        from navigation_agent.app.deployments.runtime import Deployment
        self.stack.enter_context(patch.object(api, 'deployment', Deployment(api)))
        from tests.config_support import test_runtime, legacy_search_run_task
        self.stack.enter_context(patch.object(api, 'run_task', legacy_search_run_task))
        self.stack.enter_context(patch.object(api, 'current_settings', lambda: test_runtime(self.path)))
        return self

    def stop(self):
        self.stack.close()

    __enter__ = start

    def __exit__(self, *_):
        self.stop()


def clear_navigation_check():
    """Explicit mock visual evidence for tests whose subject is not scene restrictions."""
    return {'target_area':'allowed','path_area':'allowed','passage':'clear','space':'normal',
            'evidence':'测试画面：目标与路径在允许的硬质地面上，通道开放且无玻璃阻隔'}
