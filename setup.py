"""构建独立交付包；默认资源仅从仓库配置复制，不维护副本。"""

from pathlib import Path
from shutil import copy2
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildAgent(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        # app 根目录的部署模块不随独立 Agent 分发。
        if package == 'navigation_agent.app':
            modules = [item for item in modules if item[1] == '__init__']
        return modules

    def run(self):
        super().run()
        source = Path(__file__).parent / 'configs'
        target = Path(self.build_lib) / 'navigation_agent/app/agent/resources'
        for relative in ('agent.toml', 'prompts/common.md', 'prompts/go2-office.md', 'prompts/scheduling.md'):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy2(source / relative, destination)


setup(cmdclass={'build_py': BuildAgent})
