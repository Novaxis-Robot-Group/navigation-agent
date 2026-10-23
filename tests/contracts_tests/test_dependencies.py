"""限制业务模块的依赖方向，避免目录拆开后又读取彼此实现。"""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / 'src/navigation_agent'
MODULES = {'agent', 'navigation', 'robots', 'vln', 'contracts', 'app'}
BUSINESS = MODULES | {'archive', 'experiments', 'adapters'}


class DependencyTests(unittest.TestCase):
    def test_src_contains_only_agreed_modules(self):
        directories = {
            path.name for path in SOURCE.iterdir()
            if path.is_dir() and path.name != '__pycache__' and not path.name.endswith('.egg-info')
        }
        self.assertEqual(directories, MODULES)

    def test_business_layers_only_import_their_own_code_and_contracts(self):
        for layer in sorted(MODULES):
            allowed = MODULES if layer == 'app' else {layer, 'contracts'}
            for path in (SOURCE / layer).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level:
                        modules = [node.module or '']
                    else:
                        continue
                    for module in modules:
                        root = module.removeprefix('navigation_agent.').split('.')[0]
                        if root in BUSINESS:
                            self.assertIn(root, allowed, f'{path}: {module}')
                        if layer in ('agent', 'navigation', 'contracts'):
                            self.assertFalse(
                                module.startswith(('vln_mujoco', 'mujoco', 'rclpy')),
                                f'{path}: {module}',
                            )
                        if layer == 'robots':
                            self.assertNotEqual(module, 'vln_mujoco.vln_client', str(path))


    def test_standalone_application_cannot_import_deployment_modules(self):
        from importlib.util import resolve_name
        for path in (SOURCE / 'app/agent').rglob('*.py'):
            package = 'navigation_agent.' + '.'.join(path.relative_to(SOURCE).with_suffix('').parts)
            package = package.rsplit('.', 1)[0]
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    modules = [resolve_name('.' * node.level + module, package) if node.level else module]
                for module in modules:
                    self.assertNotIn(module.removeprefix('navigation_agent.').split('.')[0], ('robots', 'navigation', 'vln', 'mujoco', 'rclpy'))
                    if module.startswith('navigation_agent.app.'):
                        self.assertTrue(module.startswith('navigation_agent.app.agent'), f'{path}: {module}')

    def test_model_layer_has_no_task_or_deployment_implementation_imports(self):
        from importlib.util import resolve_name
        for path in (SOURCE / 'agent/model').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    module = resolve_name('.' * node.level + (node.module or ''), 'navigation_agent.agent.model') if node.level else node.module or ''
                    self.assertFalse(module.removeprefix('navigation_agent.').startswith(('agent.skills', 'app', 'robots', 'navigation', 'vln')), str(path))
