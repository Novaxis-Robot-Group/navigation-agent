"""验证配置到服务侧客户端的装配，不连接真实设备或模型。"""

from unittest import TestCase
from unittest.mock import patch
import tempfile
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from navigation_agent.app.deployments import vln
from navigation_agent.app.deployments.factory import create_vln_client
from navigation_agent.app.deployments.settings import VlnSettings, load_backend, load_settings, PROJECT_ROOT


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    steps: int = Field(default=2, gt=0)


class AdapterTests(TestCase):
    def setUp(self):
        self.patch = patch.dict(vln._ADAPTERS)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.calls = []
        self.client = object()
        def create(url, options):
            self.calls.append((url, options))
            return self.client
        self.adapter = vln.VlnAdapter(
            options=Options, create=create,
            instruction_guide='使用 Approach 描述明确目标，方向和停止要求按需要添加。',
        )
        vln.register_vln('test_adapter', self.adapter)

    def test_options_reach_selected_factory_only_when_created(self):
        settings = VlnSettings(backend='test_adapter', server_url='ws://localhost:9900', options={'steps': 3})
        self.assertEqual(self.calls, [])
        self.assertIs(create_vln_client(settings), self.client)
        self.assertEqual(self.calls[0][0], settings.server_url)
        self.assertEqual(self.calls[0][1].steps, 3)

    def test_unknown_adapter_option_and_wrong_types_fail_before_connection(self):
        for backend, options in [('missing', {}), ('test_adapter', {'steps': '3'}),
                                 ('test_adapter', {'unknown': 1}), ('lightnav', {'queue_size': 1})]:
            with self.subTest(backend=backend, options=options), self.assertRaises(ValueError):
                VlnSettings(backend=backend, server_url='ws://localhost:9900', options=options)
        self.assertEqual(self.calls, [])

    def test_duplicate_registration_and_protocol_mismatch_fail(self):
        with self.assertRaises(ValueError):
            vln.register_vln('test_adapter', self.adapter)
        with self.assertRaises(ValueError):
            VlnSettings(backend='test_adapter', server_url='http://localhost:9900')

    def test_unsupported_go2_combination_rejected(self):
        settings = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        data = settings.model_dump()
        data['vln']['backend'] = 'test_adapter'
        with self.assertRaises(ValueError):
            type(settings).model_validate(data)

    def test_adapter_guide_and_explicit_override_are_in_effective_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory) / 'backend.toml'
            original = (PROJECT_ROOT / 'configs/mujoco-lightnav.toml').read_text()
            backend.write_text(original.replace('backend = "lightnav"', 'backend = "test_adapter"'))
            settings = load_settings(backend_file=backend, env_file=Path(directory) / '.env', environ={})
            self.assertEqual(settings.agent.prompt.instruction_guide, self.adapter.instruction_guide)
            backend.write_text(backend.read_text().replace('[vln]', '[vln]\ninstruction_guide = "Use Go toward with a clear destination."'))
            settings = load_settings(backend_file=backend, env_file=Path(directory) / '.env', environ={})
            self.assertEqual(settings.snapshot()['agent']['prompt']['instruction_guide'], 'Use Go toward with a clear destination.')
            self.assertEqual(self.calls, [])


    def test_lightnav_guide_is_selected_without_leaking_into_other_adapters(self):
        from navigation_agent.vln.lightnav.instructions import INSTRUCTION_GUIDE
        settings = load_settings(env_file=Path('/nonexistent.env'), environ={})
        self.assertEqual(settings.agent.prompt.instruction_guide, INSTRUCTION_GUIDE)
        self.assertNotEqual(self.adapter.instruction_guide, INSTRUCTION_GUIDE)
        with self.assertRaises(ValueError):
            vln.register_vln('empty_guide', vln.VlnAdapter(
                options=Options, create=self.adapter.create, instruction_guide=' ',
            ))
