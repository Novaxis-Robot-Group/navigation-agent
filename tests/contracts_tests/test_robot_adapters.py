"""后端在读取配置前显式登记；配置检查不连接服务。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import model_validator
from navigation_agent.app.deployments import factory
from navigation_agent.app.deployments.settings import BackendSettings, load_backend
from tests.config_support import toml_text


class RobotAdapterTests(unittest.TestCase):
    def test_custom_configuration_and_constructor_are_selected_together(self):
        class CustomSettings(BackendSettings):
            custom_scale: int

            @model_validator(mode='after')
            def positive_scale(self):
                if self.custom_scale <= 0:
                    raise ValueError('custom_scale 必须为正数')
                return self

        calls = []
        result = object()
        def create(settings, **kwargs):
            calls.append((settings, kwargs))
            return result

        with patch.dict(factory._ROBOTS), tempfile.TemporaryDirectory() as directory:
            factory.register_robot('example_robot', factory.RobotAdapter(CustomSettings, create))
            data = load_backend().model_dump(mode='json')
            data['robot']['backend'] = 'example_robot'
            data['simulation'] = None
            data['custom_scale'] = 2
            profile = Path(directory) / 'backend.toml'
            profile.write_text(toml_text(data))
            settings = load_backend(profile)
            self.assertIsInstance(settings, CustomSettings)
            self.assertEqual(calls, [])
            self.assertIs(factory.create_backend(settings, base_url='http://example:8088'), result)
            self.assertEqual(calls[0][0].custom_scale, 2)
            self.assertEqual(calls[0][1]['base_url'], 'http://example:8088')
            data['custom_scale'] = 0
            profile.write_text(toml_text(data))
            with self.assertRaises(ValueError):
                load_backend(profile)
            self.assertEqual(len(calls), 1)

    def test_duplicate_name_and_wrong_configuration_model_are_rejected(self):
        with self.assertRaises(ValueError):
            factory.register_robot('go2', factory.get_robot('go2'))
        with self.assertRaises(TypeError):
            factory.register_robot('invalid', factory.RobotAdapter(dict, lambda: None))
