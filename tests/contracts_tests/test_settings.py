"""配置验证必须在连接设备之前完成。"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import load_backend, load_settings, PROJECT_ROOT, LEGACY_VARIABLES
from navigation_agent.app.agent.settings import load_agent, read_toml
from tests.config_support import toml_text


class SettingsTests(unittest.TestCase):

    def test_go2_defaults_storage_and_navigation_are_consistent(self):
        from navigation_agent.app.deployments.runtime_overrides import motion_defaults
        runtime = load_settings(backend_file=PROJECT_ROOT / 'configs/go2-lightnav.toml', environ={})
        self.assertEqual(runtime.agent.storage.runs_dir, PROJECT_ROOT / 'runs/go2')
        self.assertFalse(runtime.backend.navigation.time_limit_enabled)
        self.assertFalse(runtime.agent.prompt.navigation_time_limit_enabled)
        self.assertEqual(motion_defaults(runtime).agent.max_seconds, runtime.backend.ui.task_defaults.max_seconds)
        self.assertEqual(motion_defaults(runtime).agent.max_steps, runtime.backend.ui.task_defaults.max_steps)


    def test_backend_detector_address_overrides_only_endpoint_and_preserves_lightnav_guide(self):
        from navigation_agent.agent.context.prompts import navigation_prompt, read_prompt
        from navigation_agent.vln.lightnav.instructions import INSTRUCTION_GUIDE
        with tempfile.TemporaryDirectory() as directory:
            agent = read_toml(PROJECT_ROOT / 'configs/agent.toml')
            agent['prompt']['path'] = str(PROJECT_ROOT / 'configs/prompts/common.md')
            agent['prompt']['algorithm_path'] = str(PROJECT_ROOT / 'configs/prompts/scheduling.md')
            agent['target_watch'].update(server_url='http://old-detector:8060', confidence=.4)
            agent_file = Path(directory) / 'agent.toml'
            agent_file.write_text(toml_text(agent))
            backend = load_backend().model_dump(mode='json')
            backend_file = Path(directory) / 'backend.toml'
            for endpoint in ('', 'http://127.0.0.1:18060'):
                backend['target_watch'] = {'server_url': endpoint}
                backend_file.write_text(toml_text(backend))
                runtime = load_settings(agent_file, backend_file,
                                        env_file=Path(directory) / '.env', environ={})
                self.assertEqual(runtime.agent.target_watch.server_url, endpoint)
                self.assertEqual(runtime.agent.target_watch.confidence, .4)
                self.assertEqual(runtime.agent.prompt.instruction_guide, INSTRUCTION_GUIDE)
                prompt = navigation_prompt(read_prompt(runtime.agent.prompt), seconds=15,
                                           guide=runtime.agent.prompt.instruction_guide)
                self.assertIn(INSTRUCTION_GUIDE, prompt)
            backend.pop('target_watch')
            backend_file.write_text(toml_text(backend))
            runtime = load_settings(agent_file, backend_file,
                                    env_file=Path(directory) / '.env', environ={})
            self.assertEqual(runtime.agent.target_watch.server_url, 'http://old-detector:8060')

    def test_backend_detector_rejects_invalid_urls_and_agent_sampling_parameters(self):
        from navigation_agent.app.deployments.settings import TargetWatchConnectionSettings
        for endpoint in ('ws://localhost:8060', 'http://user:secret@localhost:8060',
                         'http://localhost:8060/path', 'http://localhost:99999',
                         'http://localhost:8060?token=secret'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                TargetWatchConnectionSettings(server_url=endpoint)
        with self.assertRaises(ValueError):
            TargetWatchConnectionSettings(server_url='', confidence=.3)

    def test_go2_turn_angle_comes_from_profile_and_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'go2.toml'
            source = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml').model_dump(mode='json')
            for angle in (45, 0, 91, True):
                source['robot']['look_angle_deg'] = angle
                profile.write_text(toml_text(source))
                if angle == 45:
                    self.assertEqual(create_backend(load_backend(profile)).look_angle_deg, 45)
                else:
                    with self.assertRaises(ValueError):
                        load_backend(profile)

    def test_legacy_variables_are_rejected_even_if_process_masks_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            for name in LEGACY_VARIABLES:
                env.write_text(name + '=legacy\nDASHSCOPE_API_KEY=secret')
                with self.assertRaisesRegex(ValueError, name):
                    load_settings(env_file=env, environ={name:''})

    def test_default_profile_does_not_depend_on_working_directory(self):
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(load_backend().robot.backend, 'mujoco')
                self.assertTrue(load_agent().prompt.path.is_file())
                os.chdir(original)
        finally:
            os.chdir(original)

    def test_invalid_types_unknown_fields_and_nonfinite_values(self):
        from navigation_agent.agent.config import AgentSettings
        source = load_agent().model_dump()
        for section, key, value in [('llm','max_tokens',True), ('llm','request_timeout_s',float('nan')),
                ('llm','format_retries',-1), ('memory','nearby_radius_m',float('inf')),
                ('memory','unknown',1), ('task_limits','min_seconds',1000),
                ('task_limits','max_navigation_seconds',61), ('execution','navigation_seconds',100)]:
            with self.subTest(key=key):
                data = {**source,section:{**source[section],key:value}}
                with self.assertRaises(ValueError):
                    AgentSettings.model_validate(data)

    def test_unregistered_robot_or_model_rejected_before_connecting(self):
        from navigation_agent.app.deployments.settings import BackendSettings
        for group in ('robot','vln'):
            source = load_backend().model_dump()
            source[group]['backend'] = 'unregistered'
            with patch('navigation_agent.app.deployments.factory.ConsoleConnection') as channel:
                with self.assertRaises(ValueError):
                    BackendSettings.model_validate(source)
                channel.assert_not_called()

    def test_snapshot_excludes_secret_and_configuration_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / '.env'
            env.write_text(load_agent().llm.api_key_env + '=super-secret')
            runtime = load_settings(env_file=env, environ={})
            self.assertEqual(runtime.api_key, 'super-secret')
            self.assertNotIn('super-secret', str(runtime.snapshot()))
            self.assertNotIn('super-secret', repr(runtime))
            with self.assertRaises(ValueError):
                runtime.agent.execution.navigation_seconds = 12

    def test_common_and_go2_prompts_are_selected_by_backend_configuration(self):
        from navigation_agent.agent.context.prompts import read_prompt
        with tempfile.TemporaryDirectory() as directory:
            env=Path(directory)/'.env'
            for kind in ('go2','mujoco'):
                runtime=load_settings(backend_file=PROJECT_ROOT/f'configs/{kind}-lightnav.toml',
                                      env_file=env,environ={})
                prompt=read_prompt(runtime.agent.prompt)
                self.assertEqual(runtime.agent.prompt.path,PROJECT_ROOT/'configs/prompts/common.md')
                self.assertEqual('在座人员' in prompt,kind=='go2')
                if kind=='go2':
                    self.assertEqual(runtime.agent.prompt.scene.navigation_policy,'go2_office')
                    self.assertEqual(runtime.snapshot()['agent']['prompt']['scene']['path'],
                                     str(PROJECT_ROOT/'configs/prompts/go2-office.md'))
                else:
                    self.assertIsNone(runtime.agent.prompt.scene)

    def test_prompt_timing_uses_backend_configuration_for_both_modes(self):
        from navigation_agent.agent.context.prompts import navigation_prompt, read_prompt
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'backend.toml'
            for kind in ('go2', 'mujoco'):
                data = load_backend(PROJECT_ROOT / f'configs/{kind}-lightnav.toml').model_dump(mode='json')
                for enabled in (False, True):
                    with self.subTest(kind=kind, enabled=enabled):
                        data['navigation']['time_limit_enabled'] = enabled
                        profile.write_text(toml_text(data))
                        runtime = load_settings(backend_file=profile,
                                                env_file=Path(directory) / '.env', environ={})
                        self.assertEqual(runtime.agent.prompt.navigation_time_limit_enabled, enabled)
                        self.assertEqual(create_backend(runtime.backend).time_limit_enabled, enabled)
                        prompt = navigation_prompt(
                            read_prompt(runtime.agent.prompt), seconds=23,
                            guide=runtime.agent.prompt.instruction_guide,
                            time_limit_enabled=runtime.agent.prompt.navigation_time_limit_enabled,
                        )
                        self.assertIn('最多 23 秒' if enabled else '不设固定上限', prompt)
                        if not enabled:
                            self.assertNotIn('最多 23 秒', prompt)
                        self.assertNotIn('{{', prompt)

    def test_scene_prompt_paths_are_relative_and_invalid_policy_fails_early(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'prompts').mkdir()
            shutil.copyfile(PROJECT_ROOT/'configs/prompts/go2-office.md',root/'prompts/office.md')
            data=load_backend(PROJECT_ROOT/'configs/go2-lightnav.toml').model_dump(mode='json')
            data['prompt']['path']='prompts/office.md'
            profile=root/'backend.toml'
            profile.write_text(toml_text(data))
            self.assertEqual(load_backend(profile).prompt.path,root/'prompts/office.md')
            data['prompt']['navigation_policy']='unknown'
            profile.write_text(toml_text(data))
            with self.assertRaises(ValueError):load_backend(profile)
            data['prompt']['navigation_policy']='go2_office'
            data['prompt']['path']='missing.md'
            profile.write_text(toml_text(data))
            with self.assertRaisesRegex(ValueError,'Prompt 文件不存在'):load_backend(profile)

    def test_navigation_max_attempts_is_validated_and_wired(self):
        from navigation_agent.app.deployments.settings import VlnSettings
        source = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        self.assertEqual(create_backend(source).max_attempts, 3)
        single = source.model_copy(update={'vln': source.vln.model_copy(update={'max_attempts': 1})})
        self.assertEqual(create_backend(single).max_attempts, 1)
        for value in (0, -1, 6, True, 1.5, '3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                VlnSettings(backend='lightnav', server_url='ws://localhost:8052', max_attempts=value)


class TaskNavigationLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_override_reaches_backend_prompt_and_record_without_persisting(self):
        from dataclasses import replace
        from unittest.mock import AsyncMock
        from navigation_agent.app.agent import api
        from navigation_agent.app.deployments.runtime import Deployment
        from navigation_agent.app.go2 import motion_settings
        from navigation_agent.agent.context.prompts import navigation_prompt

        config_paths = [PROJECT_ROOT / 'configs' / name for name in
                        ('agent.toml', 'mujoco-lightnav.toml', 'go2-lightnav.toml')]
        original_files = {path: path.read_bytes() for path in config_paths}
        with tempfile.TemporaryDirectory() as directory:
            for robot in ('mujoco', 'go2'):
                base = load_settings(backend_file=PROJECT_ROOT / f'configs/{robot}-lightnav.toml', environ={})
                # 固定部署为开启，验证页面显式关闭确实覆盖它；所有设备交互均替换。
                backend = base.backend.model_copy(update={
                    'navigation': base.backend.navigation.model_copy(update={'time_limit_enabled': True}),
                    'recording': None,
                })
                if robot == 'mujoco':
                    backend = backend.model_copy(update={'simulation': backend.simulation.model_copy(update={
                        'recording': backend.simulation.recording.model_copy(update={'enabled': False}),
                    })})
                saved_file = Path(directory) / 'motion.json'
                if robot == 'go2':
                    backend = backend.model_copy(update={'robot': backend.robot.model_copy(update={
                        'motion_settings_file': saved_file,
                    })})
                    saved = motion_settings.MotionSettings()
                    saved.look.navigation_seconds = 23.
                    motion_settings.persist(saved, saved_file, backend.robot.base_url)
                    original_saved = saved_file.read_bytes()
                agent = base.agent.model_copy(update={
                    'prompt': base.agent.prompt.model_copy(update={'navigation_time_limit_enabled': True}),
                    'storage': base.agent.storage.model_copy(update={'runs_dir': Path(directory) / robot}),
                })
                runtime = replace(base, backend=backend, agent=agent, api_key='test-only')
                baseline = runtime.snapshot()
                deployment = Deployment(api)
                with patch.object(api, 'current_settings', return_value=runtime), \
                     patch.object(api, 'deployment', deployment), patch.object(api, 'worker', None), \
                     patch.object(api.app.state, 'monitor_http', None, create=True), \
                     patch.object(motion_settings, 'apply', new_callable=AsyncMock,
                                  side_effect=lambda http, url, value: value.model_dump()), \
                     patch.object(api, 'run_task', new_callable=AsyncMock) as runner:
                    # 连续任务必须独立；省略开关的 API 调用仍沿用部署值。
                    for mode in (False, True, False, None):
                        with self.subTest(robot=robot, mode=mode):
                            request = api.TaskRequest(goal='离线验证', navigation_seconds=31.,
                                                      navigation_time_limit_enabled=mode)
                            record = await api.start_task(request)
                            await api.worker
                            expected = True if mode is None else mode
                            tools = runner.call_args.args[2]
                            settings = runner.call_args.kwargs['settings']
                            self.assertEqual(tools.backend.time_limit_enabled, expected)
                            self.assertEqual(settings.prompt.navigation_time_limit_enabled, expected)
                            self.assertEqual(record['navigation_time_limit_enabled'], expected)
                            self.assertEqual(record['navigation_seconds'], 31.)
                            self.assertEqual(record['configuration']['backend']['navigation']['time_limit_enabled'], expected)
                            prompt = navigation_prompt('{{navigation_time_limit}}', seconds=31.,
                                guide='', time_limit_enabled=settings.prompt.navigation_time_limit_enabled)
                            self.assertEqual(prompt, '最多 31 秒' if expected else '不设固定上限')
                            self.assertEqual(runtime.snapshot(), baseline)
                            self.assertTrue(api.get_agent_settings()['navigation_time_limit_enabled'])
                            if robot == 'go2':
                                self.assertEqual(saved_file.read_bytes(), original_saved)
                                self.assertEqual(api.get_agent_settings()['navigation_seconds'], 23.)
                            api.records.pop(record['id'], None)
        for path, content in original_files.items():
            self.assertEqual(path.read_bytes(), content)

    def test_task_navigation_mode_rejects_ambiguous_values(self):
        from navigation_agent.agent.schemas import TaskRequest
        for value in ('false', 'true', 0, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TaskRequest(goal='测试', navigation_time_limit_enabled=value)

    async def test_standalone_backend_does_not_silently_ignore_mode_override(self):
        from fastapi import HTTPException
        from navigation_agent.agent.schemas import TaskRequest
        from navigation_agent.app.agent.settings import StandaloneDeployment, standalone_settings
        with self.assertRaises(HTTPException) as error:
            await StandaloneDeployment().prepare_task(
                standalone_settings(), TaskRequest(goal='测试', navigation_time_limit_enabled=False))
        self.assertEqual(error.exception.status_code, 422)
