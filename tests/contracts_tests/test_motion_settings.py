"""Runtime settings use the actual robot protocol without acquiring motion control."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from navigation_agent.app.go2 import motion_settings as motion
from navigation_agent.app.deployments.factory import create_backend
from navigation_agent.app.deployments.settings import load_backend, PROJECT_ROOT
from tests.config_support import test_runtime
from dataclasses import replace
from navigation_agent.app.agent import api
from fastapi import HTTPException
from types import SimpleNamespace


class MotionSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from navigation_agent.app.deployments.runtime import Deployment
        deployment = patch.object(api, 'deployment', Deployment(api))
        deployment.start()
        self.addCleanup(deployment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name) / 'motion.json'
        self.config = {**motion.MpcSettings().model_dump(), 'q_x':10., 'q_y':10., 'q_yaw':1., 'r_v':.1, 'r_w':.1}
        self.body = motion.BodySettings().model_dump()
        self.commands = []
        self.fail_body_once = False
        self.source = 'disabled'
        self.version = 1
        self.ignore_body_once = False
        app = web.Application()
        async def health(request):
            return web.json_response({'agent':{'motion_settings_version':self.version},
                'control_source':self.source,'mpc_config':{'available':True,**self.config},'robot':self.body})
        async def websocket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                data = message.json(); self.commands.append(data)
                kind=data['type']; ok=True
                if kind=='set_mpc_config':
                    self.config = data['config']; reply='mpc_config'
                elif kind=='set_body_limits':
                    reply='body_limits'
                    if self.fail_body_once:
                        ok=False; self.fail_body_once=False
                    elif self.ignore_body_once:
                        self.ignore_body_once=False
                    else:
                        self.body=data['config']
                else:
                    raise AssertionError('No control or motion commands are allowed')
                await socket.send_json({'type':'command_result','command':reply,'ok':ok,'message':'test rejection'})
            return socket
        app.router.add_get('/api/health',health);app.router.add_get('/ws',websocket)
        self.server=TestServer(app);await self.server.start_server()
        self.url=str(self.server.make_url('/')).rstrip('/')
        self.session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2))

    async def asyncTearDown(self):
        await self.session.close();await self.server.close()
        self.temp.cleanup()

    def changed(self):
        return motion.MotionSettings(look={'look_angle_deg':45.},
                                     mpc={'objnav_v_max':.2},body={'linear_limit':.25})

    async def test_apply_readback_persistence_and_backend_use(self):
        settings=self.changed()
        await motion.apply(self.session,self.url,settings,save=True,settings_file=self.file)
        self.assertEqual(self.config['objnav_v_max'],.2)
        self.assertEqual(self.body['linear_limit'],.25)
        self.assertEqual(motion.load_saved(self.file,self.url),settings)
        with patch.dict(os.environ,{'VLN_AGENT_CONFIG':'configs/go2-lightnav.toml'},clear=True):
            backend=create_backend(load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml'), saved_motion=settings)
            self.assertEqual(backend.look_angle_deg,45.)
            self.assertFalse(hasattr(backend, 'navigation_seconds'))
        # Simulate robot restart and reapply saved settings before a task.
        self.config['objnav_v_max']=.3;self.body['linear_limit']=.3
        await motion.apply(self.session,self.url,motion.load_saved(self.file,self.url))
        self.assertEqual(self.config['objnav_v_max'],.2)
        self.assertEqual(self.body['linear_limit'],.25)
        self.assertTrue(all(x['type'] in ('set_mpc_config','set_body_limits') for x in self.commands))

    async def test_partial_failure_rolls_back_and_preserves_saved_file(self):
        original=motion.MotionSettings();motion.persist(original,self.file,self.url)
        self.fail_body_once=True
        with self.assertRaisesRegex(ValueError,'已恢复原参数'):
            await motion.apply(self.session,self.url,self.changed(),save=True,settings_file=self.file)
        self.assertEqual(self.config['objnav_v_max'],.3)
        self.assertEqual(self.body['linear_limit'],.3)
        self.assertEqual(motion.load_saved(self.file,self.url),original)

    async def test_ack_without_actual_change_is_not_saved(self):
        self.ignore_body_once=True
        with self.assertRaisesRegex(ValueError,'已恢复原参数'):
            await motion.apply(self.session,self.url,self.changed(),save=True,settings_file=self.file)
        self.assertFalse(self.file.exists())
        self.assertEqual(self.config['objnav_v_max'],.3)

    async def test_busy_or_old_robot_receives_no_update(self):
        for source,version in [('manual',1),('auto',1),('disabled',0)]:
            self.source,self.version=source,version
            with self.assertRaises(ValueError):
                await motion.apply(self.session,self.url,self.changed(),save=True,settings_file=self.file)
        self.assertEqual(self.commands,[])
        self.assertFalse(self.file.exists())

    async def test_disk_failure_restores_robot(self):
        with patch.object(motion,'persist',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(ValueError,'disk full'):
                await motion.apply(self.session,self.url,self.changed(),save=True,settings_file=self.file)
        self.assertEqual(self.config['objnav_v_max'],.3)
        self.assertEqual(self.body['linear_limit'],.3)

    async def test_api_reads_existing_robot_values_without_overwriting_them(self):
        self.config['objnav_v_max']=.6
        backend = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        backend = backend.model_copy(update={'recording':None, 'robot':backend.robot.model_copy(update={'base_url':self.url,'motion_settings_file':self.file})})
        runtime = replace(test_runtime(), backend=backend)
        with patch.object(api,'current_settings',return_value=runtime), patch.object(
            api.app.state,'monitor_http',self.session,create=True):
            result=await api.deployment.get_motion_settings()
        self.assertEqual(result['values']['mpc']['objnav_v_max'],.6)
        self.assertFalse(result['saved'])
        self.assertEqual(self.commands,[])

    async def test_api_rejects_parameter_updates_while_task_running(self):
        with patch.object(api,'worker',SimpleNamespace(done=lambda:False)):
            with self.assertRaises(HTTPException) as caught:
                await api.deployment.set_motion_settings(self.changed(),SimpleNamespace(headers={}))
        self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(self.commands,[])

    async def test_task_restore_failure_prevents_task_and_connection(self):
        from unittest.mock import AsyncMock
        backend=load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        backend=backend.model_copy(update={'robot':backend.robot.model_copy(update={'base_url':self.url,'motion_settings_file':self.file})})
        runtime=replace(test_runtime(),backend=backend,api_key='test-only')
        motion.persist(self.changed(),self.file,self.url)
        with patch.object(api,'current_settings',return_value=runtime), patch.object(api,'worker',None), \
             patch.object(api.app.state,'monitor_http',self.session,create=True), \
             patch.object(api.deployment,'create_tools') as factory, \
             patch.object(motion,'apply',new=AsyncMock(side_effect=ValueError('restore failed'))):
            with self.assertRaises(HTTPException) as error:
                await api.start_task(api.TaskRequest(goal='测试'))
            self.assertEqual(error.exception.status_code,503)
            factory.assert_not_called()

    def test_malformed_or_incomplete_saved_state_is_rejected(self):
        for data in ([],None,{'schema_version':2,'robot_url':None,'values':{}},
                     {'schema_version':2,'robot_url':self.url,'values':{'look':{},'mpc':{},'body':{}}}):
            self.file.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                motion.load_saved(self.file,self.url)


    def test_saved_default_template_uses_backend_guide_without_changing_motion(self):
        from navigation_agent.contracts.runtime import PromptSettings
        from navigation_agent.vln.lightnav.instructions import INSTRUCTION_GUIDE

        settings = self.changed()
        motion.persist(settings, self.file, self.url)
        raw = json.loads(self.file.read_text())
        raw['values']['prompt'] = {'instruction_template': 'Walk straight toward {target} {location} and stop.'}
        self.file.write_text(json.dumps(raw))
        original = self.file.read_bytes()
        loaded = motion.load_saved(
            self.file, self.url, prompt_defaults=PromptSettings(instruction_guide=INSTRUCTION_GUIDE),
        )
        self.assertEqual(loaded.prompt.instruction_guide, INSTRUCTION_GUIDE)
        self.assertEqual(loaded.mpc, settings.mpc)
        self.assertEqual(loaded.body, settings.body)
        self.assertEqual(self.file.read_bytes(), original)
        self.assertEqual(self.commands, [])
        raw['values']['prompt']['instruction_template'] = 'Approach {target} {location}.'
        self.file.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'instruction_guide'):
            motion.load_saved(self.file, self.url)

    def test_legacy_observation_limit_preserves_saved_values_and_file(self):
        from navigation_agent.app.deployments.runtime_overrides import apply_agent_settings

        settings = self.changed()
        settings.agent.max_steps = 17
        settings.look.navigation_seconds = 23.
        settings.mpc.q_x = 73.
        motion.persist(settings, self.file, self.url)
        raw = json.loads(self.file.read_text())
        raw['values']['exploration']['max_looks'] = 6
        raw['preserve_mpc_weights'] = ['q_y']
        self.file.write_text(json.dumps(raw))
        original = self.file.read_bytes()

        loaded = motion.load_saved(self.file, self.url)
        self.assertEqual(loaded.model_dump(), settings.model_dump())
        self.assertNotIn('q_y', loaded.mpc.model_fields_set)
        self.assertNotIn('max_looks', loaded.exploration.model_dump())
        effective = apply_agent_settings(test_runtime().agent, loaded)
        self.assertEqual(effective.task_defaults.max_steps, 17)
        self.assertEqual(effective.execution.navigation_seconds, 23.)
        self.assertEqual(self.file.read_bytes(), original)
        self.assertEqual(self.commands, [])

    def test_legacy_limit_does_not_hide_invalid_saved_parameters(self):
        for invalid in ('missing_motion', 'missing_exploration', 'unknown_field',
                        'out_of_range', 'wrong_robot'):
            with self.subTest(invalid=invalid):
                motion.persist(self.changed(), self.file, self.url)
                raw = json.loads(self.file.read_text())
                raw['values']['exploration']['max_looks'] = 6
                if invalid == 'missing_motion':
                    raw['values']['body'].pop('linear_limit')
                elif invalid == 'missing_exploration':
                    raw['values']['exploration'].pop('recovery_attempts')
                elif invalid == 'unknown_field':
                    raw['values']['exploration']['unknown_limit'] = 1
                elif invalid == 'out_of_range':
                    raw['values']['body']['linear_limit'] = 99.
                else:
                    raw['robot_url'] = 'http://another:8088'
                self.file.write_text(json.dumps(raw))
                original = self.file.read_bytes()
                with self.assertRaises(ValueError):
                    motion.load_saved(self.file, self.url)
                self.assertEqual(self.file.read_bytes(), original)
        self.assertEqual(self.commands, [])

    def test_startup_reports_invalid_saved_field_without_its_value(self):
        import io
        from contextlib import redirect_stderr
        from navigation_agent.app import main

        motion.persist(self.changed(), self.file, self.url)
        raw = json.loads(self.file.read_text())
        raw['values']['body']['linear_limit'] = 'private-invalid-value'
        self.file.write_text(json.dumps(raw))
        backend = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        backend = backend.model_copy(update={'robot': backend.robot.model_copy(update={
            'base_url': self.url, 'motion_settings_file': self.file})})
        runtime = replace(test_runtime(), backend=backend)
        error = io.StringIO()
        with patch.object(main, 'load_settings', return_value=runtime), \
             patch('sys.argv', ['main.py', '--check-config']), redirect_stderr(error):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('body', error.getvalue())
        self.assertIn('linear_limit', error.getvalue())
        self.assertNotIn('private-invalid-value', error.getvalue())

    def test_invalid_and_contradictory_settings_rejected(self):
        for value in (True,0,-1,float('nan'),float('inf'),2.):
            with self.assertRaises(ValidationError):
                motion.BodySettings(linear_limit=value)
        with self.assertRaises(ValidationError):
            motion.MotionSettings(look={'look_angular_min':.5,'look_angular_max':.4})
        with self.assertRaises(ValidationError):
            motion.MotionSettings(body={'angular_limit':.1})

    async def test_saved_runtime_reaches_task_snapshot_and_preserves_explicit_budget(self):
        from unittest.mock import AsyncMock
        from navigation_agent.app.deployments.runtime_overrides import motion_defaults, apply_agent_settings
        backend = load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        backend = backend.model_copy(update={'recording':None, 'robot':backend.robot.model_copy(update={
            'base_url':self.url, 'motion_settings_file':self.file})})
        runtime = replace(test_runtime(Path(self.temp.name)/'runs'), backend=backend, api_key='test-only')
        settings = motion_defaults(runtime)
        self.assertEqual(
            apply_agent_settings(runtime.agent, settings),
            runtime.agent.model_copy(update={'task_defaults': backend.ui.task_defaults}),
        )
        settings.agent.max_steps=7
        settings.agent.max_seconds=80
        settings.agent.decision_timeout_s=31.
        settings.agent.max_tokens=1024
        settings.look.navigation_seconds=27.
        settings.look.look_gain=2.
        settings.exploration.recovery_attempts=0
        settings.memory.view_angle_deg=12.
        settings.exploration.view_angle_deg=18.
        settings.protection.command_timeout_s=4.
        settings.protection.stall_window_s=2.
        settings.prompt.instruction_guide='使用 Approach 描述明确目标，方向和停止要求按需要添加。'
        motion.persist(settings,self.file,self.url)
        with patch.object(api,'current_settings',return_value=runtime), patch.object(api,'worker',None), \
             patch.object(api.app.state,'monitor_http',self.session,create=True), \
             patch.object(api.deployment,'create_tools') as factory, patch.object(api,'run_task',new_callable=AsyncMock) as runner:
            record=await api.start_task(api.TaskRequest(goal='offline fixture',navigation_seconds=9.))
            await api.worker
            effective=runner.call_args.kwargs['settings']
            self.assertEqual((record['max_steps'], record['max_seconds'],record['navigation_seconds']),(7,80,9.))
            self.assertEqual(effective.execution.navigation_seconds,27.)
            self.assertEqual(effective.llm.request_timeout_s,31.)
            self.assertEqual(effective.llm.max_tokens,1024)
            self.assertEqual(effective.llm.base_url,runtime.agent.llm.base_url)
            self.assertEqual(effective.storage,runtime.agent.storage)
            self.assertEqual(effective.task_limits,runtime.agent.task_limits)
            self.assertEqual(effective.prompt.instruction_guide,settings.prompt.instruction_guide)
            self.assertEqual(effective.exploration.max_recovery_attempts,0)
            self.assertEqual(effective.memory.same_view_angle_deg,12.)
            self.assertEqual(effective.observation.same_view_angle_deg,18.)
            loaded=factory.call_args.kwargs['motion']
            assembled=create_backend(backend,saved_motion=loaded)
            self.assertEqual(assembled.robot.protection.command_timeout_s,4.)
            self.assertEqual(assembled.progress.settings.stall_window_s,2.)
            self.assertEqual(record['configuration']['saved_motion'],loaded.model_dump())
            self.assertNotIn('test-only',json.dumps(record))
            self.assertEqual(api.get_agent_settings()['navigation_seconds'],27.)
        api.records.pop(record['id'],None)

    async def test_saved_values_cannot_bypass_dev_task_limits(self):
        backend=load_backend(PROJECT_ROOT / 'configs/go2-lightnav.toml')
        backend=backend.model_copy(update={'robot':backend.robot.model_copy(update={
            'base_url':self.url,'motion_settings_file':self.file})})
        runtime=replace(test_runtime(),backend=backend)
        runtime=replace(runtime,agent=runtime.agent.model_copy(update={
            'task_limits':runtime.agent.task_limits.model_copy(update={'max_navigation_seconds':20.})}))
        settings=self.changed()
        settings.look.navigation_seconds=27.
        with patch.object(api,'current_settings',return_value=runtime), patch.object(api,'worker',None):
            with self.assertRaises(HTTPException):
                await api.deployment.set_motion_settings(settings,SimpleNamespace(headers={}))
        self.assertEqual(self.commands,[])
        self.assertFalse(self.file.exists())

    async def test_migrated_missing_mpc_weights_preserve_live_robot_values(self):
        settings=self.changed()
        motion.persist(settings,self.file,self.url)
        raw=json.loads(self.file.read_text())
        raw['preserve_mpc_weights']=['q_x','q_y','q_yaw','r_v','r_w']
        self.file.write_text(json.dumps(raw))
        self.config['q_x']=73.
        applied=await motion.apply(self.session,self.url,motion.load_saved(self.file,self.url),
                                   save=True,settings_file=self.file)
        self.assertEqual(self.config['q_x'],73.)
        self.assertEqual(applied['mpc']['q_x'],73.)
        self.assertEqual(motion.load_saved(self.file,self.url).mpc.q_x,73.)
        self.assertNotIn('preserve_mpc_weights',json.loads(self.file.read_text()))
